"""Feature engineering: dai mazzi grezzi (metastats.net + HSReplay, notebooks/02 e 07)
al dataset per il fine-tuning del classificatore power level. Unisce gli snapshot
(dedup per deck_id, tenendo la riga con piu' partite), scarta mazzi senza decklist o
sotto MIN_GAMES partite, incrocia le carte con HearthstoneJSON per costruire le
feature (curva di mana, rarita'/tipo, meccaniche chiave) con il winrate come label.
Azalina Soulsever (20 carte) e Timethief Rafaam (40) sono inclusi con n_cards_total/
has_special_deckbuild come feature esplicite invece di essere scartati per struttura
di mazzo diversa dalle 30 carte standard. Output: data/processed/finetune_dataset.csv.

Come eseguirlo:
    python -u notebooks/03_build_finetune_dataset.py
"""

import ast
import glob
import json
import os

import pandas as pd
import requests

RAW_DIR = os.path.join("data", "raw", "metastats")
RAW_DIR_HSREPLAY = os.path.join("data", "raw", "hsreplay")
HSJSON_DIR = os.path.join("data", "raw", "hearthstonejson")
HSJSON_PATH = os.path.join(HSJSON_DIR, "cards.json")
HSJSON_URL = "https://api.hearthstonejson.com/v1/latest/enUS/cards.json"

OUT_DIR = os.path.join("data", "processed")
OUT_PATH = os.path.join(OUT_DIR, "finetune_dataset.csv")

MIN_GAMES = 30  # sotto questa soglia il winrate è considerato troppo rumoroso

# Gli snapshot metastats.net sono a cavallo di una patch di bilanciamento (label di
# winrate incoerenti nel tempo), HSReplay.net e' internamente coerente - si usa quindi
# solo HSReplay.net; load_all_snapshots resta nel codice per un eventuale confronto.
USE_METASTATS_DATA = False

# Costo in polvere arcana per craftare una copia non dorata di ogni rarita' (valori
# fissi del gioco) - calcolabile in modo esatto da decklist+rarita', senza fonte esterna.
DUST_COST_BY_RARITY = {"COMMON": 40, "RARE": 100, "EPIC": 400, "LEGENDARY": 1600, "OTHER": 0}
# Carte che cambiano le regole di costruzione del mazzo (vedi nota nel docstring del
# modulo) - usate per marcare i mazzi con has_special_deckbuild, non per scartarli.
SPECIAL_DECKBUILD_CARD_NAMES = {"Azalina Soulsever", "Timethief Rafaam"}
MECHANICS_OF_INTEREST = [
    "TAUNT", "DEATHRATTLE", "BATTLECRY", "RUSH", "DIVINE_SHIELD", "COMBO", "LIFESTEAL",
]


def log(msg: str):
    print(msg, flush=True)


def safe_literal_eval(val):
    """cells / cards_id_count sono salvati nei CSV come repr() di liste Python —
    vanno riletti con ast.literal_eval, non con json.loads (virgolette singole)."""
    if val is None or (isinstance(val, float)):  # NaN da pandas
        return None
    val = str(val).strip()
    if not val:
        return None
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return None


def download_hearthstonejson() -> dict:
    """Scarica (o riusa la cache locale) il database completo delle carte, indicizzato
    per id HearthstoneJSON (stringa, es. "CORE_DS1_185") — è lo stesso id che leggiamo
    dal filename dell'immagine sulla pagina di metastats.net."""
    os.makedirs(HSJSON_DIR, exist_ok=True)
    if not os.path.exists(HSJSON_PATH):
        log(f"Scarico HearthstoneJSON da {HSJSON_URL} ...")
        resp = requests.get(HSJSON_URL, timeout=30)
        resp.raise_for_status()
        with open(HSJSON_PATH, "w", encoding="utf-8") as f:
            f.write(resp.text)
        log(f"Salvato in {HSJSON_PATH} ({len(resp.text)} caratteri).")
    else:
        log(f"Uso la cache locale {HSJSON_PATH} (cancellala per riscaricare una versione aggiornata).")

    with open(HSJSON_PATH, encoding="utf-8") as f:
        cards = json.load(f)

    by_id = {}
    for c in cards:
        cid = c.get("id")
        if cid:
            by_id[cid] = c
    log(f"Carte indicizzate per id: {len(by_id)}")
    return by_id


def find_special_deckbuild_ids(card_lookup: dict) -> set:
    """Trova gli id HearthstoneJSON delle carte in SPECIAL_DECKBUILD_CARD_NAMES,
    cercando per nome invece di hardcodare gli id (piu' robusto a rotazioni/ristampe)."""
    ids = {c["id"] for c in card_lookup.values() if c.get("name") in SPECIAL_DECKBUILD_CARD_NAMES}
    found_names = {card_lookup[i].get("name") for i in ids}
    missing = SPECIAL_DECKBUILD_CARD_NAMES - found_names
    if missing:
        log(f"  [WARN] Non trovate in HearthstoneJSON: {missing} (nome cambiato? controllare SPECIAL_DECKBUILD_CARD_NAMES)")
    return ids


def card_mechanics(card: dict) -> set:
    """Il campo 'mechanics' in HearthstoneJSON è una lista di dict {'name': 'TAUNT'}
    (in alcune versioni può essere una lista di stringhe) — normalizziamo a un set di nomi."""
    mechs = card.get("mechanics", []) or []
    names = set()
    for m in mechs:
        if isinstance(m, dict):
            names.add(m.get("name", ""))
        else:
            names.add(str(m))
    return names


def load_all_snapshots() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(RAW_DIR, "decks_*.csv")))
    log(f"Snapshot trovati: {files}")
    if not files:
        raise FileNotFoundError(f"Nessun file decks_*.csv in {RAW_DIR} — lancia prima notebooks/02_scrape_metastats.py")

    frames = [pd.read_csv(f) for f in files]
    all_df = pd.concat(frames, ignore_index=True)
    log(f"Righe totali (tutti gli snapshot, con duplicati per deck_id): {len(all_df)}")

    # Estrae 'games' dalla colonna 'cells' (es. ['1', 'Cannon Warrior#61013', '877', '58.39%'])
    def extract_games(cells_raw):
        cells = safe_literal_eval(cells_raw)
        if not cells or len(cells) < 3:
            return None
        try:
            return int(cells[2])
        except (ValueError, TypeError):
            return None

    def extract_winrate(cells_raw):
        cells = safe_literal_eval(cells_raw)
        if not cells or len(cells) < 4:
            return None
        try:
            return float(str(cells[3]).replace("%", "").strip())
        except (ValueError, TypeError):
            return None

    all_df["games"] = all_df["cells"].apply(extract_games)
    all_df["winrate"] = all_df["cells"].apply(extract_winrate)
    # Assunzione da verificare a mano: la pagina metastats.net non mostra un
    # selettore di formato esplicito - si assume RANKED_STANDARD (il piu' tracciato).
    all_df["game_type"] = "RANKED_STANDARD"

    # Dedup per deck_id: tiene la riga con più partite (stima più affidabile)
    all_df = all_df.sort_values("games", ascending=False).drop_duplicates(subset="deck_id", keep="first")
    log(f"Righe dopo dedup per deck_id (tiene la versione con più partite): {len(all_df)}")
    return all_df


def load_hsreplay_snapshots() -> pd.DataFrame:
    """Carica gli snapshot di notebooks/07_scrape_hsreplay.py, se presenti. A differenza
    di load_all_snapshots() (metastats.net) qui 'cards_id_count' e' gia' nel formato
    finale (lista di tuple id-stringa/quantita', vedi quel file) e 'winrate'/'games' sono
    gia' colonne dirette (non vanno estratte da 'cells' come per metastats.net). Ritorna
    un DataFrame vuoto con le colonne giuste se lo scraper non e' ancora stato lanciato,
    cosi' questo script resta utilizzabile anche a chi non l'ha ancora eseguito."""
    cols = ["deck_id", "cards_id_count", "winrate", "games", "scrape_date", "game_type"]
    files = sorted(glob.glob(os.path.join(RAW_DIR_HSREPLAY, "decks_*.csv")))
    if not files:
        log(f"Nessuno snapshot HSReplay in {RAW_DIR_HSREPLAY} (notebooks/07_scrape_hsreplay.py non ancora lanciato) - salto questa fonte.")
        return pd.DataFrame(columns=cols)

    log(f"Snapshot HSReplay trovati: {files}")
    frames = [pd.read_csv(f) for f in files]
    all_df = pd.concat(frames, ignore_index=True)
    log(f"Righe totali HSReplay (tutti gli snapshot, con duplicati per deck_id): {len(all_df)}")

    all_df = all_df.sort_values("games", ascending=False).drop_duplicates(subset="deck_id", keep="first")
    log(f"Righe HSReplay dopo dedup per deck_id: {len(all_df)}")
    return all_df[cols]


def build_deck_features(row, card_lookup: dict, special_ids: set) -> dict | None:
    cards_raw = safe_literal_eval(row["cards_id_count"])
    if not cards_raw:
        return None  # niente decklist letta per questo mazzo

    total_cards = 0
    has_special_deckbuild = False
    cost_sum = 0
    counts_by_type = {"MINION": 0, "SPELL": 0, "WEAPON": 0, "LOCATION": 0, "OTHER": 0}
    counts_by_rarity = {"COMMON": 0, "RARE": 0, "EPIC": 0, "LEGENDARY": 0, "OTHER": 0}
    cost_curve = {f"cost_{i}": 0 for i in range(8)}  # cost_7 = "7 o più"
    attack_values, health_values = [], []
    mechanic_counts = {f"n_{m.lower()}": 0 for m in MECHANICS_OF_INTEREST}
    classes_seen = {}

    for card_id, count in cards_raw:
        card = card_lookup.get(card_id)
        if card is None:
            continue  # id carta non trovato nel dump scaricato (raro, es. carte rimosse/rinominate)
        if card_id in special_ids:
            has_special_deckbuild = True
        total_cards += count
        cost = card.get("cost", 0) or 0
        cost_sum += cost * count
        bucket = min(cost, 7)
        cost_curve[f"cost_{bucket}"] += count

        ctype = card.get("type", "OTHER")
        counts_by_type[ctype if ctype in counts_by_type else "OTHER"] += count
        rarity = card.get("rarity", "OTHER")
        counts_by_rarity[rarity if rarity in counts_by_rarity else "OTHER"] += count

        if ctype == "MINION":
            if "attack" in card:
                attack_values += [card["attack"]] * count
            if "health" in card:
                health_values += [card["health"]] * count

        mechs = card_mechanics(card)
        for m in MECHANICS_OF_INTEREST:
            if m in mechs:
                mechanic_counts[f"n_{m.lower()}"] += count

        card_class = card.get("cardClass", "NEUTRAL")
        classes_seen[card_class] = classes_seen.get(card_class, 0) + count

    if total_cards == 0:
        return None  # nessuna carta valida trovata per questo mazzo

    deck_class = max(
        ((c, n) for c, n in classes_seen.items() if c != "NEUTRAL"),
        key=lambda x: x[1], default=("NEUTRAL", 0)
    )[0]

    features = {
        "deck_id": row["deck_id"],
        "deck_class": deck_class,
        "n_cards_total": total_cards,  # 30 di norma; 20/40 per Azalina Soulsever/Timethief Rafaam (vedi nota sopra)
        "has_special_deckbuild": has_special_deckbuild,  # True se il mazzo include una carta che cambia le regole di costruzione
        "formato": "Standard" if row.get("game_type") == "RANKED_STANDARD" else "Wild",  # vedi src/agent/format_rules.py per perche' serve distinguerlo esplicitamente
        "avg_cost": round(cost_sum / total_cards, 3),
        "avg_attack": round(sum(attack_values) / len(attack_values), 3) if attack_values else None,
        "avg_health": round(sum(health_values) / len(health_values), 3) if health_values else None,
        **cost_curve,
        "n_minions": counts_by_type["MINION"],
        "n_spells": counts_by_type["SPELL"],
        "n_weapons": counts_by_type["WEAPON"],
        "n_locations": counts_by_type["LOCATION"],
        "n_common": counts_by_rarity["COMMON"],
        "n_rare": counts_by_rarity["RARE"],
        "n_epic": counts_by_rarity["EPIC"],
        "n_legendary": counts_by_rarity["LEGENDARY"],
        "costo_polvere": sum(DUST_COST_BY_RARITY[r] * n for r, n in counts_by_rarity.items()),  # polvere arcana per craftare il mazzo (copie standard, vedi nota sopra)
        **mechanic_counts,
        # target + info per filtrare/pesare in fase di training:
        "winrate": row["winrate"],
        "games": row["games"],
        "scrape_date": row["scrape_date"],
    }
    return features


def main():
    card_lookup = download_hearthstonejson()
    special_ids = find_special_deckbuild_ids(card_lookup)

    if USE_METASTATS_DATA:
        metastats_df = load_all_snapshots()
    else:
        metastats_df = pd.DataFrame(columns=["deck_id", "cards_id_count", "winrate", "games", "scrape_date", "game_type"])
        log("USE_METASTATS_DATA=False (vedi commento sopra) - fonte metastats.net esclusa, "
            "0 mazzi da questa fonte.")
    hsreplay_df = load_hsreplay_snapshots()
    # Solo i DataFrame non vuoti vanno a concat (evita un FutureWarning di pandas
    # sull'unione con un DataFrame vuoto).
    non_empty = [df for df in (metastats_df, hsreplay_df) if not df.empty]
    decks_df = pd.concat(non_empty, ignore_index=True) if non_empty else metastats_df
    # deck_id delle due fonti sono namespace separati (nessuna collisione possibile),
    # ma si ri-deduplica comunque per sicurezza se una fonte viene rilanciata piu' volte.
    decks_df = decks_df.sort_values("games", ascending=False).drop_duplicates(subset="deck_id", keep="first")
    log(f"Mazzi totali dopo l'unione delle fonti (metastats.net: {len(metastats_df)}, HSReplay: {len(hsreplay_df)}): {len(decks_df)}")

    before = len(decks_df)
    decks_df = decks_df[decks_df["games"].notna() & (decks_df["games"] >= MIN_GAMES)]
    log(f"Dopo filtro games >= {MIN_GAMES}: {len(decks_df)} (scartati {before - len(decks_df)})")

    rows = []
    skipped_no_cardlist = 0
    n_special = 0
    for _, row in decks_df.iterrows():
        feats = build_deck_features(row, card_lookup, special_ids)
        if feats is None:
            skipped_no_cardlist += 1
            continue
        if feats["has_special_deckbuild"]:
            n_special += 1
        rows.append(feats)

    log(f"Mazzi senza decklist letta (scartati): {skipped_no_cardlist}")
    log(f"Mazzi con regole di costruzione speciali (inclusi, non scartati - vedi has_special_deckbuild): {n_special}")
    log(f"Mazzi nel dataset finale: {len(rows)}")

    out_df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    log(f"\n[OK] Salvato in {OUT_PATH}")

    if not out_df.empty:
        log("\n--- Statistiche rapide ---")
        log(f"Winrate: min={out_df['winrate'].min():.1f}%, media={out_df['winrate'].mean():.1f}%, max={out_df['winrate'].max():.1f}%")
        log(f"Partite: min={out_df['games'].min()}, mediana={out_df['games'].median():.0f}, max={out_df['games'].max()}")
        log("\nMazzi per classe:")
        log(out_df["deck_class"].value_counts().to_string())


if __name__ == "__main__":
    main()
