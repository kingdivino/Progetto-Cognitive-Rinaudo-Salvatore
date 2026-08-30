"""
Feature engineering — dai mazzi grezzi raccolti da metastats.net (notebooks/02) al
dataset pronto per il fine-tuning del classificatore power level/interestingness.

Input:
- data/raw/metastats/decks_<data>.csv (uno o più snapshot settimanali)
- HearthstoneJSON (scaricato/cachato in data/raw/hearthstonejson/cards.json) per le
  caratteristiche reali delle carte (costo mana, attacco/vita, rarità, tipo, meccaniche)

Cosa fa:
1. Unisce tutti gli snapshot settimanali, deduplicando per deck_id (tiene la riga con
   più partite giocate = stima del winrate più affidabile — un mazzo con 40 partite ha
   un winrate molto più rumoroso di uno con 800).
2. Scarta i mazzi senza decklist decodificata (niente deck code trovato sul sito, ~20%
   dei casi osservati — non recuperabile da qui) e quelli sotto una soglia minima di
   partite (MIN_GAMES) — sotto quella soglia il winrate è troppo rumoroso per essere
   un target affidabile.
3. Per ogni mazzo, incrocia i dbfId con HearthstoneJSON e costruisce le FEATURE (X):
   curva di mana, conteggio per rarità/tipo, statistiche medie attacco/vita, classe,
   presenza di meccaniche chiave (Taunt, Deathrattle, Battlecry, Rush, Divine Shield,
   Combo, Lifesteal).
4. Il winrate (%) è la LABEL (Y). Le partite giocate restano in output per poter
   filtrare/pesare gli esempi più avanti (fase di training).

IMPORTANTE — cosa sono davvero i "dbfId" per mazzo: NON è il decklist completo da
30 carte. metastats.net traccia le partite per archetipo (giocate da mazzi diversi
tra loro) e pubblica solo le carte "firma" che definiscono l'archetipo — verificato
decodificando a mano il deck code byte per byte (nessun byte avanza, quindi non è un
problema di parsing). Le feature calcolate qui vanno quindi lette come statistiche sul
PACCHETTO CORE dell'archetipo, non sul mazzo intero — comunque un segnale legittimo per
un classificatore di power level, ma da descrivere onestamente nel report (punto 9 della
guida fine-tuning: limiti del modello/dataset).

Output: data/processed/finetune_dataset.csv

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
HSJSON_DIR = os.path.join("data", "raw", "hearthstonejson")
HSJSON_PATH = os.path.join(HSJSON_DIR, "cards.json")
HSJSON_URL = "https://api.hearthstonejson.com/v1/latest/enUS/cards.json"

OUT_DIR = os.path.join("data", "processed")
OUT_PATH = os.path.join(OUT_DIR, "finetune_dataset.csv")

MIN_GAMES = 30  # sotto questa soglia il winrate è considerato troppo rumoroso
MECHANICS_OF_INTEREST = [
    "TAUNT", "DEATHRATTLE", "BATTLECRY", "RUSH", "DIVINE_SHIELD", "COMBO", "LIFESTEAL",
]


def log(msg: str):
    print(msg, flush=True)


def safe_literal_eval(val):
    """cells / cards_dbfid_count / heroes sono salvati nei CSV come repr() di liste
    Python — vanno riletti con ast.literal_eval, non con json.loads (virgolette singole)."""
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
    """Scarica (o riusa la cache locale) il database completo delle carte, indicizzato per dbfId."""
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

    by_dbfid = {}
    for c in cards:
        dbf = c.get("dbfId")
        if dbf is not None:
            by_dbfid[dbf] = c
    log(f"Carte indicizzate per dbfId: {len(by_dbfid)}")
    return by_dbfid


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

    # Dedup per deck_id: tiene la riga con più partite (stima più affidabile)
    all_df = all_df.sort_values("games", ascending=False).drop_duplicates(subset="deck_id", keep="first")
    log(f"Righe dopo dedup per deck_id (tiene la versione con più partite): {len(all_df)}")
    return all_df


def build_deck_features(row, card_lookup: dict) -> dict | None:
    cards_raw = safe_literal_eval(row["cards_dbfid_count"])
    if not cards_raw:
        return None  # niente decklist decodificata per questo mazzo

    total_cards = 0
    cost_sum = 0
    counts_by_type = {"MINION": 0, "SPELL": 0, "WEAPON": 0, "LOCATION": 0, "OTHER": 0}
    counts_by_rarity = {"COMMON": 0, "RARE": 0, "EPIC": 0, "LEGENDARY": 0, "OTHER": 0}
    cost_curve = {f"cost_{i}": 0 for i in range(8)}  # cost_7 = "7 o più"
    attack_values, health_values = [], []
    mechanic_counts = {f"n_{m.lower()}": 0 for m in MECHANICS_OF_INTEREST}
    classes_seen = {}

    for dbf_id, count in cards_raw:
        card = card_lookup.get(dbf_id)
        if card is None:
            continue  # carta non trovata (dbfId non nel dump scaricato, raro)
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
        return None  # nessuna carta firma valida trovata per questo mazzo

    deck_class = max(
        ((c, n) for c, n in classes_seen.items() if c != "NEUTRAL"),
        key=lambda x: x[1], default=("NEUTRAL", 0)
    )[0]

    features = {
        "deck_id": row["deck_id"],
        "deck_class": deck_class,
        "n_signature_cards": total_cards,  # NON il mazzo completo (30 carte) — vedi nota sopra
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
        **mechanic_counts,
        # target + info per filtrare/pesare in fase di training:
        "winrate": row["winrate"],
        "games": row["games"],
        "scrape_date": row["scrape_date"],
    }
    return features


def main():
    card_lookup = download_hearthstonejson()
    decks_df = load_all_snapshots()

    before = len(decks_df)
    decks_df = decks_df[decks_df["games"].notna() & (decks_df["games"] >= MIN_GAMES)]
    log(f"Dopo filtro games >= {MIN_GAMES}: {len(decks_df)} (scartati {before - len(decks_df)})")

    rows = []
    skipped_no_deckstring = 0
    for _, row in decks_df.iterrows():
        feats = build_deck_features(row, card_lookup)
        if feats is None:
            skipped_no_deckstring += 1
            continue
        rows.append(feats)

    log(f"Mazzi senza decklist decodificata (scartati): {skipped_no_deckstring}")
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
