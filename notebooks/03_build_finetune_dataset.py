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
2. Scarta i mazzi senza decklist letta dal sito (rari, non recuperabile da qui) e
   quelli sotto una soglia minima di partite (MIN_GAMES) — sotto quella soglia il
   winrate è troppo rumoroso per essere un target affidabile. NON scarta più i mazzi
   con Azalina Soulsever (20 carte) o Timethief Rafaam (40 carte) — vedi nota sotto:
   sono inclusi con n_cards_total e has_special_deckbuild come feature esplicite,
   invece di essere buttati via (dataset già piccolo, e altrimenti non avremmo modo
   di valutare il power level proprio delle due carte che cambiano più le regole).
3. Per ogni mazzo, incrocia gli id carta con HearthstoneJSON e costruisce le FEATURE
   (X): curva di mana, conteggio per rarità/tipo, statistiche medie attacco/vita,
   classe, presenza di meccaniche chiave (Taunt, Deathrattle, Battlecry, Rush, Divine
   Shield, Combo, Lifesteal).
4. Il winrate (%) è la LABEL (Y). Le partite giocate restano in output per poter
   filtrare/pesare gli esempi più avanti (fase di training).

NOTA sulla decklist: leggiamo le carte direttamente dal blocco HTML `ul.card-list`
della pagina di dettaglio mazzo (id carta HearthstoneJSON ricavato dal filename
dell'immagine + quantità), NON più dal "deck code" del sito. Verificato (30/08/2026)
che il deck code ha un bug lato metastats.net: per molti mazzi codifica solo una
manciata di carte anche quando la pagina mostra (e noi ora leggiamo) la decklist
completa da 30 carte. Vedi il docstring di notebooks/02_scrape_metastats.py per i
dettagli della verifica.

NOTA su Azalina Soulsever / Timethief Rafaam (01/09/2026): due leggendarie (Priest e
Warlock) cambiano davvero le regole di costruzione del mazzo per chi le include
(rispettivamente 20 e 40 carte invece di 30 — verificato sul testo ufficiale delle
carte). Un primo tentativo li escludeva dal dataset per confrontare solo mazzi da 30
carte; su segnalazione dell'utente si è deciso di tenerli: il dataset è già piccolo
(poche centinaia di mazzi) e buttare via il ~12% dei dati vuol dire anche non avere
mai un esempio per valutare il power level dei mazzi costruiti proprio intorno a
queste due carte. Restano però strutturalmente diversi (base di conteggio diversa),
quindi sono segnalati con `n_cards_total` (dimensione reale) e `has_special_deckbuild`
(booleano) come feature esplicite, cosicché il modello possa imparare a tenerne conto
invece di confrontarli alla cieca con un mazzo da 30.

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

    # Dedup per deck_id: tiene la riga con più partite (stima più affidabile)
    all_df = all_df.sort_values("games", ascending=False).drop_duplicates(subset="deck_id", keep="first")
    log(f"Righe dopo dedup per deck_id (tiene la versione con più partite): {len(all_df)}")
    return all_df


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
    special_ids = find_special_deckbuild_ids(card_lookup)
    decks_df = load_all_snapshots()

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
