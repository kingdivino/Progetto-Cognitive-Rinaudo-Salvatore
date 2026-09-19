"""
Segnali reali dai dati di scraping (metastats.net + HearthstoneJSON, vedi notebooks/02
e 03) da dare in pasto al Planner come spunto CONCRETO invece di far inventare topic
all'LLM a vuoto - coerente con il requisito delle specifiche di verificare l'accuratezza
delle informazioni: qui il "grounding" comincia gia' in fase di pianificazione, non solo
nel drafting.

Non e' un tool del grafo (nessuna chiamata rete, nessun LLM) - e' lettura locale di un
CSV gia' prodotto dalla pipeline di data collection, quindi puo' girare anche nella
sandbox cloud oltre che nel venv Windows dell'utente.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

PROCESSED_PATH = os.path.join("data", "processed", "finetune_dataset.csv")
RAW_DIR = os.path.join("data", "raw", "metastats")
RAW_DIR_HSREPLAY = os.path.join("data", "raw", "hsreplay")


def _latest_deck_names() -> dict:
    """deck_id -> nome leggibile (es. "Dragon Warrior", "Rafaam Warlock") dall'ultimo
    snapshot di ciascuna fonte, per non mostrare all'LLM solo numeri/classi generiche
    senza senso. Unisce metastats.net e HSReplay (07/09/2026, dopo aver notato che i
    mazzi HSReplay comparivano nei topic del Planner senza nome di archetipo, solo con
    la classe: 07_scrape_hsreplay.py ora risolve il nome via l'endpoint archetipi, ma
    va comunque letto qui) - i due namespace di deck_id (intero vs stringa) non
    collidono mai, quindi un dict.update() semplice basta."""
    names: dict = {}

    metastats_files = sorted(glob.glob(os.path.join(RAW_DIR, "decks_*.csv")))
    if metastats_files:
        df = pd.read_csv(metastats_files[-1])
        names.update(dict(zip(df["deck_id"], df["deck_name_raw"])))

    hsreplay_files = sorted(glob.glob(os.path.join(RAW_DIR_HSREPLAY, "decks_*.csv")))
    if hsreplay_files:
        df = pd.read_csv(hsreplay_files[-1])
        if "deck_name_raw" in df.columns:  # snapshot generati prima di questo fix non ce l'hanno
            names.update(dict(zip(df["deck_id"], df["deck_name_raw"])))

    return names


def load_archetype_signals(
    top_n: int = 8, covered_topics: list[str] | None = None
) -> list[dict]:
    """Seleziona una manciata di archetipi 'interessanti' dal dataset di fine-tuning:
    i migliori/peggiori per winrate (materiale per un post news/review su cosa sale e
    cosa scende nel meta) + quelli con regole di costruzione speciali (Azalina
    Soulsever / Timethief Rafaam, vedi guida di progetto - materiale naturale per un
    how-to su un archetipo particolare). Ritorna [] se il dataset non esiste ancora
    (es. primo avvio prima di aver lanciato lo scraper) - il Planner deve gestirlo.

    covered_topics (17/09/2026): elenco dei topic gia' presenti nel KG (lo stesso che
    il Planner passa gia' testualmente all'LLM). Caso reale osservato: il Planner ha
    ripetuto DUE VOLTE, parola per parola, lo stesso identico topic ("Come costruire
    un mazzo Azalina Priest Standard con un winrate del 55.7%") nonostante l'elenco
    "Topic gia' coperti dal KG" fosse gia' esplicitamente nel suo prompt - la regola
    testuale "non ripetere" da sola non e' bastata su qwen3:8b, lo stesso limite di
    compliance gia' documentato piu' volte in questo progetto. Fix strutturale invece
    dell'ennesima regola di prompt: un archetipo il cui nome compare gia' in un topic
    coperto viene escluso QUI, prima ancora di arrivare al Planner - non gli viene
    data la possibilita' di essere riproposto, il candidato successivo per
    winrate/partite prende automaticamente il suo posto. Match euristico (nome
    dell'archetipo come sottostringa case-insensitive di un topic coperto) - non
    perfetto ma coerente con altri controlli meccanici gia' presenti nel progetto
    (es. BATTLEGROUNDS_ONLY_KEYWORDS in format_rules.py); se il filtro escludesse
    TUTTI gli archetipi (dataset molto piccolo, tutti gia' coperti), si preferisce
    mostrarli comunque al Planner piuttosto che non dargli nessun dato reale."""
    if not os.path.exists(PROCESSED_PATH):
        return []
    df = pd.read_csv(PROCESSED_PATH)
    if df.empty:
        return []

    names = _latest_deck_names()
    df = df.copy()
    df["deck_name"] = df["deck_id"].map(names).fillna(df["deck_class"])
    if "formato" not in df.columns:
        # CSV generato prima del fix Standard/Wild del 07/09/2026 (rilanciare
        # 03_build_finetune_dataset.py per il dato reale) - degrada con un default
        # invece di far crashare il Planner, stesso principio del KG irraggiungibile.
        df["formato"] = "Standard"
    if "costo_polvere" not in df.columns:
        # CSV generato prima dell'aggiunta del costo in polvere (08/09/2026) -
        # stesso principio di degradare invece di crashare; None esclude il campo dal
        # dizionario finale invece di mostrare uno zero fuorviante (vedi sotto).
        df["costo_polvere"] = None

    if covered_topics:
        _covered_blob = " || ".join(str(t) for t in covered_topics).lower()
        _mask_not_covered = ~df["deck_name"].astype(str).str.lower().apply(
            lambda name: bool(name) and name in _covered_blob
        )
        _df_filtered = df[_mask_not_covered]
        if not _df_filtered.empty:
            df = _df_filtered

    top_winrate = df.nlargest(3, "winrate")
    bottom_winrate = df.nsmallest(2, "winrate")
    special = df[df["has_special_deckbuild"] == True].head(3)  # noqa: E712 (confronto esplicito piu' chiaro qui)

    picked = pd.concat([top_winrate, bottom_winrate, special]).drop_duplicates(subset="deck_id")
    picked = picked.head(top_n)

    result = []
    for _, row in picked.iterrows():
        entry = {
            "deck_name": row["deck_name"],
            "classe": row["deck_class"],
            "winrate": f"{row['winrate']:.1f}%",
            "partite": int(row["games"]),
            "numero_carte_mazzo": int(row["n_cards_total"]),
            "formato": row["formato"],  # "Standard" o "Wild" - vedi src/agent/format_rules.py:
            # in Wild sono legali tutte le carte mai pubblicate, in Standard solo un
            # sottoinsieme di espansioni in rotazione (aggiunto 07/09/2026 dopo aver
            # unito una seconda fonte dati - HSReplay.net - che include anche mazzi Wild)
        }
        # Costo in polvere arcana per craftare il mazzo (aggiunto 08/09/2026, su
        # richiesta dell'utente) - calcolato localmente dalla decklist +
        # rarita' delle carte in notebooks/03_build_finetune_dataset.py, non recuperato
        # da nessuna fonte esterna (vedi commento li' per il perche'). Omesso dal
        # dizionario (invece di un fuorviante 0/None) se il CSV in uso e' precedente
        # a questa aggiunta - un campo assente e' piu' sicuro di uno zero che
        # sembrerebbe un mazzo gratis da craftare.
        if pd.notna(row["costo_polvere"]):
            entry["costo_polvere"] = int(row["costo_polvere"])
        result.append(entry)
    return result
