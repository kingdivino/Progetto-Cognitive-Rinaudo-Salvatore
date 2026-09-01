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


def _latest_deck_names() -> dict[int, str]:
    """deck_id -> nome leggibile (es. "Dragon Warrior") dall'ultimo snapshot scaricato,
    per non mostrare all'LLM solo numeri di deck_id senza senso."""
    files = sorted(glob.glob(os.path.join(RAW_DIR, "decks_*.csv")))
    if not files:
        return {}
    df = pd.read_csv(files[-1])
    return dict(zip(df["deck_id"], df["deck_name_raw"]))


def load_archetype_signals(top_n: int = 8) -> list[dict]:
    """Seleziona una manciata di archetipi 'interessanti' dal dataset di fine-tuning:
    i migliori/peggiori per winrate (materiale per un post news/review su cosa sale e
    cosa scende nel meta) + quelli con regole di costruzione speciali (Azalina
    Soulsever / Timethief Rafaam, vedi guida di progetto - materiale naturale per un
    how-to su un archetipo particolare). Ritorna [] se il dataset non esiste ancora
    (es. primo avvio prima di aver lanciato lo scraper) - il Planner deve gestirlo."""
    if not os.path.exists(PROCESSED_PATH):
        return []
    df = pd.read_csv(PROCESSED_PATH)
    if df.empty:
        return []

    names = _latest_deck_names()
    df = df.copy()
    df["deck_name"] = df["deck_id"].map(names).fillna(df["deck_class"])

    top_winrate = df.nlargest(3, "winrate")
    bottom_winrate = df.nsmallest(2, "winrate")
    special = df[df["has_special_deckbuild"] == True].head(3)  # noqa: E712 (confronto esplicito piu' chiaro qui)

    picked = pd.concat([top_winrate, bottom_winrate, special]).drop_duplicates(subset="deck_id")
    picked = picked.head(top_n)

    return [
        {
            "deck_name": row["deck_name"],
            "classe": row["deck_class"],
            "winrate": f"{row['winrate']:.1f}%",
            "partite": int(row["games"]),
            "regola_speciale": bool(row["has_special_deckbuild"]),
        }
        for _, row in picked.iterrows()
    ]
