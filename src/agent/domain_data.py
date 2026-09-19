"""Carica dal CSV di data/processed gli archetipi/winrate reali da dare al Planner
come spunto per i topic dei post. Sola lettura locale, nessun tool del grafo."""
from __future__ import annotations

import glob
import os

import pandas as pd

PROCESSED_PATH = os.path.join("data", "processed", "finetune_dataset.csv")
RAW_DIR = os.path.join("data", "raw", "metastats")
RAW_DIR_HSREPLAY = os.path.join("data", "raw", "hsreplay")


def _latest_deck_names() -> dict:
    """deck_id -> nome leggibile (es. "Dragon Warrior")."""
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
    """Seleziona una manciata di archetipi 'interessanti' dal dataset di fine-tuning
    (migliori/peggiori per winrate + regole di costruzione speciali) per il Planner.
    Ritorna [] se il dataset non esiste ancora. covered_topics esclude qui, prima del
    Planner, gli archetipi il cui nome compare gia' in un topic coperto (match
    euristico case-insensitive) - se il filtro li escluderebbe tutti, li mostra
    comunque piuttosto che non dare nessun dato reale."""
    if not os.path.exists(PROCESSED_PATH):
        return []
    df = pd.read_csv(PROCESSED_PATH)
    if df.empty:
        return []

    names = _latest_deck_names()
    df = df.copy()
    df["deck_name"] = df["deck_id"].map(names).fillna(df["deck_class"])
    if "formato" not in df.columns:
        # CSV precedente al fix Standard/Wild - degrada con un default invece di
        # crashare il Planner (rilanciare 03_build_finetune_dataset.py per il dato reale).
        df["formato"] = "Standard"
    if "costo_polvere" not in df.columns:
        # Stesso principio: None esclude il campo invece di mostrare uno zero fuorviante.
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
            "formato": row["formato"],  # "Standard" o "Wild" (vedi format_rules.py)
        }
        # Costo in polvere (calcolato localmente, notebooks/03) - omesso se assente
        # invece di un fuorviante 0/None.
        if pd.notna(row["costo_polvere"]):
            entry["costo_polvere"] = int(row["costo_polvere"])
        result.append(entry)
    return result
