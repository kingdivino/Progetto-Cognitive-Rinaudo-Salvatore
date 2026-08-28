"""
Script di verifica schema — HSDataset (Zenodo, DOI 10.5281/zenodo.10198504)
NON è parte della pipeline finale. Serve solo a controllare se decks.csv / cards.csv
contengono un campo di frequenza/conteggio uso (proxy di popolarità/forza) prima di
decidere se questo dataset può servire da fonte di label per il classificatore
power level/interestingness.

Come eseguirlo:
    1. Scarica manualmente decks.csv e cards.csv da https://zenodo.org/records/10198504
       (pulsante Download sulla pagina) e mettili in data/raw/hsdataset/
       ATTENZIONE: cards.csv è ~564 MB, decks.csv è ~56 MB — non versionarli su git
       (data/raw/* è già in .gitignore).
    2. Attiva il venv, poi:
       python notebooks/01_check_hsdataset_schema.py
"""

import os
import pandas as pd

DATA_DIR = os.path.join("data", "raw", "hsdataset")
DECKS_PATH = os.path.join(DATA_DIR, "decks.csv")
CARDS_PATH = os.path.join(DATA_DIR, "cards.csv")


def inspect_decks():
    if not os.path.exists(DECKS_PATH):
        print(f"[SKIP] {DECKS_PATH} non trovato — scaricalo prima da Zenodo.")
        return None
    df = pd.read_csv(DECKS_PATH)
    print("\n=== decks.csv ===")
    print(f"Righe: {len(df)}")
    print(f"Colonne: {list(df.columns)}")
    print(df.dtypes)
    print("\nPrime righe:")
    print(df.head())
    print("\nValori unici per colonna (utile per capire se qualcosa è un conteggio/frequenza):")
    for col in df.columns:
        nunique = df[col].nunique()
        print(f"  {col}: {nunique} valori unici" + (f", range [{df[col].min()}, {df[col].max()}]" if pd.api.types.is_numeric_dtype(df[col]) else ""))
    return df


def inspect_cards(sample_rows=200_000):
    if not os.path.exists(CARDS_PATH):
        print(f"[SKIP] {CARDS_PATH} non trovato — scaricalo prima da Zenodo.")
        return None
    # cards.csv è grande (~564MB): leggiamo solo un campione per l'ispezione dello schema
    df = pd.read_csv(CARDS_PATH, nrows=sample_rows)
    print(f"\n=== cards.csv (campione di {sample_rows} righe) ===")
    print(f"Colonne: {list(df.columns)}")
    print(df.dtypes)
    print("\nPrime righe:")
    print(df.head())
    if "idDeck" in df.columns or "deckId" in df.columns:
        id_col = "idDeck" if "idDeck" in df.columns else "deckId"
        print(f"\nCarte per mazzo (nel campione, colonna '{id_col}'):")
        print(df.groupby(id_col).size().describe())
    return df


if __name__ == "__main__":
    decks_df = inspect_decks()
    cards_df = inspect_cards()

    print("\n=== VERDETTO ===")
    if decks_df is not None:
        numeric_cols = [c for c in decks_df.columns if pd.api.types.is_numeric_dtype(decks_df[c]) and decks_df[c].nunique() > 5]
        print("Colonne numeriche con abbastanza varietà da essere un possibile proxy di popolarità/forza:")
        print(numeric_cols if numeric_cols else "NESSUNA — probabile conferma che il dataset non ha label di potenza/winrate.")
