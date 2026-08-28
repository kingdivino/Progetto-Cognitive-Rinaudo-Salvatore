"""
Script di verifica schema — HSDataset (Zenodo, DOI 10.5281/zenodo.10198504)
NON è parte della pipeline finale. Serve a controllare se decks.csv / cards.csv
contengono un campo utilizzabile come proxy di popolarità/forza per il classificatore
power level/interestingness, dato che il dataset non include winrate/tier espliciti.

Schema reale (verificato ispezionando il file, NON è quello ipotizzato inizialmente):
    decks.csv: idDeck;Minions;Spells;Weapons;DeckType;DeckArchetype;CraftingCost;Date;Set
    cards.csv: idDeck;idCard
Separatore: ';' (punto e virgola), non ','.

Come eseguirlo:
    1. Scarica decks.csv e cards.csv da https://zenodo.org/records/10198504
       e mettili in data/raw/hsdataset/ (già escluso da git, sono pesanti)
    2. python notebooks/01_check_hsdataset_schema.py
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
    df = pd.read_csv(DECKS_PATH, sep=";")
    print("\n=== decks.csv ===")
    print(f"Righe: {len(df)}")
    print(f"Colonne: {list(df.columns)}")
    print(df.dtypes)
    print("\nPrime righe:")
    print(df.head())
    return df


def analyze_archetype_frequency_proxy(df: pd.DataFrame):
    """
    Nessuna colonna di winrate/tier esiste in questo dataset.
    Questa funzione valuta se la FREQUENZA di un archetipo tra i mazzi Ranked
    può fare da proxy debole di rilevanza/competitività nel meta.
    """
    print("\n=== Proxy candidato: frequenza archetipo (solo Ranked Deck) ===")
    ranked = df[df["DeckType"] == "Ranked Deck"].copy()
    ranked["DeckArchetype"] = ranked["DeckArchetype"].str.strip()
    print(f"Mazzi Ranked totali: {len(ranked)}")

    unknown_mask = ranked["DeckArchetype"].str.lower() == "unknown"
    print(f"Di cui 'Unknown' (da scartare): {unknown_mask.sum()} ({unknown_mask.mean():.1%})")

    clean = ranked[~unknown_mask]
    freq = clean["DeckArchetype"].value_counts()
    print(f"\nArchetipi distinti (puliti): {freq.shape[0]}")
    print("\nTop 15 per frequenza:")
    print(freq.head(15))
    print("\nBottom 15 per frequenza (coda lunga — probabile rumore/varianti rare):")
    print(freq.tail(15))
    print(f"\nArchetipi con meno di 30 occorrenze: {(freq < 30).sum()} su {freq.shape[0]}"
          " (troppo pochi esempi per un label affidabile)")
    return freq


if __name__ == "__main__":
    decks_df = inspect_decks()
    if decks_df is not None:
        analyze_archetype_frequency_proxy(decks_df)

    print("\n=== VERDETTO ===")
    print("Nessuna colonna di winrate/tier/power level in questo dataset.")
    print("La frequenza archetipo è un proxy debole (popolarità != forza) e va ripulita")
    print("(scartare 'Unknown', normalizzare spazi, escludere archetipi con troppe poche occorrenze).")
    print("Da confrontare con l'alternativa: label reali da Vicious Syndicate Data Reaper Report.")
