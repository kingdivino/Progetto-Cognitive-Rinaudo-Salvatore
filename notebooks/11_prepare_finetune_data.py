"""
Preparazione dataset per il fine-tuning (roadmap punto 6, step 1-5 della guida delle
specifiche - vedi claude/specifiche-progetto.md).

STEP 1 - Definizione del problema: il componente scelto per il fine-tuning e' un
classificatore del "power level" (basso/medio/alto) di un mazzo Hearthstone, a partire
dalla sua composizione (classe, formato, curva di mana, rarita', meccaniche, costo in
polvere) - MAI dal winrate/partite stesso, che e' esattamente cio' che il modello deve
imparare a stimare (userlo come input sarebbe una fuga di target, non un input lecito).
Uso a valle: il modello fine-tuned viene esposto come tool dell'agente
(src/tools/power_level_tool.py, roadmap: "almeno un tool aggiuntivo basato sul modello
fine-tuned", requisito esplicito delle specifiche) - il nodo Research puo' chiamarlo per
arricchire un post how-to/review con una valutazione indipendente del mazzo di cui sta
scrivendo, oltre a cio' che trova via RAG/search.

STEP 3 - Metrica: classificazione a 3 classi (non regressione diretta sul winrate) -
scelta deliberata per la dimensione del dataset (curca 570 righe, split realistici da
poche decine di esempi per classe): un problema a 3 classi e' molto piu' robusto al
rumore campionario di una regressione fine-grained, e resta comunque un segnale
significativo di "power level" per un lettore del blog. Le soglie sono TERZILI
calcolati SOLO sul train set (mai su tutto il dataset, altrimenti sarebbe leakage:
useremmo informazione dei set di validation/test per definire le classi) - basso/medio/
alto restano quindi bilanciati per costruzione sul train, ma non necessariamente su
val/test (le soglie sono fisse, applicate agli stessi bucket).

STEP 4 - Dataset: data/processed/finetune_dataset.csv, generato dalla pipeline di
scraping di questo stesso progetto (metastats.net + HSReplay.net, vedi guida di
progetto) - non da HuggingFace, ma dati REALI (non sintetici), gia' usati e verificati
per il Planner. Formato: input = descrizione testuale del mazzo (composizione, non
winrate), output = etichetta "basso"/"medio"/"alto".

STEP 5 - Split: 70/15/15 stratificato sull'etichetta di power level, per garantire che
ogni split abbia una distribuzione simile delle 3 classi nonostante N piccolo.
"""
from __future__ import annotations

import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_PATH = os.path.join("data", "processed", "finetune_dataset.csv")
OUT_DIR = os.path.join("data", "processed", "finetune_powerlevel")

RANDOM_STATE = 42


def build_deck_description(row: pd.Series) -> str:
    """Testo di input per il modello - SOLO caratteristiche di composizione del mazzo,
    mai winrate/partite (che sono esattamente il segnale da cui deriva l'etichetta,
    vedi docstring del modulo)."""
    special_note = ""
    if row["has_special_deckbuild"]:
        special_note = (
            " (numero di carte non standard: una carta leggendaria specifica altera "
            "la regola di costruzione del mazzo)"
        )
    return (
        f"Mazzo {row['deck_class']}, formato {row['formato']}, "
        f"{int(row['n_cards_total'])} carte totali{special_note}.\n"
        f"Curva di mana (numero di carte per costo): "
        f"0={int(row['cost_0'])}, 1={int(row['cost_1'])}, 2={int(row['cost_2'])}, "
        f"3={int(row['cost_3'])}, 4={int(row['cost_4'])}, 5={int(row['cost_5'])}, "
        f"6={int(row['cost_6'])}, 7+={int(row['cost_7'])}.\n"
        f"Composizione per tipo: {int(row['n_minions'])} minions, "
        f"{int(row['n_spells'])} spells, {int(row['n_weapons'])} weapons, "
        f"{int(row['n_locations'])} locations.\n"
        f"Rarita': {int(row['n_common'])} common, {int(row['n_rare'])} rare, "
        f"{int(row['n_epic'])} epic, {int(row['n_legendary'])} legendary.\n"
        f"Meccaniche presenti: {int(row['n_taunt'])} taunt, "
        f"{int(row['n_deathrattle'])} deathrattle, {int(row['n_battlecry'])} battlecry, "
        f"{int(row['n_rush'])} rush, {int(row['n_divine_shield'])} divine shield, "
        f"{int(row['n_combo'])} combo, {int(row['n_lifesteal'])} lifesteal.\n"
        f"Costo medio: {row['avg_cost']:.1f}. Attacco medio: {row['avg_attack']:.1f}. "
        f"Salute media: {row['avg_health']:.1f}.\n"
        f"Costo per craftare in polvere arcana: {int(row['costo_polvere'])}."
    )


def main() -> None:
    df = pd.read_csv(PROCESSED_PATH)
    print(f"Dataset caricato: {len(df)} mazzi.")

    # Split 70/15/15 stratificato - le soglie di bucket vengono calcolate SOLO sul
    # train risultante da questo split (mai prima), per evitare qualunque fuga di
    # informazione da val/test nella definizione delle classi. Stratifichiamo qui
    # sulla classe del mazzo (deck_class) come proxy grezzo di power level per lo
    # split iniziale - le vere etichette basso/medio/alto vengono assegnate DOPO,
    # quindi non possiamo ancora stratificare su di esse in questo primo split.
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=RANDOM_STATE, stratify=df["deck_class"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_df["deck_class"]
    )

    # Soglie di bucket (terzili) calcolate SOLO sul train.
    q1 = train_df["winrate"].quantile(1 / 3)
    q2 = train_df["winrate"].quantile(2 / 3)
    print(f"Soglie di bucket (terzili sul train): basso < {q1:.2f} <= medio < {q2:.2f} <= alto")

    def bucket(winrate: float) -> str:
        if winrate < q1:
            return "basso"
        if winrate < q2:
            return "medio"
        return "alto"

    os.makedirs(OUT_DIR, exist_ok=True)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    summary = {}
    for name, split_df in splits.items():
        split_df = split_df.copy()
        split_df["power_level"] = split_df["winrate"].apply(bucket)
        records = []
        for _, row in split_df.iterrows():
            records.append({
                "deck_id": row["deck_id"],
                "input": build_deck_description(row),
                "label": row["power_level"],
                "winrate": row["winrate"],  # tenuto SOLO per l'analisi degli errori,
                # mai passato come input al modello durante training/inferenza.
            })
        out_path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts = split_df["power_level"].value_counts().to_dict()
        summary[name] = {"n": len(split_df), "distribuzione": counts}
        print(f"{name}: {len(split_df)} esempi scritti in {out_path} - distribuzione: {counts}")

    with open(os.path.join(OUT_DIR, "soglie_bucket.json"), "w", encoding="utf-8") as f:
        json.dump({"q1_basso_medio": q1, "q2_medio_alto": q2, "summary": summary}, f, indent=2)

    print("\nEsempio di input generato (prima riga del train):")
    print(build_deck_description(train_df.iloc[0]))


if __name__ == "__main__":
    main()
