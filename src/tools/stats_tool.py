"""Secondo tool aggiuntivo richiesto dalla specifica (il primo e'
assess_deck_power_level, fine-tuned): interroga i dati reali HSReplay per il winrate/
popolarita' di una classe/formato, invece di farli stimare al modello o cercare sul
web. Usa sempre lo snapshot piu' recente su disco."""
from __future__ import annotations

import os

import pandas as pd
from langchain_core.tools import tool

PROCESSED_PATH = os.path.join("data", "processed", "finetune_dataset.csv")


def _load_df() -> pd.DataFrame:
    # Nessun caching (a differenza di power_level_tool.py): il CSV e' piccolo, meglio
    # rileggerlo ad ogni chiamata che rischiare dati non aggiornati da uno scrape
    # rilanciato durante la stessa sessione.
    return pd.read_csv(PROCESSED_PATH)


@tool
def get_archetype_stats(deck_class: str, formato: str, justification: str) -> str:
    """Restituisce statistiche REALI (winrate e numero di mazzi/partite osservate,
    non una stima) per una classe di Hearthstone in un formato, calcolate sui dati
    veri raccolti da HSReplay (la stessa fonte usata per il fine-tuning). Usa questo
    tool quando il post deve citare un numero (winrate, popolarita') su una classe/
    archetipo - MAI inventare o dedurre questi numeri da search_web/search_card_knowledge,
    che non contengono statistiche di winrate reali e aggiornate. Se il post riguarda
    un singolo mazzo specifico con nome (es. "Azalina Priest"), usa comunque la classe
    di quel mazzo (es. "PRIEST") - questo tool non ha granularita' per singolo
    archetipo con nome, solo per classe/formato (vedi anche assess_deck_power_level
    per una valutazione del modello sulla composizione di UN mazzo specifico, cosa
    diversa da queste statistiche aggregate). Se lo usi, il claim corrispondente DEVE
    avere come source esattamente il letterale 'DATI-HSREPLAY'.

    Args:
        deck_class: classe Hearthstone, es. "PRIEST", "WARLOCK", "WARRIOR" (case-insensitive).
        formato: "Standard", "Wild", o "" per aggregare entrambi.
        justification: perche' serve questa statistica ora, per quale claim del post.
    """
    try:
        df = _load_df()
    except Exception as e:
        return f"[ERROR] impossibile leggere {PROCESSED_PATH!r}: {e}"

    classe_norm = (deck_class or "").strip().upper()
    subset = df[df["deck_class"].str.upper() == classe_norm]
    formato_norm = (formato or "").strip()
    if formato_norm:
        subset = subset[subset["formato"].str.lower() == formato_norm.lower()]

    if subset.empty:
        classi_disponibili = ", ".join(sorted(df["deck_class"].str.upper().unique()))
        return (
            f"[INFO] Nessun mazzo trovato per classe={deck_class!r}, formato={formato!r} "
            f"nei dati reali attualmente raccolti. Classi disponibili: {classi_disponibili}. "
            "Non inventare un numero: se questa classe/formato non e' nel dataset, dillo "
            "esplicitamente nel post invece di stimare un winrate."
        )

    n_mazzi = len(subset)
    winrate_medio = subset["winrate"].mean()
    winrate_mediano = subset["winrate"].median()
    partite_totali = int(subset["games"].sum())
    formato_desc = formato_norm or "Standard+Wild"
    # Stesso principio di trasparenza usato altrove (es. research.py): un campione
    # piccolo non va presentato con la stessa sicurezza di uno ampio.
    campione_note = (
        f" ATTENZIONE: campione molto piccolo (solo {n_mazzi} mazzi), il numero e' "
        "reale ma poco rappresentativo - non presentarlo nel post come un dato "
        "solido senza questa cautela."
        if n_mazzi < 5 else ""
    )
    return (
        f"Statistiche REALI (HSReplay, {n_mazzi} mazzi osservati, {partite_totali} "
        f"partite totali, snapshot piu' recente su disco) per classe={classe_norm}, "
        f"formato={formato_desc}: winrate medio {winrate_medio:.1f}%, winrate mediano "
        f"{winrate_mediano:.1f}%.{campione_note} (giustificazione: {justification})"
    )
