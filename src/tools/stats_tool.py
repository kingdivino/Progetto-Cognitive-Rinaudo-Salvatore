"""
Secondo tool aggiuntivo richiesto dalla specifica (claude/specifiche-progetto.md:
"Tool minimi obbligatori... 4. Almeno 2 tool aggiuntivi progettati dal team, di cui
almeno uno basato sul modello fine-tuned") - il primo e' assess_deck_power_level
(src/tools/power_level_tool.py, basato sul modello fine-tuned), questo e' il secondo,
non fine-tuned: interroga direttamente i dati REALI raccolti da HSReplay
(data/processed/finetune_dataset.csv, la stessa fonte usata per il fine-tuning) per
restituire winrate/popolarita' VERI di una classe, invece di lasciare che il modello
li stimi (assess_deck_power_level) o li cerchi sul web (search_web, dove i risultati
di questo progetto si sono ripetutamente rivelati privi di data o non recenti - vedi
addendum del 18/09/2026). E' un dato di fatto verificabile, non un'opinione: zero
rischio di allucinazione sui numeri, a differenza di un claim numerico scritto a
parole dal modello dopo una ricerca web.

Nota di scope: il dataset processato non conserva un nome di archetipo per mazzo (solo
la classe e il formato, oltre alla composizione) - la granularita' di questo tool e'
quindi per classe/formato, non per singolo archetipo con nome (es. "Priest" in
Standard, non "Azalina Priest" nello specifico). Coerente con quanto Research rileva
gia' da solo per ogni post (detected_class/detected_format in research.py) - stessa
granularita', nessuna informazione in piu' da inventare.

Usa sempre lo snapshot PIU' RECENTE su disco (non il train/val/test.jsonl congelato
usato per il fine-tuning, che resta intenzionalmente fermo a 572 mazzi per la validita'
del confronto tra i 3 run - vedi addendum fine-tuning): per un dato citato in un post
del blog vogliamo il numero piu' aggiornato disponibile, non quello congelato per
motivi sperimentali.
"""
from __future__ import annotations

import os

import pandas as pd
from langchain_core.tools import tool

PROCESSED_PATH = os.path.join("data", "processed", "finetune_dataset.csv")


def _load_df() -> pd.DataFrame:
    # Nessun caching in memoria (a differenza del modello fine-tuned in
    # power_level_tool.py): leggere un CSV di poche centinaia di righe ad ogni
    # chiamata costa pochi millisecondi, molto meno del rischio di servire dati
    # non aggiornati se lo scrape viene rilanciato durante la stessa sessione lunga
    # dell'agente (vedi 07_scrape_hsreplay.py, rilanciato piu' volte in questo
    # progetto proprio mentre l'agente restava attivo).
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
    # Stesso principio di trasparenza gia' usato altrove nel progetto (es. i warning
    # su fonti non datate in research.py): un campione piccolo non va presentato con
    # la stessa sicurezza di uno ampio, anche se il numero e' comunque reale.
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
