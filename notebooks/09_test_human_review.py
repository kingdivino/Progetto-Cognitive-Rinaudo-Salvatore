"""
Test end-to-end del grafo LangGraph completo, Planner -> Research/ReAct ->
Format/Draft -> Human Review (roadmap punto 7).

Cosa verifica in piu' rispetto a 08_test_format.py:
- Il nodo Human Review sospende davvero il grafo (interrupt() di LangGraph) invece di
  scrivere subito sul KG - requisito esplicito della specifica ("Human-in-the-loop":
  mostra il post, permette approvazione/modifica/rigenerazione, il KG si aggiorna
  SOLO dopo approvazione).
- Le decisioni possibili (approva/modifica/rigenera la bozza, o scarta l'intero post)
  e il loro effetto sullo stato e sul routing del grafo (vedi
  src/agent/human_review.py e src/agent/graph.py).

ATTENZIONE (11/09/2026): il grafo ora elabora l'INTERO piano pianificato dal Planner,
un post alla volta, tornando qui per la revisione di OGNI post (vedi
src/agent/orchestrator.py) - non solo del primo come in versioni precedenti di questo
test. Con 6 post di default, questo script chiedera' quindi una decisione da tastiera
fino a 6 volte (piu' un giro extra per ogni "rigenera"/"scarta"). Per il test completo
che include anche la scrittura sul KG e la rilettura finale, vedi
`10_test_kg_update.py` - questo file resta utile per un controllo piu' rapido delle
sole meccaniche di Human Review.

Questo test e' INTERATTIVO: a un certo punto si ferma e chiede una decisione da
tastiera (non e' un errore se il processo sembra "in attesa" - lo e' davvero).

Stessi prerequisiti di 08_test_format.py (indice RAG costruito, TAVILY_API_KEY nel
.env, Ollama/Neo4j raggiungibili dal proprio venv Windows).

Come eseguirlo (dalla cartella del progetto, con il venv attivo):
    python -u notebooks/09_test_human_review.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(override=True)

from langgraph.types import Command  # noqa: E402

from src.agent.graph import build_graph  # noqa: E402 (import dopo sys.path/load_dotenv di proposito)


def _leggi_testo_multilinea(prompt: str) -> str | None:
    """Legge testo su piu' righe da tastiera, terminato da una riga con solo 'FINE'.
    Ritorna None se l'utente non scrive nulla (prima riga vuota) - il chiamante lo
    interpreta come "lascia invariato questo campo"."""
    print(prompt)
    print("(per CAMBIARE: scrivi il nuovo testo, anche su piu' righe, poi su una riga "
          "da sola scrivi FINE e premi invio; per LASCIARE INVARIATO: scrivi subito "
          "FINE come prima cosa, senza scrivere altro prima)")
    righe = []
    while True:
        riga = input()
        if riga.strip() == "FINE":
            break
        righe.append(riga)
    testo = "\n".join(righe).strip()
    return testo or None


def _chiedi_decisione_umana(payload: dict) -> dict:
    """Mostra il payload dell'interrupt (bozza + warning prioritari) e raccoglie la
    decisione umana da tastiera, nello stesso formato atteso da src/agent/human_review.py."""
    print("\n" + "=" * 70)
    print("=== REVISIONE UMANA RICHIESTA ===")
    print("=" * 70)
    print(f"\nTitolo: {payload.get('titolo')}\n")
    print(payload.get("corpo"))
    print(f"\nFonti citate ({len(payload.get('fonti_citate', []))}):")
    for f in payload.get("fonti_citate", []):
        flag = "  [NON RICONOSCIUTA]" if f in payload.get("fonti_non_riconosciute", []) else ""
        print(f"- {f}{flag}")

    warning_prioritari = payload.get("warning_prioritari", [])
    if warning_prioritari:
        print(f"\n--- Warning da Research/Format da considerare ({len(warning_prioritari)}) ---")
        for w in warning_prioritari:
            print(f"! {w}")

    while True:
        scelta = input(
            "\nDecisione - [a]pprova / [m]odifica / [r]igenera bozza / [s]carta intero post: "
        ).strip().lower()
        if scelta in ("a", "approva"):
            return {"azione": "approva"}
        if scelta in ("m", "modifica"):
            nuovo_titolo = _leggi_testo_multilinea("Nuovo titolo:")
            nuovo_corpo = _leggi_testo_multilinea("Nuovo corpo:")
            decisione = {"azione": "modifica"}
            if nuovo_titolo:
                decisione["titolo"] = nuovo_titolo
            if nuovo_corpo:
                decisione["corpo"] = nuovo_corpo
            return decisione
        if scelta in ("r", "rigenera"):
            return {"azione": "rigenera"}
        if scelta in ("s", "scarta"):
            return {"azione": "scarta"}
        print("Non riconosciuto - rispondi 'a', 'm', 'r' o 's'.")


def main():
    graph = build_graph()
    # thread_id: identifica QUESTA esecuzione del grafo per il checkpointer - deve
    # restare lo stesso tra l'invoke iniziale e ogni invoke di ripresa
    # (Command(resume=...)) qui sotto, altrimenti LangGraph non trova lo stato
    # sospeso da cui riprendere.
    config = {"configurable": {"thread_id": "test-09-human-review"}}
    initial_state = {
        "user_input": "pianifica i prossimi post del blog, fai ricerca su ognuno, scrivine le bozze e sottoponile a revisione",
        "reasoning_trace": [],
    }

    wall_start = time.perf_counter()
    result = graph.invoke(initial_state, config=config)

    n_round = 0
    while "__interrupt__" in result:
        n_round += 1
        payload = result["__interrupt__"][0].value
        idx = result.get("current_post_index")
        n_posts = len(result.get("post_plan", []))
        print(f"\n[Round di revisione #{n_round} - post {idx + 1 if idx is not None else '?'}/{n_posts} del piano]")
        decisione = _chiedi_decisione_umana(payload)
        result = graph.invoke(Command(resume=decisione), config=config)

    wall_elapsed = time.perf_counter() - wall_start

    print("\n=== Reasoning trace ===")
    for line in result["reasoning_trace"]:
        print("-", line)

    print(f"\n=== Esito del piano: {n_round} round di revisione totali tra tutti i post ===")
    print(f"=== Esito dell'ULTIMO post elaborato: review_decision = {result.get('review_decision')!r} ===")
    draft = result.get("draft")
    if draft is None:
        print("\n[ATTENZIONE] Nessuna bozza finale disponibile - controlla il reasoning_trace sopra.")
    else:
        print(f"\n=== Bozza finale (dopo {n_round} round di revisione) ===\nTitolo: {draft.get('titolo')}\n")
        print(draft.get("corpo"))
        print(f"\nFonti citate ({len(draft.get('fonti_citate', []))}):")
        for f in draft.get("fonti_citate", []):
            print(f"- {f}")

    timings = result.get("timings", {})
    reasoning_env = os.environ.get("OLLAMA_REASONING", "(default del modello)")
    print(f"\n=== Tempi (modello: {os.environ.get('OLLAMA_MODEL', 'llama3.1:8b')}, "
          f"OLLAMA_REASONING={reasoning_env}) ===")
    for key, seconds in timings.items():
        print(f"- {key}: {seconds:.1f}s")
    print(f"- tempo totale wall-clock (incluso il tempo di attesa della revisione umana): {wall_elapsed:.1f}s")


if __name__ == "__main__":
    main()
