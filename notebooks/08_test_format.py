"""
Test end-to-end del grafo LangGraph con Planner -> Research/ReAct -> Format/Draft
(roadmap punto 6).

Cosa verifica in piu' rispetto a 06_test_research.py:
- Il nodo Format riceve i claim verificati dal Research, li filtra (solo quelli con
  source_well_formed=True, source_grounded=True, format_valid/class_valid non False -
  vedi src/agent/format_draft.py) e scrive il testo del post SOLO con quelli.
- Chiamata KG fissa per la coerenza in fase di drafting (distinta da quella gia'
  fatta in fase di ricerca).
- Verifica in codice che le fonti dichiarate nel post corrispondano davvero a quelle
  dei claim usati (campo 'fonti_non_riconosciute').

Stessi prerequisiti di 06_test_research.py (indice RAG costruito, TAVILY_API_KEY nel
.env, Ollama/Neo4j raggiungibili dal proprio venv Windows).

Per testare un post diverso dal primo del piano (di default sempre post_plan[0] -
vedi RESEARCH_POST_INDEX in src/agent/research.py), impostare la variabile
d'ambiente RESEARCH_POST_INDEX (0-based) prima di lanciare, es.:
    set RESEARCH_POST_INDEX=3 (PowerShell: $env:RESEARCH_POST_INDEX=3)

Come eseguirlo (dalla cartella del progetto, con il venv attivo):
    python -u notebooks/08_test_format.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(override=True)

from src.agent.graph import build_graph  # noqa: E402 (import dopo sys.path/load_dotenv di proposito)


def main():
    graph = build_graph()
    # thread_id richiesto da LangGraph perche' il grafo ora ha un checkpointer
    # (serve al nodo Human Review, roadmap punto 7 - vedi src/agent/graph.py).
    # Questo test non gestisce la revisione umana: si ferma quando il grafo la
    # raggiunge e la segnala sotto, invece di rispondere all'interrupt.
    config = {"configurable": {"thread_id": "test-08-format"}}
    initial_state = {
        "user_input": "pianifica i prossimi post del blog, fai ricerca sul primo e scrivine la bozza",
        "reasoning_trace": [],
    }
    wall_start = time.perf_counter()
    result = graph.invoke(initial_state, config=config)
    wall_elapsed = time.perf_counter() - wall_start

    print("\n=== Reasoning trace ===")
    for line in result["reasoning_trace"]:
        print("-", line)

    current_post = result.get("current_post")
    if current_post:
        print(f"\n=== Post pianificato ===\n[{current_post['tipo'].upper()}] {current_post['topic']}")
        print(f"Motivazione: {current_post.get('justification')}")

    research_summary = result.get("research_summary", {})
    claims = research_summary.get("claims", [])
    n_trusted = sum(
        1 for c in claims
        if c.get("source_well_formed") is True
        and c.get("source_grounded") is True
        and c.get("format_valid") is not False
        and c.get("class_valid") is not False
    )
    print(f"\n=== Claim dal Research: {len(claims)} totali, {n_trusted} usati per il drafting ===")

    draft = result.get("draft")
    if draft is None:
        print("\n[ATTENZIONE] Nessuna bozza generata - controlla il reasoning_trace sopra per l'errore.")
    else:
        print(f"\n=== Bozza generata ===\nTitolo: {draft.get('titolo')}\n")
        print(draft.get("corpo"))
        print(f"\nFonti citate ({len(draft.get('fonti_citate', []))}):")
        for f in draft.get("fonti_citate", []):
            flag = "  [NON RICONOSCIUTA - non corrisponde a un claim verificato]" if f in draft.get(
                "fonti_non_riconosciute", []
            ) else ""
            print(f"- {f}{flag}")

    timings = result.get("timings", {})
    reasoning_env = os.environ.get("OLLAMA_REASONING", "(default del modello)")
    print(f"\n=== Tempi (modello: {os.environ.get('OLLAMA_MODEL', 'llama3.1:8b')}, "
          f"OLLAMA_REASONING={reasoning_env}) ===")
    for key, seconds in timings.items():
        print(f"- {key}: {seconds:.1f}s")
    print(f"- tempo totale wall-clock (graph.invoke): {wall_elapsed:.1f}s")

    if "__interrupt__" in result:
        print(
            "\n[INFO] Il grafo si e' fermato al nodo Human Review (interrupt) dopo "
            "aver generato la bozza sopra - normale, questo test non gestisce la "
            "revisione umana. Vedi notebooks/09_test_human_review.py per il test "
            "end-to-end con approvazione/modifica/rigenerazione."
        )


if __name__ == "__main__":
    main()
