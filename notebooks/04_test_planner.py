"""
Test end-to-end del grafo LangGraph con il nodo Planner (roadmap punto 4).

Cosa verifica:
- Il grafo (src/agent/graph.py) si costruisce e si invoca correttamente.
- Connessione al KG (Neo4j): se Neo4j Desktop non e' avviato il Planner non deve
  bloccarsi - deve loggarlo nel reasoning_trace e procedere assumendo nessuno storico.
- Recupero di archetipi reali da data/processed/finetune_dataset.csv come spunto.
- L'LLM (Ollama) pianifica una sequenza di post con output strutturato, ognuno con
  tipo/topic/justification.
- Tracing su LangSmith (se configurato nel .env, vedi guida di progetto).
- Cronometraggio: stampa il tempo totale e, per nodo, quanto va in chiamate LLM vs
  tool vs overhead (utile per confrontare modelli Ollama diversi, vedi guida di
  progetto). NOTA: dato che src/agent/graph.py collega ormai Planner -> Research,
  questo script esegue di fatto anche il nodo Research, non solo il Planner - per
  un test mirato solo sul Research vedi notebooks/06_test_research.py.

Come eseguirlo (dalla cartella del progetto, con il venv attivo):
    python -u notebooks/04_test_planner.py
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
    config = {"configurable": {"thread_id": "test-04-planner"}}
    initial_state = {
        "user_input": "pianifica i prossimi post del blog",
        "reasoning_trace": [],
    }
    wall_start = time.perf_counter()
    result = graph.invoke(initial_state, config=config)
    wall_elapsed = time.perf_counter() - wall_start

    print("\n=== Reasoning trace ===")
    for line in result["reasoning_trace"]:
        print("-", line)

    print(f"\n=== Piano post ({len(result['post_plan'])}) ===")
    for i, post in enumerate(result["post_plan"], start=1):
        print(f"\n{i}. [{post['tipo'].upper()}] {post['topic']}")
        print(f"   Motivazione: {post['justification']}")

    if not result["post_plan"]:
        print("\n[ATTENZIONE] Nessun post pianificato - controlla il reasoning_trace sopra per l'errore.")

    timings = result.get("timings", {})
    print(f"\n=== Tempi (modello: {os.environ.get('OLLAMA_MODEL', 'llama3.1:8b')}) ===")
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
