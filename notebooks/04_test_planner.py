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

Come eseguirlo (dalla cartella del progetto, con il venv attivo):
    python -u notebooks/04_test_planner.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(override=True)

from src.agent.graph import build_graph  # noqa: E402 (import dopo sys.path/load_dotenv di proposito)


def main():
    graph = build_graph()
    initial_state = {
        "user_input": "pianifica i prossimi post del blog",
        "reasoning_trace": [],
    }
    result = graph.invoke(initial_state)

    print("\n=== Reasoning trace ===")
    for line in result["reasoning_trace"]:
        print("-", line)

    print(f"\n=== Piano post ({len(result['post_plan'])}) ===")
    for i, post in enumerate(result["post_plan"], start=1):
        print(f"\n{i}. [{post['tipo'].upper()}] {post['topic']}")
        print(f"   Motivazione: {post['justification']}")

    if not result["post_plan"]:
        print("\n[ATTENZIONE] Nessun post pianificato - controlla il reasoning_trace sopra per l'errore.")


if __name__ == "__main__":
    main()
