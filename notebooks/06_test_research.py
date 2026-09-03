"""
Test end-to-end del grafo LangGraph con Planner -> Research/ReAct (roadmap punto 5).

Cosa verifica in piu' rispetto a 04_test_planner.py:
- Il nodo Research riceve il primo post pianificato e avvia il ciclo ReAct.
- Selezione dinamica tra i 3 tool minimi (query_knowledge_graph, search_card_knowledge,
  search_web), ognuno chiamato con `justification` - controllare il reasoning_trace
  per vedere Thought->Action->Observation di ogni chiamata.
- L'indice RAG locale (data/rag/cards.faiss) deve esistere: se manca, lanciare prima
  'python -u notebooks/05_build_rag_index.py' (serve sentence-transformers/faiss-cpu,
  gia' in requirements.txt).
- TAVILY_API_KEY deve essere configurata nel .env per search_web (altrimenti quel
  tool ritorna un errore leggibile invece di bloccare l'intero nodo - vedi
  src/tools/search_tool.py).
- Estrazione finale strutturata dei claim raccolti, ognuno con la fonte.

Come eseguirlo (dalla cartella del progetto, con il venv attivo):
    python -u notebooks/06_test_research.py
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
        "user_input": "pianifica i prossimi post del blog e fai ricerca sul primo",
        "reasoning_trace": [],
    }
    result = graph.invoke(initial_state)

    print("\n=== Reasoning trace ===")
    for line in result["reasoning_trace"]:
        print("-", line)

    print(f"\n=== Piano post ({len(result.get('post_plan', []))}) ===")
    for i, post in enumerate(result.get("post_plan", []), start=1):
        print(f"\n{i}. [{post['tipo'].upper()}] {post['topic']}")
        print(f"   Motivazione: {post['justification']}")

    current_post = result.get("current_post")
    if current_post:
        print(f"\n=== Post ricercato ===\n[{current_post['tipo'].upper()}] {current_post['topic']}")

    research_summary = result.get("research_summary", {})
    tools_used = research_summary.get("tools_used", [])
    claims = research_summary.get("claims", [])

    print(f"\n=== Tool usati ({len(tools_used)}) ===")
    for t in tools_used:
        print("-", t)

    print(f"\n=== Claim raccolti ({len(claims)}) ===")
    for i, c in enumerate(claims, start=1):
        print(f"\n{i}. {c.get('claim')}")
        print(f"   Fonte: {c.get('source')}")

    if not claims:
        print("\n[ATTENZIONE] Nessun claim raccolto - controlla il reasoning_trace sopra per l'errore "
              "(indice RAG mancante? TAVILY_API_KEY assente? Ollama non raggiungibile?).")


if __name__ == "__main__":
    main()
