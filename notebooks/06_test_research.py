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

Per testare un post diverso dal primo del piano (di default sempre post_plan[0] -
vedi RESEARCH_POST_INDEX in src/agent/research.py), impostare la variabile
d'ambiente RESEARCH_POST_INDEX (0-based) prima di lanciare, es.:
    set RESEARCH_POST_INDEX=3 (PowerShell: $env:RESEARCH_POST_INDEX=3)

Come eseguirlo (dalla cartella del progetto, con il venv attivo):
    python -u notebooks/06_test_research.py
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
    config = {"configurable": {"thread_id": "test-06-research"}}
    initial_state = {
        "user_input": "pianifica i prossimi post del blog e fai ricerca sul primo",
        "reasoning_trace": [],
    }
    wall_start = time.perf_counter()
    result = graph.invoke(initial_state, config=config)
    wall_elapsed = time.perf_counter() - wall_start

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

    tool_outputs = result.get("tool_outputs", [])
    print(f"\n=== Tool usati ({len(tools_used)}) ===")
    for t in tools_used:
        print("-", t)

    print(f"\n=== Osservazioni complete dei tool ({len(tool_outputs)}) ===")
    # Stampate per intero (non troncate a 300 char come nel reasoning_trace) - servono a
    # verificare a occhio se i claim del riassunto finale sono davvero fondati su quello
    # che i tool hanno trovato, o se l'estrazione ha aggiunto dettagli non presenti qui.
    for i, entry in enumerate(tool_outputs, start=1):
        print(f"\n{i}. [{entry.get('tool')}] justification: {entry.get('justification')}")
        print(f"   Observation: {entry.get('observation')}")

    print(f"\n=== Claim raccolti ({len(claims)}) ===")
    for i, c in enumerate(claims, start=1):
        flag = "" if c.get("source_well_formed", True) else "  [FONTE NON CONFORME - verificare a mano]"
        if c.get("format_valid") is False:
            flag += "  [CARTA NON LEGALE IN STANDARD - verificare a mano]"
        if c.get("class_valid") is False:
            flag += "  [CARTA DI CLASSE SBAGLIATA PER IL MAZZO - verificare a mano]"
        if c.get("source_tier") == "opinione_singola":
            flag += "  [FONTE = OPINIONE INDIVIDUALE - verificare se generalizza a torto]"
        if c.get("source_well_formed", True) and c.get("source_grounded") is False:
            flag += "  [FONTE NON RISCONTRATA IN NESSUNA OSSERVAZIONE REALE - probabile fabbricazione, SCARTARE]"
        print(f"\n{i}. {c.get('claim')}")
        print(f"   Fonte: {c.get('source')}{flag}  (tier: {c.get('source_tier', '?')}, grounded: {c.get('source_grounded')})")

    if not claims:
        print("\n[ATTENZIONE] Nessun claim raccolto - controlla il reasoning_trace sopra per l'errore "
              "(indice RAG mancante? TAVILY_API_KEY assente? Ollama non raggiungibile?).")

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
