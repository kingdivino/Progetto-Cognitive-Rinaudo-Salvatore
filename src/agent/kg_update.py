"""Nodo KG Update: scrive il post approvato nel Knowledge Graph (chiamata fissa,
solo se review_decision == "approved")."""
from __future__ import annotations

import time
import uuid

from src.agent.format_rules import detect_deck_class, detect_format
from src.agent.research import classify_source_tier
from src.agent.state import AgentState
from src.tools.kg_tool import update_knowledge_graph


def update_kg(state: AgentState) -> AgentState:
    """Scrive il post approvato sul Knowledge Graph, ricontrollando review_decision
    == "approved" invece di fidarsi solo del routing del grafo."""
    node_start = time.perf_counter()
    reasoning_trace = list(state.get("reasoning_trace", []))
    kg_summary = dict(state.get("kg_summary", {}))

    if state.get("review_decision") != "approved":
        reasoning_trace.append(
            "[KGUpdate] review_decision non e' 'approved' - nodo saltato, nessuna "
            "scrittura sul KG (richiesto dalla specifica: il grafo si aggiorna SOLO "
            "dopo approvazione umana esplicita)."
        )
        return {**state, "reasoning_trace": reasoning_trace}

    draft = state.get("draft")
    current_post = state.get("current_post")
    if draft is None or current_post is None:
        reasoning_trace.append(
            "[KGUpdate] [WARNING] review_decision e' 'approved' ma draft o "
            "current_post sono assenti - impossibile scrivere sul KG, nodo saltato "
            "(anomalia di stato, andrebbe indagata)."
        )
        return {**state, "reasoning_trace": reasoning_trace}

    topic_text = f"{current_post.get('topic', '')} {current_post.get('justification', '')}"
    classe = detect_deck_class(topic_text)
    formato = detect_format(topic_text)

    # Solo le fonti ancora riconosciute (vedi format_draft.py) - non si scrive una
    # fonte che potrebbe essere stata riformulata o inventata in fase di drafting.
    fonti_non_riconosciute = set(draft.get("fonti_non_riconosciute", []))
    fonti_valide = [f for f in draft.get("fonti_citate", []) if f not in fonti_non_riconosciute]
    sources = [{"ref": f, "tier": classify_source_tier(f)} for f in fonti_valide]

    # Solo claim gia' verificati (stesso filtro di format_draft.py) con fonte ancora
    # tra quelle valide sopra.
    research_summary = state.get("research_summary") or {}
    fonti_valide_set = set(fonti_valide)
    claims = [
        {"text": c.get("claim"), "source": c.get("source")}
        for c in research_summary.get("claims", [])
        if c.get("source_well_formed") is True
        and c.get("source_grounded") is True
        and c.get("format_valid") is not False
        and c.get("class_valid") is not False
        and c.get("source") in fonti_valide_set
    ]

    post_id = str(uuid.uuid4())
    _tool_start = time.perf_counter()
    try:
        observation = update_knowledge_graph.invoke({
            "justification": (
                "Passo fisso del workflow: scrivere sul KG il post appena approvato "
                "dalla revisione umana (nessun giudizio richiesto, e' una conseguenza "
                "diretta dell'approvazione)."
            ),
            "post_id": post_id,
            "tipo": current_post.get("tipo"),
            "topic": current_post.get("topic"),
            "summary": draft.get("corpo"),
            "sources": sources,
            "claims": claims,
            "classe": classe,
            "formato": formato,
        })
    except Exception as e:
        observation = f"[ERROR] esecuzione tool 'update_knowledge_graph' fallita: {e}"
    tool_elapsed = time.perf_counter() - _tool_start

    kg_summary["kg_update_result"] = str(observation)
    reasoning_trace.append(f"[KGUpdate] {observation}")

    node_elapsed = time.perf_counter() - node_start
    reasoning_trace.append(f"[KGUpdate] Tempo nodo: {node_elapsed:.1f}s totali (tool: {tool_elapsed:.1f}s).")
    timings = dict(state.get("timings", {}))
    timings["kg_update_tool_s"] = round(tool_elapsed, 2)
    timings["kg_update_total_s"] = round(node_elapsed, 2)

    return {**state, "reasoning_trace": reasoning_trace, "kg_summary": kg_summary, "timings": timings}
