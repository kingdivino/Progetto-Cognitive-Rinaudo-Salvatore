"""Costruzione del grafo LangGraph: Planner -> [select_next_post -> Research ->
Format -> Human Review -> KG Update] -> torna a select_next_post finche' il piano
non e' esaurito. Chi chiama graph.invoke/stream deve passare un 'thread_id' esplicito
in config (richiesto dal checkpointer) - vedi notebooks/09_test_human_review.py per
il pattern invoke/interrupt/resume."""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from src.agent.format_draft import draft_post
from src.agent.human_review import human_review
from src.agent.kg_update import update_kg
from src.agent.orchestrator import route_after_select, select_next_post
from src.agent.planner import plan_posts
from src.agent.research import research_topic
from src.agent.state import AgentState


def _route_after_review(state: AgentState) -> str:
    """Instradamento dopo Human Review: "regenerate" torna a Format per una nuova
    bozza sullo stesso post; qualunque altro valore va a KG Update (che scrive solo
    se 'review_decision' e' davvero "approved")."""
    if state.get("review_decision") == "regenerate":
        return "format"
    return "kg_update"


def build_graph(checkpointer=None):
    """checkpointer: BaseCheckpointSaver di LangGraph, o None per crearne uno in
    memoria (InMemorySaver)."""
    if checkpointer is None:
        checkpointer = InMemorySaver()

    graph = StateGraph(AgentState)
    graph.add_node("planner", plan_posts)
    graph.add_node("select_next_post", select_next_post)
    graph.add_node("research", research_topic)
    graph.add_node("format", draft_post)
    graph.add_node("human_review", human_review)
    graph.add_node("kg_update", update_kg)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "select_next_post")
    graph.add_conditional_edges("select_next_post", route_after_select, {"research": "research", END: END})
    graph.add_edge("research", "format")
    graph.add_edge("format", "human_review")
    graph.add_conditional_edges("human_review", _route_after_review, {"format": "format", "kg_update": "kg_update"})
    # kg_update chiude sempre il post corrente (che sia stato scritto sul KG o
    # saltato) tornando a select_next_post per decidere se c'e' un post successivo
    # nel piano o se il grafo deve terminare - MAI un edge fisso verso END da qui,
    # altrimenti il ciclo si fermerebbe sempre dopo il primo post.
    graph.add_edge("kg_update", "select_next_post")
    return graph.compile(checkpointer=checkpointer)
