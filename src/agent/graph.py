"""
Costruzione del grafo LangGraph. Per ora ha un solo nodo (Planner) - e' letteralmente
lo "scheletro minimo funzionante" indicato al punto 4 della roadmap nelle specifiche,
pensato per crescere: i prossimi nodi (Research/ReAct, Format/Draft, Human Review,
KG Update) si aggiungono qui con altri add_node/add_edge, senza toccare planner.py.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agent.planner import plan_posts
from src.agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", plan_posts)
    graph.set_entry_point("planner")
    graph.add_edge("planner", END)
    return graph.compile()
