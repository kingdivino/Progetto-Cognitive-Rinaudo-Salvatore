"""
Costruzione del grafo LangGraph. Nodi presenti: Planner -> Research/ReAct (roadmap
punti 4-5). E' pensato per crescere: i prossimi nodi (Format/Draft, Human Review,
KG Update) si aggiungono qui con altri add_node/add_edge, senza toccare planner.py
ne' research.py.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agent.planner import plan_posts
from src.agent.research import research_topic
from src.agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", plan_posts)
    graph.add_node("research", research_topic)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "research")
    graph.add_edge("research", END)
    return graph.compile()
