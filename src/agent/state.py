"""Stato condiviso del grafo LangGraph: un dizionario esplicito aggiornato ad ogni
nodo e passato al successivo.

Campi: user_input, reasoning_trace (log human-readable), tool_outputs, kg_summary,
planning_info, post_plan (Planner); current_post/research_summary/timings (Research);
draft (Format); review_decision - "approved"/"regenerate"/"discarded" (Human Review);
current_post_index (ciclo sui post, azzerato insieme a research_summary/draft/
review_decision/tool_outputs ad ogni avanzamento - vedi select_next_post)."""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    user_input: str
    reasoning_trace: list[str]
    tool_outputs: list[dict[str, Any]]
    kg_summary: dict[str, Any]
    planning_info: dict[str, Any]
    post_plan: list[dict[str, Any]]
    current_post: dict[str, Any] | None
    current_post_index: int | None
    research_summary: dict[str, Any] | None
    timings: dict[str, float]
    draft: dict[str, Any] | None
    review_decision: str | None
