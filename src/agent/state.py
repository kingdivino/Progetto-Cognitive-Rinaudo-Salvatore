"""
Stato condiviso del grafo LangGraph (gestione dello stato "MCP-style" richiesta dalle
specifiche): un dizionario esplicito aggiornato ad ogni nodo e passato come input al
nodo successivo, cosi' ogni nodo vede la storia completa del ragionamento fin qui.

Campi richiesti dalle specifiche:
- user_input: input dell'utente che ha avviato il grafo
- reasoning_trace: log testuale human-readable del ragionamento, un elemento per step
  (utile sia per debug sia per l'osservabilita' in LangSmith)
- tool_outputs: output grezzi di ogni chiamata a un tool (search/RAG/KG/fine-tuned)
- kg_summary: riassunto di cosa e' stato recuperato dal Knowledge Graph in questo run
- planning_info: dettagli della pianificazione (numero di post richiesti, topic gia'
  coperti al momento della pianificazione, ecc.)
- post_plan: l'output vero e proprio del Planner - sequenza di post futuri pianificati
"""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    user_input: str
    reasoning_trace: list[str]
    tool_outputs: list[dict[str, Any]]
    kg_summary: dict[str, Any]
    planning_info: dict[str, Any]
    post_plan: list[dict[str, Any]]
