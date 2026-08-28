"""
Script di verifica ambiente — NON è parte della pipeline finale.
Serve solo a controllare che LangGraph + Ollama + un tool funzionino insieme
prima di costruire l'architettura vera del progetto.

Come eseguirlo (dopo aver seguito il setup):
    1. Assicurati che Ollama sia avviato e che tu abbia scaricato un modello, es.:
       ollama pull llama3.1:8b
    2. Attiva il virtual environment e installa requirements.txt
    3. python notebooks/00_hello_langgraph.py
"""

import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

load_dotenv()  # legge il file .env

# 1. Lo STATE: cosa si porta dietro l'agente lungo il grafo.
# "messages" è la cronologia della conversazione; add_messages sa come
# accumulare i nuovi messaggi invece di sovrascriverli.
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# 2. Un TOOL di prova — finge di cercare "eventi Hearthstone".
# In un caso reale sarebbe una vera chiamata a Tavily.
@tool
def fake_search_tool(query: str) -> str:
    """Cerca informazioni su un argomento legato a Hearthstone (VERSIONE FINTA, solo per test)."""
    return f"[risultato finto] Non ci sono eventi rilevanti trovati per: '{query}'"


# 3. Il modello locale via Ollama, con il tool collegato
llm = ChatOllama(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    temperature=0,
)
llm_with_tools = llm.bind_tools([fake_search_tool])


# 4. Un NODE: riceve lo stato, chiama il modello, restituisce lo stato aggiornato
def call_model(state: AgentState) -> AgentState:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# 5. Costruzione del grafo: un solo nodo, che parte e finisce (per ora)
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.set_entry_point("agent")
graph.add_edge("agent", END)
app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({
        "messages": [("user", "Ci sono eventi Hearthstone interessanti questa settimana?")]
    })
    print("\n--- Risposta finale ---")
    print(result["messages"][-1].content)
    print("\n--- Ha chiamato il tool? ---")
    last_msg = result["messages"][-1]
    print(bool(getattr(last_msg, "tool_calls", None)))
