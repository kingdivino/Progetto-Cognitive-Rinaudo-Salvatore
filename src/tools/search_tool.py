"""
Search tool (uno dei 3 tool minimi obbligatori dalla specifica) - ricerca web via
Tavily, per verificare fatti recenti/attuali che il RAG locale (statico, basato su
HearthstoneJSON) non puo' coprire: espansioni, patch, tornei, annunci ufficiali.

Il parametro `justification` e' obbligatorio per requisito di progetto ("ogni
invocazione di tool va giustificata") - non e' un dettaglio decorativo: il nodo
Research/ReAct lo registra nel reasoning_trace cosi' si vede *perche'* l'agente ha
deciso di cercare quella cosa, non solo cosa ha trovato. Il tool ufficiale
langchain_tavily.TavilySearch non ha questo parametro nel suo schema, quindi lo
wrappiamo in un tool nostro invece di esporlo direttamente all'LLM.
"""
from __future__ import annotations

from langchain_core.tools import tool


def _run_tavily_search(query: str):
    from langchain_tavily import TavilySearch

    # TAVILY_API_KEY letta automaticamente dall'ambiente (deve essere nel .env).
    tavily = TavilySearch(max_results=5, search_depth="advanced", topic="general")
    return tavily.invoke({"query": query})


@tool
def search_web(query: str, justification: str) -> str:
    """Cerca sul web informazioni recenti/attuali su Hearthstone (espansioni, patch,
    tornei, annunci ufficiali) che un corpus statico locale non potrebbe coprire.
    Usa questo tool SOLO per fatti specifici e verificabili di cui non hai gia' i dati
    (non per informazioni generiche sulle meccaniche delle carte, per quelle usa il
    tool RAG locale, piu' verificabile e senza rischio di risultati di bassa qualita').

    Args:
        query: la query di ricerca, specifica e mirata (non generica).
        justification: perche' serve questa ricerca ora - quale gap informativo
            colma, per quale claim del post servira'. Obbligatoria.
    """
    try:
        raw = _run_tavily_search(query)
    except Exception as e:
        return f"[ERROR] Ricerca fallita per {query!r}: {e}"

    hits = raw.get("results", []) if isinstance(raw, dict) else []
    if not hits:
        return f"Nessun risultato trovato per la query: {query!r}"

    formatted = [f"Risultati di ricerca per {query!r} (giustificazione: {justification}):"]
    for i, hit in enumerate(hits, start=1):
        title = hit.get("title", "(senza titolo)")
        url = hit.get("url", "")
        content = (hit.get("content") or "")[:500]
        formatted.append(f"{i}. {title} — {url}\n   {content}")
    return "\n".join(formatted)
