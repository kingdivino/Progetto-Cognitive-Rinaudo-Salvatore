"""Tool di ricerca web via Tavily, per fatti recenti che il RAG locale non copre
(uno dei 3 tool minimi richiesti). Wrappato per esporre il parametro
`justification`, assente nello schema di langchain_tavily.TavilySearch."""
from __future__ import annotations

from langchain_core.tools import tool

# Livelli di affidabilita' delle fonti web per dominio (pubblici, non in
# research.py, perche' servono anche a restringere Tavily sotto): "aggregato"
# (statistiche/decklist su molti giocatori), "ufficiale" (domini Blizzard di
# annunci/patch notes), "opinione_singola" (contenuto di un singolo autore, esclusa
# da TRUSTED_SEARCH_DOMAINS).
SOURCE_TIER_DOMAINS = {
    "aggregato": ("hsreplay.net", "hearthpwn.com", "metastats.net", "hearthstone.wiki.gg"),
    "ufficiale": ("playhearthstone.com", "news.blizzard.com"),
    "opinione_singola": (
        "youtube.com", "youtu.be", "reddit.com", "facebook.com",
        "forums.blizzard.com", "twitter.com", "x.com",
    ),
}

# Domini a cui restringere Tavily: solo "aggregato" + "ufficiale", mai
# "opinione_singola" - piu' affidabile impedire a Tavily di restituire quei domini
# che chiedere all'LLM di ignorarli una volta visti.
TRUSTED_SEARCH_DOMAINS = SOURCE_TIER_DOMAINS["aggregato"] + SOURCE_TIER_DOMAINS["ufficiale"]


def _run_tavily_search(query: str):
    from langchain_tavily import TavilySearch

    # TAVILY_API_KEY letta automaticamente dall'ambiente (deve essere nel .env).
    kwargs = dict(
        max_results=5, search_depth="advanced", topic="general",
        include_domains=list(TRUSTED_SEARCH_DOMAINS),
    )
    try:
        tavily = TavilySearch(**kwargs)
    except TypeError:
        # Fallback se questa versione di langchain_tavily non accetta
        # include_domains - meglio una ricerca non ristretta che un errore.
        kwargs.pop("include_domains", None)
        tavily = TavilySearch(**kwargs)
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
        # Tavily include a volte una data di pubblicazione ("published_date"): senza
        # esporla il modello non ha modo strutturato di sapere quanto sia vecchia una
        # fonte (osservato: un articolo datato presentato come notizia recente). Non
        # sempre presente - la si aggiunge solo quando c'e'.
        published = hit.get("published_date")
        date_note = f" [pubblicato: {published}]" if published else ""
        formatted.append(f"{i}. {title}{date_note} — {url}\n   {content}")
    return "\n".join(formatted)
