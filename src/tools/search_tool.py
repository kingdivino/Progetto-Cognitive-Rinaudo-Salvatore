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

# Livelli di affidabilita' delle fonti web (solo URL - RAG/KG sono dati locali
# strutturati, questione diversa). Pubblici (non in research.py) perche' servono a
# due scopi: classificare a posteriori una fonte gia' trovata (research.py importa
# da qui) e restringere a priori cosa Tavily puo' restituire (sotto) - un'unica
# lista invece di due copie che rischierebbero di disallinearsi.
#
# Classificazione MECCANICA dal dominio dell'URL (nessun giudizio semantico, per
# questo verificata in codice e non lasciata al prompt):
# - "aggregato": siti di statistiche/decklist aggregate su molti giocatori
#   (hsreplay.net, hearthpwn.com, metastats.net, hearthstone.wiki.gg) - un dato su
#   un campione ampio, non l'opinione di una persona sola.
# - "ufficiale": domini Blizzard di annunci/patch notes (non i forum, che restano
#   contenuto generato dagli utenti anche se ospitato su un dominio Blizzard).
# - "opinione_singola": contenuto generato da un singolo autore senza aggregazione
#   (video, forum, reddit, social) - un'esperienza/opinione individuale, non un dato
#   di popolazione. Esclusa di proposito da TRUSTED_SEARCH_DOMAINS sotto.
SOURCE_TIER_DOMAINS = {
    "aggregato": ("hsreplay.net", "hearthpwn.com", "metastats.net", "hearthstone.wiki.gg"),
    "ufficiale": ("playhearthstone.com", "news.blizzard.com"),
    "opinione_singola": (
        "youtube.com", "youtu.be", "reddit.com", "facebook.com",
        "forums.blizzard.com", "twitter.com", "x.com",
    ),
}

# Domini a cui restringere Tavily: SOLO "aggregato" + "ufficiale", MAI
# "opinione_singola" (dopo un run in cui search_web restituiva quasi solo video
# YouTube/post Reddit) - piu' affidabile chiedere a Tavily di non restituire quei
# domini che chiedere all'LLM di ignorarli una volta visti.
#
# Compromesso reale: restringere aumenta la probabilita' di "nessun risultato" per
# query molto specifiche non coperte da questi domini - preferibile comunque a un
# risultato di bassa qualita' citato come autorevole; se capita spesso in pratica,
# vale la pena ampliare la lista invece di rimuovere la restrizione.
#
# _run_tavily_search sotto e' protetto con un fallback (TypeError) nel caso la
# versione installata di langchain_tavily non supporti include_domains.
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
        # Vedi nota sopra: fallback se questa versione di langchain_tavily non
        # accetta include_domains - meglio una ricerca non restretta che un errore.
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
