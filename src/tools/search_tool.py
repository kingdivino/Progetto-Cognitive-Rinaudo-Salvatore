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

# Livelli di affidabilita' delle fonti web (solo per fonti di tipo URL - RAG/KG sono
# dati locali strutturati, questione diversa). Vivevano prima come copia privata in
# src/agent/research.py (classify_source_tier) - spostati QUI e resi pubblici
# (senza underscore) l'11/09/2026 perche' ora servono a DUE scopi, non solo uno:
# classificare a posteriori una fonte gia' trovata (research.py, che li importa da
# qui) E restringere A PRIORI cosa Tavily puo' restituire (search_web sotto, vedi
# perche' sotto) - un'unica lista invece di due copie che rischierebbero di
# disallinearsi (stesso principio "una sola fonte di verita'" gia' seguito altrove
# nel progetto, es. il conteggio RELATED_TO thread-ato dal solo punto di scrittura).
#
# Classificazione MECCANICA dal dominio dell'URL (nessun giudizio semantico
# richiesto - per questo verificata in codice e non lasciata al prompt, stesso
# principio di source_well_formed in research.py):
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

# Domini a cui restringere Tavily (richiesta esplicita dell'utente l'11/09/2026, dopo
# un run in cui search_web ha restituito quasi solo video YouTube/post Reddit,
# classificati "opinione_singola" - vedi guida di progetto): SOLO "aggregato" +
# "ufficiale", MAI "opinione_singola" - e' esattamente questo secondo gruppo che si
# vuole escludere. "codice invece di prompt", stesso principio del resto del
# progetto: piu' affidabile chiedere a Tavily di NON restituire altri domini che
# chiedere all'LLM di ignorarli una volta che li ha gia' visti nei risultati.
#
# ATTENZIONE - compromesso reale, non solo teorico: restringere aumenta la
# probabilita' di "nessun risultato trovato" per query molto specifiche non coperte
# da questi 6 domini (es. un torneo minore, un annuncio molto recente non ancora
# indicizzato li'). Preferibile comunque a un risultato di bassa qualita' che il
# modello potrebbe citare come se fosse autorevole - se in pratica capita spesso,
# vale la pena ampliare questa lista (o farla configurabile) invece di rimuovere la
# restrizione del tutto.
#
# NOTA - non verificato dal vivo: il parametro include_domains di
# langchain_tavily.TavilySearch non e' stato testato in questa sandbox (PyPI non e'
# raggiungibile da qui - vedi guida di progetto) - risulta dalla documentazione
# dell'API Tavily, che lo supporta da tempo. _run_tavily_search sotto e' comunque
# protetto con un fallback (TypeError) nel caso la versione installata non lo
# accetti, per non far crashare il nodo Research per questo.
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
        # Tavily a volte include una data di pubblicazione per risultato (campo
        # "published_date") - fino al 10/09/2026 veniva scartata qui, quindi il
        # modello non aveva NESSUN modo strutturato di sapere quanto fosse vecchia una
        # fonte (osservato: un articolo del 2023 presentato nel post come notizia
        # "degli ultimi mesi", con l'anno riconoscibile solo per caso perche' compariva
        # nell'URL). Non e' sempre presente (dipende dal sito indicizzato da Tavily) -
        # la si aggiunge solo quando c'e', senza inventare un "data non disponibile"
        # per ogni risultato che ne e' privo.
        published = hit.get("published_date")
        date_note = f" [pubblicato: {published}]" if published else ""
        formatted.append(f"{i}. {title}{date_note} — {url}\n   {content}")
    return "\n".join(formatted)
