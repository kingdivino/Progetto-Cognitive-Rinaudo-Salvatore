"""
Knowledge Graph tool (uno dei 3 tool minimi obbligatori dalla specifica: "Knowledge
Graph tool (query + update)") - espone sia la query (usata da Research/Format) sia
l'update (usato dal nodo KG Update, roadmap punto 8) sullo stesso grafo Neo4j.

query_knowledge_graph serve al Research/ReAct per leggere lo stato attuale del grafo
(coerenza con post precedenti, gap di copertura) e per il K-RAG: usare quello che
trova qui per raffinare la query da mandare al RAG (search_card_knowledge) o al
search tool - istruzione esplicita nel prompt del nodo Research, non logica
automatica dentro al tool stesso.

update_knowledge_graph scrive un post nel grafo - MA va chiamato SOLO dal nodo KG
Update (src/agent/kg_update.py), e SOLO dopo che il nodo Human Review ha registrato
un'approvazione esplicita (requisito della specifica: "Il KG si aggiorna SOLO dopo
approvazione"). Stesso principio "chiamata fissa in codice, non lasciata alla
discrezione di un ciclo ReAct" gia' consolidato in questo progetto per le chiamate
fisse a query_knowledge_graph in Research/Format - qui a maggior ragione, dato che
scrivere sul KG senza una vera approvazione violerebbe un requisito esplicito.

Il parametro `justification` e' obbligatorio su entrambi i tool per lo stesso motivo
degli altri tool minimi (ogni invocazione di tool va giustificata).
"""
from __future__ import annotations

import datetime

from langchain_core.tools import tool

from src.kg import connection as kg


@tool
def query_knowledge_graph(justification: str) -> str:
    """Interroga il Knowledge Graph editoriale: topic gia' coperti dai post
    precedenti e gli ultimi post pubblicati (topic/tipo/data). Usa questo tool
    all'INIZIO della ricerca per un post, prima di search_web o
    search_card_knowledge: il risultato aiuta a capire cosa e' gia' stato trattato
    (per evitare ripetizioni/coerenza con contenuti precedenti) e puo' suggerire
    termini piu' mirati da usare nelle ricerche successive (K-RAG: il contesto del
    KG raffina la query di retrieval, non e' un lookup a vuoto).

    Se il KG non e' raggiungibile (es. Neo4j Desktop non avviato) o e' ancora vuoto
    (nessun nodo "KG Update" lo ha mai popolato), lo dice esplicitamente invece di
    fallire silenziosamente - il grafo vuoto e' uno stato atteso finche' non esiste
    ancora il nodo KG Update.

    Args:
        justification: perche' serve interrogare il KG in questo momento.
            Obbligatoria.
    """
    if not kg.check_connection():
        return (
            "[KG non raggiungibile] Neo4j non e' avviato o il .env non e' configurato "
            "- si procede assumendo nessuno storico pregresso."
        )

    covered_topics = kg.get_covered_topics()
    recent_posts = kg.get_recent_posts(limit=5)

    if not covered_topics and not recent_posts:
        return (
            "[KG raggiungibile ma vuoto] Nessun topic o post ancora presente nel grafo "
            "(normale finche' il nodo KG Update non ha mai scritto nulla) - nessuna "
            "ripetizione possibile per ora, procedi senza vincoli di copertura pregressa."
        )

    lines = [f"Topic gia' coperti nel KG ({len(covered_topics)}): {', '.join(covered_topics) or 'nessuno'}"]
    lines.append(f"Post recenti ({len(recent_posts)}):")
    for post in recent_posts:
        lines.append(f"  - [{post.get('tipo')}] {post.get('topic')} ({post.get('created_at')})")
    return "\n".join(lines)


@tool
def update_knowledge_graph(
    justification: str,
    post_id: str,
    tipo: str,
    topic: str,
    summary: str,
    sources: list[dict],
    claims: list[dict],
    classe: str | None = None,
    formato: str | None = None,
) -> str:
    """Scrive un post APPROVATO nel Knowledge Graph editoriale: nodo Post, nodo Topic
    (con classe/formato se noti), fonti citate, claim chiave con la fonte a supporto,
    e relazioni RELATED_TO tra Topic della stessa classe di mazzo.

    ATTENZIONE (vedi docstring del modulo): questo tool va invocato SOLO dal nodo KG
    Update, SOLO dopo un'approvazione umana esplicita registrata dal nodo Human
    Review - mai in autonomia da un ciclo ReAct.

    Args:
        justification: perche' si sta scrivendo ora (per coerenza con gli altri
            tool - qui la chiamata e' comunque sempre fissa in codice, mai una
            decisione del modello).
        post_id: identificatore univoco del post (generato dal chiamante).
        tipo: tipo di post (how-to/review/news/eventi).
        topic: argomento del post.
        summary: testo del post approvato.
        sources: fonti effettivamente citate nel post pubblicato, come
            [{"ref": "<stringa fonte>", "tier": "<esito classify_source_tier>"}].
        claims: claim chiave del post con la fonte a supporto, come
            [{"text": "...", "source": "<deve corrispondere a un ref in sources>"}].
        classe: classe Hearthstone del mazzo trattato, se nota (es. "PRIEST").
        formato: formato di gioco del post, se noto ("standard"/"wild"/"misto").
    """
    if not kg.check_connection():
        return (
            "[KG non raggiungibile] Neo4j non e' avviato o il .env non e' configurato "
            "- il post NON e' stato scritto sul grafo (nessuna perdita silenziosa: "
            "l'aggiornamento va semplicemente rilanciato quando Neo4j e' di nuovo "
            "raggiungibile)."
        )
    kg.ensure_schema()
    n_related = kg.write_approved_post(
        post_id=post_id,
        tipo=tipo,
        topic=topic,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        summary=summary,
        classe=classe,
        formato=formato,
        sources=sources,
        claims=claims,
    )
    # Il messaggio riflette il conteggio reale di n_related, non solo se 'classe' e'
    # nota. Caso singolare gestito a parte: "relazione"/"creata" non condividono lo
    # stesso suffisso plurale di "relazioni"/"create", quindi serve un branch esplicito
    # invece di un'unica variabile plurale applicata a entrambe le parole.
    if classe and n_related == 1:
        extra = f", 1 relazione RELATED_TO creata con altri topic della classe {classe}"
    elif classe and n_related:
        extra = f", {n_related} relazioni RELATED_TO create con altri topic della classe {classe}"
    elif classe:
        extra = f", nessuna relazione RELATED_TO creata (nessun altro topic della classe {classe} ancora nel grafo)"
    else:
        extra = ""
    return f"Post '{topic}' scritto nel KG (id={post_id}): {len(sources)} fonti, {len(claims)} claim collegati{extra}."

