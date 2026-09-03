"""
Knowledge Graph tool (uno dei 3 tool minimi obbligatori dalla specifica) - query
(non scrittura) sul Knowledge Graph Neo4j per il nodo Research/ReAct.

Solo QUERY qui, non update: le specifiche impongono che il KG si aggiorni SOLO dopo
approvazione umana del post (human-in-the-loop) - la scrittura vera e propria vive nel
futuro nodo "KG Update" (roadmap punto 5, dopo il nodo Human Review), non qui. Questo
tool serve al Research/ReAct per leggere lo stato attuale del grafo (coerenza con post
precedenti, gap di copertura) e per il K-RAG: usare quello che trova qui per
raffinare la query da mandare al RAG (search_card_knowledge) o al search tool -
istruzione esplicita nel prompt del nodo Research (src/agent/research.py), non
logica automatica dentro al tool stesso.

Il parametro `justification` e' obbligatorio per lo stesso motivo degli altri due
tool minimi.
"""
from __future__ import annotations

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
