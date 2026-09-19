"""Connessione a Neo4j + query di base per il Knowledge Graph editoriale, scritto
per la prima volta dal nodo KG Update solo dopo approvazione umana. Schema:

    (:Post {id, tipo, topic, created_at, summary})
    (:Topic {name, classe, formato})
    (:Source {ref, tier})
    (:Claim {text})
    (Post)-[:ABOUT]->(Topic)
    (Post)-[:CITES]->(Source)
    (Post)-[:MAKES_CLAIM]->(Claim)
    (Claim)-[:SUPPORTED_BY]->(Source)
    (Topic)-[:RELATED_TO]->(Topic)   # topic con la stessa classe di mazzo
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(override=True)

_driver = None


def get_driver():
    """Driver Neo4j, creato una sola volta (lazy singleton) e riusato tra le chiamate."""
    global _driver
    if _driver is None:
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USERNAME", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "")
        _driver = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


def check_connection() -> bool:
    """True se Neo4j e' raggiungibile e le credenziali sono valide. Non solleva mai
    eccezioni: i nodi del grafo (es. il Planner) devono poter continuare a funzionare
    anche se Neo4j Desktop non e' avviato, trattando il KG come vuoto."""
    try:
        get_driver().verify_connectivity()
        return True
    except Exception:
        return False


def ensure_schema() -> None:
    """Crea i vincoli di unicita' se non esistono gia' (idempotente - va chiamato una
    volta all'avvio, prima di scrivere nel grafo)."""
    with get_driver().session() as session:
        session.run("CREATE CONSTRAINT post_id IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE")
        session.run("CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE")


def get_covered_topics() -> list[str]:
    """Tutti i topic gia' presenti nel KG - usato dal Planner per trovare gap di
    copertura ed evitare di ripianificare lo stesso argomento."""
    with get_driver().session() as session:
        result = session.run("MATCH (t:Topic) RETURN t.name AS name ORDER BY t.name")
        return [record["name"] for record in result]


def get_recent_posts(limit: int = 5) -> list[dict]:
    """Gli ultimi N post pubblicati (per topic/tipo/data) - usato dal Planner per non
    riproporre lo stesso argomento troppo a ridosso di uno gia' pubblicato."""
    with get_driver().session() as session:
        result = session.run(
            "MATCH (p:Post)-[:ABOUT]->(t:Topic) "
            "RETURN p.topic AS topic, p.tipo AS tipo, p.created_at AS created_at "
            "ORDER BY p.created_at DESC LIMIT $limit",
            limit=limit,
        )
        return [dict(record) for record in result]


def write_approved_post(
    *,
    post_id: str,
    tipo: str,
    topic: str,
    created_at: str,
    summary: str,
    classe: str | None,
    formato: str | None,
    sources: list[dict],
    claims: list[dict],
) -> int:
    """Scrive un post approvato nel KG - unico punto di scrittura del grafo,
    chiamato solo da update_knowledge_graph dopo approvazione umana. Idempotente sul
    post_id (MERGE): una doppia scrittura aggiorna le stesse proprieta' invece di
    duplicare il nodo.

    Args:
        post_id: identificatore univoco del post.
        tipo, topic, created_at, summary: proprieta' del nodo Post.
        classe: classe Hearthstone del mazzo trattato dal post, se nota - usata anche
            per collegare il Topic ad altri Topic della stessa classe (RELATED_TO).
        formato: formato di gioco del post, se noto ("standard"/"wild"/"misto").
        sources: fonti citate nel post pubblicato, come [{"ref": ..., "tier": ...}].
        claims: claim chiave del post, come [{"text": ..., "source": ...}] - 'source'
            deve corrispondere esattamente a un 'ref' in sources per essere collegato
            (SUPPORTED_BY); se non corrisponde a nessuna fonte nota, il claim viene
            comunque scritto ma senza quel collegamento.

    Returns:
        Il numero di altri Topic della stessa 'classe' effettivamente collegati via
        RELATED_TO (0 se 'classe' non e' nota, o se questo e' il primo Topic mai
        scritto per quella classe - non c'e' ancora nulla a cui collegarlo). Il
        chiamante (update_knowledge_graph in kg_tool.py) usa questo numero per non
        dichiarare relazioni create quando in realta' non ce ne sono state.
    """
    with get_driver().session() as session:
        return session.execute_write(
            _write_approved_post_tx,
            post_id, tipo, topic, created_at, summary, classe, formato, sources, claims,
        )


def _write_approved_post_tx(tx, post_id, tipo, topic, created_at, summary, classe, formato, sources, claims):
    tx.run(
        "MERGE (p:Post {id: $post_id}) "
        "SET p.tipo = $tipo, p.topic = $topic, p.created_at = $created_at, p.summary = $summary",
        post_id=post_id, tipo=tipo, topic=topic, created_at=created_at, summary=summary,
    )
    tx.run(
        "MERGE (t:Topic {name: $topic}) "
        "SET t.classe = coalesce($classe, t.classe), t.formato = coalesce($formato, t.formato) "
        "WITH t MATCH (p:Post {id: $post_id}) MERGE (p)-[:ABOUT]->(t)",
        topic=topic, classe=classe, formato=formato, post_id=post_id,
    )
    for source in sources:
        tx.run(
            "MERGE (s:Source {ref: $ref}) SET s.tier = coalesce($tier, s.tier) "
            "WITH s MATCH (p:Post {id: $post_id}) MERGE (p)-[:CITES]->(s)",
            ref=source.get("ref"), tier=source.get("tier"), post_id=post_id,
        )
    for claim in claims:
        tx.run(
            "MERGE (c:Claim {text: $text}) "
            "WITH c MATCH (p:Post {id: $post_id}) MERGE (p)-[:MAKES_CLAIM]->(c) "
            "WITH c "
            "OPTIONAL MATCH (s:Source {ref: $source}) "
            "FOREACH (_ IN CASE WHEN s IS NULL THEN [] ELSE [1] END | MERGE (c)-[:SUPPORTED_BY]->(s))",
            text=claim.get("text"), source=claim.get("source"), post_id=post_id,
        )
    if not classe:
        return 0
    result = tx.run(
        "MATCH (t:Topic {name: $topic}) "
        "MATCH (other:Topic) WHERE other.classe = $classe AND other.name <> $topic "
        "MERGE (t)-[:RELATED_TO]->(other) MERGE (other)-[:RELATED_TO]->(t) "
        "RETURN count(other) AS n",
        topic=topic, classe=classe,
    )
    record = result.single()
    return record["n"] if record else 0

