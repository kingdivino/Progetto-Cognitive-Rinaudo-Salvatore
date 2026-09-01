"""
Connessione a Neo4j + query di base per il Knowledge Graph editoriale.

Schema minimo del grafo (creato incrementalmente man mano che i nodi che lo popolano
vengono costruiti - per ora solo letto dal Planner, che quindi vede un grafo vuoto
finche' non esiste ancora un nodo "KG Update"):

    (:Post {id, tipo, topic, created_at, summary})
    (:Topic {name})
    (:Source {url, title})
    (:Claim {text})
    (Post)-[:ABOUT]->(Topic)
    (Post)-[:CITES]->(Source)
    (Post)-[:MAKES_CLAIM]->(Claim)
    (Claim)-[:SUPPORTED_BY]->(Source)

Gotcha da questo progetto (vedi guida di progetto per i dettagli): load_dotenv va
chiamato con override=True per evitare che variabili d'ambiente di sistema (Windows)
con lo stesso nome abbiano precedenza silenziosa sul file .env del progetto.
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
