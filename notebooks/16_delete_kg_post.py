"""
Cancellazione manuale e sicura di un post gia' approvato/scritto nel Knowledge
Graph (roadmap: strumento di correzione a posteriori, richiesto dall'utente il
19/09/2026 dopo che notebooks/15_audit_kg_claims.py ha confermato un post
fabbricato gia' approvato - "Tempo Priest 26.1%", vedi addendum di progetto).

NON e' pensato per uso ricorrente nella pipeline (nessun nodo del grafo
LangGraph lo chiama) - e' un'utility da lanciare a mano quando l'audit trova un
post che va tolto dal grafo. Mostra SEMPRE un'anteprima di cosa verrebbe
cancellato e chiede conferma esplicita (stesso principio di
"digita per confermare" gia' usato in Human Review, notebooks/09 e 10) prima di
eseguire qualunque DELETE - un'azione distruttiva e irreversibile sul KG non
deve mai partire senza una conferma inequivocabile.

Cosa cancella, e cosa NO:
- Il nodo Post (con tutte le sue relazioni: ABOUT, CITES, MAKES_CLAIM).
- Il nodo Topic collegato, MA SOLO se nessun ALTRO post lo usa (altrimenti
  cancellarlo romperebbe il collegamento di un post legittimo) - in quel caso
  viene lasciato intatto e lo si segnala.
- I nodi Claim collegati, MA SOLO se nessun ALTRO post condivide lo stesso
  claim (stessa logica di sicurezza del Topic).
- MAI i nodi Source (es. 'DATI-HSREPLAY', 'RAG: <carta>') - sono etichette
  generiche condivise da molti post legittimi, cancellarle romperebbe le
  citazioni di post corretti.

Uso:
    python -u notebooks/16_delete_kg_post.py <post_id>

Esempio (il post fabbricato scoperto il 19/09/2026):
    python -u notebooks/16_delete_kg_post.py 42683e12-57be-45c6-ba4f-04acceb52cec
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kg import connection as kg


def _anteprima(session, post_id: str) -> dict | None:
    record = session.run(
        "MATCH (p:Post {id: $post_id}) "
        "OPTIONAL MATCH (p)-[:ABOUT]->(t:Topic) "
        "OPTIONAL MATCH (p)-[:MAKES_CLAIM]->(c:Claim) "
        "OPTIONAL MATCH (p)-[:CITES]->(s:Source) "
        "OPTIONAL MATCH (other:Post)-[:ABOUT]->(t) WHERE other <> p "
        "OPTIONAL MATCH (otherc:Post)-[:MAKES_CLAIM]->(c) WHERE otherc <> p "
        "RETURN p.topic AS topic, p.tipo AS tipo, p.created_at AS created_at, "
        "t.name AS topic_name, count(DISTINCT other) AS altri_post_su_topic, "
        "collect(DISTINCT c.text) AS claims, "
        "count(DISTINCT otherc) AS altri_post_su_claim, "
        "collect(DISTINCT s.ref) AS fonti",
        post_id=post_id,
    ).single()
    return dict(record) if record else None


def _elimina(session, post_id: str) -> None:
    # Claim: cancellati solo se non condivisi con NESSUN altro post.
    session.run(
        "MATCH (p:Post {id: $post_id})-[:MAKES_CLAIM]->(c:Claim) "
        "WHERE NOT EXISTS { MATCH (other:Post)-[:MAKES_CLAIM]->(c) WHERE other <> p } "
        "DETACH DELETE c",
        post_id=post_id,
    )
    # Topic: cancellato solo se nessun altro post lo referenzia.
    session.run(
        "MATCH (p:Post {id: $post_id})-[:ABOUT]->(t:Topic) "
        "WHERE NOT EXISTS { MATCH (other:Post)-[:ABOUT]->(t) WHERE other <> p } "
        "DETACH DELETE t",
        post_id=post_id,
    )
    # Post: sempre cancellato (con le relazioni residue, es. CITES verso Source
    # condivisi - i nodi Source non vengono mai toccati).
    session.run("MATCH (p:Post {id: $post_id}) DETACH DELETE p", post_id=post_id)


def main() -> None:
    if len(sys.argv) != 2:
        print("Uso: python -u notebooks/16_delete_kg_post.py <post_id>")
        return
    post_id = sys.argv[1]

    if not kg.check_connection():
        print("[ERRORE] Neo4j non raggiungibile - avvia Neo4j Desktop e riprova.")
        return

    with kg.get_driver().session() as session:
        info = _anteprima(session, post_id)
        if info is None or info.get("topic") is None:
            print(f"[INFO] Nessun post con id={post_id!r} trovato nel KG - niente da fare.")
            return

        print("=== Anteprima di cio' che verra' cancellato ===")
        print(f"Post: {info['topic']!r} (tipo={info['tipo']}, creato={info['created_at']})")
        print(f"Fonti citate (NON verranno toccate, sono condivise): {info['fonti']}")
        print(f"Claim collegati ({len(info['claims'])}):")
        for c in info["claims"]:
            print(f"  - {c!r}")
        if info["altri_post_su_claim"] > 0:
            print(
                f"  ATTENZIONE: {info['altri_post_su_claim']} altro/i post condivide/ono "
                "almeno uno di questi claim per testo IDENTICO - quel/quei claim NON "
                "verra'/verranno cancellato/i, solo scollegato/i da questo post."
            )
        print(f"Topic: {info['topic_name']!r}")
        if info["altri_post_su_topic"] > 0:
            print(
                f"  ATTENZIONE: {info['altri_post_su_topic']} altro/i post usa/no lo stesso "
                "Topic - NON verra' cancellato, solo scollegato da questo post."
            )
        print()

        conferma = input(
            "Digita ESATTAMENTE 'elimina' per procedere con la cancellazione "
            "(qualunque altra cosa annulla): "
        ).strip()
        if conferma != "elimina":
            print("Annullato - nessuna modifica al KG.")
            return

        _elimina(session, post_id)
        print(f"[OK] Post id={post_id!r} cancellato dal KG (con Topic/Claim non condivisi).")


if __name__ == "__main__":
    main()
