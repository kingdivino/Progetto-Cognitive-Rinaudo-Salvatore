"""
Audit dei post GIA' APPROVATI e scritti sul Knowledge Graph (roadmap: verifica
a posteriori, richiesta dall'utente il 19/09/2026 dopo aver notato che un post
approvato - "Come costruire un mazzo Tempo Priest Standard con un winrate del
26.1%" - citava un winrate palesemente inventato dal Planner, mai confermato da
nessun dato reale, ma comunque passato indenne dal grounding a causa di un bug
scoperto lo stesso giorno - vedi addendum di progetto per i dettagli).

Cosa fa: legge OGNI post gia' scritto nel KG (quindi gia' approvato da un umano
in Human Review) insieme ai suoi claim, estrae qualunque percentuale citata nel
testo di ciascun claim, e la confronta con le statistiche REALI calcolate sullo
stesso identico dataset usato dal tool get_archetype_stats (data/processed/
finetune_dataset.csv) per la classe/formato di quel post. Se un claim cita una
percentuale che NON corrisponde (con una tolleranza di arrotondamento) al
winrate medio o mediano reale di quella classe/formato, viene segnalato come
sospetto - stesso identico principio di grounding usato in tempo reale da
_source_is_grounded() in src/agent/research.py, applicato qui A POSTERIORI su
tutto cio' che e' gia' stato scritto nel grafo, per scoprire se il bug della
giustificazione echeggiata (vedi addendum) ha lasciato passare qualcos'altro
oltre al caso "Tempo Priest 26.1%" gia' scoperto dal vivo.

NON modifica il KG - e' un controllo di sola lettura. Se emergono post sospetti,
la decisione su cosa farne (correggere il testo, cancellare il nodo, lasciarlo
con una nota per il report finale come "caso di studio di un fix imperfetto")
resta dell'utente.

Uso: eseguire direttamente (richiede Neo4j Desktop avviato, come per gli altri
notebook che toccano il KG):
    python -u notebooks/15_audit_kg_claims.py
"""
from __future__ import annotations

import os
import re
import sys

# Permette di eseguire lo script sia da dentro notebooks/ sia dalla root del
# progetto, stesso accorgimento usato negli altri file in questa cartella.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.kg import connection as kg
from src.tools.stats_tool import PROCESSED_PATH

_PCT_RE = re.compile(r"\d+(?:[.,]\d+)?%")


def _estrai_percentuali(testo: str) -> set[str]:
    return {m.replace(",", ".") for m in _PCT_RE.findall(testo or "")}


def _statistiche_reali_per_classe(df: pd.DataFrame) -> dict[tuple[str, str], dict]:
    """Precalcola winrate medio/mediano REALI per ogni combinazione (classe,
    formato) presente nel dataset - stessa aggregazione esatta di
    get_archetype_stats in src/tools/stats_tool.py, cosi' il confronto usa
    davvero gli stessi numeri che il tool avrebbe restituito in ricerca."""
    stats: dict[tuple[str, str], dict] = {}
    for (classe, formato), gruppo in df.groupby([df["deck_class"].str.upper(), "formato"]):
        stats[(classe, str(formato).strip().lower())] = {
            "winrate_medio": round(gruppo["winrate"].mean(), 1),
            "winrate_mediano": round(gruppo["winrate"].median(), 1),
            "n_mazzi": len(gruppo),
        }
    return stats


def main() -> None:
    if not kg.check_connection():
        print(
            "[ERRORE] Neo4j non raggiungibile - avvia Neo4j Desktop e controlla il "
            ".env prima di rilanciare questo audit."
        )
        return

    try:
        df = pd.read_csv(PROCESSED_PATH)
    except Exception as e:
        print(f"[ERRORE] impossibile leggere {PROCESSED_PATH!r}: {e}")
        return
    stats_reali = _statistiche_reali_per_classe(df)

    with kg.get_driver().session() as session:
        risultati = session.run(
            "MATCH (p:Post)-[:ABOUT]->(t:Topic) "
            "OPTIONAL MATCH (p)-[:MAKES_CLAIM]->(c:Claim) "
            "RETURN p.id AS id, p.tipo AS tipo, p.topic AS topic, "
            "p.created_at AS created_at, t.classe AS classe, t.formato AS formato, "
            "collect(c.text) AS claims"
        )
        posts = [dict(r) for r in risultati]

    print(f"=== Audit di {len(posts)} post approvati presenti nel KG ===\n")

    n_sospetti = 0
    for post in posts:
        classe = (post.get("classe") or "").strip().upper()
        formato = (post.get("formato") or "").strip().lower()
        reali = stats_reali.get((classe, formato)) if classe else None
        _pcts_reali_valide = (
            {str(reali["winrate_medio"]), str(reali["winrate_mediano"])} if reali else set()
        )

        claims_sospetti = []
        for claim_text in post.get("claims") or []:
            pcts_claim = _estrai_percentuali(claim_text)
            if not pcts_claim:
                continue
            if not reali or not pcts_claim.issubset(_pcts_reali_valide):
                claims_sospetti.append((claim_text, pcts_claim))

        if claims_sospetti:
            n_sospetti += 1
            print(f"--- SOSPETTO: post id={post.get('id')} ---")
            print(f"    Topic: {post.get('topic')} (tipo={post.get('tipo')}, "
                  f"creato={post.get('created_at')})")
            print(f"    Classe/formato rilevati: {classe or '?'}/{formato or '?'}"
                  + (f" (dati reali: medio {reali['winrate_medio']}%, "
                     f"mediano {reali['winrate_mediano']}%, {reali['n_mazzi']} mazzi)"
                     if reali else " (nessun dato reale disponibile per confronto)"))
            for testo, pcts in claims_sospetti:
                print(f"    - claim: {testo!r}")
                print(f"      percentuali citate: {sorted(pcts)} - NON confermate dai dati reali")
            print()

    print(f"=== Fine audit: {n_sospetti}/{len(posts)} post con almeno un claim sospetto ===")
    if n_sospetti == 0:
        print("Nessun claim numerico sospetto trovato nei post gia' approvati.")


if __name__ == "__main__":
    main()
