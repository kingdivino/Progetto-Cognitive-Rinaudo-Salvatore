"""
Test end-to-end del grafo LangGraph completo e del ciclo sull'INTERO piano pianificato:
Planner -> [select_next_post -> Research/ReAct -> Format/Draft -> Human Review ->
KG Update] -> torna a select_next_post per il post successivo, finche' il piano non
e' esaurito (roadmap punti 4-8, grafo completo - vedi src/agent/orchestrator.py per
il ciclo, aggiunto l'11/09/2026 al posto della vecchia scorciatoia
RESEARCH_POST_INDEX che elaborava sempre un solo post scelto a mano).

Cosa verifica in piu' rispetto a 09_test_human_review.py:
- Il nodo KG Update scrive sul Knowledge Graph SOLO quando la revisione umana di un
  post si conclude con "approva"/"modifica" (review_decision == "approved") -
  requisito esplicito della specifica ("Il KG si aggiorna SOLO dopo approvazione").
  "rigenera" e "scarta" lo saltano entrambi, ma con routing diverso: "rigenera" torna
  al nodo Format per lo STESSO post, "scarta" passa al post SUCCESSIVO del piano.
- Il grafo elabora TUTTI i post pianificati in sequenza, non solo il primo - questo
  test mostra live (stampando via via il reasoning_trace) l'avanzamento da un post al
  successivo, e alla fine rilegge il KG con query_knowledge_graph per dimostrare che
  TUTTI i post approvati durante questo run compaiono davvero nel grafo (non solo il
  messaggio di conferma di ogni singolo tool).

Questo test e' INTERATTIVO, stessa modalita' di 09_test_human_review.py: si ferma e
chiede una decisione da tastiera per OGNI post del piano (non e' un errore se il
processo sembra "in attesa" - lo e' davvero, e con un piano di 6 post di default
succedera' fino a 6 volte in questo singolo run, oltre a un giro extra per ogni
"rigenera"/"scarta").

Stessi prerequisiti di 09_test_human_review.py (indice RAG costruito, TAVILY_API_KEY
nel .env, Ollama/Neo4j raggiungibili dal proprio venv Windows - qui Neo4j serve
davvero, non solo per la query iniziale del Research ma per la scrittura finale).

Come eseguirlo (dalla cartella del progetto, con il venv attivo):
    python -u notebooks/10_test_kg_update.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(override=True)

from langgraph.types import Command  # noqa: E402

from src.agent.graph import build_graph  # noqa: E402 (import dopo sys.path/load_dotenv di proposito)
from src.tools.kg_tool import query_knowledge_graph  # noqa: E402


def _leggi_testo_multilinea(prompt: str) -> str | None:
    """Legge testo su piu' righe da tastiera, terminato da una riga con solo 'FINE'.
    Ritorna None se l'utente non scrive nulla (prima riga vuota) - il chiamante lo
    interpreta come "lascia invariato questo campo"."""
    print(prompt)
    print("(per CAMBIARE: scrivi il nuovo testo, anche su piu' righe, poi su una riga "
          "da sola scrivi FINE e premi invio; per LASCIARE INVARIATO: scrivi subito "
          "FINE come prima cosa, senza scrivere altro prima)")
    righe = []
    while True:
        riga = input()
        if riga.strip() == "FINE":
            break
        righe.append(riga)
    testo = "\n".join(righe).strip()
    return testo or None


def _chiedi_decisione_umana(payload: dict) -> dict:
    """Mostra il payload dell'interrupt (bozza + warning prioritari) e raccoglie la
    decisione umana da tastiera, nello stesso formato atteso da src/agent/human_review.py.
    Quattro azioni possibili: approva/modifica/rigenera (la bozza) o scarta (l'intero
    post - passa al successivo del piano senza scrivere nulla sul KG)."""
    print("\n" + "=" * 70)
    print("=== REVISIONE UMANA RICHIESTA ===")
    print("=" * 70)
    print(f"\nTitolo: {payload.get('titolo')}\n")
    print(payload.get("corpo"))
    print(f"\nFonti citate ({len(payload.get('fonti_citate', []))}):")
    for f in payload.get("fonti_citate", []):
        flag = "  [NON RICONOSCIUTA]" if f in payload.get("fonti_non_riconosciute", []) else ""
        print(f"- {f}{flag}")

    warning_prioritari = payload.get("warning_prioritari", [])
    if warning_prioritari:
        print(f"\n--- Warning da Research/Format da considerare ({len(warning_prioritari)}) ---")
        for w in warning_prioritari:
            print(f"! {w}")

    while True:
        scelta = input(
            "\nDecisione - [a]pprova / [m]odifica / [r]igenera bozza / [s]carta intero post: "
        ).strip().lower()
        if scelta in ("a", "approva"):
            return {"azione": "approva"}
        if scelta in ("m", "modifica"):
            nuovo_titolo = _leggi_testo_multilinea("Nuovo titolo:")
            nuovo_corpo = _leggi_testo_multilinea("Nuovo corpo:")
            decisione = {"azione": "modifica"}
            if nuovo_titolo:
                decisione["titolo"] = nuovo_titolo
            if nuovo_corpo:
                decisione["corpo"] = nuovo_corpo
            return decisione
        if scelta in ("r", "rigenera"):
            return {"azione": "rigenera"}
        if scelta in ("s", "scarta"):
            return {"azione": "scarta"}
        print("Non riconosciuto - rispondi 'a', 'm', 'r' o 's'.")


def _stampa_trace_incrementale(result: dict, gia_stampate: int) -> int:
    """Stampa solo le righe di reasoning_trace NUOVE da quando e' stata chiamata
    l'ultima volta (evita di ristampare tutto il trace da capo a ogni round, utile
    ora che un run puo' attraversare piu' post in sequenza). Ritorna il nuovo
    conteggio di righe stampate, da passare alla chiamata successiva."""
    trace = result.get("reasoning_trace", [])
    for line in trace[gia_stampate:]:
        print("-", line)
    return len(trace)


def main():
    graph = build_graph()
    # thread_id: identifica QUESTA esecuzione del grafo per il checkpointer - deve
    # restare lo stesso tra l'invoke iniziale e ogni invoke di ripresa
    # (Command(resume=...)) qui sotto, altrimenti LangGraph non trova lo stato
    # sospeso da cui riprendere.
    config = {"configurable": {"thread_id": "test-10-kg-update"}}
    initial_state = {
        "user_input": "pianifica i prossimi post del blog, fai ricerca su ognuno, scrivine le bozze e sottoponile a revisione",
        "reasoning_trace": [],
    }

    wall_start = time.perf_counter()
    print("\n=== Planner in corso (puo' richiedere qualche minuto) ===")
    result = graph.invoke(initial_state, config=config)
    _trace_stampate = _stampa_trace_incrementale(result, 0)

    n_round = 0
    while "__interrupt__" in result:
        n_round += 1
        payload = result["__interrupt__"][0].value
        idx = result.get("current_post_index")
        n_posts = len(result.get("post_plan", []))
        print(f"\n[Round di revisione #{n_round} - post {idx + 1 if idx is not None else '?'}/{n_posts} del piano]")
        decisione = _chiedi_decisione_umana(payload)
        result = graph.invoke(Command(resume=decisione), config=config)
        _trace_stampate = _stampa_trace_incrementale(result, _trace_stampate)

    wall_elapsed = time.perf_counter() - wall_start

    # Distinzione aggiunta il 15/09/2026 (vedi MAX_POSTS_TO_PROCESS in
    # src/agent/orchestrator.py): il ciclo sopra finisce sia quando il piano e'
    # DAVVERO esaurito, sia quando MAX_POSTS_TO_PROCESS tronca il run apposta prima -
    # "piano esaurito" sarebbe un'affermazione sbagliata nel secondo caso (il piano
    # pianificato dal Planner puo' avere piu' post di quelli davvero elaborati qui).
    _plan_len = len(result.get("post_plan", []))
    _final_idx = result.get("current_post_index")
    _piano_esaurito = _final_idx is not None and _final_idx >= _plan_len
    if _piano_esaurito:
        print(f"\n=== Piano esaurito dopo {n_round} round di revisione totali (tra tutti i post) ===")
    else:
        print(
            f"\n=== Run troncato apposta dopo {n_round} round di revisione (MAX_POSTS_TO_PROCESS) - "
            f"il piano pianificato dal Planner prevede {_plan_len} post in totale, elaborati solo "
            f"i primi {_final_idx if _final_idx is not None else '?'} ==="
        )

    kg_result_lines = [line for line in result.get("reasoning_trace", []) if line.startswith("[KGUpdate]")]
    print(f"\n=== Riepilogo scritture sul KG in questo run ({len(kg_result_lines)}) ===")
    if kg_result_lines:
        for line in kg_result_lines:
            print("-", line)
    else:
        print("(nessuna scrittura sul KG in questo run - controlla sopra se tutti i post "
              "sono stati scartati/rigenerati senza mai approvarne uno)")

    print("\n=== Verifica indipendente: rileggo il KG con query_knowledge_graph ===")
    print("(mostra lo stato ATTUALE del grafo, non solo i post approvati in questo run - "
          "stessa lettura che farebbe il nodo Research alla prossima esecuzione)")
    verifica = query_knowledge_graph.invoke({
        "justification": "Verifica manuale post-test: controllare lo stato del KG dopo "
                          "aver elaborato l'intero piano di questo run."
    })
    print(verifica)

    timings = result.get("timings", {})
    reasoning_env = os.environ.get("OLLAMA_REASONING", "(default del modello)")
    print(f"\n=== Tempi dell'ULTIMO post elaborato (modello: {os.environ.get('OLLAMA_MODEL', 'llama3.1:8b')}, "
          f"OLLAMA_REASONING={reasoning_env}) ===")
    print("(nota: 'timings' non e' ancora per-post - mostra solo l'ultimo post del piano, "
          "limite noto, vedi guida di progetto)")
    for key, seconds in timings.items():
        print(f"- {key}: {seconds:.1f}s")
    print(f"- tempo totale wall-clock dell'INTERO run (piano + tutti i post, incluso il "
          f"tempo di attesa di ogni revisione umana): {wall_elapsed:.1f}s")


if __name__ == "__main__":
    main()
