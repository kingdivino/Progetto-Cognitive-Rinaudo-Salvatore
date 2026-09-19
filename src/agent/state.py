"""
Stato condiviso del grafo LangGraph (gestione dello stato "MCP-style" richiesta dalle
specifiche): un dizionario esplicito aggiornato ad ogni nodo e passato come input al
nodo successivo, cosi' ogni nodo vede la storia completa del ragionamento fin qui.

Campi richiesti dalle specifiche:
- user_input: input dell'utente che ha avviato il grafo
- reasoning_trace: log testuale human-readable del ragionamento, un elemento per step
  (utile sia per debug sia per l'osservabilita' in LangSmith)
- tool_outputs: output grezzi di ogni chiamata a un tool (search/RAG/KG/fine-tuned)
- kg_summary: riassunto di cosa e' stato recuperato dal Knowledge Graph in questo run
- planning_info: dettagli della pianificazione (numero di post richiesti, topic gia'
  coperti al momento della pianificazione, ecc.)
- post_plan: l'output vero e proprio del Planner - sequenza di post futuri pianificati

Campi aggiunti per il nodo Research/ReAct (roadmap punto 5):
- current_post: il post (da post_plan) su cui il nodo Research sta lavorando in
  questo run - la pipeline Research -> Format -> HITL -> KG Update opera su UN post
  alla volta, non sull'intero piano insieme. Dall'11/09/2026 e' normalmente popolato
  dal nodo di orchestrazione `select_next_post` (src/agent/orchestrator.py, vedi
  sotto) PRIMA che Research parta, non da Research stesso - Research lo usa
  cosi' com'e' se e' gia' presente, e ricade sulla vecchia euristica
  (RESEARCH_POST_INDEX/post_plan[0]) solo se viene invocato in isolamento, fuori dal
  grafo completo (es. un test diretto del nodo).
- research_summary: l'output del nodo Research - claim raccolti con le fonti a
  supporto di ciascuno, pronti per il nodo Format/Draft
- timings: cronometraggio per nodo (secondi, float), popolato in modo incrementale
  da ogni nodo (ognuno aggiunge le proprie chiavi, es. "planner_total_s",
  "research_llm_s") - permette di confrontare a colpo d'occhio quanto tempo va via
  in chiamate LLM vs. chiamate tool (rete/RAG) vs. resto, utile per confrontare
  modelli Ollama diversi sullo stesso prompt senza dover cronometrare a mano

Campo aggiunto per il nodo Format/Draft (roadmap punto 6):
- draft: l'output del nodo Format/Draft - titolo/corpo/fonti_citate del post
  generato (dict, vedi DraftPost in src/agent/format_draft.py), None se il nodo e'
  stato saltato (nessun current_post disponibile) o se la generazione e' fallita

Campo aggiunto per il nodo Human Review (roadmap punto 7, richiesto dalla specifica:
"prima di aggiornare il KG: mostra il post generato, permette approvazione/modifica/
rigenerazione. Il KG si aggiorna SOLO dopo approvazione." - vedi
src/agent/human_review.py):
- review_decision: esito della revisione umana su 'draft' - "approved" (accettata,
  eventualmente dopo una modifica manuale che ha gia' aggiornato 'draft' stesso),
  "regenerate" (rifiutata SOLO per la bozza, il grafo torna al nodo Format per
  riprovare sugli stessi claim), o "discarded" (rifiutato l'intero POST, non solo la
  bozza - aggiunto l'11/09/2026 su richiesta dell'utente per il caso in cui il
  topic stesso non vale la pubblicazione, non serve rigenerare all'infinito - vedi
  src/agent/human_review.py). None finche' il nodo Human Review non e' ancora stato
  eseguito. Il nodo KG Update (roadmap punto 8) scrive sul grafo SOLO quando questo
  campo vale "approved" - sia "regenerate" che "discarded" lo saltano, la differenza
  tra i due e' solo nel routing (vedi src/agent/graph.py): "regenerate" torna a
  Format per lo STESSO post, "discarded"/"approved" avanzano entrambi al prossimo
  post del piano.

Campo aggiunto per il ciclo sui post pianificati (src/agent/orchestrator.py,
11/09/2026 - completa il roadmap punto 4 "pianifica una sequenza", che fino a quel
momento ne elaborava sempre e solo UNO tramite la scorciatoia RESEARCH_POST_INDEX):
- current_post_index: indice (0-based) del post di post_plan attualmente in corso
  tra Research/Format/HITL/KG Update. None prima che il ciclo sia mai partito
  (il nodo `select_next_post` lo interpreta come "primo post, indice 0"). Ad ogni
  avanzamento al post successivo, `select_next_post` deve anche RESETTARE
  research_summary/draft/review_decision/tool_outputs a vuoto - altrimenti
  claim/bozze/decisioni del post PRECEDENTE resterebbero visibili (e nel caso di
  tool_outputs, farebbero superare erroneamente il controllo di grounding di
  research.py) durante l'elaborazione del post nuovo.
"""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    user_input: str
    reasoning_trace: list[str]
    tool_outputs: list[dict[str, Any]]
    kg_summary: dict[str, Any]
    planning_info: dict[str, Any]
    post_plan: list[dict[str, Any]]
    current_post: dict[str, Any] | None
    current_post_index: int | None
    research_summary: dict[str, Any] | None
    timings: dict[str, float]
    draft: dict[str, Any] | None
    review_decision: str | None
