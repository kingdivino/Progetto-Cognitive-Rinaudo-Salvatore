"""Nodo Human Review: sospende l'esecuzione (interrupt()/Command(resume=...)) per
far approvare/modificare/rigenerare/scartare la bozza a un umano prima
dell'aggiornamento del KG. Nessun side-effect prima di interrupt(): il nodo viene
rieseguito da capo quando il grafo riprende."""
from __future__ import annotations

from langgraph.types import interrupt

from src.agent.state import AgentState


def human_review(state: AgentState) -> AgentState:
    """Sospende l'esecuzione con interrupt() mostrando la bozza e i warning, poi
    applica la decisione umana al resume: "approva", "modifica" (sovrascrive
    titolo/corpo), "rigenera" (torna a Format sullo stesso post) o "scarta" (passa al
    post successivo)."""
    reasoning_trace = list(state.get("reasoning_trace", []))
    draft = state.get("draft")

    if draft is None:
        reasoning_trace.append(
            "[HumanReview] Nessuna bozza disponibile (draft assente) - nodo saltato, "
            "nessuna scrittura possibile sul KG per questo post."
        )
        return {**state, "reasoning_trace": reasoning_trace, "review_decision": "regenerate"}

    # Warning gia' calcolati da Research/Format, ripresentati qui limitati al post
    # corrente (Research) e al solo ultimo tentativo di drafting (Format), cosi'
    # un "rigenera" ripetuto non ripropone warning di bozze gia' scartate.
    try:
        _ultimo_research_idx = max(
            i for i, line in enumerate(reasoning_trace) if "[Research] Avvio ricerca per il post" in line
        )
    except ValueError:
        _ultimo_research_idx = 0
    research_warnings = [
        line for line in reasoning_trace[_ultimo_research_idx:]
        if "[Research]" in line and "[WARNING]" in line
    ]
    try:
        _ultimo_draft_idx = max(
            i for i, line in enumerate(reasoning_trace) if "[Format] Bozza generata" in line
        )
    except ValueError:
        _ultimo_draft_idx = 0
    format_warnings = [
        line for line in reasoning_trace[_ultimo_draft_idx:]
        if "[Format]" in line and "[WARNING]" in line
    ]
    warning_prioritari = research_warnings + format_warnings

    decision = interrupt({
        "titolo": draft.get("titolo"),
        "corpo": draft.get("corpo"),
        "fonti_citate": draft.get("fonti_citate", []),
        "fonti_non_riconosciute": draft.get("fonti_non_riconosciute", []),
        "warning_prioritari": warning_prioritari,
        "istruzioni_per_il_client": (
            "Riprendere il grafo con Command(resume=<dict>), dove <dict> e' uno di: "
            "{'azione': 'approva'} - accetta la bozza cosi' com'e'; "
            "{'azione': 'modifica', 'titolo': ..., 'corpo': ...} - sovrascrive uno o "
            "entrambi i campi (omettere quello che non cambia) e poi accetta; "
            "{'azione': 'rigenera'} - scarta SOLO questa bozza, il grafo torna al nodo "
            "Format per generarne una nuova con gli stessi claim verificati (stesso "
            "post); {'azione': 'scarta'} - scarta l'intero POST (non solo la bozza): "
            "nessuna scrittura sul KG, il grafo passa direttamente al post successivo "
            "del piano (se presente) - usarla quando il topic stesso non merita la "
            "pubblicazione, non solo la bozza generata."
        ),
    })

    azione = decision.get("azione") if isinstance(decision, dict) else None

    if azione == "approva":
        reasoning_trace.append("[HumanReview] Bozza approvata dall'utente senza modifiche.")
        return {**state, "reasoning_trace": reasoning_trace, "review_decision": "approved"}

    if azione == "modifica":
        edited_draft = dict(draft)
        nuovo_titolo = decision.get("titolo")
        nuovo_corpo = decision.get("corpo")
        if nuovo_titolo:
            edited_draft["titolo"] = nuovo_titolo
        if nuovo_corpo:
            edited_draft["corpo"] = nuovo_corpo
        reasoning_trace.append(
            "[HumanReview] Bozza approvata DOPO modifica manuale dell'utente "
            f"(titolo modificato: {bool(nuovo_titolo)}, corpo modificato: {bool(nuovo_corpo)})."
        )
        return {
            **state,
            "reasoning_trace": reasoning_trace,
            "draft": edited_draft,
            "review_decision": "approved",
        }

    if azione == "rigenera":
        reasoning_trace.append(
            "[HumanReview] Bozza rifiutata dall'utente - richiesta una nuova generazione "
            "(il grafo torna al nodo Format)."
        )
        return {**state, "reasoning_trace": reasoning_trace, "review_decision": "regenerate"}

    if azione == "scarta":
        # Diverso da "rigenera": qui e' il TOPIC che l'utente non vuole pubblicare, non
        # solo la bozza - il routing (graph.py) avanza al post successivo del piano.
        reasoning_trace.append(
            "[HumanReview] Post scartato dall'utente (non solo la bozza) - nessuna "
            "scrittura sul KG per questo post, si passa al prossimo del piano pianificato "
            "(se presente)."
        )
        return {**state, "reasoning_trace": reasoning_trace, "review_decision": "discarded"}

    # Risposta non riconosciuta: mai un default rischioso (es. approvare per errore) -
    # trattata come rigenerazione e loggata.
    reasoning_trace.append(
        f"[HumanReview] [WARNING] Risposta di resume non riconosciuta ({decision!r}) - "
        "trattata come rigenerazione per sicurezza (nessuna scrittura sul KG senza "
        "un'approvazione esplicita e riconosciuta)."
    )
    return {**state, "reasoning_trace": reasoning_trace, "review_decision": "regenerate"}
