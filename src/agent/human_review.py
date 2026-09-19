"""
Human Review node (roadmap punto 7, tra Format/Draft e KG Update).

Requisito esplicito della specifica ("Human-in-the-loop"): "Prima di aggiornare il
KG: mostra il post generato, permette approvazione / modifica / rigenerazione. Il KG
si aggiorna SOLO dopo approvazione." Questo nodo e' il punto in cui questo requisito
viene implementato - il nodo KG Update (roadmap punto 8) scrive sul grafo SOLO quando
'review_decision' vale "approved".

Azione aggiuntiva "scarta" (11/09/2026, non richiesta letteralmente dalla specifica
ma non in contraddizione con essa - richiesta dall'utente dopo che il grafo ha
iniziato a elaborare un piano intero di post, vedi src/agent/orchestrator.py): scarta
l'intero POST (non solo la bozza attuale), utile quando il topic stesso non merita
la pubblicazione e "rigenera" all'infinito non risolverebbe nulla. Vedi
route_after_select/select_next_post in src/agent/orchestrator.py per come il grafo
avanza al post successivo dopo uno scarto.

Implementato con il meccanismo nativo di human-in-the-loop di LangGraph
(interrupt()/Command(resume=...), vedi src/agent/graph.py per il checkpointer
richiesto): il nodo sospende l'esecuzione qui, il chiamante (per ora un test da
terminale, in futuro un'eventuale interfaccia) mostra la bozza a un umano e ne
raccoglie la decisione, poi il grafo riprende quando viene re-invocato con
Command(resume=<decisione>).

ATTENZIONE (comportamento documentato di interrupt() in LangGraph): quando il grafo
riprende da un interrupt, QUESTO NODO VIENE RIESEGUITO DA CAPO fino al punto
dell'interrupt. Per questo il nodo resta deliberatamente privo di side-effect prima
della chiamata a interrupt() (nessuna chiamata LLM/tool qui) - si limita a leggere lo
stato gia' calcolato dai nodi precedenti e a preparare il payload da mostrare.
"""
from __future__ import annotations

from langgraph.types import interrupt

from src.agent.state import AgentState


def human_review(state: AgentState) -> AgentState:
    """Nodo Human Review del grafo LangGraph. Riceve 'draft' (gia' popolato dal nodo
    Format), sospende l'esecuzione con interrupt() mostrando la bozza e i warning ad
    alta priorita' gia' calcolati a monte, poi applica la decisione umana ricevuta al
    resume: "approva" (nessuna modifica), "modifica" (sovrascrive titolo/corpo con il
    testo fornito, poi tratta come approvata), "rigenera" (scarta SOLO la bozza, il
    grafo torna al nodo Format per lo stesso post) o "scarta" (scarta l'intero post,
    il grafo passa al post successivo del piano). Ritorna lo stato aggiornato con
    reasoning_trace/draft/review_decision popolati."""
    reasoning_trace = list(state.get("reasoning_trace", []))
    draft = state.get("draft")

    if draft is None:
        reasoning_trace.append(
            "[HumanReview] Nessuna bozza disponibile (draft assente) - nodo saltato, "
            "nessuna scrittura possibile sul KG per questo post."
        )
        return {**state, "reasoning_trace": reasoning_trace, "review_decision": "regenerate"}

    # Warning ad alta priorita' gia' calcolati dai nodi precedenti (Research/Format) -
    # ripresentati qui cosi' chi revisiona parte gia' sapendo dove guardare con piu'
    # attenzione, invece di dover rileggere tutto il reasoning_trace da capo.
    #
    # I warning di Research restano validi per qualunque bozza DELLO STESSO post (i
    # claim non cambiano finche' Research non viene rieseguito) - ma vanno limitati
    # al post CORRENTE: dall'11/09/2026 il grafo elabora un piano intero di post in
    # sequenza (src/agent/orchestrator.py), quindi reasoning_trace contiene ormai
    # anche i warning di Research di post PRECEDENTI gia' conclusi (approvati o
    # scartati) - senza questo limite, revisionando il post 3 si vedrebbero ancora i
    # warning di Research del post 1, stesso tipo di bug gia' risolto sotto per i
    # warning di Format tra un "rigenera" e l'altro dello stesso post.
    # I warning di Format vanno invece limitati al SOLO tentativo di drafting PIU'
    # RECENTE: in caso di "rigenera" (loop Format -> Human Review ripetuto),
    # reasoning_trace accumula anche i warning di bozze gia' scartate nei tentativi
    # precedenti - mostrarli tutti farebbe credere al revisore che un problema di un
    # tentativo precedente valga ancora per la bozza attuale, quando magari il nuovo
    # tentativo non lo ripete piu'.
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
        # Diverso da "rigenera": qui non e' la BOZZA il problema (potrebbe anche
        # essere scritta bene), e' il TOPIC stesso che l'utente ha deciso di non voler
        # pubblicare - continuare a rigenerare all'infinito non lo risolverebbe.
        # Aggiunto l'11/09/2026 su richiesta esplicita dell'utente: non e' uno dei tre
        # esiti nominati dalla specifica (approvazione/modifica/rigenerazione), ma non
        # la contraddice nemmeno - resta comunque vero che "il KG si aggiorna SOLO
        # dopo approvazione" (qui non c'e' approvazione, quindi nessuna scrittura,
        # identico a "rigenera" su questo punto specifico). Il routing (vedi
        # src/agent/graph.py) fa la differenza: "rigenera" torna a Format per lo
        # STESSO post, "discarded" avanza al post successivo del piano.
        reasoning_trace.append(
            "[HumanReview] Post scartato dall'utente (non solo la bozza) - nessuna "
            "scrittura sul KG per questo post, si passa al prossimo del piano pianificato "
            "(se presente)."
        )
        return {**state, "reasoning_trace": reasoning_trace, "review_decision": "discarded"}

    # Risposta non riconosciuta (formato inatteso dal client che ha fatto il resume) -
    # stesso principio anti-assunzione gia' visto altrove in questo progetto: non si
    # assume un'azione di default rischiosa (es. approvare per errore, che scriverebbe
    # sul KG senza una vera approvazione) - si tratta come "da rigenerare" e si logga
    # con chiarezza, cosi' l'anomalia e' visibile nel reasoning_trace.
    reasoning_trace.append(
        f"[HumanReview] [WARNING] Risposta di resume non riconosciuta ({decision!r}) - "
        "trattata come rigenerazione per sicurezza (nessuna scrittura sul KG senza "
        "un'approvazione esplicita e riconosciuta)."
    )
    return {**state, "reasoning_trace": reasoning_trace, "review_decision": "regenerate"}
