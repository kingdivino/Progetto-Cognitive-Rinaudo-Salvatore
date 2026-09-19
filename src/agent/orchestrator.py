"""Nodo select_next_post: fa avanzare il grafo sui post pianificati, uno alla
volta (chiamato dopo il Planner e dopo ogni KG Update). MAX_POSTS_TO_PROCESS
(opzionale) tronca l'ESECUZIONE a N post per velocizzare demo/test, senza accorciare
il piano pianificato/giustificato dal Planner."""
from __future__ import annotations

import os

from langgraph.graph import END

from src.agent.state import AgentState


def select_next_post(state: AgentState) -> AgentState:
    """Legge post_plan/current_post_index: se il piano e' esaurito o si e' raggiunto
    MAX_POSTS_TO_PROCESS ritorna current_post=None (il grafo instrada verso END);
    altrimenti imposta current_post sul post successivo e RESETTA research_summary/
    draft/review_decision/tool_outputs, cosi' il post nuovo non eredita nulla del
    precedente."""
    reasoning_trace = list(state.get("reasoning_trace", []))
    post_plan = state.get("post_plan") or []

    _raw_index = state.get("current_post_index")
    # None = prima chiamata (subito dopo il Planner), si parte dall'indice 0;
    # altrimenti quel post e' concluso, si passa al successivo.
    idx = 0 if _raw_index is None else _raw_index + 1

    # MAX_POSTS_TO_PROCESS: stessa gestione difensiva di RESEARCH_POST_INDEX in
    # research.py (None = non impostata affatto, stringa vuota o non numerica =
    # ignorata con un avviso, mai un crash del nodo).
    _raw_max = os.environ.get("MAX_POSTS_TO_PROCESS")
    max_posts: int | None = None
    if _raw_max is not None and _raw_max.strip():
        try:
            max_posts = int(_raw_max)
        except ValueError:
            reasoning_trace.append(
                f"[Orchestrator] MAX_POSTS_TO_PROCESS={_raw_max!r} non e' un numero "
                "valido - ignorato, nessun limite applicato."
            )

    # Rete di sicurezza parallela al fix in domain_data.py (che copre solo i topic
    # legati a un archetipo dei dati di scraping, non quelli tipo evento/review/news
    # inventati liberamente dal Planner). Controllo meccanico (stringa esatta
    # normalizzata, nessun giudizio semantico): se il topic del prossimo post e'
    # IDENTICO a uno gia' nel KG o a un post precedente di QUESTO piano, lo si salta
    # senza avviare Research/Format - inutile spendere minuti di LLM su un contenuto
    # che verrebbe comunque scartato in revisione.
    _covered_norm = {
        (t or "").strip().lower()
        for t in (state.get("planning_info", {}).get("covered_topics") or [])
    }
    _covered_norm.update((p.get("topic") or "").strip().lower() for p in post_plan[:idx])
    while idx < len(post_plan):
        _topic_norm = (post_plan[idx].get("topic") or "").strip().lower()
        if _topic_norm and _topic_norm in _covered_norm:
            reasoning_trace.append(
                f"[Orchestrator] [WARNING] Post {idx + 1}/{len(post_plan)} del piano "
                "saltato automaticamente: il suo topic e' IDENTICO (parola per parola) "
                "a uno gia' presente nel KG o gia' incontrato in questo stesso piano - "
                "nessuna ricerca/draft/revisione avviata per non sprecare tempo su un "
                f"contenuto che verrebbe comunque scartato. Topic: "
                f"{post_plan[idx].get('topic')!r}."
            )
            idx += 1
            continue
        break

    if max_posts is not None and idx >= max_posts:
        reasoning_trace.append(
            f"[Orchestrator] Limite MAX_POSTS_TO_PROCESS={max_posts} raggiunto - run "
            f"troncato di proposito qui, NON per piano esaurito (il piano pianificato "
            f"e giustificato dal Planner ne prevede {len(post_plan)} in totale, ma solo "
            f"i primi {max_posts} vengono davvero elaborati - research/draft/revisione/"
            "KG - in questo run). Fine del grafo."
        )
        return {
            **state,
            "reasoning_trace": reasoning_trace,
            "current_post_index": idx,
            "current_post": None,
        }

    if idx >= len(post_plan):
        reasoning_trace.append(
            f"[Orchestrator] Piano completato: tutti i {len(post_plan)} post pianificati "
            "sono stati elaborati (approvati, scartati o falliti) - fine del grafo."
            if post_plan else
            "[Orchestrator] post_plan e' vuoto (Planner non ha prodotto nulla) - nessun "
            "post da elaborare, fine del grafo."
        )
        return {
            **state,
            "reasoning_trace": reasoning_trace,
            "current_post_index": idx,
            "current_post": None,
        }

    prossimo_post = post_plan[idx]
    reasoning_trace.append(
        f"[Orchestrator] Avanzamento al post {idx + 1}/{len(post_plan)} del piano: "
        f"[{prossimo_post.get('tipo')}] {prossimo_post.get('topic')}"
    )
    return {
        **state,
        "reasoning_trace": reasoning_trace,
        "current_post_index": idx,
        "current_post": prossimo_post,
        # Reset esplicito per il nuovo post (vedi docstring sopra e la nota su
        # current_post_index in src/agent/state.py per il perche' e' necessario).
        "research_summary": None,
        "draft": None,
        "review_decision": None,
        "tool_outputs": [],
    }


def route_after_select(state: AgentState) -> str:
    """Instradamento dopo select_next_post: se ha impostato un current_post nuovo si
    procede con la ricerca di quel post (nodo 'research'), altrimenti il piano e'
    esaurito e il grafo termina qui (END)."""
    return "research" if state.get("current_post") is not None else END
