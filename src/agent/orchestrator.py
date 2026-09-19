"""
Orchestrator - nodo `select_next_post` (11/09/2026), completa il roadmap punto 4
("pianifica una sequenza di post futuri") facendo davvero avanzare il grafo su TUTTI
i post pianificati, uno alla volta, invece della scorciatoia RESEARCH_POST_INDEX che
elaborava sempre lo stesso post indicato a mano (comoda per i test manuali di un
singolo nodo, mai pensata come soluzione definitiva - vedi commenti in
src/agent/research.py).

Punto di innesto nel grafo (src/agent/graph.py): chiamato sia SUBITO DOPO il Planner
(prima iterazione, current_post_index ancora None) sia SUBITO DOPO il KG Update
(ogni iterazione successiva, per passare al post che segue nel piano) - stesso nodo
per entrambi i casi, la logica e' identica ("dato l'indice corrente, prepara il
prossimo post o segnala che il piano e' finito").

Perche' un nodo dedicato invece di farlo dentro Research: Research deve restare
concentrato sulla ricerca di UN post che gli viene dato, non decidere lui stesso
QUALE post viene dopo - stessa separazione di responsabilita' gia' seguita per gli
altri nodi di questo grafo (es. Format non decide se rigenerare, lo fa Human Review).

MAX_POSTS_TO_PROCESS (opzionale, 15/09/2026): per l'esame basta dimostrare la
pipeline su al massimo 2 post, non su tutti e 6 quelli pianificati - ma il Planner
deve CONTINUARE a pianificare/giustificare una sequenza di 6 (richiesto dalla
specifica "pianifica una sequenza", e vedi DEFAULT_N_POSTS in planner.py: una
sequenza troppo corta e' segnalata li' come probabile causa del voto non massimo di
GymAssistant). Quindi la riduzione va fatta qui, non in planner.py: questo nodo
pianifica per intero ma SMETTE DI ELABORARE (research/draft/revisione/scrittura KG)
dopo N post, anche se il piano ne prevede di piu' - una scelta esplicita per velocizzare
i test/la demo d'esame (ogni post costa ~10-15 minuti con qwen3:8b), non per pigrizia:
il piano completo resta comunque visibile e giustificato nel reasoning_trace, solo la
sua ESECUZIONE viene troncata. Se non impostata, nessun limite (comportamento
originale, elabora l'intero piano)."""
from __future__ import annotations

import os

from langgraph.graph import END

from src.agent.state import AgentState


def select_next_post(state: AgentState) -> AgentState:
    """Nodo del grafo LangGraph. Legge post_plan/current_post_index, e:
    - se il piano e' vuoto o e' stato esaurito (indice >= len(post_plan)), OPPURE se
      e' stato raggiunto il limite MAX_POSTS_TO_PROCESS (vedi commento in cima al
      modulo), ritorna current_post=None (il chiamante instrada verso END, vedi
      _route_after_select in graph.py) e logga il motivo esatto (i due casi sono
      distinti nel reasoning_trace - "piano esaurito" non e' la stessa cosa di "run
      troncato apposta");
    - altrimenti imposta current_post sul post successivo del piano e RESETTA
      research_summary/draft/review_decision/tool_outputs a vuoto, cosi' il post
      nuovo parte da uno stato pulito e non eredita claim/bozze/decisioni (o
      osservazioni di tool) del post PRECEDENTE gia' concluso."""
    reasoning_trace = list(state.get("reasoning_trace", []))
    post_plan = state.get("post_plan") or []

    _raw_index = state.get("current_post_index")
    # None = prima chiamata di questo nodo in questo run (subito dopo il Planner),
    # non ancora stato elaborato nessun post - si parte dall'indice 0. Un indice gia'
    # presente (chiamata successiva, dopo un KG Update) significa invece "quel post e'
    # concluso (approvato/scartato/fallito), passa al successivo" - da qui il +1.
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

    # Rete di sicurezza (17/09/2026), parallela al fix in domain_data.py: quel fix
    # impedisce che un archetipo gia' coperto venga PROPOSTO al Planner, ma non copre
    # i topic tipo evento/review/news (inventati liberamente dall'LLM, non legati a
    # un archetipo dei dati di scraping) - caso reale osservato lo stesso giorno,
    # "Analisi dei nuovi percorsi di missioni in Hearthstone" ripetuto parola per
    # parola nonostante fosse gia' un topic coperto ed esplicitamente elencato nel
    # prompt del Planner. Controllo puramente meccanico (stringa esatta normalizzata,
    # nessun giudizio semantico): se il topic del prossimo post e' IDENTICO a uno
    # gia' presente nel KG (fotografato dal Planner a inizio run) o a uno di un post
    # precedente di QUESTO STESSO piano, lo si salta senza nemmeno avviare Research/
    # Format - inutile spendere 5-10 minuti di LLM su un contenuto che verrebbe
    # comunque scartato in revisione, come e' successo in entrambi i casi reali.
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
