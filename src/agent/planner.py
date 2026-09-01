"""
Planner — primo nodo vero del grafo LangGraph (roadmap punto 4, dopo lo scheletro di
notebooks/00_hello_langgraph.py).

Cosa fa (per soddisfare i requisiti delle specifiche sul Planning):
1. Interroga il KG (Neo4j) per i topic gia' coperti e i post recenti - se il KG non e'
   raggiungibile (es. Neo4j Desktop non avviato) non blocca l'esecuzione: logga
   l'avviso nel reasoning_trace e procede assumendo nessuno storico. Questo e'
   importante ora che il KG e' vuoto (nessun nodo "KG Update" esiste ancora): il
   Planner deve comunque essere testabile end-to-end.
2. Recupera una manciata di archetipi reali dai dati di scraping (vedi domain_data.py)
   come materiale concreto su cui basare le proposte, invece di far inventare topic a
   vuoto all'LLM.
3. Chiede all'LLM (Ollama locale) di pianificare una sequenza di post futuri con
   output STRUTTURATO (Pydantic): ogni post ha tipo/topic/justification - la
   justification e' obbligatoria per requisito di progetto ("giustifica ordine e
   selezione dei post").

Nodi successivi da collegare qui in futuro: Research/ReAct (tool search/RAG/KG/
fine-tuned) -> Format/Draft -> Human Review (interrupt) -> KG Update.
"""
from __future__ import annotations

import json
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.agent.domain_data import load_archetype_signals
from src.agent.state import AgentState
from src.kg import connection as kg

DEFAULT_N_POSTS = 6  # >2 di proposito: le "lezioni apprese da GymAssistant" nella
# guida di progetto segnalano un planning troppo corto (solo 2 post) come uno dei
# probabili motivi del voto non massimo del progetto di riferimento.


class PlannedPost(BaseModel):
    tipo: str = Field(description="uno tra: evento, how-to, review, news")
    topic: str = Field(description="argomento specifico e concreto del post, non generico")
    justification: str = Field(
        description="perche' questo post ora: gap di copertura nel KG, rilevanza dei "
        "dati reali forniti, novita' nel meta, diversificazione rispetto agli altri "
        "post pianificati. Niente giustificazioni vaghe tipo 'e' un argomento interessante'."
    )


class PostPlan(BaseModel):
    posts: list[PlannedPost] = Field(
        description="sequenza di post futuri pianificati, in ordine di pubblicazione consigliato"
    )


PLANNER_SYSTEM_PROMPT = """Sei il planner editoriale di un blog su Hearthstone (gioco di carte Blizzard).
Il tuo compito e' pianificare una sequenza di {n_posts} post futuri, diversificati e non ripetitivi.

Tipi di post ammessi (usa esattamente queste etichette in "tipo"):
- evento: tornei/esport, uscita espansioni, patch di bilanciamento
- how-to: guide a mazzi/archetipi specifici, mulligan guide
- review: review di espansioni, carte o meccaniche nuove
- news: cambi di meta dopo una patch, nerf/buff

Regole:
- Non ripetere un topic gia' presente in "Topic gia' coperti dal KG" ne' uno uguale a
  "Post recenti".
- Copri piu' tipi diversi tra i {n_posts} post, non concentrarti su uno solo.
- Ogni post deve avere una justification concreta e specifica (gap di copertura, dato
  reale citato, novita' nel meta) - non giustificazioni generiche.
- Se sono forniti "Archetipi reali dai dati di scraping", usali come base concreta per
  almeno un paio di post (sono numeri veri, non inventarne altri)."""


def plan_posts(state: AgentState) -> AgentState:
    """Nodo Planner del grafo LangGraph. Riceve lo stato condiviso, ritorna lo stato
    aggiornato con reasoning_trace/planning_info/post_plan popolati."""
    reasoning_trace = list(state.get("reasoning_trace", []))

    kg_reachable = kg.check_connection()
    if kg_reachable:
        covered_topics = kg.get_covered_topics()
        recent_posts = kg.get_recent_posts(limit=5)
        reasoning_trace.append(
            f"[Planner] KG raggiungibile: {len(covered_topics)} topic gia' coperti, "
            f"{len(recent_posts)} post recenti recuperati."
        )
    else:
        covered_topics, recent_posts = [], []
        reasoning_trace.append(
            "[Planner] KG non raggiungibile (Neo4j non attivo o .env non configurato) "
            "- si procede assumendo nessuno storico. Verificare che Neo4j Desktop sia avviato."
        )

    archetype_signals = load_archetype_signals(top_n=8)
    if archetype_signals:
        reasoning_trace.append(
            f"[Planner] {len(archetype_signals)} archetipi reali recuperati da "
            f"data/processed/finetune_dataset.csv come spunto per i post."
        )
    else:
        reasoning_trace.append(
            "[Planner] Nessun archetipo disponibile (dataset di scraping non ancora "
            "generato) - pianifico solo da topic generici del dominio."
        )

    n_posts = state.get("planning_info", {}).get("n_posts", DEFAULT_N_POSTS)

    llm = ChatOllama(
        model=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.4,
    )
    structured_llm = llm.with_structured_output(PostPlan)

    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM_PROMPT.format(n_posts=n_posts)),
        ("human",
         "Topic gia' coperti dal KG:\n{covered_topics}\n\n"
         "Post recenti (da evitare):\n{recent_posts}\n\n"
         "Archetipi reali dai dati di scraping:\n{archetype_signals}\n\n"
         "Pianifica {n_posts} post."),
    ])
    chain = prompt | structured_llm

    try:
        plan: PostPlan = chain.invoke({
            "covered_topics": json.dumps(covered_topics, ensure_ascii=False) if covered_topics else "nessuno",
            "recent_posts": json.dumps(recent_posts, ensure_ascii=False) if recent_posts else "nessuno",
            "archetype_signals": json.dumps(archetype_signals, ensure_ascii=False) if archetype_signals else "nessuno",
            "n_posts": n_posts,
        })
        post_plan = [p.model_dump() for p in plan.posts]
        reasoning_trace.append(f"[Planner] Pianificati {len(post_plan)} post.")
    except Exception as e:
        # Un modello locale (llama3.1:8b) puo' occasionalmente non rispettare lo schema
        # strutturato richiesto - non deve far crashare il grafo, va segnalato.
        reasoning_trace.append(f"[Planner] [ERROR] Fallita generazione del piano: {e}")
        post_plan = []

    return {
        **state,
        "reasoning_trace": reasoning_trace,
        "planning_info": {
            **state.get("planning_info", {}),
            "kg_reachable": kg_reachable,
            "covered_topics": covered_topics,
            "n_posts": n_posts,
        },
        "post_plan": post_plan,
    }
