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
from typing import Literal

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
    # Literal invece di str: constraint imposto dallo schema (Ollama/Pydantic
    # rifiutano/ricampionano un valore fuori da questi 4), non delegato interamente
    # all'LLM che seguendo solo l'istruzione testuale ha talvolta prodotto "event"
    # invece di "evento" (osservato nel run del 01/09/2026 con qwen3:14b).
    tipo: Literal["evento", "how-to", "review", "news"] = Field(
        description="uno tra: evento, how-to, review, news (esattamente queste 4 stringhe)"
    )
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
- evento: tornei/esport, uscita espansioni, patch di bilanciamento, eventi in-game a
  tempo (es. percorsi/tracker con missioni e ricompense aggiuntive che escono a rotazione)
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
  almeno un paio di post (sono numeri veri, non inventarne altri).
- NON INVENTARE fatti specifici e verificabili che non ti sono stati forniti: nomi di
  espansioni, date di patch, nomi/date di tornei, percentuali di winrate diverse da
  quelle fornite, e neppure il fatto che un evento specifico sia gia' accaduto (es. non
  scrivere che un'espansione/patch/torneo e' stato "annunciato", "rilasciato" o "in
  corso" - non lo sai). Non sai con certezza quale sia l'ultima espansione o il
  calendario esport attuale. Questo vale sia per il campo "topic" SIA per la
  "justification": e' inutile lasciare il topic generico se poi la justification
  afferma comunque un fatto specifico non verificato (es. topic generico "analisi
  della prossima espansione" ma justification che dice "e' stata annunciata da
  Blizzard" - anche questo va evitato). Per un topic di tipo evento/news di cui non
  hai il dato reale, resta GENERICO sia nel topic sia nella motivazione (es. "analisi
  della prossima espansione in arrivo", con motivazione tipo "le nuove espansioni
  cambiano regolarmente il meta, vale la pena preparare un'analisi non appena
  disponibili i dettagli" - SENZA affermare che sia gia' successo qualcosa di
  specifico) invece di inventare un nome, una data o un evento plausibili ma falsi: la
  verifica del fatto reale spetta al nodo di ricerca a valle (Search/RAG), non a te in
  fase di pianificazione.
- Quando descrivi un archetipo preso da "Archetipi reali dai dati di scraping", ogni
  caratteristica che affermi deve essere derivabile SOLO dai campi forniti (winrate,
  numero di partite, regola_speciale):
  - "popolarita'" o "quanto e' giocato" puoi stimarli SOLO confrontando il numero di
    partite di un archetipo con quello degli ALTRI archetipi forniti nella stessa lista
    (es. numero di partite alto rispetto agli altri = relativamente piu' popolare/
    giocato in questo campione; numero basso rispetto agli altri = relativamente meno
    popolare). Non usare MAI il winrate come indicatore di popolarita' o forza: sono
    dati indipendenti, un winrate basso non implica ne' un archetipo debole ne' uno
    popolare.
  - NON affermare una "tendenza" (in crescita, in calo, in aumento di recente) per
    nessun archetipo: i dati forniti sono uno scatto in un solo momento, non una serie
    storica, quindi non c'e' alcun dato da cui dedurre un andamento nel tempo.
  - Prima di scrivere un'affermazione quantitativa o comparativa su un archetipo,
    verifica che sia coerente con il numero esatto fornito per quell'archetipo, non con
    un'impressione generica sul suo nome.
  - Ogni voce degli "Archetipi reali dai dati di scraping" e' UN mazzo specifico (un
    decklist preciso, identificato dal numero dopo il cancelletto es. "Control
    Warrior#33012"), non l'intero archetipo/nome mazzo in generale: altri mazzi con lo
    stesso nome possono avere statistiche molto diverse (es. tre mazzi chiamati
    "Control Warrior" nel dataset hanno winrate 54.76%, 53.29% e 21.61% - il singolo
    campione basso non descrive l'archetipo nel suo complesso). Riferisciti quindi al
    mazzo/campione specifico fornito (es. "questo Control Warrior", "questa build",
    "il mazzo #33012"), non generalizzare le sue statistiche a "l'archetipo Control
    Warrior" o "i mazzi Control Warrior" in generale, a meno che i dati forniti non
    contengano piu' campioni concordanti dello stesso nome.
  - Il campo "regola_speciale" indica che il mazzo usa una regola di costruzione non
    standard (dimensione di 20 o 40 carte invece delle 30 tipiche) dovuta a una carta
    leggendaria specifica (es. Azalina Soulsever o Timethief Rafaam) che altera quante
    carte si costruiscono. Quando lo citi in topic/justification, descrivilo con parole
    tue (es. "una regola di costruzione mazzo non standard", "una meccanica che
    modifica la dimensione del mazzo") invece di scrivere letteralmente il nome del
    campo "regola_speciale" tra virgolette, che e' un'etichetta interna dei dati, non
    un termine che i lettori del blog riconoscerebbero.
  - PRIMA di scrivere che un valore e' "unico", "il solo" o "l'unico" tra gli archetipi
    forniti (es. "l'unico mazzo con regola_speciale attiva", "l'unico archetipo con
    winrate sotto il 30%"), controlla ESPLICITAMENTE ogni singola voce della lista
    fornita per verificare che nessun'altra voce condivida quel valore. Non assumere
    l'unicita' guardando solo la voce di cui stai scrivendo: un'affermazione di
    unicita' sbagliata e' un errore grave perche' contraddice dati che hai gia'
    ricevuto nello stesso messaggio.
  - Gli "Archetipi reali dai dati di scraping" sono un CAMPIONE (una manciata di
    mazzi selezionati), non l'intero dataset di scraping: non scrivere affermazioni
    tipo "nel dataset" o "nel meta attuale" quando intendi solo il campione che ti e'
    stato fornito - usa invece "tra gli archetipi forniti" o "in questo campione"."""


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

    # num_gpu = quanti layer del modello mandare sulla GPU (nome fuorviante: NON e'
    # il numero di GPU). Lasciato non impostato (None) di default: Ollama stima da solo
    # quanti layer entrano in VRAM, in modo prudente. Su hardware con poca VRAM dedicata
    # (es. 4GB) puo' lasciare margine libero non sfruttato - override manuale via
    # OLLAMA_NUM_GPU nel .env se si osserva VRAM libera con `ollama ps` durante un run
    # (vedi commento in .env.example per come tarare il valore).
    _num_gpu_override = os.environ.get("OLLAMA_NUM_GPU")
    llm_kwargs = dict(
        model=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.4,
    )
    if _num_gpu_override:
        llm_kwargs["num_gpu"] = int(_num_gpu_override)
    llm = ChatOllama(**llm_kwargs)
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
