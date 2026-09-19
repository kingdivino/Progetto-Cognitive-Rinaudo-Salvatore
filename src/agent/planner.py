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
import time
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agent.domain_data import load_archetype_signals
from src.agent.format_rules import fix_meta_gender
from src.agent.llm_config import build_llm
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
  numero di partite, numero_carte_mazzo):
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
  - Il campo "numero_carte_mazzo" e' il numero REALE di carte di quel mazzo (30 e' la
    dimensione standard di Hearthstone; 20 o 40 significano che una carta leggendaria
    specifica altera la regola di costruzione, es. Azalina Soulsever per i mazzi da 20,
    Timethief Rafaam per quelli da 40). E' gia' un dato comprensibile di per se' - non
    e' un'etichetta interna ne' un jargon, quindi citalo direttamente (es. "questo
    mazzo ha 20 carte invece delle 30 standard") senza bisogno di parafrasarlo.
  - PRIMA di scrivere che un valore e' "unico", "il solo" o "l'unico" tra gli archetipi
    forniti (es. "l'unico mazzo con 20 carte invece di 30", "l'unico archetipo con
    winrate sotto il 30%"), controlla ESPLICITAMENTE ogni singola voce della lista
    fornita per verificare che nessun'altra voce condivida quel valore. Non assumere
    l'unicita' guardando solo la voce di cui stai scrivendo: un'affermazione di
    unicita' sbagliata e' un errore grave perche' contraddice dati che hai gia'
    ricevuto nello stesso messaggio.
  - Il campo "formato" ("Standard" o "Wild") indica in quale formato e' stato
    giocato quel mazzo specifico. E' una distinzione IMPORTANTE, non un dettaglio: in
    Wild sono legali tutte le carte mai pubblicate in Hearthstone, in Standard solo un
    sottoinsieme (le espansioni attualmente in rotazione). Quando proponi un topic su
    un mazzo con "formato": "Standard", specifica sempre il formato nel topic/
    justification (es. "guida al Dragon Warrior Standard", non solo "guida al Dragon
    Warrior") - il nodo di ricerca a valle usa questa indicazione per non suggerire
    carte non legali in quel formato. Non mescolare mai dati di mazzi con "formato"
    diversi nello stesso post come se fossero comparabili/dello stesso meta.
  - Gli "Archetipi reali dai dati di scraping" sono un CAMPIONE (una manciata di
    mazzi selezionati), non l'intero dataset di scraping: non scrivere affermazioni
    tipo "nel dataset" o "nel meta attuale" quando intendi solo il campione che ti e'
    stato fornito - usa invece "tra gli archetipi forniti" o "in questo campione".
  - Il campo "costo_polvere" (quando presente) e' il costo in polvere arcana per
    craftare quel mazzo specifico (copie standard, non dorate) - un dato puramente
    economico/di collezione, NON un indicatore di forza o qualita': un mazzo costoso
    non e' automaticamente piu' forte o piu' interessante di uno economico, e
    viceversa. Puoi citarlo (es. per aiutare un lettore a valutare se vale la pena
    craftarlo) ma non dedurne affermazioni su winrate o popolarita' - quelle restano
    supportate solo dai campi dedicati (vedi sopra). Se assente per un archetipo, non
    inventare un valore ne' assumere che costi 0.
- Quando parli del "meta" (il metagame competitivo del gioco), usa SEMPRE l'articolo
  MASCHILE: "il meta", "del meta", "nel meta", "un meta" - MAI "la meta"/"della
  meta"/"nella meta"/"una meta" (in italiano standard "meta" al femminile
  significherebbe "traguardo/obiettivo", un significato diverso - la community
  italiana di Hearthstone usa questo prestito al maschile), sia nel campo "topic" sia
  nella "justification"."""


def plan_posts(state: AgentState) -> AgentState:
    """Nodo Planner del grafo LangGraph. Riceve lo stato condiviso, ritorna lo stato
    aggiornato con reasoning_trace/planning_info/post_plan popolati."""
    node_start = time.perf_counter()
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

    # covered_topics passato qui (17/09/2026): esclude a monte, in codice, gli
    # archetipi il cui nome e' gia' comparso in un topic pubblicato - vedi il
    # commento in load_archetype_signals per il caso reale che ha motivato il fix
    # (il solo elenco testuale nel prompt sotto non e' bastato a impedire una
    # ripetizione parola per parola).
    archetype_signals = load_archetype_signals(top_n=8, covered_topics=covered_topics)
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

    # Costruzione del LLM centralizzata in src/agent/llm_config.py (letta da li' anche
    # OLLAMA_NUM_GPU/OLLAMA_REASONING - vedi commenti li' per il significato di ognuna).
    llm = build_llm(temperature=0.4)
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

    llm_start = time.perf_counter()
    try:
        plan: PostPlan = chain.invoke({
            "covered_topics": json.dumps(covered_topics, ensure_ascii=False) if covered_topics else "nessuno",
            "recent_posts": json.dumps(recent_posts, ensure_ascii=False) if recent_posts else "nessuno",
            "archetype_signals": json.dumps(archetype_signals, ensure_ascii=False) if archetype_signals else "nessuno",
            "n_posts": n_posts,
        })
        llm_elapsed = time.perf_counter() - llm_start
        post_plan = [p.model_dump() for p in plan.posts]

        # Rete di sicurezza in codice (18/09/2026), stesso principio di
        # fix_meta_gender sotto: la regola esplicita nel prompt ("Non ripetere un
        # topic gia' presente...") NON e' bastata da sola, osservato piu' volte con
        # topic ripetuti PAROLA PER PAROLA nonostante la lista dei topic coperti
        # fosse esplicitamente nel messaggio (es. "Analisi dei nuovi percorsi di
        # missioni in Hearthstone" ripianificato identico il 18/09/2026, due volte
        # nella stessa sessione di test). Finora la ripetizione veniva intercettata
        # solo a valle, in src/agent/orchestrator.py (select_next_post), che salta il
        # post senza avviare Research - corretto per evitare spreco di tempo, ma non
        # risolve il vero requisito della specifica ("il Planner tiene conto del KG
        # per evitare ridondanza": e' il Planner stesso che deve evitarla, non solo
        # un controllo successivo che nasconde il sintomo). Qui si rimuovono quindi
        # gia' in fase di pianificazione i post il cui topic e' IDENTICO (stessa
        # normalizzazione .strip().lower() usata in orchestrator.py, nessun giudizio
        # semantico) a un topic gia' nel KG o a un altro topic dello stesso piano
        # appena generato - il piano finale puo' quindi risultare piu' corto di
        # n_posts quando questo succede, l'orchestrator gestisce gia' un piano piu'
        # corto senza problemi.
        _covered_norm_planner = {(t or "").strip().lower() for t in covered_topics}
        _post_plan_dedup = []
        _seen_topic_norm: set[str] = set()
        _n_duplicati_rimossi = 0
        for post in post_plan:
            _topic_norm = (post.get("topic") or "").strip().lower()
            if _topic_norm and (_topic_norm in _covered_norm_planner or _topic_norm in _seen_topic_norm):
                _n_duplicati_rimossi += 1
                reasoning_trace.append(
                    "[Planner] [WARNING] Post pianificato scartato in fase di planning: "
                    f"topic IDENTICO (parola per parola) a uno gia' nel KG o a un altro "
                    f"post di questo stesso piano, nonostante la regola esplicita nel "
                    f"prompt - topic: {post.get('topic')!r}."
                )
                continue
            if _topic_norm:
                _seen_topic_norm.add(_topic_norm)
            _post_plan_dedup.append(post)
        post_plan = _post_plan_dedup
        if _n_duplicati_rimossi:
            reasoning_trace.append(
                f"[Planner] Rimossi {_n_duplicati_rimossi} post duplicati dal piano in fase "
                f"di planning (piano finale: {len(post_plan)}/{n_posts} post richiesti) - "
                "stessa filosofia di fix_meta_gender sotto: un controllo in codice, non solo "
                "un'istruzione nel prompt, per un requisito esplicito della specifica "
                "(evitare ridondanza tenendo conto del KG)."
            )

        # Rete di sicurezza in codice (segnalato dall'utente l'11/09/2026) per il
        # genere di "il meta"/"la meta" - vedi fix_meta_gender in format_rules.py per
        # il perche': la regola esplicita nel prompt sopra non basta da sola, stesso
        # limite di compliance testuale gia' documentato altrove nel progetto. Va
        # corretto qui (non solo nel nodo Format) perche' il "topic" pianificato qui
        # e' spesso ripreso quasi alla lettera come titolo del post finale.
        _n_fix_totale = 0
        for post in post_plan:
            post["topic"], _n = fix_meta_gender(post.get("topic", ""))
            _n_fix_totale += _n
            post["justification"], _n = fix_meta_gender(post.get("justification", ""))
            _n_fix_totale += _n
        if _n_fix_totale:
            reasoning_trace.append(
                f"[Planner] Corretto il genere di 'meta' (la -> il) {_n_fix_totale} volta/e tra "
                "topic/justification dei post pianificati (l'LLM lo aveva scritto al femminile "
                "nonostante la regola esplicita nel prompt)."
            )

        reasoning_trace.append(
            f"[Planner] Pianificati {len(post_plan)} post (chiamata LLM: {llm_elapsed:.1f}s)."
        )
    except Exception as e:
        llm_elapsed = time.perf_counter() - llm_start
        # Un modello locale (llama3.1:8b) puo' occasionalmente non rispettare lo schema
        # strutturato richiesto - non deve far crashare il grafo, va segnalato.
        reasoning_trace.append(
            f"[Planner] [ERROR] Fallita generazione del piano dopo {llm_elapsed:.1f}s: {e}"
        )
        post_plan = []

    node_elapsed = time.perf_counter() - node_start
    timings = dict(state.get("timings", {}))
    timings["planner_llm_s"] = round(llm_elapsed, 2)
    timings["planner_total_s"] = round(node_elapsed, 2)

    return {
        **state,
        "reasoning_trace": reasoning_trace,
        "timings": timings,
        "planning_info": {
            **state.get("planning_info", {}),
            "kg_reachable": kg_reachable,
            "covered_topics": covered_topics,
            "n_posts": n_posts,
        },
        "post_plan": post_plan,
    }
