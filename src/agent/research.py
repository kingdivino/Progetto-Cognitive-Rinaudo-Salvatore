"""Nodo Research/ReAct: dato 'current_post', esegue un ciclo ReAct (Thought ->
Action -> Observation) con selezione dinamica tra i tool (KG, RAG, search web,
modello fine-tuned, statistiche reali), interrogando prima il KG per il K-RAG, e
chiude con un'estrazione strutturata dei claim raccolti con le rispettive fonti."""
from __future__ import annotations

import datetime
import os
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from src.agent import format_rules
from src.agent.llm_config import build_llm
from src.agent.state import AgentState
from src.tools.kg_tool import query_knowledge_graph
from src.tools.power_level_tool import assess_deck_power_level
from src.tools.rag_tool import get_card_info_by_name, search_card_knowledge
from src.tools.search_tool import SOURCE_TIER_DOMAINS, search_web
from src.tools.stats_tool import get_archetype_stats

MAX_TOOL_ITERATIONS = 6  # tetto di sicurezza contro cicli infiniti del modello

# I due tool aggiuntivi richiesti dalla specifica: assess_deck_power_level (fine-
# tuned, fonte "MODELLO-FINETUNED") e get_archetype_stats (dati reali HSReplay,
# fonte "DATI-HSREPLAY").
TOOLS = [
    query_knowledge_graph, search_card_knowledge, search_web,
    assess_deck_power_level, get_archetype_stats,
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# query_knowledge_graph escluso dal ciclo ReAct (chiamato a parte, in codice): il
# modello lo ririchiamava nonostante il prompt lo vietasse, sprecando iterazioni.
REACT_TOOLS = [
    search_card_knowledge, search_web,
    assess_deck_power_level, get_archetype_stats,
]


class SourcedClaim(BaseModel):
    claim: str = Field(description="un'affermazione specifica e verificabile per il post, basata SOLO su quanto trovato con i tool in questa ricerca")
    source: str = Field(
        description="fonte a supporto del claim - ESATTAMENTE una di queste 5 forme, "
        "corrispondenti ai 5 tool disponibili, mai altro: un URL (da search_web), "
        "'RAG: <nome carta>' (da search_card_knowledge), 'KG' (da "
        "query_knowledge_graph), il letterale 'MODELLO-FINETUNED' (da "
        "assess_deck_power_level - una VALUTAZIONE del modello fine-tuned, non un "
        "fatto verificabile con una fonte terza, ma comunque tracciata come le "
        "altre), oppure il letterale 'DATI-HSREPLAY' (da get_archetype_stats - un "
        "dato reale osservato, non una stima). Non usare fonti come 'Planner' o "
        "altri campi dello stato - se un dato viene dal piano del Planner e non da "
        "un tool chiamato in QUESTA ricerca, non e' un claim di questo nodo e va "
        "escluso."
    )


class ResearchSummary(BaseModel):
    claims: list[SourcedClaim] = Field(
        description="claim raccolti durante la ricerca, ognuno con la propria fonte - "
        "SOLO claim supportati da un'osservazione effettiva di un tool in questa "
        "conversazione, non aggiungerne altri"
    )


# Domini importati da search_tool.py (usato anche da search_web) per non tenere due
# copie disallineabili.
def classify_source_tier(source: str) -> str:
    """Classifica una fonte (valore del campo 'source' di SourcedClaim) in un
    livello di affidabilita' meccanico, solo per fonti URL - vedi commento sopra.
    RAG/KG sono sempre 'dati_locali' (fonte strutturata e verificabile del progetto
    stesso, questione diversa dall'autorevolezza di un URL web di terzi)."""
    s = (source or "").strip()
    if s == "KG" or s.startswith("RAG:") or s == "MODELLO-FINETUNED" or s == "DATI-HSREPLAY":
        return "dati_locali"
    if not s.startswith(("http://", "https://")):
        return "sconosciuta"
    for tier, domains in SOURCE_TIER_DOMAINS.items():
        if any(d in s for d in domains):
            return tier
    return "sconosciuta"


# Verifica che la fonte dichiarata corrisponda DAVVERO a un'osservazione di un tool
# in questa ricerca (non solo che sia scritta nel formato giusto, gia' controllato
# da source_well_formed) - vedi addendum di progetto per il caso reale che lo ha
# motivato.
def _source_is_grounded(
    source: str, tools_used: list[str], tool_outputs: list[dict], claim_text: str = ""
) -> bool:
    src = (source or "").strip()

    # Controllo universale sulle percentuali, per QUALUNQUE fonte: deve comparire in
    # un'osservazione reale di un tool, altrimenti il claim e' respinto.
    _pct_re = re.compile(r"\d+(?:[.,]\d+)?%")
    _claim_pcts = {m.replace(",", ".") for m in _pct_re.findall(claim_text or "")}
    if _claim_pcts:
        # ROOT CAUSE (vedi addendum di progetto): va rimossa la justification (testo
        # del modello) dall'osservazione prima di cercare percentuali "vere", altrimenti
        # un numero fabbricato nella justification passerebbe per dato reale.
        _real_pcts_global: set = set()
        for o in tool_outputs:
            _obs_text = str(o.get("observation", ""))
            _justification_text = str(o.get("justification", ""))
            if _justification_text:
                _obs_text = _obs_text.replace(_justification_text, "")
            _real_pcts_global |= {m.replace(",", ".") for m in _pct_re.findall(_obs_text)}
        if not _claim_pcts.issubset(_real_pcts_global):
            return False

    if src == "KG":
        return "query_knowledge_graph" in tools_used
    if src == "MODELLO-FINETUNED":
        return "assess_deck_power_level" in tools_used
    if src == "DATI-HSREPLAY":
        return "get_archetype_stats" in tools_used
    if src.startswith("RAG:"):
        if "search_card_knowledge" not in tools_used:
            return False
        card_name = src[len("RAG:"):].strip().lower()
        if not card_name:
            return False
        # Solo i nomi di carta DAVVERO elencati come risultato (non un match contro
        # l'osservazione intera, che accetterebbe anche la query stessa come nome).
        _card_name_re_grounding = re.compile(r"\[([^—\]]+)\s—\sid:")
        for o in tool_outputs:
            if o.get("tool") != "search_card_knowledge":
                continue
            _obs = str(o.get("observation", ""))
            for m in _card_name_re_grounding.finditer(_obs):
                if m.group(1).strip().lower() == card_name:
                    return True
        return False
    if src.startswith(("http://", "https://")):
        if "search_web" not in tools_used:
            return False
        return any(
            o.get("tool") == "search_web" and src in str(o.get("observation", ""))
            for o in tool_outputs
        )
    return False


RESEARCH_SYSTEM_PROMPT = """Sei il ricercatore di un blog su Hearthstone (gioco di carte Blizzard).
Il tuo compito e' raccogliere informazioni VERIFICATE per supportare il post che ti
viene indicato, usando gli strumenti disponibili, con uno stile ReAct: prima pensi
(Thought) a cosa ti serve e perche', poi agisci (Action) chiamando UN tool alla
volta, poi osservi il risultato (Observation) prima di decidere il prossimo passo.

Strumenti disponibili:
- query_knowledge_graph: legge il Knowledge Graph editoriale (topic gia' coperti,
  post recenti). Il suo risultato ti aiuta a capire cosa e' gia' stato trattato e a
  formulare query piu' mirate per i tool successivi (K-RAG: il contesto del KG deve
  informare le ricerche dopo, non essere un lookup a vuoto). QUESTO TOOL E' GIA'
  STATO CHIAMATO AUTOMATICAMENTE PER TE prima che tu iniziassi: vedi il suo esito
  nel messaggio precedente della conversazione. NON e' tra i tool che puoi
  scegliere in questo ciclo (rimosso il 18/09/2026 dopo un run in cui e' stato
  richiamato 6 volte su 6 iterazioni disponibili, senza mai arrivare a cercare
  materiale reale sul topic) - usa direttamente uno degli altri tool sotto.
- search_card_knowledge: ricerca semantica in un corpus locale e verificabile di
  carte Hearthstone (nome, costo, statistiche, testo, classe, espansione). Usalo per
  qualunque domanda su meccaniche/testo di carte esistenti.
- search_web: ricerca web (Tavily) per fatti recenti/attuali (espansioni, patch,
  tornei, annunci ufficiali) che il corpus locale non copre. Usalo SOLO per questo
  tipo di fatti specifici, non per meccaniche di carte (per quelle usa search_card_knowledge).
- assess_deck_power_level: usa il modello fine-tuned del progetto per stimare il
  "power level" (basso/medio/alto) di un mazzo dalla sua composizione (classe,
  formato, curva di mana, rarita', meccaniche - MAI includere winrate/partite
  nella deck_description, il modello non le ha mai viste in training). E' una
  VALUTAZIONE del modello, non un fatto verificabile con una fonte esterna: usalo
  come segnale aggiuntivo per un post how-to/review che deve giudicare quanto un
  mazzo sia forte/interessante, non per fatti specifici sulle carte (per quelli usa
  search_card_knowledge). IMPORTANTE (errore reale osservato il 18/09/2026): la
  deck_description deve descrivere UN SOLO mazzo di UNA SOLA classe - quella del
  post (vedi "Classe del mazzo di questo post" sotto, se rilevata). NON unire MAI
  carte/mazzi di classi diverse trovati durante la ricerca in una descrizione
  inventata (e' successo: un finto mazzo "Priest+Warlock" che mescolava due carte
  di due mazzi diversi - il modello ha comunque prodotto un'etichetta, ma senza
  significato, perche' l'input non era un mazzo reale nel formato su cui e' stato
  addestrato). Se non conosci la composizione esatta del mazzo di questo post,
  descrivi solo cio' che sai per certo dal topic/dalla motivazione del Planner
  (classe, formato, eventuale numero di carte non standard) - meglio una
  descrizione essenziale ma coerente di un solo mazzo che una dettagliata ma
  incoerente su piu' mazzi. Se il post non riguarda un mazzo specifico (es. un
  post di tipo "evento" su meccaniche generali), non chiamare questo tool. Se lo
  usi, il claim corrispondente DEVE avere come source esattamente il letterale
  'MODELLO-FINETUNED'.
- get_archetype_stats: restituisce winrate/popolarita' REALI (non stimati) di una
  classe in un formato, dai dati veri raccolti da HSReplay. Usalo SEMPRE che il post
  debba citare un numero di winrate/popolarita' su una classe/archetipo - MAI
  inventare o dedurre questi numeri da search_web/search_card_knowledge, che non
  contengono statistiche di winrate reali e aggiornate. Ha granularita' per classe/
  formato, non per singolo archetipo con nome (per un mazzo specifico usa comunque
  la sua classe, es. "PRIEST" per "Azalina Priest"). Se lo usi, il claim
  corrispondente DEVE avere come source esattamente il letterale 'DATI-HSREPLAY'.

Regole:
- OGNI chiamata a un tool richiede il parametro `justification`: perche' ti serve
  questa informazione ora, per quale claim del post. Mai una justification vaga.
- Non inventare MAI un fatto che non hai effettivamente trovato con un tool in
  questa conversazione - se un tool non trova nulla di utile, dillo e prosegui
  diversamente, non riempire il vuoto con dettagli plausibili ma non verificati.
- Il Knowledge Graph (gia' interrogato per te, vedi sopra) non parla mai del
  contenuto specifico del post (carte, mazzi, meccaniche) - serve solo a evitare
  ripetizioni. DEVI comunque chiamare almeno un altro tool (search_card_knowledge
  e/o search_web) per raccogliere materiale reale sul topic, prima di poter
  concludere la ricerca.
- Fermati quando hai raccolto abbastanza materiale per il post (di norma bastano
  1-3 chiamate a tool oltre al KG) - non continuare a cercare senza motivo.
- QUALUNQUE tool (non solo search_card_knowledge/search_web - osservato il 18/09/2026 anche con get_archetype_stats, richiamato 6 volte di fila con ARGOMENTI IDENTICI): se un risultato e' poco pertinente o non ti da' quello che ti serve, NON limitarti a ripetere la stessa chiamata sperando in un esito diverso - una chiamata con gli stessi argomenti da' SEMPRE la stessa risposta, ripeterla e' uno spreco di iterazioni che non aggiunge mai informazione. Per search_card_knowledge/search_web: prova query DIVERSE e piu' specifiche (un termine di meccanica concreto come 'pesca carte', 'rimozione', 'costo basso', 'buff', oppure il nome di una carta specifica che conosci per questa classe/archetipo). Per get_archetype_stats in particolare: ha granularita' SOLO per classe/formato, mai per un singolo archetipo con nome (es. non puo' mai confermare il winrate specifico di "Temporal Priest", solo quello aggregato di tutta la classe PRIEST) - richiamarlo di nuovo con la stessa classe/formato non produrra' MAI un numero diverso o piu' specifico: se il claim che ti serve riguarda un archetipo con nome specifico, usa il dato aggregato che hai gia' ottenuto (chiarendo che e' un dato di classe, non dell'archetipo specifico) oppure prova un tool diverso (es. search_web), invece di richiamarlo di nuovo.
- Se un risultato di search_web mostra una data di pubblicazione ("[pubblicato:
  ...]"), usala per giudicare se l'informazione e' ancora attuale rispetto a OGGI
  (vedi la data di oggi nel messaggio con il post da ricercare) prima di scriverla
  nel claim - un articolo vecchio di mesi o anni non va MAI presentato come una
  notizia recente/"di questi giorni" solo perche' compare in una ricerca web. Se non
  c'e' una data, trattalo con cautela come potenzialmente non recente, specialmente
  per un post di tipo "news".
- Quando hai finito, produci il riassunto finale SOLO con claim che hai
  effettivamente verificato con i tool sopra, ognuno con la sua fonte esplicita.
- Il topic/motivazione del post che ricevi puo' menzionare un mazzo con un numero
  di carte diverso da 30 (es. 20 o 40) - non e' un errore, e' una regola di
  costruzione non standard dovuta a una carta leggendaria specifica (vedi glossario
  sotto). NON cercare mai un tool con query generiche tipo "numero carte non
  standard" o simili: cerca DIRETTAMENTE il nome della carta leggendaria responsabile
  (vedi glossario), che e' un termine di gioco reale e trova risultati pertinenti nel
  corpus RAG.

Glossario meccaniche di gioco rilevanti:
- Un mazzo da 20 carte invece di 30 e' dovuto a **Azalina Soulsever** (Priest,
  mazzo di 20 carte + 20 copiate dall'avversario). Un mazzo da 40 carte e' dovuto a
  **Timethief Rafaam** (Warlock, mazzo di 40 carte con fino a 10 leggendarie
  "Rafaam"). Se il post parla di questa meccanica senza nominare la carta
  leggendaria specifica, cerca il nome esatto della carta (es. "Azalina Soulsever",
  "Timethief Rafaam" per intero, non solo "Azalina" o "Rafaam" - il corpus contiene
  piu' carte con nomi simili e una query troppo generica puo' non trovare quella
  giusta) su search_card_knowledge.

Formato Standard vs Wild e classe delle carte - IMPORTANTE, leggi con attenzione
prima di usare i tool: in Wild sono legali TUTTE le carte mai pubblicate, in Standard
solo le espansioni indicate qui sotto (se presenti in questo messaggio) piu' il
Basic/Core Set; ogni mazzo e' inoltre di UNA classe giocatore e puo' includere SOLO
carte di quella classe piu' le Neutrali. Quando cerchi una carta con
search_card_knowledge, il risultato mostra GIA' la sua "espansione" e "classe" - ti
bastano quei due campi del risultato che hai gia' ottenuto, letti una volta sola:
- NON serve una seconda ricerca per "controllare" espansione o classe - quel dato e'
  gia' nella risposta che hai ricevuto, rileggila invece di richiamare di nuovo il tool.
- NON costruire MAI una query con parole come "Standard", "Wild", "espansione", il
  nome del mazzo o il numero di carte (es. "Azalina Priest Standard (20 carte)
  espansione") - search_card_knowledge cerca il nome di UNA carta, non mazzi interi:
  una query simile non trova nulla di utile e spreca un'iterazione su 6 disponibili.
  Cerca sempre e solo il nome esatto della carta (es. "Azalina Soulsever").
- Un controllo finale automatico (in codice, dopo la tua risposta) verifichera' di
  nuovo espansione e classe di ogni carta citata - il tuo compito e' solo raccogliere
  claim pertinenti con la fonte corretta, non verificarli tu stesso piu' volte."""


def research_topic(state: AgentState) -> AgentState:
    """Nodo Research/ReAct del grafo LangGraph. Riceve lo stato condiviso (deve gia'
    contenere post_plan, prodotto dal Planner), ritorna lo stato aggiornato con
    reasoning_trace/tool_outputs/kg_summary/current_post/research_summary popolati."""
    node_start = time.perf_counter()
    llm_time_total = 0.0
    tool_time_total = 0.0
    reasoning_trace = list(state.get("reasoning_trace", []))
    tool_outputs = list(state.get("tool_outputs", []))
    kg_summary = dict(state.get("kg_summary", {}))

    post_plan = state.get("post_plan", [])
    # RESEARCH_POST_INDEX: fallback per invocare questo nodo in isolamento, quando
    # 'current_post' non arriva gia' da select_next_post (percorso normale).
    _raw_post_index = os.environ.get("RESEARCH_POST_INDEX")
    if _raw_post_index is None:
        _post_index = 0
    else:
        try:
            _post_index = int(_raw_post_index or "0")
        except ValueError:
            reasoning_trace.append(
                f"[Research] RESEARCH_POST_INDEX={_raw_post_index!r} non e' un numero "
                "valido - uso il post 0 del piano."
            )
            _post_index = 0
    _current_post_from_state = state.get("current_post")
    current_post = _current_post_from_state or (
        post_plan[_post_index] if post_plan and _post_index < len(post_plan)
        else (post_plan[0] if post_plan else None)
    )
    if current_post is None:
        reasoning_trace.append(
            "[Research] Nessun post disponibile da ricercare (post_plan vuoto) - nodo saltato."
        )
        return {
            **state,
            "reasoning_trace": reasoning_trace,
            "research_summary": {"claims": [], "tools_used": []},
        }

    # Quando 'current_post' arriva dallo stato si riporta l'indice reale da
    # 'current_post_index', non quello (fuorviante) della variabile d'ambiente.
    if _current_post_from_state is not None:
        _display_index = state.get("current_post_index")
        _display_index = 0 if _display_index is None else _display_index
        reasoning_trace.append(
            f"[Research] Avvio ricerca per il post {_display_index + 1}/{len(post_plan) or 1} "
            f"del piano (indice fornito da select_next_post): "
            f"[{current_post.get('tipo')}] {current_post.get('topic')}"
        )
    else:
        reasoning_trace.append(
            f"[Research] Avvio ricerca per il post {_post_index + 1}/{len(post_plan) or 1} del piano "
            f"(RESEARCH_POST_INDEX={_post_index}, variabile d'ambiente "
            f"{'NON impostata nel processo - uso il default' if _raw_post_index is None else 'rilevata'}): "
            f"[{current_post.get('tipo')}] {current_post.get('topic')}"
        )

    llm = build_llm(temperature=0.3)
    llm_with_tools = llm.bind_tools(REACT_TOOLS)

    # Euristica sul testo (nessun campo strutturato collega il post al mazzo): in
    # caso di Standard si fornisce l'elenco aggiornato delle espansioni legali, che
    # cambia con le rotazioni e la conoscenza del modello sarebbe stale.
    detected_format = format_rules.detect_format(
        f"{current_post.get('topic', '')} {current_post.get('justification', '')}"
    )
    standard_sets = format_rules.fetch_standard_legal_sets() if detected_format == "standard" else None
    if detected_format == "standard" and standard_sets:
        format_note = (
            f"\n\nFormato di questo post: Standard. Espansioni attualmente legali in "
            f"Standard (oltre al Basic/Core Set): {', '.join(sorted(standard_sets))}. "
            "Vedi la regola sul formato nelle istruzioni di sistema."
        )
    elif detected_format == "standard":
        format_note = (
            "\n\nFormato di questo post: Standard (elenco espansioni legali non "
            "disponibile in questo run - fai comunque attenzione a non presentare "
            "carte palesemente storiche/rimosse dalla rotazione come parte attuale "
            "del mazzo)."
        )
    elif detected_format == "wild":
        format_note = "\n\nFormato di questo post: Wild (tutte le carte mai pubblicate sono legali)."
    else:  # "misto" - il post copre esplicitamente sia Standard che Wild
        format_note = (
            "\n\nFormato di questo post: misto Standard/Wild (il post tratta entrambi "
            "i meta insieme). Puoi citare carte di entrambi i formati - specifica sempre "
            "a quale formato si riferisce ogni affermazione, non dare per scontato che "
            "una carta sia Standard-legal solo perche' e' rilevante per questo post."
        )

    # Stessa euristica per la classe del mazzo (osservato un claim che proponeva una
    # carta Warlock per un mazzo Priest).
    detected_class = format_rules.detect_deck_class(
        f"{current_post.get('topic', '')} {current_post.get('justification', '')}"
    )
    if detected_class:
        class_note = (
            f"\n\nClasse del mazzo di questo post: {detected_class}. Puo' includere "
            "solo carte di questa classe o carte Neutrali - vedi la regola sulla "
            "classe delle carte nelle istruzioni di sistema."
        )
    else:
        class_note = ""

    reasoning_trace.append(
        f"[Research] Formato rilevato dal topic/justification: {detected_format}"
        + (" (elenco espansioni Standard recuperato)" if standard_sets else "")
        + f"; classe rilevata: {detected_class or 'non riconosciuta'}"
    )

    # Data reale iniettata qui (il system prompt statico non puo' saperla): senza,
    # il modello ha generato query con un anno vecchio (stima da training).
    _oggi = datetime.date.today().strftime("%d/%m/%Y")

    messages = [
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Data di oggi: {_oggi}. Usala per giudicare la recenza di qualunque "
                "fonte web che consulterai - non presumere l'anno corrente dalla tua "
                "conoscenza pregressa, potrebbe essere superata.\n\n"
                f"Post da ricercare:\n"
                f"Tipo: {current_post.get('tipo')}\n"
                f"Topic: {current_post.get('topic')}\n"
                f"Motivazione dal Planner: {current_post.get('justification')}"
                f"{format_note}{class_note}\n\n"
                "Raccogli le informazioni necessarie usando i tool disponibili, "
                "poi fermati quando hai materiale sufficiente."
            )
        ),
    ]

    tools_used: list[str] = []
    # Dedup delle chiamate a tool identiche (justification esclusa dal confronto,
    # contano solo tool e argomenti) - vedi addendum di progetto per il caso reale.
    _seen_tool_calls: set[tuple] = set()

    # Ban a due colpi: un tool viene escluso solo dopo la SECONDA ripetizione della
    # stessa esatta combinazione (tool, argomenti) - mai per argomenti diversi.
    _banned_tools: set[str] = set()
    _duplicate_hits: dict[tuple, int] = {}

    # KG interrogato direttamente in codice (non lasciato alla scelta dell'LLM):
    # osservato un run che rifiutava di chiamare qualunque tool producendo comunque
    # claim con fonte mai invocata - vedi addendum di progetto.
    forced_justification = (
        "Passo fisso del workflow (non richiede una decisione dell'LLM): controllare "
        "sempre il Knowledge Graph editoriale per primo, per evitare ripetizioni "
        "rispetto a post gia' pubblicati e per informare le ricerche successive (K-RAG)."
    )
    _kg_call_start = time.perf_counter()
    try:
        kg_observation = query_knowledge_graph.invoke({"justification": forced_justification})
    except Exception as e:
        kg_observation = f"[ERROR] esecuzione tool 'query_knowledge_graph' fallita: {e}"
    _kg_elapsed = time.perf_counter() - _kg_call_start
    tool_time_total += _kg_elapsed
    reasoning_trace.append(
        f"[Research] Observation (query_knowledge_graph, {_kg_elapsed:.1f}s, chiamata fissa non "
        f"richiesta dall'LLM, justification: {forced_justification}): {str(kg_observation)[:300]}"
    )
    tool_outputs.append({
        "tool": "query_knowledge_graph",
        "args": {"justification": forced_justification},
        "justification": forced_justification,
        "observation": str(kg_observation),
    })
    tools_used.append("query_knowledge_graph")
    kg_summary["research_query_result"] = str(kg_observation)
    _forced_call_id = "forced_kg_call"
    messages.append(AIMessage(content="", tool_calls=[{
        "name": "query_knowledge_graph",
        "args": {"justification": forced_justification},
        "id": _forced_call_id,
    }]))
    messages.append(ToolMessage(content=str(kg_observation), tool_call_id=_forced_call_id))

    # Promemoria anti-confusione con topic simili gia' coperti nel KG (osservato un
    # caso reale, vedi addendum di progetto) - ripetuto ad ogni iterazione, non solo
    # qui, perche' la confusione scattava a meta' ciclo.
    _topic_reminder = (
        "Promemoria: la lista 'Topic gia' coperti' qui sopra elenca ARGOMENTI DI ALTRI "
        "POST, gia' pubblicati - non hanno nulla a che fare con la tua ricerca attuale, "
        "anche se un nome ti sembra simile. Il topic che DEVI ricercare in questa "
        "conversazione, e SOLO quello, resta:\n"
        f"Tipo: {current_post.get('tipo')}\n"
        f"Topic: {current_post.get('topic')}"
        + (f"\nClasse del mazzo: {detected_class}" if detected_class else "")
    )
    messages.append(HumanMessage(content=_topic_reminder))

    for iteration in range(MAX_TOOL_ITERATIONS):
        _llm_call_start = time.perf_counter()
        if _banned_tools:
            _remaining_tools = [t for t in REACT_TOOLS if t.name not in _banned_tools]
            # Tutti i tool banditi (caso limite): si evita bind_tools([]) (alcuni
            # backend si comportano in modo inatteso) lasciando rispondere senza tool.
            _active_llm_with_tools = llm.bind_tools(_remaining_tools) if _remaining_tools else llm
        else:
            _active_llm_with_tools = llm_with_tools
        try:
            response = _active_llm_with_tools.invoke(messages)
        except Exception as e:
            llm_time_total += time.perf_counter() - _llm_call_start
            reasoning_trace.append(
                f"[Research] [ERROR] chiamata LLM fallita all'iterazione {iteration + 1}: {e}"
            )
            break
        llm_time_total += time.perf_counter() - _llm_call_start
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            # Il KG e' gia' garantito (chiamata forzata sopra): il vincolo reale e'
            # "almeno un tool OLTRE al KG", che da solo non parla del contenuto del post.
            non_kg_tools_used = [t for t in tools_used if t != "query_knowledge_graph"]
            if not non_kg_tools_used:
                reasoning_trace.append(
                    f"[Research] [WARNING] Nessun tool oltre al KG usato all'iterazione "
                    f"{iteration + 1} - il KG da solo non basta come ricerca sul contenuto "
                    "del post, forzo un altro giro."
                )
                messages.append(
                    HumanMessage(
                        content=(
                            "Non puoi concludere la ricerca avendo usato solo il Knowledge "
                            "Graph: non contiene alcuna informazione sul contenuto specifico "
                            "del post (carte, mazzi, meccaniche), serve solo a evitare "
                            "ripetizioni. Usa search_card_knowledge e/o search_web per "
                            "raccogliere materiale reale sul topic prima di concludere."
                        )
                    )
                )
                continue
            reasoning_trace.append(
                f"[Research] Nessun altro tool richiesto (iterazione {iteration + 1}) - ricerca conclusa."
            )
            break

        for call in tool_calls:
            name = call.get("name")
            args = call.get("args", {}) or {}
            justification = args.get("justification", "(nessuna giustificazione fornita)")
            reasoning_trace.append(f"[Research] Thought->Action: '{name}' con args={args}")

            _dedup_args = {k: v for k, v in args.items() if k != "justification"}
            try:
                _call_key = (name, tuple(sorted(_dedup_args.items())))
            except TypeError:
                _call_key = None  # argomenti non confrontabili in modo semplice - salta il dedup

            if _call_key is not None and _call_key in _seen_tool_calls:
                _duplicate_hits[_call_key] = _duplicate_hits.get(_call_key, 0) + 1
                _hit_count = _duplicate_hits[_call_key]
                if _hit_count >= 2:
                    # Terzo tentativo identico: si bandisce il tool intero (l'API non
                    # permette di escludere una singola combinazione argomenti).
                    _banned_tools.add(name)
                    observation = (
                        "[INFO] Chiamata gia' ripetuta identica per la seconda volta - non fornisce "
                        "nessuna informazione nuova ed e' ormai chiaro che non stai variando gli "
                        "argomenti. Da questo momento il tool "
                        f"'{name}' non e' piu' selezionabile in questa ricerca: usa un tool diverso "
                        "tra quelli ancora disponibili, oppure concludi la ricerca se non ne restano "
                        "di utili."
                    )
                    _tool_elapsed = 0.0
                    reasoning_trace.append(
                        f"[Research] Observation ({name}, chiamata duplicata per la 2a volta - "
                        f"tool bandito per il resto della ricerca, justification: {justification}): "
                        f"{observation}"
                    )
                else:
                    # Prima ripetizione di questa esatta combinazione: avviso forte ma
                    # NESSUN bando ancora - lascia al modello la possibilita' di essere
                    # stata una svista isolata (vedi commento su _duplicate_hits sopra).
                    observation = (
                        "[INFO] Chiamata gia' eseguita in questa ricerca con questi stessi argomenti "
                        "(vedi Observation precedente per lo stesso tool/query) - non fornisce nessuna "
                        "informazione nuova. Prova argomenti REALMENTE diversi con questo stesso tool "
                        "(es. un'altra carta, un'altra classe/formato), oppure passa a un altro tool: "
                        "se ripeti IDENTICA questa chiamata ancora una volta, il tool verra' escluso "
                        "dalla scelta per il resto di questa ricerca."
                    )
                    _tool_elapsed = 0.0
                    reasoning_trace.append(
                        f"[Research] Observation ({name}, chiamata duplicata - NON rieseguita, "
                        f"1a ripetizione (avviso, tool non ancora bandito), justification: "
                        f"{justification}): {observation}"
                    )
            else:
                if _call_key is not None:
                    _seen_tool_calls.add(_call_key)
                tool_fn = TOOLS_BY_NAME.get(name)
                _tool_call_start = time.perf_counter()
                if tool_fn is None:
                    observation = f"[ERROR] tool sconosciuto richiesto dall'LLM: {name}"
                else:
                    try:
                        observation = tool_fn.invoke(args)
                    except Exception as e:
                        observation = f"[ERROR] esecuzione tool '{name}' fallita: {e}"
                _tool_elapsed = time.perf_counter() - _tool_call_start
                tool_time_total += _tool_elapsed
                reasoning_trace.append(
                    f"[Research] Observation ({name}, {_tool_elapsed:.1f}s, justification: {justification}): "
                    f"{str(observation)[:300]}"
                )

            tool_outputs.append(
                {"tool": name, "args": args, "justification": justification, "observation": str(observation)}
            )
            tools_used.append(name)
            if name == "query_knowledge_graph":
                kg_summary["research_query_result"] = str(observation)

            messages.append(ToolMessage(content=str(observation), tool_call_id=call.get("id") or name))

        # Ripetuto ad ogni iterazione (vedi commento sopra, dopo la prima osservazione
        # del KG) - vicino al punto in cui la confusione tra topic simili e' stata
        # osservata scattare dal vivo, a meta' delle iterazioni disponibili.
        messages.append(HumanMessage(content=_topic_reminder))
    else:
        reasoning_trace.append(
            f"[Research] Raggiunto il limite di {MAX_TOOL_ITERATIONS} iterazioni tool - fermo qui."
        )

    structured_llm = llm.with_structured_output(ResearchSummary)
    extraction_messages = messages + [
        HumanMessage(
            content=(
                "Riassumi ora i claim raccolti in questa ricerca, ognuno con la fonte "
                "esplicita a supporto. Includi SOLO claim supportati da quello che hai "
                "effettivamente trovato con i tool sopra in questa conversazione - non "
                "includere dati che vengono solo dal piano del Planner (es. winrate/"
                "partite gia' presenti nella justification del post) se non li hai "
                "anche verificati/ritrovati con un tool in questa ricerca.\n\n"
                "Attenzione in particolare ai claim che caratterizzano l'INTERO mazzo/"
                "archetipo in un modo specifico (es. il suo stile di gioco - combo, "
                "aggro, control, OTK - o un giudizio sulla sua forza generale): prima "
                "di scriverli, controlla se le ALTRE osservazioni raccolte in questa "
                "stessa ricerca (RAG o gli altri risultati web) lo confermano o lo "
                "contraddicono. Se un singolo risultato di tipo opinione individuale "
                "(un video, un post di forum o social) descrive il mazzo in un modo "
                "che le altre fonti - specialmente dati aggregati come HSReplay o "
                "Hearthpwn, o piu' fonti indipendenti concordi - non confermano o "
                "addirittura contraddicono, NON generalizzare quell'opinione come "
                "fatto sull'archetipo: ometti il claim, oppure formulalo "
                "esplicitamente come opinione di quella fonte specifica (es. "
                "'secondo un video pubblicato da un singolo giocatore...') invece "
                "che come caratteristica del mazzo.\n\n"
                "Quando un'osservazione reale (in particolare search_card_knowledge) ti da' un "
                "fatto specifico su una carta - il suo testo/effetto, costo, "
                "statistiche - non limitarti a ripetere il dato grezzo come claim: "
                "quando ha senso per il tipo di post, deriva un'implicazione "
                "strategica concreta e utile (es. per una guida al mulligan: se una "
                "carta pesca carte, costa poco, o e' una rimozione efficiente, e' "
                "ragionevole indicare che conviene tenerla in mano nelle fasi "
                "iniziali; per una recensione: valuta se il suo effetto reale la "
                "rende adatta al ritmo tipico dell'archetipo). L'implicazione DEVE "
                "restare ancorata esplicitamente all'effetto REALE che hai visto "
                "nell'osservazione - non inventare un effetto diverso da quello "
                "osservato (vedi il caso 'Azalina Soulsever' descritta come carta "
                "Hero quando l'osservazione diceva chiaramente 'minion': un errore "
                "di questo tipo invalida la conclusione anche se il nome della carta "
                "era corretto). Cita nello stesso claim sia il nome della carta sia, "
                "in breve, il suo effetto reale, cosi' la conclusione resta "
                "verificabile. Un claim breve ma specifico (nome carta reale + "
                "effetto reale + implicazione concreta) e' sempre meglio di una "
                "frase generica priva di riferimenti verificabili: preferisci "
                "qualcosa come 'Anetheron costa meno con la mano piena (RAG: "
                "Anetheron), quindi conviene tenerlo per le fasi centrali della "
                "partita quando si accumulano carte in mano' a una frase come "
                "'e' importante gestire bene le risorse in mano'.\n\n"
                "IMPORTANTE sul campo source (errore reale osservato il 18/09/2026): "
                "per OGNI claim, controlla a quale SPECIFICA chiamata tool devi quel "
                "contenuto - non scrivere per abitudine il nome dell'ultimo tool "
                "chiamato o del tool usato piu' spesso in questa ricerca per TUTTI i "
                "claim, indipendentemente da quale abbia davvero prodotto quel "
                "contenuto (e' successo: 5 claim diversi, provenienti da tool diversi "
                "inclusa una statistica numerica reale da get_archetype_stats, tutti "
                "con source scritta 'search_card_knowledge' anche quando quel tool "
                "non c'entrava). La fonte di ogni claim deve essere ESATTAMENTE una "
                "di queste forme, in base al tool che ha DAVVERO prodotto quel "
                "contenuto specifico: un URL (search_web), 'RAG: <nome carta>' "
                "(search_card_knowledge), 'KG' (query_knowledge_graph), il letterale "
                "'MODELLO-FINETUNED' (assess_deck_power_level, mai altro testo), il "
                "letterale 'DATI-HSREPLAY' (get_archetype_stats, mai altro testo) - "
                "mai il nome del tool scritto come testo libero."
            )
        )
    ]
    _extraction_start = time.perf_counter()
    try:
        summary: ResearchSummary = structured_llm.invoke(extraction_messages)
        llm_time_total += time.perf_counter() - _extraction_start
        claims_dicts = []
        n_malformed = 0
        n_ungrounded = 0
        # Diagnostica: si raccolgono fino a 3 esempi REALI di fonte malformata (non
        # solo un conteggio) per poter distinguere un problema normalizzabile in
        # codice da un problema di compliance del prompt.
        _esempi_fonte_malformata: list[str] = []
        for c in summary.claims:
            claim_dict = c.model_dump()
            # Validazione strutturale del campo source (URL, "RAG: <...>" o "KG") -
            # un modello piu' piccolo non rispetta sempre il formato richiesto nel prompt.
            src = (claim_dict.get("source") or "").strip()
            well_formed = (
                src == "KG"
                or src.startswith("RAG:")
                or src.startswith(("http://", "https://"))
                or src == "MODELLO-FINETUNED"
                or src == "DATI-HSREPLAY"
            )

            # Riparazione automatica: il modello spesso copia l'intera riga del
            # risultato Tavily invece del solo URL - si estrae l'URL con una regex e
            # lo si accetta solo se compare in un'osservazione search_web reale.
            if not well_formed:
                _url_match = re.search(r"https?://\S+", src)
                if _url_match:
                    _candidate_url = _url_match.group(0).rstrip(").,;:\"'")
                    if any(
                        o.get("tool") == "search_web" and _candidate_url in str(o.get("observation", ""))
                        for o in tool_outputs
                    ):
                        reasoning_trace.append(
                            "[Research] Fonte riparata automaticamente per un claim: la stringa "
                            f"scritta dal modello ({src[:80]!r}) conteneva un URL che compare per "
                            f"davvero in un'osservazione search_web di questa ricerca - normalizzata "
                            f"a {_candidate_url!r} invece di scartare il claim solo per formato."
                        )
                        src = _candidate_url
                        claim_dict["source"] = src
                        well_formed = True

            # Riparazione automatica, variante RAG: si cerca tra i nomi DAVVERO
            # restituiti da search_card_knowledge quello che compare come sottostringa
            # nella fonte scritta dal modello, e si usa il nome esatto dell'osservazione.
            if not well_formed:
                _card_name_re = re.compile(r"\[([^—\]]+)\s—\sid:")
                for o in tool_outputs:
                    if well_formed or o.get("tool") != "search_card_knowledge":
                        continue
                    for m in _card_name_re.finditer(str(o.get("observation", ""))):
                        _candidate_name = m.group(1).strip()
                        if _candidate_name and _candidate_name.lower() in src.lower():
                            reasoning_trace.append(
                                "[Research] Fonte riparata automaticamente per un claim: la "
                                f"stringa scritta dal modello ({src[:80]!r}) contiene il nome di "
                                f"una carta ({_candidate_name!r}) davvero restituita da "
                                "search_card_knowledge in questa ricerca - normalizzata a "
                                f"'RAG: {_candidate_name}' invece di scartare il claim solo per "
                                "formato."
                            )
                            src = f"RAG: {_candidate_name}"
                            claim_dict["source"] = src
                            well_formed = True
                            break

            # Riparazione automatica, terza variante RAG: query non un nome di carta
            # (es. "Neutral Standard") - si rilegge l'osservazione reale per quella
            # query e si estrae il nome alla posizione indicata dal modello.
            if not well_formed:
                _ref_match = re.search(
                    r"query:\s*['\"]([^'\"]+)['\"]\s*,\s*risultato\s*(\d+)", src, re.IGNORECASE
                )
                if _ref_match:
                    _ref_query = _ref_match.group(1).strip()
                    _ref_index = _ref_match.group(2)
                    for o in tool_outputs:
                        if well_formed or o.get("tool") != "search_card_knowledge":
                            continue
                        _obs = str(o.get("observation", ""))
                        if f"'{_ref_query}'" not in _obs and f'"{_ref_query}"' not in _obs:
                            continue
                        _entry_match = re.search(
                            rf"^{_ref_index}\.\s*\[([^—\]]+)\s—\sid:", _obs, re.MULTILINE
                        )
                        if _entry_match:
                            _candidate_name = _entry_match.group(1).strip()
                            reasoning_trace.append(
                                "[Research] Fonte riparata automaticamente per un claim: la "
                                f"stringa scritta dal modello ({src[:80]!r}) fa riferimento al "
                                f"risultato {_ref_index} della query {_ref_query!r}, che in "
                                f"un'osservazione reale di questa ricerca corrisponde davvero alla "
                                f"carta {_candidate_name!r} - normalizzata a "
                                f"'RAG: {_candidate_name}' invece di scartare il claim solo per "
                                "formato."
                            )
                            src = f"RAG: {_candidate_name}"
                            claim_dict["source"] = src
                            well_formed = True
                            break

            # Riparazione automatica, quarta variante: il modello scrive il nome nudo
            # del tool (es. 'get_archetype_stats') invece del letterale fisso richiesto
            # - riparabile solo se quel tool e' stato davvero invocato (tools_used).
            if not well_formed:
                _src_norm = src.strip().lower()
                _fixed_literal_by_tool = {
                    "get_archetype_stats": "DATI-HSREPLAY",
                    "assess_deck_power_level": "MODELLO-FINETUNED",
                }
                _repaired_literal = _fixed_literal_by_tool.get(_src_norm)
                if _repaired_literal and _src_norm in tools_used:
                    reasoning_trace.append(
                        "[Research] Fonte riparata automaticamente per un claim: la "
                        f"stringa scritta dal modello ({src[:80]!r}) e' il nome nudo del "
                        f"tool '{_src_norm}', che e' stato davvero invocato in questa "
                        f"ricerca - normalizzata al letterale richiesto "
                        f"{_repaired_literal!r} invece di scartare il claim solo per "
                        "formato."
                    )
                    src = _repaired_literal
                    claim_dict["source"] = src
                    well_formed = True

            claim_dict["source_well_formed"] = well_formed
            if not well_formed:
                n_malformed += 1
                if len(_esempi_fonte_malformata) < 3:
                    _esempi_fonte_malformata.append(repr(src)[:80])

            claim_dict["source_tier"] = classify_source_tier(src) if well_formed else "sconosciuta"

            # Grounding (vedi _source_is_grounded sopra): controllo distinto e piu'
            # severo, solo per fonti gia' ben formate.
            claim_dict["source_grounded"] = (
                _source_is_grounded(
                    src, tools_used, tool_outputs, claim_dict.get("claim", "")
                )
                if well_formed else False
            )
            if well_formed and not claim_dict["source_grounded"]:
                n_ungrounded += 1

            # Verifica di formato/classe solo per claim RAG (solo li' sappiamo il nome
            # esatto della carta). None = non applicabile/verificabile, non "valido".
            claim_dict["format_valid"] = None
            claim_dict["class_valid"] = None
            if well_formed and src.startswith("RAG:"):
                card_name = src[len("RAG:"):].strip()
                card_info = get_card_info_by_name(card_name)
                if card_info is not None:
                    if detected_format == "standard" and standard_sets:
                        claim_dict["format_valid"] = card_info.get("set") in standard_sets
                    if detected_class:
                        card_class = card_info.get("cardClass")
                        claim_dict["class_valid"] = card_class in (detected_class, "NEUTRAL")

            # A differenza di format_valid/class_valid si applica a OGNI claim (vedi
            # BATTLEGROUNDS_ONLY_KEYWORDS in format_rules.py).
            claim_dict["mode_valid"] = not format_rules.mentions_battlegrounds_only(
                claim_dict.get("claim", "")
            )

            claims_dicts.append(claim_dict)

        research_summary = {"claims": claims_dicts, "tools_used": tools_used}
        if n_malformed:
            reasoning_trace.append(
                f"[Research] [WARNING] {n_malformed}/{len(summary.claims)} claim con fonte non nel "
                "formato atteso (URL/RAG:/KG) - controllare a mano prima di usarli nel post "
                "(campo 'source_well_formed': False su questi claim). Esempi REALI di fonte "
                f"scritta dal modello (fino a 3, per capire il pattern esatto): "
                f"{'; '.join(_esempi_fonte_malformata)}"
            )
        if n_ungrounded:
            reasoning_trace.append(
                f"[Research] [WARNING] {n_ungrounded}/{len(summary.claims)} claim hanno una fonte nel "
                "formato giusto ma NON riscontrabile in nessuna osservazione realmente ottenuta in "
                "questa ricerca (tool mai chiamato, o contenuto citato mai comparso in un'osservazione) "
                "- probabile fabbricazione (campo 'source_grounded': False)."
            )
        n_format_invalid = sum(1 for c in claims_dicts if c.get("format_valid") is False)
        if n_format_invalid:
            reasoning_trace.append(
                f"[Research] [WARNING] {n_format_invalid} claim citano una carta la cui espansione "
                "NON risulta legale in Standard, in un post identificato come Standard - "
                "controllare a mano (campo 'format_valid': False su questi claim)."
            )
        n_class_invalid = sum(1 for c in claims_dicts if c.get("class_valid") is False)
        if n_class_invalid:
            reasoning_trace.append(
                f"[Research] [WARNING] {n_class_invalid} claim citano una carta di una classe "
                f"diversa da quella del mazzo (classe rilevata dal topic: {detected_class}) e non "
                "Neutrale - non giocabile in quel mazzo, controllare a mano (campo 'class_valid': "
                "False su questi claim)."
            )
        n_opinione_singola = sum(1 for c in claims_dicts if c.get("source_tier") == "opinione_singola")
        if n_opinione_singola:
            reasoning_trace.append(
                f"[Research] [WARNING] {n_opinione_singola} claim si basano su una fonte di "
                "opinione individuale (video/forum/social, es. YouTube o Reddit) - controllare a "
                "mano se generalizzano l'esperienza di una sola persona come fatto sull'intero "
                "archetipo/mazzo, specialmente se in contraddizione con altre fonti trovate nella "
                "stessa ricerca (campo 'source_tier': 'opinione_singola' su questi claim)."
            )
        n_mode_invalid = sum(1 for c in claims_dicts if c.get("mode_valid") is False)
        if n_mode_invalid:
            # Caso reale che ha motivato questo controllo: vedi riferimento-moduli-per-report.md.
            reasoning_trace.append(
                f"[Research] [WARNING] {n_mode_invalid} claim nominano un termine esclusivo di "
                "Battlegrounds (es. 'Trinket', 'Dark Gift') in un post su un mazzo costruito - "
                "SCARTARE questi claim prima di usarli nel post (campo 'mode_valid': False)."
            )

        # Controllo "novita' presunta ma non confermata dalle date reali delle fonti"
        # (vedi addendum di progetto per il razionale): warning non bloccante se il
        # topic presuppone attualita' ma nessuna fonte search_web ha una data recente.
        _novelty_keywords = (
            "nuovo", "nuova", "nuovi", "nuove",
            "recente", "recenti", "ultimo", "ultima", "ultimi", "ultime",
        )
        _topic_text_novelty = (
            f"{current_post.get('topic', '')} {current_post.get('justification', '')}"
        ).lower()
        if any(kw in _topic_text_novelty for kw in _novelty_keywords):
            _date_re = re.compile(r"\[pubblicato:\s*(\d{4})(?:-(\d{2}))?")
            _has_recent_source = False
            _has_any_dated_source = False
            _oggi_obj = datetime.date.today()
            for o in tool_outputs:
                if o.get("tool") != "search_web":
                    continue
                for m in _date_re.finditer(str(o.get("observation", ""))):
                    _has_any_dated_source = True
                    _anno = int(m.group(1))
                    _mese = int(m.group(2)) if m.group(2) else 6  # meta anno se manca il mese
                    try:
                        _data_fonte = datetime.date(_anno, _mese, 1)
                    except ValueError:
                        continue
                    if 0 <= (_oggi_obj - _data_fonte).days <= 180:
                        _has_recent_source = True
            if not _has_recent_source:
                # Caso reale che ha motivato questo controllo: vedi riferimento-moduli-per-report.md.
                reasoning_trace.append(
                    "[Research] [WARNING] Il topic/motivazione di questo post presuppone che "
                    "l'argomento sia 'nuovo'/'recente', ma nessuna fonte search_web di questa "
                    "ricerca ha una data di pubblicazione confermata negli ultimi ~6 mesi"
                    + (
                        " (le fonti web trovate non riportano affatto una data)"
                        if not _has_any_dated_source
                        else " (le date trovate sono piu' vecchie)"
                    )
                    + " - non e' detto che il contenuto sia sbagliato, ma la premessa di attualita' "
                    "va controllata a mano prima di pubblicare."
                )
    except Exception as e:
        llm_time_total += time.perf_counter() - _extraction_start
        reasoning_trace.append(f"[Research] [ERROR] Fallita l'estrazione del riassunto finale: {e}")
        research_summary = {"claims": [], "tools_used": tools_used}

    node_elapsed = time.perf_counter() - node_start
    reasoning_trace.append(
        f"[Research] Tempo nodo: {node_elapsed:.1f}s totali (LLM: {llm_time_total:.1f}s, "
        f"tool: {tool_time_total:.1f}s, resto/overhead: {node_elapsed - llm_time_total - tool_time_total:.1f}s)."
    )
    timings = dict(state.get("timings", {}))
    timings["research_llm_s"] = round(llm_time_total, 2)
    timings["research_tool_s"] = round(tool_time_total, 2)
    timings["research_total_s"] = round(node_elapsed, 2)

    return {
        **state,
        "reasoning_trace": reasoning_trace,
        "tool_outputs": tool_outputs,
        "kg_summary": kg_summary,
        "current_post": current_post,
        "research_summary": research_summary,
        "timings": timings,
    }
