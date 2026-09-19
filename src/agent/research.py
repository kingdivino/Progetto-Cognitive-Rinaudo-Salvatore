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

MAX_TOOL_ITERATIONS = 6  # tetto di sicurezza - il prompt istruisce l'LLM a fermarsi
# da solo quando ha materiale sufficiente, ma un limite esplicito evita cicli
# infiniti se il modello continuasse a richiedere tool senza necessita'.

# assess_deck_power_level: tool basato sul modello fine-tuned (power_level_tool.py) -
# requisito di specifica ("almeno un tool aggiuntivo basato sul modello fine-tuned").
# Produce una VALUTAZIONE del modello (un'inferenza, non un fatto verificabile con
# una fonte terza), tracciata con la fonte dedicata "MODELLO-FINETUNED" (vedi
# SourcedClaim.source sotto) per non confonderla con URL/RAG/KG.
# get_archetype_stats: secondo dei "2 tool aggiuntivi" richiesti - non fine-tuned,
# interroga direttamente i dati reali HSReplay (stats_tool.py) per un winrate/
# popolarita' VERO invece di farlo stimare al modello o cercare sul web. Fonte
# dedicata "DATI-HSREPLAY", stesso principio di tracciamento di MODELLO-FINETUNED.
TOOLS = [
    query_knowledge_graph, search_card_knowledge, search_web,
    assess_deck_power_level, get_archetype_stats,
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# query_knowledge_graph e' escluso dal ciclo ReAct vero e proprio (resta disponibile
# solo per la chiamata forzata via codice, invocata direttamente sotto): osservato un
# run che lo richiamava di nuovo nonostante il prompt lo vietasse esplicitamente,
# sprecando tutte le iterazioni senza fare ricerca reale. Un controllo in codice (non
# offrire la scelta) batte un'istruzione ripetuta nel prompt quando il modello non la
# rispetta in modo affidabile - stesso principio usato altrove in questo file.
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


# Livelli di affidabilita' delle fonti web (solo URL - RAG/KG sono sempre affidabili
# in quanto dati strutturati locali). Distingue una fonte aggregata/ufficiale da
# un'opinione di un singolo creator. Dizionario dei domini importato da
# src/tools/search_tool.py (usato anche da search_web) per non tenere due copie
# disallineabili - vedi quel file per il significato di ogni livello.
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


# Verifica che la fonte dichiarata di un claim corrisponda DAVVERO a un'osservazione
# ottenuta in QUESTA ricerca - non solo che sia scritta nel formato giusto (quello lo
# fa gia' source_well_formed sopra, ma controlla solo la STRINGA, non i fatti). Caso
# reale che ha motivato il fix: un run senza nessuna chiamata a tool ha comunque
# prodotto claim con fonte "search_card_knowledge" mai invocato - un modello che
# scriva "RAG: <nome carta mai cercata>" passerebbe indenne da source_well_formed
# (che controlla solo il formato) e sembrerebbe una fonte valida.
#
# Il controllo e' meccanico: per KG basta che query_knowledge_graph compaia in
# tools_used; per RAG/URL il tool giusto deve essere stato usato E il contenuto
# citato deve comparire per davvero in un'osservazione registrata in tool_outputs.
def _source_is_grounded(
    source: str, tools_used: list[str], tool_outputs: list[dict], claim_text: str = ""
) -> bool:
    src = (source or "").strip()

    # Controllo UNIVERSALE, prima della verifica specifica della fonte: qualunque
    # percentuale citata in un claim (di QUALUNQUE fonte, non solo DATI-HSREPLAY/KG)
    # deve comparire in almeno un'osservazione reale di un tool in questa ricerca,
    # altrimenti il claim e' respinto (prima di questo fix un claim RAG/URL/
    # MODELLO-FINETUNED poteva contenere una cifra inventata senza verifica).
    _pct_re = re.compile(r"\d+(?:[.,]\d+)?%")
    _claim_pcts = {m.replace(",", ".") for m in _pct_re.findall(claim_text or "")}
    if _claim_pcts:
        # ROOT CAUSE (vedi addendum di progetto): ogni tool ripete alla lettera il
        # testo di 'justification' (scelto liberamente dal modello) dentro la propria
        # 'observation'. Va rimosso PRIMA di cercare percentuali "vere", altrimenti un
        # numero fabbricato nella justification (che spesso ripete proprio la cifra
        # che il modello sta cercando di "confermare") passerebbe per dato reale.
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
        # L'osservazione REALE di query_knowledge_graph (vedi kg_tool.py) contiene
        # solo nomi di topic e metadati editoriali, mai percentuali - quindi un
        # claim 'KG' con una percentuale viene gia' respinto dal controllo
        # universale sopra prima ancora di arrivare qui, in ogni caso pratico.
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
        # Un substring-match contro l'INTERA osservazione (incluso l'header "Risultati
        # RAG per '<query>' (...)" che rag_tool.py antepone) farebbe risultare
        # "grounded" per costruzione un claim con source 'RAG: <query esatta>' - la
        # query stessa scritta come se fosse un nome di carta - anche quando nessuna
        # carta con quel nome compare davvero tra i risultati. Si estraggono quindi
        # SOLO i nomi di carta DAVVERO elencati come risultato (stesso pattern regex
        # usato nelle riparazioni automatiche sopra) e si richiede una corrispondenza
        # con uno di questi, non con l'osservazione intera.
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
    # RESEARCH_POST_INDEX: rimasto solo come fallback per invocare questo nodo in
    # isolamento (es. un test diretto di un singolo post) - il percorso normale passa
    # da select_next_post (orchestrator.py) che imposta gia' 'current_post', usato
    # direttamente sotto senza guardare questa variabile. Il try/except protegge da
    # un valore non numerico (es. errore di battitura nel .env) senza far crashare il
    # nodo. os.environ.get(...) torna None se la variabile non e' mai arrivata al
    # processo (causa tipica su Windows: "set VAR=..." e' sintassi cmd.exe, non fa
    # nulla in PowerShell - serve "$env:VAR=..."), tenuto distinto da "impostata a 0"
    # cosi' il reasoning_trace dice se l'override e' stato visto o no.
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

    # Bug corretto: il messaggio riportava sempre l'indice di RESEARCH_POST_INDEX per
    # il "post N/M" anche quando 'current_post' arrivava da select_next_post (percorso
    # normale) - il post ricercato era corretto, ma il numero nel trace restava
    # fisso alla variabile d'ambiente, ingannevole nel trace/LangSmith. Quando
    # 'current_post' arriva dallo stato si riporta l'indice REALE da
    # 'current_post_index' (impostato da select_next_post), non quello della
    # variabile d'ambiente, che a quel punto non ha influenzato nulla.
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

    # Nessun campo strutturato collega il post al mazzo specifico che lo ha ispirato
    # (solo testo libero) - euristica sul testo per capire Standard/Wild, e in caso di
    # Standard si fornisce esplicitamente l'elenco aggiornato delle espansioni legali
    # (format_rules.py), perche' cambia con le rotazioni e la conoscenza pregressa del
    # modello sarebbe ferma al training e quasi certamente stale.
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

    # Stessa euristica e stesso limite del formato sopra (nessun campo strutturato
    # collega il post alla classe del mazzo). Osservato un claim che proponeva una
    # carta Warlock per un mazzo Priest - qui il gap non era "il modello non ha il
    # dato giusto" (le classi non cambiano nel tempo) ma "nessuno gli chiede mai di
    # controllarlo".
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

    # Data reale di oggi, iniettata qui (non nel system prompt statico, che non puo'
    # saperla): senza ancoraggio alla data corrente il modello ha generato una query
    # search_web con un anno vecchio (la sua stima interna da training di "adesso") e
    # presentato quel risultato come notizia recente - serve dirgli esplicitamente
    # che giorno e' davvero.
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
    # Dedup delle chiamate a tool identiche entro questa ricerca: osservate fino a 5
    # chiamate CONSECUTIVE a search_web con query IDENTICA, che consumavano quasi
    # tutte le iterazioni disponibili senza aggiungere informazione nuova. La
    # justification e' esclusa dalla chiave di confronto (puo' variare in
    # formulazione restando la stessa richiesta) - contano solo tool e argomenti.
    _seen_tool_calls: set[tuple] = set()

    # Segnalare solo a parole una chiamata duplicata non basta a far smettere il
    # modello di ripeterla (osservato piu' volte, anche 5 iterazioni consecutive
    # identiche) - serve un vincolo strutturale. _banned_tools esclude un tool dalla
    # scelta per il resto di QUESTA ricerca, ma solo dopo la SECONDA ripetizione
    # della stessa esatta combinazione (tool, argomenti): un singolo duplicato puo'
    # essere una svista isolata, non merita di bruciare l'intero tool; due sono un
    # ciclo vero. Non blocca mai chiamate allo stesso tool con argomenti diversi
    # (es. due carte diverse), che non contano mai come duplicato.
    _banned_tools: set[str] = set()
    _duplicate_hits: dict[tuple, int] = {}

    # Passo fisso e obbligatorio del workflow, chiamato direttamente in codice invece
    # di essere lasciato alla scelta dell'LLM: il prompt chiede gia' di interrogare
    # SEMPRE il KG per primo, sempre la stessa identica azione, zero giudizio
    # richiesto. Osservato un run in cui il modello ha rifiutato di chiamare
    # QUALUNQUE tool per tutte le iterazioni disponibili nonostante l'istruzione
    # ripetuta, producendo comunque claim con fonte "search_card_knowledge" mai
    # invocato - una fabbricazione piu' grave del "nessuna ricerca fatta", perche'
    # finge un'osservazione mai avvenuta. Chiamare il tool direttamente elimina la
    # classe di errore alla radice per questo primo passo, e garantisce che
    # tools_used non sia mai vuoto quando inizia il ciclo LLM sotto (che resta
    # comunque come rete di sicurezza per le iterazioni successive).
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

    # Promemoria anti-confusione tra topic/mazzi simili: osservato un caso in cui il
    # modello ha abbandonato il topic assegnato a meta' ciclo per cercarne uno diverso
    # ma strutturalmente simile a uno gia' "coperto nel KG" (stesso pattern di frase),
    # sprecando la maggior parte delle iterazioni su un archetipo estraneo. Si ripete
    # esplicitamente il topic assegnato ad OGNI iterazione (non solo qui all'inizio),
    # perche' nel run osservato la confusione e' scattata quando il promemoria
    # iniziale era ormai lontano nel contesto - resta un'istruzione testuale, quindi
    # un'ipotesi sul comportamento del modello, non una garanzia.
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
            # Se sono stati banditi tutti i tool disponibili (caso limite) si evita
            # bind_tools([]) - alcuni backend si comportano in modo inatteso con una
            # lista vuota - lasciando che il modello risponda senza strumenti: la
            # ricerca si conclude qui, gestita dal ramo "nessun tool_call" sotto.
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
            # Il KG e' ora sempre chiamato in automatico prima di questo ciclo, quindi
            # tools_used non e' mai vuoto qui - il vincolo reale non e' piu' "almeno un
            # tool" (gia' garantito) ma "almeno un tool OLTRE al KG", perche' il KG da
            # solo non parla mai del contenuto specifico del post (carte, mazzi,
            # meccaniche).
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
                    # Stessa combinazione (tool, argomenti) ritentata una SECONDA volta
                    # come duplicato (terzo tentativo identico) - non e' piu' una svista
                    # isolata, e' un ciclo. Si bandisce il tool intero (l'API di
                    # tool-calling non permette di escludere una singola combinazione)
                    # per liberare le iterazioni residue; non impedisce chiamate allo
                    # stesso tool con argomenti DIVERSI, che non sono mai un duplicato.
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
        # Diagnostica: il trace mostrava solo un CONTEGGIO di claim con fonte non nel
        # formato atteso, mai la stringa vera scritta dal modello - senza vederla e'
        # impossibile distinguere un problema risolvibile in codice (es. un prefisso
        # extra normalizzabile) da un problema di compliance del prompt. Si raccolgono
        # fino a 3 esempi REALI (non tutti, per non gonfiare il trace) da leggere
        # prima di decidere come correggere.
        _esempi_fonte_malformata: list[str] = []
        for c in summary.claims:
            claim_dict = c.model_dump()
            # Validazione strutturale (in codice, non delegata al prompt) del campo
            # source: deve essere un URL, "RAG: <...>" o esattamente "KG" - un modello
            # piu' piccolo non rispetta sempre l'istruzione testuale sul formato, quindi
            # lo si intercetta e segnala invece di fidarsi ciecamente.
            src = (claim_dict.get("source") or "").strip()
            well_formed = (
                src == "KG"
                or src.startswith("RAG:")
                or src.startswith(("http://", "https://"))
                or src == "MODELLO-FINETUNED"
                or src == "DATI-HSREPLAY"
            )

            # Riparazione automatica: per fonti search_web, il modello spesso non scrive
            # SOLO l'URL nel campo source ma copia (parte del)la riga intera del
            # risultato Tavily (formattata "{titolo} — {url}\n   {contenuto}" in
            # search_tool.py) - l'URL vero e' quindi spesso presente dentro la stringa
            # malformata, solo non all'inizio. Prima di scartare il claim per un
            # problema di sola formattazione, si prova a estrarne l'URL con una regex
            # e lo si accetta SOLO se compare per davvero in un'osservazione search_web
            # di questa ricerca (stesso controllo di _source_is_grounded sotto). Se
            # nessun URL e' riscontrabile, il claim resta malformato come prima -
            # nessuna riparazione "ottimistica" che inventi una fonte.
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

            # Riparazione automatica, variante RAG: stesso principio di sopra ma per
            # search_card_knowledge (es. 'search_card_knowledge - Lyra the Sunshard
            # (id: UNG_963)' invece del prefisso richiesto 'RAG: Lyra the Sunshard').
            # Il nome della carta e' quasi sempre presente per intero nella stringa
            # malformata; si cerca tra i nomi DAVVERO restituiti da search_card_
            # knowledge in questa ricerca (estratti dalle osservazioni reali, non
            # dalla stringa del modello) quello che compare come sottostringa, e si usa
            # il nome ESATTO dell'osservazione per normalizzare a "RAG: <nome>".
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

            # Riparazione automatica, terza variante RAG: caso in cui la query NON e'
            # un nome di carta (es. "Neutral Standard") e il modello cita tool, query e
            # posizione nella lista invece del nome (es. "search_card_knowledge -
            # query: 'Neutral Standard', risultato 1"). Si rilegge l'osservazione REALE
            # di questa ricerca per quella stessa query e si estrae il nome alla
            # posizione indicata; se non trova corrispondenza, il claim resta
            # malformato come prima.
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

            # Riparazione automatica, quarta variante: il modello scrive il nome NUDO
            # del tool come source (es. 'get_archetype_stats') invece del letterale
            # richiesto. A differenza delle varianti RAG/web sopra, questi due tool
            # hanno un letterale FISSO unico ('DATI-HSREPLAY'/'MODELLO-FINETUNED'),
            # quindi il nome del tool e' gia' l'unica informazione che serve -
            # riparabile SOLO se quel tool e' stato DAVVERO invocato in questa ricerca
            # (tools_used); altrimenti e' una fabbricazione e il claim resta
            # malformato.
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

            # Livello di affidabilita' della fonte (vedi classify_source_tier sopra
            # per il perche') - calcolato per ogni claim con fonte ben formata,
            # indipendentemente dal tool di origine (a differenza di format_valid/
            # class_valid, che si applicano solo a fonti RAG).
            claim_dict["source_tier"] = classify_source_tier(src) if well_formed else "sconosciuta"

            # Verifica di grounding (vedi _source_is_grounded sopra per il perche'
            # e' un controllo distinto e piu' severo di source_well_formed): solo
            # per fonti gia' ben formate ha senso controllare se sono anche vere -
            # una fonte malformata e' gia' segnalata da source_well_formed.
            claim_dict["source_grounded"] = (
                _source_is_grounded(
                    src, tools_used, tool_outputs, claim_dict.get("claim", "")
                )
                if well_formed else False
            )
            if well_formed and not claim_dict["source_grounded"]:
                n_ungrounded += 1

            # Verifica strutturale di formato E classe, solo per claim con fonte RAG
            # (solo li' sappiamo il nome esatto della carta). None = controllo non
            # applicabile o non verificabile, non "valido" (mazzo/formato non
            # riconosciuto, elenco Standard non disponibile, o carta non nell'indice).
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

            # Controllo di modalita' di gioco (vedi BATTLEGROUNDS_ONLY_KEYWORDS in
            # format_rules.py): a differenza di format_valid/class_valid si applica a
            # OGNI claim, non solo a fonte RAG, perche' controlla il testo del claim
            # stesso - questa pipeline pianifica solo mazzi costruiti, quindi un
            # termine esclusivo di Battlegrounds e' fuori dominio a prescindere dal
            # formato rilevato per il post.
            claim_dict["mode_valid"] = not format_rules.mentions_battlegrounds_only(
                claim_dict.get("claim", "")
            )

            claims_dicts.append(claim_dict)

        research_summary = {"claims": claims_dicts, "tools_used": tools_used}
        reasoning_trace.append(f"[Research] Riassunto finale: {len(summary.claims)} claim raccolti.")
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
                "- probabile fabbricazione, piu' grave di un semplice formato non conforme: SCARTARE "
                "questi claim prima di usarli nel post (campo 'source_grounded': False)."
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
            reasoning_trace.append(
                f"[Research] [WARNING] {n_mode_invalid} claim nominano un termine esclusivo di "
                "Battlegrounds (es. 'Trinket', 'Dark Gift') in un post su un mazzo costruito - "
                "caso reale osservato il 16/09/2026: una patch che copriva sia Standard/Wild sia "
                "Battlegrounds nello stesso articolo, con un claim che ha preso per errore le "
                "modifiche Battlegrounds come se riguardassero il meta Wild. SCARTARE questi claim "
                "prima di usarli nel post (campo 'mode_valid': False)."
            )

        # Controllo "novita' presunta ma non confermata dalle date reali delle fonti":
        # un topic puo' presupporre attualita' (es. "nuovi percorsi di missioni") ed
        # essere ricercato con successo con fonti vere, ma quelle fonti possono
        # descrivere una funzionalita' vecchia di molte patch - nessuna fabbricazione,
        # ma il post finale la presenta comunque come novita' perche' nessuno confronta
        # la data reale della fonte con la premessa di "novita'" gia' nel topic (la
        # data di oggi nel prompt copre solo il caso in cui e' il MODELLO a scrivere
        # "recentemente" di sua iniziativa). Controllo meccanico, nessun giudizio
        # semantico: se topic/motivazione presuppongono attualita' e nessuna fonte
        # search_web riporta una data "[pubblicato: ...]" recente (~6 mesi), un
        # warning invita a un controllo umano prima di pubblicare, senza bloccare.
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
                reasoning_trace.append(
                    "[Research] [WARNING] Il topic/motivazione di questo post presuppone che "
                    "l'argomento sia 'nuovo'/'recente', ma nessuna fonte search_web di questa "
                    "ricerca ha una data di pubblicazione confermata negli ultimi ~6 mesi"
                    + (
                        " (le fonti web trovate non riportano affatto una data)"
                        if not _has_any_dated_source
                        else " (le date trovate sono piu' vecchie)"
                    )
                    + " - caso reale osservato il 17/09/2026: un post su 'nuovi percorsi di "
                    "missioni' basato su articoli Blizzard reali ma di una patch molto piu' "
                    "vecchia (24.2) della serie attuale, presentati come una novita' senza che "
                    "nessuna data confermasse la recenza. Non e' detto che il contenuto sia "
                    "sbagliato (l'evento potrebbe non avere una data ufficiale chiara), ma la "
                    "premessa di attualita' va controllata a mano prima di pubblicare."
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
