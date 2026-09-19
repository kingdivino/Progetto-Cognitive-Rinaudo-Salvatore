"""Nodo Format/Draft: genera il testo del post usando solo i claim verificati da
Research, interrogando di nuovo il KG per coerenza con i post gia' pubblicati."""
from __future__ import annotations

import datetime
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agent.format_rules import fix_meta_gender
from src.agent.llm_config import build_llm
from src.agent.state import AgentState
from src.tools.kg_tool import query_knowledge_graph


class DraftPost(BaseModel):
    titolo: str = Field(description="Titolo del post, chiaro e specifico sul topic assegnato")
    corpo: str = Field(
        description="Testo completo del post (alcuni paragrafi in prosa), che cita "
        "esplicitamente la fonte subito dopo ogni fatto specifico verificato - es. "
        "'il mazzo ha un winrate del 54% (fonte: hsreplay.net)'."
    )
    fonti_citate: list[str] = Field(
        description="Elenco delle fonti effettivamente usate nel corpo del post, "
        "nello stesso formato ESATTO del campo 'source' dei claim ricevuti (un URL, "
        "'RAG: <nome carta>' o 'KG') - una per fonte distinta, senza duplicati, senza "
        "riformulare o abbreviare la stringa originale."
    )


FORMAT_SYSTEM_PROMPT = """Sei il redattore di un blog su Hearthstone (gioco di carte Blizzard).
Il tuo compito e' scrivere il TESTO VERO E PROPRIO di un post, a partire da un topic
gia' pianificato e da un elenco di claim GIA' VERIFICATI da un nodo di ricerca a monte.

Regole fondamentali, nello stesso spirito di quelle gia' seguite nei nodi precedenti
di questo agente:
- Usa ESCLUSIVAMENTE i fatti presenti nell'elenco di claim verificati che ricevi -
  non aggiungere MAI un fatto specifico (numeri, nomi di carte, statistiche, eventi)
  che non sia in quell'elenco, anche se ti sembra plausibile o lo ricordi dal tuo
  addestramento (potrebbe essere superato o sbagliato - il meta di Hearthstone
  cambia spesso ed e' esattamente per questo che esiste un nodo di ricerca a monte).
- Ogni volta che usi un fatto da un claim, cita ESPLICITAMENTE la sua fonte nel testo,
  copiando la stringa della fonte esattamente come te la abbiamo data (non
  riformularla, non abbreviarla, non inventarne una piu' leggibile) - il lettore deve
  poter distinguere un'affermazione verificata da un commento generico tuo, e un
  controllo automatico dopo la tua risposta confrontera' le fonti che dichiari con
  quelle che ti sono state davvero date.
- Se l'elenco di claim verificati e' VUOTO o molto scarno, NON inventare contenuto
  specifico per riempire il vuoto: scrivi un post piu' breve e generico sul topic
  (spiegando concetti generali del gioco, se pertinenti), senza numeri o fatti
  specifici non supportati - stesso principio gia' richiesto al Planner quando
  mancano dati concreti per un topic.
- Se ricevi un contesto dal Knowledge Graph su post/topic precedenti, collegati ad
  esso SOLO se naturale e pertinente per il lettore (es. "abbiamo gia' parlato di X,
  qui approfondiamo Y") - non forzare un collegamento se non c'e' relazione reale, e
  non trattarlo come materiale su cui basare nuovi fatti (il KG editoriale non
  contiene mai dati di gioco, solo la cronologia dei post).
- Il post deve restare nell'ambito del topic e del formato/classe assegnati (se
  indicati nel topic/motivazione) - non generalizzare oltre quello che il topic
  chiede.
- NON descrivere un fatto come "recente", "degli ultimi mesi", "di questi giorni" o
  simili SOLO perche' compare in un claim verificato - un claim verificato e' vero,
  non necessariamente attuale (puo' provenire da una fonte web vecchia di mesi o
  anni). Usa un linguaggio temporale ("recente"/"nuovo") SOLO se il claim o la sua
  fonte indicano esplicitamente una data vicina a oggi (vedi la data di oggi nel
  messaggio con il post da scrivere); altrimenti descrivi il fatto senza collocarlo
  nel tempo (es. "il mazzo X ha un winrate del..." invece di "recentemente il mazzo
  X..."), o menziona esplicitamente quando risale se e' rilevante per il lettore.
- Quando parli del "meta" (il metagame competitivo del gioco - i mazzi/strategie
  dominanti in un dato momento), usa SEMPRE l'articolo MASCHILE: "il meta", "del meta",
  "nel meta", "un meta" - MAI "la meta"/"della meta"/"nella meta"/"una meta" (in
  italiano standard "meta" al femminile significherebbe "traguardo/obiettivo", un
  significato diverso da quello inteso qui - la community italiana di Hearthstone usa
  questo prestito al maschile).
- Quando un claim verificato contiene gia' un ragionamento specifico (es. "la carta X ha l'effetto Y, quindi conviene Z"), PRESERVA quella specificita' nel testo - nome della carta, effetto reale, implicazione concreta - invece di riformularla in una frase piu' generica o vaga: e' proprio questa specificita' verificabile, e non la genericita', il valore che distingue il post da un testo scritto senza ricerca a monte.
"""


def draft_post(state: AgentState) -> AgentState:
    """Nodo Format/Draft del grafo LangGraph. Riceve current_post e research_summary
    (gia' popolati dai nodi precedenti), ritorna lo stato aggiornato con
    reasoning_trace/kg_summary/draft/timings popolati."""
    node_start = time.perf_counter()
    llm_time_total = 0.0
    tool_time_total = 0.0
    reasoning_trace = list(state.get("reasoning_trace", []))
    kg_summary = dict(state.get("kg_summary", {}))

    current_post = state.get("current_post")
    research_summary = state.get("research_summary") or {}
    claims = research_summary.get("claims", [])

    if current_post is None:
        reasoning_trace.append(
            "[Format] Nessun post corrente disponibile (current_post assente) - nodo saltato."
        )
        return {**state, "reasoning_trace": reasoning_trace, "draft": None}

    # Filtro in codice (non lasciato all'LLM): solo i claim che hanno superato TUTTE
    # le verifiche gia' fatte a monte nel nodo Research arrivano qui. None (non
    # verificabile) e' tollerato per format_valid/class_valid (significa "non
    # applicabile", non "invalido" - es. claim con fonte web, non RAG) ma MAI per
    # source_well_formed/source_grounded, che devono essere esplicitamente True.
    trusted_claims = [
        c for c in claims
        if c.get("source_well_formed") is True
        and c.get("source_grounded") is True
        and c.get("format_valid") is not False
        and c.get("class_valid") is not False
        and c.get("mode_valid") is not False
    ]
    n_discarded = len(claims) - len(trusted_claims)
    reasoning_trace.append(
        f"[Format] {len(trusted_claims)}/{len(claims)} claim ricevuti dal Research superano tutte "
        "le verifiche e vengono usati per il drafting"
        + (f" ({n_discarded} scartati: fonte non conforme, non riscontrata in un'osservazione reale, "
           "carta fuori formato/classe, o termine esclusivo di Battlegrounds fuori dominio)."
           if n_discarded else ".")
    )

    # Chiamata KG fissa per la coerenza in fase di drafting - stesso principio della
    # chiamata KG fissa nel nodo Research (08/09/2026): e' sempre lo stesso passo
    # necessario (richiesto esplicitamente dalla specifica per la fase di drafting),
    # nessun giudizio dell'LLM serve per decidere se farla.
    _kg_call_start = time.perf_counter()
    try:
        kg_observation = query_knowledge_graph.invoke({
            "justification": "Passo fisso del workflow: controllare il Knowledge Graph in fase di "
            "drafting per collegarsi a contenuti precedenti pertinenti, se presenti."
        })
    except Exception as e:
        kg_observation = f"[ERROR] esecuzione tool 'query_knowledge_graph' fallita: {e}"
    tool_time_total += time.perf_counter() - _kg_call_start
    kg_summary["draft_query_result"] = str(kg_observation)
    reasoning_trace.append(f"[Format] Contesto KG per la coerenza col post: {str(kg_observation)[:300]}")

    claims_block = "\n".join(
        f"- {c.get('claim')} (fonte: {c.get('source')})" for c in trusted_claims
    ) or "(nessun claim verificato disponibile per questo post - vedi regola sul caso vuoto)"

    # Data reale di oggi, iniettata qui per lo stesso motivo del nodo Research
    # senza data, questo nodo non ha modo di giudicare se un claim verificato ma non
    # recente vada presentato come "recente" nel testo - osservato un articolo datato
    # descritto come notizia "degli ultimi mesi" nella bozza finale.
    _oggi = datetime.date.today().strftime("%d/%m/%Y")

    # La "motivazione" del Planner spesso contiene cifre prese dal dataset locale -
    # utili come contesto quando ci sono claim verificati a supportarle, ma un canale
    # di fuga quando non ce ne sono (osservato un post a 0 claim che riprendeva quelle
    # cifre invece di restare generico). Con trusted_claims vuoto, il campo e' OMESSO.
    if trusted_claims:
        motivazione_line = f"Motivazione dal Planner: {current_post.get('justification')}\n\n"
    else:
        motivazione_line = (
            "Motivazione dal Planner: NON fornita in questo prompt di proposito - nessun claim "
            "e' sopravvissuto alla verifica per questo post (vedi sotto), e il testo del Planner "
            "puo' contenere cifre/statistiche non ancora verificate da questo nodo. Basati SOLO "
            "su Tipo e Topic, scrivendo in modo generico come da regola sul caso vuoto. "
            "ATTENZIONE: anche il campo Topic qui sotto puo' contenere una cifra/percentuale "
            "specifica scritta dal Planner (es. un winrate) - quella cifra NON e' stata "
            "verificata da nessun claim, quindi non va MAI ripresa ne' nel titolo ne' nel corpo, "
            "nemmeno se compare testualmente nel Topic.\n\n"
        )

    messages = [
        SystemMessage(content=FORMAT_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Data di oggi: {_oggi}. Usala per decidere se e' corretto descrivere "
                "un fatto come recente (vedi regola sul linguaggio temporale).\n\n"
                f"Post da scrivere:\n"
                f"Tipo: {current_post.get('tipo')}\n"
                f"Topic: {current_post.get('topic')}\n"
                "ATTENZIONE (vale SEMPRE, non solo quando i claim verificati sono zero - errore reale osservato il 18/09/2026: un post con un claim verificato vero, DATI-HSREPLAY, e' finito comunque per citare la cifra ESATTA presente nel Topic - 26.1% di winrate dell'archetipo - come se fosse quella confermata dal claim, quando il claim verificato diceva in realta' un numero diverso, 54.2%/55.4% aggregato di classe, non dell'archetipo specifico): sia il Topic sia la Motivazione qui sotto possono contenere una cifra/percentuale scritta dal Planner (spesso presa dal dataset locale, MAI verificata da un tool in questa ricerca). Usa un numero nel testo SOLO se quello STESSO numero compare anche in uno dei claim verificati elencati sotto - se il Topic/la Motivazione menzionano una cifra che i claim verificati non confermano (anche quando un claim verificato esiste, ma per un dato correlato e diverso, es. la media di classe invece che quella dell'archetipo specifico), NON usare quella cifra nel testo, anche se hai una citazione valida per il dato correlato: cita solo il dato che il claim verificato dice davvero.\n\n"
                f"{motivazione_line}"
                f"Claim verificati disponibili ({len(trusted_claims)}):\n{claims_block}\n\n"
                f"Contesto dal Knowledge Graph (post/topic precedenti): {kg_observation}\n\n"
                "Scrivi ora il post seguendo le regole del messaggio di sistema."
            )
        ),
    ]

    llm = build_llm(temperature=0.4)
    structured_llm = llm.with_structured_output(DraftPost)
    _llm_start = time.perf_counter()
    draft_dict = None
    try:
        draft: DraftPost = structured_llm.invoke(messages)
        llm_time_total += time.perf_counter() - _llm_start
        draft_dict = draft.model_dump()

        # Dedup in codice delle fonti dichiarate (il prompt lo richiede gia', ma non
        # basta da solo - osservate fonti ripetute nonostante la regola). dict.fromkeys
        # preserva l'ordine di prima apparizione, a differenza di un set puro.
        _fonti_originali = draft_dict.get("fonti_citate", [])
        _fonti_deduplicate = list(dict.fromkeys(_fonti_originali))
        if len(_fonti_deduplicate) < len(_fonti_originali):
            reasoning_trace.append(
                f"[Format] {len(_fonti_originali) - len(_fonti_deduplicate)} fonte/i duplicata/e "
                "rimossa/e da 'fonti_citate' (l'LLM le ha dichiarate piu' volte nonostante la regola "
                "di non farlo)."
            )
        draft_dict["fonti_citate"] = _fonti_deduplicate

        # Rete di sicurezza in codice per il genere di "il meta"/"la meta" (vedi
        # fix_meta_gender in format_rules.py) - la regola nel prompt non basta da sola.
        draft_dict["titolo"], _n_fix_titolo = fix_meta_gender(draft_dict.get("titolo", ""))
        draft_dict["corpo"], _n_fix_corpo = fix_meta_gender(draft_dict.get("corpo", ""))
        if _n_fix_titolo or _n_fix_corpo:
            reasoning_trace.append(
                f"[Format] Corretto il genere di 'meta' (la -> il) {_n_fix_titolo + _n_fix_corpo} "
                "volta/e nel titolo/corpo (l'LLM lo aveva scritto al femminile nonostante la regola "
                "esplicita nel prompt)."
            )

        reasoning_trace.append(
            f"[Format] Bozza generata: '{draft_dict.get('titolo')}' "
            f"({len(draft_dict.get('corpo', ''))} caratteri, "
            f"{len(draft_dict.get('fonti_citate', []))} fonti dichiarate)."
        )
        if not trusted_claims:
            reasoning_trace.append(
                "[Format] [WARNING] Nessun claim verificato era disponibile per questo post - "
                "questa bozza e' generica per costruzione, va trattata con priorita' alta nella "
                "revisione umana (roadmap: nodo Human Review)."
            )
            # Rete di sicurezza in codice: a 0 claim verificati il post dovrebbe restare
            # generico, ma il fallback puo' non bastare se il testo contiene comunque
            # cifre statistiche (percentuali, "N partite") - riprese ad es. dal campo
            # Topic (sempre nel prompt, scritto dal Planner e non verificato). Si
            # controllano titolo E corpo insieme: una cifra sospetta puo' finire in uno
            # solo dei due mentre l'altro resta correttamente generico.
            _sospetti = re.findall(
                r"\d+[.,]?\d*\s?%|\b\d{2,}\s+partite\b",
                f"{draft_dict.get('titolo', '')} {draft_dict.get('corpo', '')}",
            )
            if _sospetti:
                reasoning_trace.append(
                    "[Format] [WARNING] ATTENZIONE PRIORITARIA: 0 claim verificati per questo post, "
                    f"ma il testo generato (titolo e/o corpo) contiene comunque cifre che sembrano "
                    f"statistiche ({_sospetti[:5]}) - il fallback 'post generico senza numeri' NON ha "
                    "funzionato come previsto. NON pubblicare senza controllare a mano l'origine di "
                    "questi numeri."
                )

            # Seconda rete di sicurezza, stesso principio ma per un leak diverso: a 0
            # claim verificati il corpo puo' restare senza numeri (niente warning sopra)
            # eppure nominare carte specifiche inventate (caso reale: nomi italiani non
            # riscontrabili come traduzione ufficiale di nessuna carta, ne' nei risultati
            # RAG di quella ricerca). Non e' verificabile in codice se un nome e' una
            # carta reale (l'indice RAG e' solo in inglese, senza localizzazione italiana
            # da confrontare), quindi si usa un segnale piu' debole ma utile: il modello
            # tende a evidenziare i nomi di carta tra asterischi (enfasi Markdown), cosa
            # che un post davvero generico non avrebbe motivo di fare. Falsi negativi
            # possibili se il modello non usa asterischi, ma meglio di un leak silenzioso.
            _entita_enfatizzate = re.findall(
                r"\*([^*\n]{3,60})\*",
                f"{draft_dict.get('titolo', '')} {draft_dict.get('corpo', '')}",
            )
            if _entita_enfatizzate:
                reasoning_trace.append(
                    "[Format] [WARNING] ATTENZIONE PRIORITARIA: 0 claim verificati per questo post, "
                    f"ma il testo generato enfatizza (tra asterischi) delle entita' specifiche "
                    f"({_entita_enfatizzate[:5]}) - potrebbero essere nomi di carte o dati inventati "
                    "invece che concetti generali di gioco. NON pubblicare senza controllare a mano "
                    "se queste entita' sono reali e pertinenti."
                )

        # Verifica minima in codice (stesso principio gia' visto in research.py): le
        # fonti dichiarate nel post devono comparire ESATTAMENTE tra quelle dei
        # claim verificati che gli sono stati dati - non fidarsi che l'LLM non ne
        # abbia riformulate o aggiunte di nuove/inventate.
        trusted_sources = {c.get("source") for c in trusted_claims}
        fonti_non_riconosciute = [
            f for f in draft_dict.get("fonti_citate", []) if f not in trusted_sources
        ]
        draft_dict["fonti_non_riconosciute"] = fonti_non_riconosciute
        if fonti_non_riconosciute:
            reasoning_trace.append(
                f"[Format] [WARNING] {len(fonti_non_riconosciute)} fonte/i dichiarate nel post NON "
                "corrispondono esattamente a nessun claim verificato ricevuto - probabile fonte "
                "riformulata o aggiunta dall'LLM, controllare a mano (campo 'fonti_non_riconosciute')."
            )

        # Rete di sicurezza in codice, stesso principio dei due controlli sopra (che
        # pero' girano solo a trusted_claims vuoto) - questo gira SEMPRE: una
        # percentuale che compare sia nel testo generato sia nel Topic/Motivazione del
        # Planner, ma non nel testo dei claim verificati, e' probabilmente una cifra
        # non verificata agganciata a una citazione vera per un dato diverso (vedi
        # _source_is_grounded in research.py per la protezione equivalente a monte).
        def _estrai_percentuali(testo: str) -> set:
            return {
                m.replace(",", ".")
                for m in re.findall(r"\d+(?:[.,]\d+)?%", testo or "")
            }

        _cifre_planner = _estrai_percentuali(
            f"{current_post.get('topic', '')} {current_post.get('justification', '')}"
        )
        _cifre_verificate = _estrai_percentuali(claims_block)
        _cifre_draft = _estrai_percentuali(
            f"{draft_dict.get('titolo', '')} {draft_dict.get('corpo', '')}"
        )
        _cifre_sospette = sorted((_cifre_draft & _cifre_planner) - _cifre_verificate)
        if _cifre_sospette:
            reasoning_trace.append(
                "[Format] [WARNING] ATTENZIONE PRIORITARIA: il post contiene percentuali "
                f"({_cifre_sospette}) presenti nel Topic/Motivazione scritti dal Planner ma "
                "NON confermate dal testo di nessun claim verificato (anche quando un claim "
                "verificato esiste, potrebbe riguardare un dato correlato ma diverso, es. la "
                "media di classe invece dell'archetipo specifico) - probabile cifra non "
                "verificata agganciata a una citazione vera per un altro dato. NON pubblicare "
                "senza controllare a mano l'origine di questi numeri."
            )
    except Exception as e:
        llm_time_total += time.perf_counter() - _llm_start
        reasoning_trace.append(f"[Format] [ERROR] Fallita la generazione della bozza: {e}")

    node_elapsed = time.perf_counter() - node_start
    reasoning_trace.append(
        f"[Format] Tempo nodo: {node_elapsed:.1f}s totali (LLM: {llm_time_total:.1f}s, "
        f"tool: {tool_time_total:.1f}s, resto/overhead: {node_elapsed - llm_time_total - tool_time_total:.1f}s)."
    )
    timings = dict(state.get("timings", {}))
    timings["format_llm_s"] = round(llm_time_total, 2)
    timings["format_tool_s"] = round(tool_time_total, 2)
    timings["format_total_s"] = round(node_elapsed, 2)

    return {
        **state,
        "reasoning_trace": reasoning_trace,
        "kg_summary": kg_summary,
        "draft": draft_dict,
        "timings": timings,
    }
