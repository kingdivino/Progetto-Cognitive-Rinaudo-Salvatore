"""
Research/ReAct node (roadmap punto 5, dopo il Planner) - secondo nodo vero del grafo.

Cosa fa (per soddisfare i requisiti delle specifiche su Reasoning/Tooling e K-RAG):
1. Prende il primo post di post_plan (output del Planner) come argomento da
   ricercare - per ora la pipeline lavora su UN post alla volta; un'orchestrazione
   che itera l'intero piano post per post e' lavoro futuro (vedi state.py).
2. Esegue un ciclo ReAct (Thought -> Action -> Observation) con selezione dinamica
   tra 3 tool: query_knowledge_graph, search_card_knowledge (RAG locale su
   HearthstoneJSON), search_web (Tavily) - ognuno richiede un parametro
   `justification` obbligatorio, loggato nel reasoning_trace.
3. K-RAG: il prompt istruisce esplicitamente di interrogare PRIMA il KG e usare
   quel contesto per raffinare le query successive al RAG/search - non e' un
   comportamento automatico dentro ai tool, e' orchestrato dal ragionamento
   dell'LLM stesso (stesso pattern usato da GymAssistant, vedi guida di progetto).
4. Chiude con un'estrazione strutturata (Pydantic) dei claim raccolti, ognuno con
   la fonte esplicita a supporto - materiale pronto per il nodo Format/Draft, che
   dovra' citare queste fonti nel post generato (richiesto dalla specifica).

Nodi successivi da collegare qui in futuro: Format/Draft -> Human Review (interrupt)
-> KG Update.
"""
from __future__ import annotations

import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from src.agent.llm_config import build_llm
from src.agent.state import AgentState
from src.tools.kg_tool import query_knowledge_graph
from src.tools.rag_tool import search_card_knowledge
from src.tools.search_tool import search_web

MAX_TOOL_ITERATIONS = 6  # tetto di sicurezza - il prompt istruisce l'LLM a fermarsi
# da solo quando ha materiale sufficiente, ma un limite esplicito evita cicli
# infiniti se il modello continuasse a richiedere tool senza necessita'.

TOOLS = [query_knowledge_graph, search_card_knowledge, search_web]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


class SourcedClaim(BaseModel):
    claim: str = Field(description="un'affermazione specifica e verificabile per il post, basata SOLO su quanto trovato con i tool in questa ricerca")
    source: str = Field(
        description="fonte a supporto del claim - ESATTAMENTE una di queste 3 forme, "
        "corrispondenti ai 3 tool disponibili, mai altro: un URL (da search_web), "
        "'RAG: <nome carta>' (da search_card_knowledge), oppure 'KG' (da "
        "query_knowledge_graph). Non usare fonti come 'Planner' o altri campi dello "
        "stato - se un dato viene dal piano del Planner e non da un tool chiamato "
        "in QUESTA ricerca, non e' un claim di questo nodo e va escluso."
    )


class ResearchSummary(BaseModel):
    claims: list[SourcedClaim] = Field(
        description="claim raccolti durante la ricerca, ognuno con la propria fonte - "
        "SOLO claim supportati da un'osservazione effettiva di un tool in questa "
        "conversazione, non aggiungerne altri"
    )


RESEARCH_SYSTEM_PROMPT = """Sei il ricercatore di un blog su Hearthstone (gioco di carte Blizzard).
Il tuo compito e' raccogliere informazioni VERIFICATE per supportare il post che ti
viene indicato, usando gli strumenti disponibili, con uno stile ReAct: prima pensi
(Thought) a cosa ti serve e perche', poi agisci (Action) chiamando UN tool alla
volta, poi osservi il risultato (Observation) prima di decidere il prossimo passo.

Strumenti disponibili:
- query_knowledge_graph: legge il Knowledge Graph editoriale (topic gia' coperti,
  post recenti). USALO SEMPRE PER PRIMO: il suo risultato ti aiuta a capire cosa e'
  gia' stato trattato e a formulare query piu' mirate per i tool successivi (K-RAG:
  il contesto del KG deve informare le ricerche dopo, non essere un lookup a vuoto).
- search_card_knowledge: ricerca semantica in un corpus locale e verificabile di
  carte Hearthstone (nome, costo, statistiche, testo, classe, espansione). Usalo per
  qualunque domanda su meccaniche/testo di carte esistenti.
- search_web: ricerca web (Tavily) per fatti recenti/attuali (espansioni, patch,
  tornei, annunci ufficiali) che il corpus locale non copre. Usalo SOLO per questo
  tipo di fatti specifici, non per meccaniche di carte (per quelle usa search_card_knowledge).

Regole:
- OGNI chiamata a un tool richiede il parametro `justification`: perche' ti serve
  questa informazione ora, per quale claim del post. Mai una justification vaga.
- Non inventare MAI un fatto che non hai effettivamente trovato con un tool in
  questa conversazione - se un tool non trova nulla di utile, dillo e prosegui
  diversamente, non riempire il vuoto con dettagli plausibili ma non verificati.
- Fermati quando hai raccolto abbastanza materiale per il post (di norma bastano
  2-4 chiamate a tool) - non continuare a cercare senza motivo.
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
  giusta) su search_card_knowledge."""


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
    current_post = state.get("current_post") or (post_plan[0] if post_plan else None)
    if current_post is None:
        reasoning_trace.append(
            "[Research] Nessun post disponibile da ricercare (post_plan vuoto) - nodo saltato."
        )
        return {
            **state,
            "reasoning_trace": reasoning_trace,
            "research_summary": {"claims": [], "tools_used": []},
        }

    reasoning_trace.append(
        f"[Research] Avvio ricerca per il post: [{current_post.get('tipo')}] {current_post.get('topic')}"
    )

    llm = build_llm(temperature=0.3)
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = [
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Post da ricercare:\n"
                f"Tipo: {current_post.get('tipo')}\n"
                f"Topic: {current_post.get('topic')}\n"
                f"Motivazione dal Planner: {current_post.get('justification')}\n\n"
                "Raccogli le informazioni necessarie usando i tool disponibili, "
                "poi fermati quando hai materiale sufficiente."
            )
        ),
    ]

    tools_used: list[str] = []

    for iteration in range(MAX_TOOL_ITERATIONS):
        _llm_call_start = time.perf_counter()
        try:
            response = llm_with_tools.invoke(messages)
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
            reasoning_trace.append(
                f"[Research] Nessun altro tool richiesto (iterazione {iteration + 1}) - ricerca conclusa."
            )
            break

        for call in tool_calls:
            name = call.get("name")
            args = call.get("args", {}) or {}
            justification = args.get("justification", "(nessuna giustificazione fornita)")
            reasoning_trace.append(f"[Research] Thought->Action: '{name}' con args={args}")

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
                "anche verificati/ritrovati con un tool in questa ricerca."
            )
        )
    ]
    _extraction_start = time.perf_counter()
    try:
        summary: ResearchSummary = structured_llm.invoke(extraction_messages)
        llm_time_total += time.perf_counter() - _extraction_start
        claims_dicts = []
        n_malformed = 0
        for c in summary.claims:
            claim_dict = c.model_dump()
            # Validazione strutturale (in codice, non delegata al prompt) del campo
            # source: deve essere un URL, "RAG: <...>" o esattamente "KG". Osservato
            # (03/09/2026, qwen3:8b) che un modello piu' piccolo non rispetta sempre
            # l'istruzione testuale sul formato - qui lo intercettiamo e lo segnaliamo
            # invece di fidarci ciecamente, cosi' un futuro nodo Format/Draft (o un
            # revisore umano) puo' scartare/verificare a mano i claim non conformi.
            src = (claim_dict.get("source") or "").strip()
            well_formed = src == "KG" or src.startswith("RAG:") or src.startswith(("http://", "https://"))
            claim_dict["source_well_formed"] = well_formed
            if not well_formed:
                n_malformed += 1
            claims_dicts.append(claim_dict)

        research_summary = {"claims": claims_dicts, "tools_used": tools_used}
        reasoning_trace.append(f"[Research] Riassunto finale: {len(summary.claims)} claim raccolti.")
        if n_malformed:
            reasoning_trace.append(
                f"[Research] [WARNING] {n_malformed}/{len(summary.claims)} claim con fonte non nel "
                "formato atteso (URL/RAG:/KG) - controllare a mano prima di usarli nel post "
                "(campo 'source_well_formed': False su questi claim)."
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
