"""
Costruzione del grafo LangGraph. Nodi presenti: Planner -> [select_next_post ->
Research/ReAct -> Format/Draft -> Human Review -> KG Update] -> torna a
select_next_post finche' il piano non e' esaurito (roadmap punti 4-8, grafo
completo). Il nodo KG Update scrive sul Knowledge Graph SOLO quando 'review_decision'
e' "approved" (vedi src/agent/kg_update.py); "regenerate" e "discarded" lo saltano.

Ciclo sui post pianificati (src/agent/orchestrator.py, 11/09/2026): il Planner
produce l'INTERO piano una sola volta, poi select_next_post fa avanzare il grafo un
post alla volta attraverso Research/Format/Human Review/KG Update, finche' tutti i
post del piano non sono stati elaborati (approvati, scartati con "rigenera"/"scarta",
o falliti) - prima di questa data la pipeline elaborava sempre e solo UN post,
scelto a mano con la variabile d'ambiente RESEARCH_POST_INDEX (rimasta come
fallback solo per invocare research_topic() in isolamento, vedi li').

Checkpointer richiesto dal nodo Human Review (interrupt()/Command(resume=...), vedi
src/agent/human_review.py) - senza, interrupt() solleva un errore a runtime. Per i
test di questo progetto (un singolo processo Python, mai riavviato tra la sospensione
e la ripresa) basta un checkpointer in memoria (InMemorySaver) - un checkpointer su
disco/DB servirebbe solo per riprendere una revisione dopo aver chiuso il processo,
non necessario per gli obiettivi di questo progetto universitario.

IMPORTANTE per chi chiama graph.invoke(...)/graph.stream(...): con un checkpointer
impostato, LangGraph richiede un 'thread_id' esplicito in config (altrimenti solleva
un errore) - vedi notebooks/09_test_human_review.py per il pattern completo
(invoke -> se il risultato contiene '__interrupt__', mostrare la bozza e richiamare
invoke con Command(resume=...) sullo stesso thread_id; con il ciclo su tutti i post,
questo pattern generico gestisce automaticamente TUTTI i round di revisione di TUTTI
i post, non serve nessuna modifica al notebook per questo).
"""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from src.agent.format_draft import draft_post
from src.agent.human_review import human_review
from src.agent.kg_update import update_kg
from src.agent.orchestrator import route_after_select, select_next_post
from src.agent.planner import plan_posts
from src.agent.research import research_topic
from src.agent.state import AgentState


def _route_after_review(state: AgentState) -> str:
    """Instradamento dopo il nodo Human Review, basato su 'review_decision' (vedi
    src/agent/human_review.py): "regenerate" torna al nodo Format per una nuova
    bozza sullo STESSO post (stessi claim gia' verificati da Research); qualunque
    altro valore ("approved", "discarded", o un default di sicurezza) va al nodo
    KG Update, che scrive sul grafo solo se davvero 'review_decision' e' "approved"
    (doppio controllo, vedi src/agent/kg_update.py) e poi (in ogni caso) passa la
    mano a select_next_post per il post successivo del piano."""
    if state.get("review_decision") == "regenerate":
        return "format"
    return "kg_update"


def build_graph(checkpointer=None):
    """checkpointer: oggetto compatibile con l'interfaccia BaseCheckpointSaver di
    LangGraph. Se None (default), ne viene creato uno in memoria (InMemorySaver) -
    sufficiente per i test in un singolo processo di questo progetto. Passare un
    checkpointer esplicito e' utile solo per condividerne uno tra piu' invocazioni
    fatte da script diversi nello stesso processo (non un caso d'uso di questo
    progetto per ora)."""
    if checkpointer is None:
        checkpointer = InMemorySaver()

    graph = StateGraph(AgentState)
    graph.add_node("planner", plan_posts)
    graph.add_node("select_next_post", select_next_post)
    graph.add_node("research", research_topic)
    graph.add_node("format", draft_post)
    graph.add_node("human_review", human_review)
    graph.add_node("kg_update", update_kg)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "select_next_post")
    graph.add_conditional_edges("select_next_post", route_after_select, {"research": "research", END: END})
    graph.add_edge("research", "format")
    graph.add_edge("format", "human_review")
    graph.add_conditional_edges("human_review", _route_after_review, {"format": "format", "kg_update": "kg_update"})
    # kg_update chiude sempre il post corrente (che sia stato scritto sul KG o
    # saltato) tornando a select_next_post per decidere se c'e' un post successivo
    # nel piano o se il grafo deve terminare - MAI un edge fisso verso END da qui,
    # altrimenti il ciclo si fermerebbe sempre dopo il primo post.
    graph.add_edge("kg_update", "select_next_post")
    return graph.compile(checkpointer=checkpointer)
