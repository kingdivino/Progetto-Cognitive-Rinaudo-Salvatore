"""
RAG retrieval tool (uno dei 3 tool minimi obbligatori dalla specifica) - ricerca
semantica sull'indice FAISS locale costruito da notebooks/05_build_rag_index.py
(corpus: carte collezionabili di HearthstoneJSON).

Limite noto (vedi anche guida di progetto): questo corpus copre bene i post
how-to/review su carte/archetipi ma NON patch notes/espansioni/tornei - per quelli
serve il search tool (src/tools/search_tool.py). Il RAG e' statico e verificabile
(nessun rischio di risultati di bassa qualita' come una ricerca web), va preferito
quando l'informazione che serve riguarda testo/statistiche di carte esistenti.

Il parametro `justification` e' obbligatorio per lo stesso motivo del search tool:
tracciare nel reasoning_trace perche' l'agente ha deciso di interrogare il RAG.
"""
from __future__ import annotations

import json
import os

from langchain_core.tools import tool

INDEX_DIR = os.path.join("data", "rag")
INDEX_PATH = os.path.join(INDEX_DIR, "cards.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "cards_metadata.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Lazy singleton (come il driver Neo4j in src/kg/connection.py): l'indice e il
# modello di embedding si caricano una sola volta, non ad ogni chiamata del tool.
_index = None
_metadata = None
_embedder = None


def _load_index():
    global _index, _metadata, _embedder
    if _index is not None:
        return _index, _metadata, _embedder

    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise RuntimeError(
            "Indice RAG non trovato - lanciare prima 'python -u "
            "notebooks/05_build_rag_index.py' per costruirlo da HearthstoneJSON."
        )

    import faiss
    from sentence_transformers import SentenceTransformer

    _index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, encoding="utf-8") as f:
        _metadata = json.load(f)
    _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _index, _metadata, _embedder


def get_card_info_by_name(name: str) -> dict | None:
    """Cerca una carta per nome ESATTO (case-insensitive) nei metadati gia' caricati
    dall'indice RAG e ritorna {'set': ..., 'cardClass': ...}, o None se non trovata o
    se l'indice non e' disponibile. Usato dal nodo Research per verificare in codice -
    non solo a parole nel prompt - che una carta suggerita in un claim con fonte
    'RAG: <nome>' sia davvero legale nel formato (Standard/Wild, src/agent/
    format_rules.py) E della classe giusta (o Neutrale) per il mazzo del post in
    corso - una carta di classe sbagliata non e' giocabile in quel mazzo, a
    prescindere dal formato."""
    try:
        _, metadata, _ = _load_index()
    except Exception:
        return None
    name_lower = name.strip().lower()
    for card in metadata:
        if (card.get("name") or "").strip().lower() == name_lower:
            return {"set": card.get("set"), "cardClass": card.get("cardClass")}
    return None


@tool
def search_card_knowledge(query: str, justification: str, top_k: int = 5) -> str:
    """Cerca nel corpus locale (RAG) informazioni su carte Hearthstone: testo,
    costo, statistiche, classe, rarita', espansione. Usa questo tool per qualsiasi
    domanda su meccaniche/testo di carte esistenti - e' locale e verificabile, da
    preferire al search tool quando l'informazione riguarda carte gia' pubblicate
    invece di eventi/annunci recenti.

    Args:
        query: cosa cercare (es. nome di una carta, o una descrizione della
            meccanica che si sta cercando).
        justification: perche' serve questa ricerca ora - quale claim del post
            deve supportare. Obbligatoria.
        top_k: quanti risultati restituire (default 5).
    """
    try:
        index, metadata, embedder = _load_index()
    except Exception as e:
        return f"[ERROR] {e}"

    import numpy as np

    query_vec = embedder.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.asarray(query_vec, dtype="float32"), top_k)

    formatted = [f"Risultati RAG per {query!r} (giustificazione: {justification}):"]
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx < 0:
            continue
        card = metadata[idx]
        formatted.append(
            f"{rank}. [{card['name']} — id: {card['id']}, score {score:.2f}] {card['document']}"
        )
    if len(formatted) == 1:
        return f"Nessun risultato rilevante trovato per: {query!r}"
    return "\n".join(formatted)
