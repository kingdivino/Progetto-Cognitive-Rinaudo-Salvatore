"""
Costruisce l'indice RAG (FAISS + sentence-transformers) a partire da HearthstoneJSON
(roadmap punto 5, K-RAG). Corpus scelto: le carte collezionabili di HearthstoneJSON
(nome, classe, tipo, costo, statistiche, testo) - e' l'unico corpus locale gia'
disponibile nel progetto (data/raw/hearthstonejson/cards.json, gia' usato per il
feature engineering del dataset di fine-tuning, vedi notebooks/03).

IMPORTANTE - limite noto (documentato anche nella guida di progetto): questo corpus
copre bene i post di tipo how-to/review su carte/archetipi, ma NON copre patch notes,
annunci di espansioni o tornei - per quelli servirebbe un corpus aggiuntivo (patch
notes ufficiali Blizzard, community) non ancora raccolto. Per ora il Search tool
(Tavily) e' l'unica fonte per quei topic; il RAG resta locale e verificabile.

Uso (dal venv, una tantum o quando cards.json viene aggiornato):
    python -u notebooks/05_build_rag_index.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CARDS_PATH = os.path.join("data", "raw", "hearthstonejson", "cards.json")
INDEX_DIR = os.path.join("data", "rag")
INDEX_PATH = os.path.join(INDEX_DIR, "cards.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "cards_metadata.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # piccolo (~80MB) e veloce
# su CPU - qui non serve un modello di embedding grande: i documenti sono brevi
# (poche righe per carta) e la ricerca semantica non richiede sfumature complesse.


def log(msg: str) -> None:
    print(msg, flush=True)


def clean_card_text(text: str | None) -> str:
    """Ripulisce il testo carta HearthstoneJSON dai tag di formattazione (<b>, <i>) e
    dai placeholder di valori scalabili con la spell power ($25 -> 25, #3 -> 3), che
    altrimenti confonderebbero sia l'embedding sia la lettura da parte dell'LLM."""
    if not text:
        return ""
    cleaned = re.sub(r"</?b>|</?i>", "", text)
    cleaned = re.sub(r"[$#](\d+)", r"\1", cleaned)
    cleaned = cleaned.replace("\n", " ").strip()
    return cleaned


def card_to_document(card: dict) -> str:
    """Un documento testuale per carta, pensato per essere sia un buon target di
    ricerca semantica sia leggibile/citabile direttamente nel post generato a valle."""
    parts = [card["name"]]
    if card.get("cardClass"):
        parts.append(f"classe {card['cardClass'].title()}")
    if card.get("type"):
        parts.append(card["type"].lower())
    if card.get("cost") is not None:
        parts.append(f"costo {card['cost']}")
    if card.get("attack") is not None or card.get("health") is not None:
        parts.append(f"{card.get('attack', '?')}/{card.get('health', '?')}")
    if card.get("rarity"):
        parts.append(card["rarity"].lower())
    if card.get("set"):
        parts.append(f"espansione {card['set']}")
    header = ", ".join(parts)
    text = clean_card_text(card.get("text"))
    return f"{header}. {text}" if text else header


def load_collectible_cards() -> list[dict]:
    log(f"Carico {CARDS_PATH}...")
    with open(CARDS_PATH, encoding="utf-8") as f:
        cards = json.load(f)
    collectible = [c for c in cards if c.get("collectible") and c.get("name")]
    log(f"  {len(cards)} carte totali, {len(collectible)} collezionabili (corpus RAG).")
    return collectible


def main() -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)

    cards = load_collectible_cards()
    documents = [card_to_document(c) for c in cards]
    metadata = [
        {
            "id": c["id"],
            "name": c["name"],
            "cardClass": c.get("cardClass"),
            "set": c.get("set"),
            "document": doc,
        }
        for c, doc in zip(cards, documents)
    ]

    log(f"Carico il modello di embedding ({EMBEDDING_MODEL})...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)

    log(f"Calcolo {len(documents)} embedding (puo' richiedere qualche minuto su CPU)...")
    embeddings = model.encode(
        documents, show_progress_bar=True, normalize_embeddings=True, batch_size=64
    )

    log("Costruisco l'indice FAISS (Inner Product su embedding normalizzati = cosine similarity)...")
    import faiss
    import numpy as np

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.asarray(embeddings, dtype="float32"))

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    log(f"Indice salvato: {INDEX_PATH} ({index.ntotal} vettori, dim {dim})")
    log(f"Metadata salvati: {METADATA_PATH}")
    log("Fatto. Il tool RAG (src/tools/rag_tool.py) legge questi due file.")


if __name__ == "__main__":
    main()
