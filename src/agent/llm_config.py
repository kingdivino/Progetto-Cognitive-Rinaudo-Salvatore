"""
Costruzione centralizzata del ChatOllama usato da tutti i nodi del grafo (Planner,
Research/ReAct, e i futuri Format/Draft) - prima ogni nodo duplicava la stessa logica
di lettura delle variabili d'ambiente, ora e' in un solo posto.

Variabili d'ambiente lette (tutte opzionali, vedi .env.example per i dettagli):
- OLLAMA_MODEL, OLLAMA_BASE_URL: quale modello/endpoint usare.
- OLLAMA_NUM_GPU: quanti layer del modello forzare sulla GPU (nome fuorviante di
  Ollama: NON e' il numero di GPU). Da tarare in base alla VRAM dedicata reale
  (osservato: 40 su un modello 14B satura 4GB di VRAM e crasha il driver CUDA - 20
  funziona, vedi guida di progetto). Lasciato non impostato di default: Ollama stima
  da solo, in modo prudente.
- OLLAMA_REASONING: alcuni modelli (es. tutta la famiglia Qwen3) fanno "thinking"
  interno esteso di default prima di rispondere, il che aumenta molto la latenza
  indipendentemente dalla dimensione del modello (osservato: qwen3:14b >10 min anche
  su prompt semplici). Impostare a "false" per disattivarlo (parametro nativo di
  langchain-ollama, ChatOllama(reasoning=False) - mappa al parametro "think" dell'API
  Ollama). Lasciato non impostato di default (nessun comportamento forzato, ogni
  modello usa il proprio default).
"""
from __future__ import annotations

import os

from langchain_ollama import ChatOllama


def build_llm(temperature: float) -> ChatOllama:
    """Costruisce un ChatOllama con le variabili d'ambiente correnti. `temperature`
    resta un parametro esplicito (non da env) perche' il valore giusto dipende dal
    compito del nodo che chiama (es. planning piu' creativo vs. research piu'
    conservativo), non e' una preferenza globale dell'hardware/modello."""
    kwargs = dict(
        model=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
    )

    num_gpu_override = os.environ.get("OLLAMA_NUM_GPU")
    if num_gpu_override:
        kwargs["num_gpu"] = int(num_gpu_override)

    reasoning_override = os.environ.get("OLLAMA_REASONING")
    if reasoning_override is not None:
        kwargs["reasoning"] = reasoning_override.strip().lower() not in ("false", "0", "no")

    return ChatOllama(**kwargs)
