"""Costruzione centralizzata del client ChatOllama usato da tutti i nodi del grafo.
Variabili d'ambiente lette (tutte opzionali, vedi .env.example): OLLAMA_MODEL,
OLLAMA_BASE_URL, OLLAMA_NUM_GPU, OLLAMA_REASONING."""
from __future__ import annotations

import os

from langchain_ollama import ChatOllama


def build_llm(temperature: float) -> ChatOllama:
    """Costruisce un ChatOllama con le variabili d'ambiente correnti;
    `temperature` resta esplicito perche' dipende dal compito del nodo chiamante."""
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
