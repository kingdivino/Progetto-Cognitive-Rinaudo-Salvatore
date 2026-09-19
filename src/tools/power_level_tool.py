"""Tool basato sul modello fine-tuned: carica il modello base + l'adapter LoRA
addestrato e stima il power level (basso/medio/alto) di un mazzo."""
from __future__ import annotations

import os

from langchain_core.tools import tool

BASE_MODEL = os.environ.get("FINETUNE_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
RUN_NAME = os.environ.get("LORA_RUN_NAME", "run")
RUN_DIR = os.path.join("data", "processed", "finetune_powerlevel", RUN_NAME)
LABELS = ("basso", "medio", "alto")

# Stesso identico prompt di notebooks/13_finetune_train.py - deve restare identico
# a quello usato in training.
_PROMPT = """Sei un esperto di Hearthstone (gioco di carte Blizzard). Classifica il \
potenziale competitivo ("power level") del mazzo descritto sotto in una di queste tre \
categorie: basso, medio, alto.

{deck_description}

Rispondi con UNA SOLA PAROLA tra: basso, medio, alto. Nessun'altra spiegazione."""

# Lazy singleton (come l'indice RAG e il driver Neo4j): il modello si carica una
# sola volta.
_model = None
_tokenizer = None
_device = None


def _load_model():
    global _model, _tokenizer, _device
    if _model is not None:
        return _model, _tokenizer, _device

    if not os.path.exists(RUN_DIR):
        raise RuntimeError(
            f"Adapter LoRA non trovato in {RUN_DIR!r} - lanciare prima "
            "notebooks/11_prepare_finetune_data.py e notebooks/13_finetune_train.py "
            "(scegliendo la configurazione migliore con notebooks/13b_compare_runs.py "
            "e impostando LORA_RUN_NAME di conseguenza)."
        )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(RUN_DIR)
    # Forziamo la CPU: questo tool gira mentre Ollama potrebbe gia' usare la GPU
    # (solo 4GB VRAM) - condividerla rischierebbe un CUDA OOM.
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    _model = PeftModel.from_pretrained(base_model, RUN_DIR)
    _device = "cpu"
    _model.to(_device)
    _model.eval()
    return _model, _tokenizer, _device


def _parse_label(generated_text: str) -> str:
    """Stessa logica di parsing usata nei notebook di training/valutazione,
    duplicata deliberatamente per non far dipendere i tool di produzione dal codice
    sperimentale di training."""
    t = generated_text.lower()
    for label in LABELS:
        if label in t:
            return label
    return "medio"


@tool
def assess_deck_power_level(deck_description: str, justification: str) -> str:
    """Stima il "power level" (basso/medio/alto) di un mazzo Hearthstone usando il
    modello fine-tuned (LoRA su un piccolo LLM, addestrato su mazzi reali con
    winrate/composizione noti). Usa questo tool quando il post che stai scrivendo
    (how-to/review) deve valutare quanto un mazzo sia forte/interessante, come segnale
    ULTERIORE e indipendente da RAG/search - non sostituisce una fonte reale per fatti
    specifici sulle carte (il testo/le regole di una carta vanno comunque verificate
    con search_card_knowledge), e' una valutazione del modello sulla composizione
    complessiva del mazzo.

    Args:
        deck_description: descrizione testuale della composizione del mazzo (classe,
            formato, curva di mana, rarita', meccaniche - NON includere winrate/
            partite, che il modello non ha mai visto come input in training e
            userebbe come scorciatoia invece di valutare davvero la composizione).
        justification: perche' serve questa valutazione ora - quale parte del post
            deve supportare. Obbligatoria, come per gli altri tool di questo progetto.
    """
    try:
        model, tokenizer, device = _load_model()
    except Exception as e:
        return f"[ERROR] {e}"

    try:
        import torch

        messages = [{"role": "user", "content": _PROMPT.format(deck_description=deck_description)}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    except Exception as e:
        return f"[ERROR] esecuzione del modello fine-tuned fallita: {e}"

    label = _parse_label(generated)
    return (
        f"Power level stimato dal modello fine-tuned (LoRA su {BASE_MODEL}): {label} "
        f"(giustificazione: {justification})"
    )
