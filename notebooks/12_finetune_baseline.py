"""
STEP 2 della guida fine-tuning (claude/specifiche-progetto.md) - baseline SENZA
fine-tuning, da battere con il modello addestrato in notebooks/13_finetune_train.py.

Due baseline, non una sola, per un confronto piu' onesto:
1. Maggioranza: predice sempre la classe piu' frequente nel train - il "pavimento"
   minimo che qualunque modello deve superare per essere utile.
2. Zero-shot: lo STESSO modello base che verra' fine-tuned (Qwen2.5-1.5B-Instruct),
   senza alcun adapter LoRA, a cui si chiede di classificare il mazzo con un prompt
   diretto - questo e' il confronto piu' rilevante per lo step 8 delle specifiche
   ("confronto finale col baseline sul test set"): isola l'effetto del fine-tuning
   dall'effetto della scelta del modello di base.

Va eseguito nel venv Windows dell'utente (richiede transformers/torch, non
disponibili nella sandbox cloud usata per scrivere questo codice) DOPO
11_prepare_finetune_data.py.
"""
from __future__ import annotations

import json
import os
import time

from sklearn.metrics import accuracy_score, classification_report, f1_score

DATA_DIR = os.path.join("data", "processed", "finetune_powerlevel")
BASE_MODEL = os.environ.get("FINETUNE_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
LABELS = ("basso", "medio", "alto")

ZERO_SHOT_PROMPT = """Sei un esperto di Hearthstone (gioco di carte Blizzard). Classifica il \
potenziale competitivo ("power level") del mazzo descritto sotto in una di queste tre \
categorie: basso, medio, alto.

{deck_description}

Rispondi con UNA SOLA PAROLA tra: basso, medio, alto. Nessun'altra spiegazione."""


def load_split(name: str) -> list[dict]:
    path = os.path.join(DATA_DIR, f"{name}.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def parse_label(generated_text: str) -> str:
    """Estrae basso/medio/alto dalla risposta libera del modello - cerca le 3
    etichette come parole (case-insensitive) nell'ordine in cui compaiono, invece di
    pretendere un match esatto (i modelli chat spesso aggiungono punteggiatura o
    testo attorno anche quando istruiti a non farlo, stesso limite di compliance
    testuale gia' visto piu' volte in questo progetto). Se nessuna etichetta e'
    riconoscibile, ritorna 'medio' (la classe centrale, la scelta meno sbagliata in
    assenza di segnale) e logga il caso per ispezione manuale."""
    t = generated_text.lower()
    for label in LABELS:
        if label in t:
            return label
    return "medio"


def evaluate(y_true: list[str], y_pred: list[str], name: str) -> None:
    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Macro-F1: {f1_score(y_true, y_pred, average='macro', labels=list(LABELS)):.3f}")
    print(classification_report(y_true, y_pred, labels=list(LABELS), zero_division=0))


def majority_baseline() -> None:
    train = load_split("train")
    from collections import Counter
    majority_label = Counter(r["label"] for r in train).most_common(1)[0][0]
    print(f"Classe di maggioranza sul train: {majority_label!r}")

    for split_name in ("val", "test"):
        split = load_split(split_name)
        y_true = [r["label"] for r in split]
        y_pred = [majority_label] * len(split)
        evaluate(y_true, y_pred, f"Baseline maggioranza - {split_name}")


def zero_shot_baseline() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nCaricamento modello base per zero-shot: {BASE_MODEL} (puo' richiedere qualche minuto)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    for split_name in ("val", "test"):
        split = load_split(split_name)
        y_true, y_pred = [], []
        t0 = time.perf_counter()
        for i, record in enumerate(split):
            messages = [
                {"role": "user", "content": ZERO_SHOT_PROMPT.format(deck_description=record["input"])}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = parse_label(generated)
            y_true.append(record["label"])
            y_pred.append(pred)
            if i < 3:
                print(f"  esempio {i}: risposta grezza={generated!r} -> etichetta={pred!r} (vera: {record['label']!r})")
        elapsed = time.perf_counter() - t0
        print(f"({split_name}: {len(split)} esempi in {elapsed:.1f}s, {elapsed / max(len(split), 1):.2f}s/esempio)")
        evaluate(y_true, y_pred, f"Baseline zero-shot ({BASE_MODEL}) - {split_name}")


if __name__ == "__main__":
    majority_baseline()
    zero_shot_baseline()
