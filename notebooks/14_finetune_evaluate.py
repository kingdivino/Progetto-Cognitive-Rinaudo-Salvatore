"""
STEP 8-9 della guida fine-tuning (claude/specifiche-progetto.md) - da lanciare UNA
SOLA VOLTA, dopo aver scelto la configurazione migliore sul validation set con
13b_compare_runs.py (guardare il test set piu' di una volta per scegliere tra
configurazioni sarebbe la stessa fuga di informazione che il test set esiste apposta
per evitare).

STEP 8 - Confronto finale: il modello fine-tuned scelto (LORA_RUN_NAME) viene
valutato sul test set e confrontato con le due baseline di 12_finetune_baseline.py
(maggioranza e zero-shot) sullo STESSO test set.

STEP 9 - Analisi dei risultati: oltre alle metriche aggregate, questo script salva
ogni predizione sbagliata con il winrate reale e il flag has_special_deckbuild (letto
da finetune_dataset.csv via deck_id) - per rispondere concretamente a "il modello
sbaglia sistematicamente su qualche categoria di mazzi?" invece di fermarsi al numero
di accuracy da solo, come richiesto dalla specifica ("successi ed errori, limiti del
modello").

Imposta LORA_RUN_NAME alla configurazione scelta (stessa variabile di
13_finetune_train.py) prima di lanciare.
"""
from __future__ import annotations

import json
import os

import pandas as pd

DATA_DIR = os.path.join("data", "processed", "finetune_powerlevel")
PROCESSED_PATH = os.path.join("data", "processed", "finetune_dataset.csv")
BASE_MODEL = os.environ.get("FINETUNE_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
RUN_NAME = os.environ.get("LORA_RUN_NAME", "run")
RUN_DIR = os.path.join(DATA_DIR, RUN_NAME)
LABELS = ("basso", "medio", "alto")

TRAIN_PROMPT = """Sei un esperto di Hearthstone (gioco di carte Blizzard). Classifica il \
potenziale competitivo ("power level") del mazzo descritto sotto in una di queste tre \
categorie: basso, medio, alto.

{deck_description}

Rispondi con UNA SOLA PAROLA tra: basso, medio, alto. Nessun'altra spiegazione."""


def load_split(name: str) -> list[dict]:
    path = os.path.join(DATA_DIR, f"{name}.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def parse_label(generated_text: str) -> str:
    t = generated_text.lower()
    for label in LABELS:
        if label in t:
            return label
    return "medio"


def main() -> None:
    import torch
    from peft import PeftModel
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from transformers import AutoModelForCausalLM, AutoTokenizer

    test_records = load_split("test")

    print(f"Caricamento modello base {BASE_MODEL} + adapter LoRA da {RUN_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(RUN_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model = PeftModel.from_pretrained(base_model, RUN_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    errors = []
    for record in test_records:
        messages = [{"role": "user", "content": TRAIN_PROMPT.format(deck_description=record["input"])}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_label(generated)
        y_true.append(record["label"])
        y_pred.append(pred)
        if pred != record["label"]:
            errors.append({
                "deck_id": record["deck_id"],
                "vero": record["label"],
                "predetto": pred,
                "winrate_reale": record["winrate"],
                "risposta_grezza": generated,
            })

    print("\n=== Modello fine-tuned - TEST SET (unica valutazione, non ripetere) ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Macro-F1: {f1_score(y_true, y_pred, average='macro', labels=list(LABELS)):.3f}")
    print(classification_report(y_true, y_pred, labels=list(LABELS), zero_division=0))
    print("Matrice di confusione (righe=vero, colonne=predetto, ordine basso/medio/alto):")
    print(confusion_matrix(y_true, y_pred, labels=list(LABELS)))

    # Arricchimento degli errori con has_special_deckbuild - per rispondere a "il
    # modello sbaglia di piu' sui mazzi con regola di costruzione non standard?"
    full_df = pd.read_csv(PROCESSED_PATH)
    special_by_id = dict(zip(full_df["deck_id"], full_df["has_special_deckbuild"]))
    for e in errors:
        e["has_special_deckbuild"] = bool(special_by_id.get(e["deck_id"], False))

    n_special_in_test = sum(1 for r in test_records if special_by_id.get(r["deck_id"], False))
    n_special_in_errors = sum(1 for e in errors if e["has_special_deckbuild"])
    print(
        f"\nErrori totali: {len(errors)}/{len(test_records)}. "
        f"Di cui su mazzi has_special_deckbuild=True: {n_special_in_errors} "
        f"(su {n_special_in_test} presenti nel test set) - confrontare questa "
        "proporzione con quella nel test set nel suo complesso per capire se il "
        "modello fatica di piu' su questa categoria (piu' rara nel dataset)."
    )

    errors_path = os.path.join(RUN_DIR, "test_errors.json")
    with open(errors_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    print(f"Dettaglio di tutti gli errori (con winrate reale) salvato in {errors_path} "
          "- da guardare a mano per lo step 9 (analisi qualitativa).")


if __name__ == "__main__":
    main()
