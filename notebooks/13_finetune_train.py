"""STEP 6-7 della guida fine-tuning (claude/specifiche-progetto.md): training LoRA
(peft+trl) su Qwen2.5-1.5B-Instruct, parametrizzato via env (LORA_RANK, LEARNING_RATE,
NUM_EPOCHS, LORA_RUN_NAME) per confrontare piu' run sul validation set senza toccare
il codice. Va eseguito nel venv Windows dell'utente dopo 11_prepare_finetune_data.py."""
from __future__ import annotations

import json
import os
import time

DATA_DIR = os.path.join("data", "processed", "finetune_powerlevel")
BASE_MODEL = os.environ.get("FINETUNE_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
LABELS = ("basso", "medio", "alto")

LORA_RANK = int(os.environ.get("LORA_RANK", "8"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
RUN_NAME = os.environ.get("LORA_RUN_NAME", "run")

OUTPUT_DIR = os.path.join(DATA_DIR, RUN_NAME)

TRAIN_PROMPT = """Sei un esperto di Hearthstone (gioco di carte Blizzard). Classifica il \
potenziale competitivo ("power level") del mazzo descritto sotto in una di queste tre \
categorie: basso, medio, alto.

{deck_description}

Rispondi con UNA SOLA PAROLA tra: basso, medio, alto. Nessun'altra spiegazione."""


def load_split(name: str) -> list[dict]:
    path = os.path.join(DATA_DIR, f"{name}.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_chat_text(tokenizer, record: dict, include_answer: bool) -> str:
    """Formatta un esempio come conversazione chat; risposta inclusa solo in training
    (include_answer=True), turno assistente vuoto in inferenza."""
    messages = [{"role": "user", "content": TRAIN_PROMPT.format(deck_description=record["input"])}]
    if include_answer:
        messages.append({"role": "assistant", "content": record["label"]})
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_label(generated_text: str) -> str:
    """Stessa logica di 12_finetune_baseline.py, duplicata deliberatamente (notebook
    autosufficienti)."""
    t = generated_text.lower()
    for label in LABELS:
        if label in t:
            return label
    return "medio"


def main() -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    print(
        f"Configurazione: modello={BASE_MODEL}, rank={LORA_RANK}, lr={LEARNING_RATE}, "
        f"epoche={NUM_EPOCHS}, output={OUTPUT_DIR}"
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_records = load_split("train")
    val_records = load_split("val")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_texts = [to_chat_text(tokenizer, r, include_answer=True) for r in train_records]
    train_dataset = Dataset.from_dict({"text": train_texts})

    # Dataset di valutazione per il Trainer, stesso formato del train (risposta
    # inclusa, serve per una loss confrontabile) - usato SOLO per scegliere il
    # checkpoint migliore (vedi load_best_model_at_end sotto). La valutazione "vera"
    # (accuracy/macro-F1, stessa metrica delle baseline) resta quella fatta a mano
    # piu' sotto, dopo il training.
    val_texts = [to_chat_text(tokenizer, r, include_answer=True) for r in val_records]
    val_dataset = Dataset.from_dict({"text": val_texts})

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        # 'dtype' (non 'torch_dtype', deprecato). bf16 su GPU: piu' stabile di fp16,
        # non serve loss scaling.
    )

    # target_modules q/k/v/o_proj: i moduli di attenzione, miglior rapporto tra
    # parametri allenabili e qualita' per modelli di questa taglia.
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # batch effettivo 8, meno picco memoria/passo
        logging_steps=10,
        save_strategy="steps",
        save_steps=10,  # non a fine epoca: primo salvataggio troppo tardi altrimenti
        save_total_limit=3,  # ultimi 3 checkpoint (load_best_model_at_end protegge il migliore)
        eval_strategy="steps",
        eval_steps=10,  # deve essere multiplo di save_steps per load_best_model_at_end
        load_best_model_at_end=True,  # evita overfitting sull'ultima epoca, ripristina
        # il checkpoint con eval_loss piu' bassa invece dell'ultimo.
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],  # niente wandb - LangSmith copre l'agente, non il training
        dataset_text_field="text",
        max_length=512,  # descrizioni brevi, 512 token bastano (param 'max_length',
        # non 'max_seq_length', rimosso in questa versione di trl)
        bf16=torch.cuda.is_available(),  # default True in SFTConfig, va disattivato su CPU
        use_cpu=not torch.cuda.is_available(),
        gradient_checkpointing=False,  # con PEFT/LoRA senza enable_input_require_grads()
        # causa un errore in backward (bug noto) - disattivato, dataset piccolo.
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Ripresa automatica da un checkpoint precedente se una run e' stata interrotta.
    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
    if last_checkpoint:
        print(f"Trovato un checkpoint precedente in {last_checkpoint!r} - riprendo il training da li' invece di ripartire da zero.")
    else:
        print("Nessun checkpoint precedente trovato - training da zero.")

    t0 = time.perf_counter()
    trainer.train(resume_from_checkpoint=last_checkpoint)
    train_elapsed = time.perf_counter() - t0
    print(f"Training completato in {train_elapsed:.1f}s.")
    print(
        f"Checkpoint migliore selezionato (eval_loss piu' bassa): "
        f"{trainer.state.best_model_checkpoint!r} (eval_loss={trainer.state.best_metric:.4f}) "
        "- i pesi in memoria sono gia' stati ripristinati a questo checkpoint "
        "(load_best_model_at_end=True), la valutazione sotto usa questi pesi."
    )

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Valutazione sul validation set, stesso schema di 12_finetune_baseline.py.
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_true, y_pred = [], []
    for record in val_records:
        prompt = to_chat_text(tokenizer, record, include_answer=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        y_true.append(record["label"])
        y_pred.append(parse_label(generated))

    val_accuracy = accuracy_score(y_true, y_pred)
    val_macro_f1 = f1_score(y_true, y_pred, average="macro", labels=list(LABELS))
    print(f"Validation - accuracy: {val_accuracy:.3f}, macro-F1: {val_macro_f1:.3f}")

    metrics = {
        "run_name": RUN_NAME,
        "base_model": BASE_MODEL,
        "lora_rank": LORA_RANK,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "train_elapsed_s": round(train_elapsed, 1),
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_checkpoint_eval_loss": trainer.state.best_metric,
    }
    with open(os.path.join(OUTPUT_DIR, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metriche salvate in {os.path.join(OUTPUT_DIR, 'val_metrics.json')}")


if __name__ == "__main__":
    main()
