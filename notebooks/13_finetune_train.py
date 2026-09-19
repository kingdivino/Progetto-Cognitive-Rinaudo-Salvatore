"""
STEP 6-7 della guida fine-tuning (claude/specifiche-progetto.md).

STEP 6 - Tecnica: LoRA (libreria peft + trl) su un modello base piccolo
(Qwen2.5-1.5B-Instruct di default, stesso modello/template gia' usato nel progetto di
riferimento GymAssistant per il loro componente fine-tuned - vedi guida di progetto).
Motivazione: full fine-tuning di un LLM, anche piccolo, non e' realistico sull'hardware
disponibile (laptop con 4GB di VRAM dedicata, vedi nota hardware nella guida di
progetto - lo stesso vincolo che ha guidato la scelta di modelli Ollama piccoli per il
resto della pipeline); LoRA alza solo un numero ridotto di parametri allenabili, quindi
resta fattibile anche con risorse limitate e con un dataset piccolo (~400 esempi di
train).

STEP 7 - Model selection: questo script e' parametrizzato via variabili d'ambiente
(stesso principio gia' consolidato nel resto del progetto per MAX_POSTS_TO_PROCESS/
OLLAMA_NUM_GPU/OLLAMA_REASONING - un valore diverso per run diverso, senza toccare il
codice) cosi' da poter lanciare piu' configurazioni e confrontarle sul validation set
SENZA modificare lo script:
- LORA_RANK (default 8): rank della decomposizione LoRA.
- LEARNING_RATE (default 2e-4): learning rate.
- NUM_EPOCHS (default 3): epoche di training.
- LORA_RUN_NAME (default "run"): nome della cartella di output, per non sovrascrivere
  un run precedente quando se ne lancia un secondo per il confronto.

Ogni run scrive le sue metriche di validation in
data/processed/finetune_powerlevel/<LORA_RUN_NAME>/val_metrics.json, cosi' i vari run
si possono confrontare senza doverli tenere tutti in memoria/RAM del terminale.

Va eseguito nel venv Windows dell'utente (richiede torch/transformers/peft/trl, non
disponibili nella sandbox cloud usata per scrivere questo codice) DOPO
11_prepare_finetune_data.py.
"""
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
    """Formatta un esempio come conversazione chat - durante il training la risposta
    (l'etichetta vera) e' inclusa nel testo (il modello impara a generarla), durante
    l'inferenza no (add_generation_prompt=True lascia il turno dell'assistente vuoto,
    pronto per essere generato)."""
    messages = [{"role": "user", "content": TRAIN_PROMPT.format(deck_description=record["input"])}]
    if include_answer:
        messages.append({"role": "assistant", "content": record["label"]})
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_label(generated_text: str) -> str:
    """Stessa logica di 12_finetune_baseline.py - duplicata deliberatamente (file
    indipendenti, eseguiti in momenti diversi) invece di un import incrociato tra
    notebook, per restare coerenti con lo stile gia' usato per gli altri notebook di
    questo progetto (ognuno autosufficiente)."""
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

    # Dataset di valutazione per il Trainer (17/09/2026) - stesso formato del train
    # (risposta inclusa nel testo, serve per calcolare una loss confrontabile),
    # usato SOLO per scegliere il checkpoint migliore durante il training (vedi
    # load_best_model_at_end sotto) - la valutazione "vera" (accuracy/macro-F1 per
    # generazione autoregressiva, la stessa metrica delle baseline) resta quella
    # fatta a mano piu' sotto, dopo il training.
    val_texts = [to_chat_text(tokenizer, r, include_answer=True) for r in val_records]
    val_dataset = Dataset.from_dict({"text": val_texts})

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        # 'dtype' non 'torch_dtype' - rinominato nella versione di transformers
        # installata il 17/09/2026 (5.16.1); 'torch_dtype' resta accettato ma da'
        # un warning di deprecazione ad ogni run. bfloat16 (non float16) sulla GPU:
        # verificato il 17/09/2026 che l'hardware dell'utente (RTX 3050 Ti, Ampere)
        # supporta bf16 nativamente - piu' stabile di fp16 per il training (stesso
        # range di esponente di fp32, non serve loss scaling per evitare NaN).
    )

    # target_modules q/k/v/o_proj: stessa scelta gia' documentata nel template
    # GymAssistant (vedi guida di progetto) - i moduli di attenzione sono dove LoRA
    # da' tipicamente il miglior rapporto tra parametri allenabili e qualita' per
    # modelli di questa taglia.
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
        gradient_accumulation_steps=8,  # stesso batch effettivo di prima (8), ma
        # meno picco di memoria per singolo passo - importante passando a una GPU
        # con solo 4GB di VRAM dedicata (rischio di out-of-memory con batch_size=4).
        logging_steps=10,
        save_strategy="steps",
        save_steps=10,  # checkpoint ogni 10 step (non a fine epoca) - 17/09/2026:
        # con "epoch" il primo salvataggio arriva solo a fine prima epoca (~15-20
        # minuti su questo hardware), troppo tardi se la sessione va interrotta
        # prima (es. utente che deve chiudere il PC). Con step=10 il primo
        # checkpoint utile arriva in pochi minuti.
        save_total_limit=3,  # tiene solo gli ultimi 3 checkpoint - un adapter LoRA
        # e' piccolo (pochi MB), ma senza limite si accumulerebbero comunque tanti
        # checkpoint su una run lunga. NOTA: con load_best_model_at_end=True sotto,
        # HF Trainer protegge sempre il checkpoint migliore dalla cancellazione
        # anche se non e' tra gli ultimi 3 per ordine cronologico.
        eval_strategy="steps",
        eval_steps=10,  # stesso intervallo di save_steps - obbligatorio per
        # load_best_model_at_end (deve essere un multiplo esatto).
        load_best_model_at_end=True,  # 17/09/2026, aggiunto dopo aver osservato un
        # overfitting netto nel primo run (accuracy sul train 93%, sul validation
        # solo 43%): invece di tenere i pesi dell'ULTIMA epoca (i piu' overfittati),
        # il Trainer tiene traccia della eval_loss ad ogni valutazione e alla fine
        # ripristina automaticamente i pesi del checkpoint con la eval_loss piu'
        # bassa - una forma di early stopping "morbido", che permette di allenare
        # per piu' epoche senza il rischio di consegnare il checkpoint peggiore.
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],  # niente wandb/altri tracker - LangSmith copre l'agente, non
        # il training, che qui logga solo su stdout/file locali
        dataset_text_field="text",
        max_length=512,  # le descrizioni di mazzo sono brevi, 512 token bastano
        # abbondantemente e tengono il training leggero sull'hardware limitato -
        # nome del parametro 'max_length' (non 'max_seq_length', rimosso nella
        # versione di trl installata il 17/09/2026, trl 1.13.0 - verificato
        # leggendo direttamente trl/trainer/sft_config.py sul venv dell'utente,
        # dato che la libreria ha binari compilati per Windows non eseguibili
        # nella sandbox Linux usata per scrivere/verificare questo codice)
        bf16=torch.cuda.is_available(),  # SFTConfig di default mette bf16=True
        # (vedi commento nella sua docstring) - richiede una GPU con supporto bf16
        # (Ampere+). Su CPU (nessuna GPU rilevata) va disattivato esplicitamente,
        # altrimenti il training si rifiuta di partire ("Your setup doesn't
        # support bf16/gpu..."); su GPU Ampere (verificato: RTX 3050 Ti) invece va
        # tenuto attivo, coerente con il dtype bf16 del modello caricato sopra.
        use_cpu=not torch.cuda.is_available(),  # esplicito invece di lasciare
        # l'auto-detection - coerente con la scelta di dtype del modello sopra.
        gradient_checkpointing=False,  # SFTConfig lo attiva di default (True), ma
        # con un modello PEFT/LoRA senza model.enable_input_require_grads() causa
        # tipicamente un errore "element 0 of tensors does not require grad" in
        # backward - problema noto della combinazione PEFT+gradient_checkpointing.
        # Disattivato: il dataset e' piccolo (~400 esempi) e non serve risparmiare
        # memoria GPU (CPU-only, nessuna VRAM in gioco).
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Ripresa automatica da un checkpoint precedente (17/09/2026) - se una run
    # precedente e' stata interrotta a meta' (es. sessione chiusa per mancanza di
    # tempo), riparte da li' invece che da zero. Cerca l'ultimo checkpoint in
    # OUTPUT_DIR; se non ce n'e' nessuno (prima volta che si lancia questo
    # LORA_RUN_NAME), parte normalmente da zero.
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

    # Valutazione sul validation set - stesso schema di 12_finetune_baseline.py, per
    # essere direttamente confrontabile con le due baseline.
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
