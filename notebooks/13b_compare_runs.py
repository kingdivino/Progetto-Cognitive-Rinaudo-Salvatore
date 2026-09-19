"""
STEP 7 della guida fine-tuning - riepilogo dei run di 13_finetune_train.py per il
model selection (confronto SUL VALIDATION SET, mai sul test, per non "sbirciare" il
test set prima del confronto finale dello step 8). Nessuna dipendenza pesante (solo
lettura di file JSON gia' scritti da run precedenti) - eseguibile ovunque, anche nella
sandbox cloud, per controllare rapidamente l'andamento senza dover rileggere ogni
singolo val_metrics.json a mano.

Uso: lanciare 13_finetune_train.py piu' volte con LORA_RUN_NAME/LORA_RANK/
LEARNING_RATE/NUM_EPOCHS diversi (es. rank8_lr2e4, rank16_lr2e4, rank8_lr5e5), poi
lanciare questo script per vedere quale configurazione vince sul validation set prima
di scegliere quella da valutare (una volta sola) sul test set in 14_finetune_evaluate.py.
"""
from __future__ import annotations

import glob
import json
import os

DATA_DIR = os.path.join("data", "processed", "finetune_powerlevel")


def main() -> None:
    pattern = os.path.join(DATA_DIR, "*", "val_metrics.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(
            f"Nessun run trovato in {pattern} - lanciare prima "
            "13_finetune_train.py (anche piu' volte, con LORA_RUN_NAME diversi)."
        )
        return

    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))

    rows.sort(key=lambda r: r["val_macro_f1"], reverse=True)

    header = f"{'run':<20}{'rank':>6}{'lr':>10}{'epoche':>8}{'accuracy':>10}{'macro-F1':>10}{'tempo(s)':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['run_name']:<20}{r['lora_rank']:>6}{r['learning_rate']:>10.0e}"
            f"{r['num_epochs']:>8}{r['val_accuracy']:>10.3f}{r['val_macro_f1']:>10.3f}"
            f"{r['train_elapsed_s']:>10.1f}"
        )

    best = rows[0]
    print(f"\nMigliore per macro-F1 su validation: {best['run_name']!r} "
          f"(rank={best['lora_rank']}, lr={best['learning_rate']}, "
          f"epoche={best['num_epochs']}) - macro-F1={best['val_macro_f1']:.3f}")
    print(
        "Usare questo LORA_RUN_NAME in 14_finetune_evaluate.py per il confronto "
        "finale (una sola volta) sul test set."
    )


if __name__ == "__main__":
    main()
