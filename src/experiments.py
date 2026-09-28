"""Experiment tracking. Append one row per run to results/metrics/experiments.csv.

Never delete failed or weak experiments - they are evidence for the discussion.
"""
import csv
from datetime import date

from .utils import RESULTS_DIR

EXPERIMENTS_CSV = RESULTS_DIR / "metrics" / "experiments.csv"
COLUMNS = [
    "experiment_id", "date", "model", "preprocessing", "tokenizer", "sequence_length",
    "embedding", "learning_rate", "batch_size", "epochs", "optimizer", "dropout",
    "train_time", "validation_accuracy", "macro_precision", "macro_recall",
    "macro_f1", "weighted_f1", "notes", "next_action",
]


def log_experiment(**fields) -> None:
    """Append a run. Unspecified columns are written as 'NOT YET MEASURED'."""
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise ValueError(f"Unknown experiment columns: {sorted(unknown)}")
    row = {c: fields.get(c, "NOT YET MEASURED") for c in COLUMNS}
    row["date"] = fields.get("date", date.today().isoformat())
    write_header = not EXPERIMENTS_CSV.exists() or EXPERIMENTS_CSV.stat().st_size == 0
    with open(EXPERIMENTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
