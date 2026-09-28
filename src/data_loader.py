"""Dataset ingestion and the single shared train/validation split. Owner: Engineer 1.

Every model MUST use `make_split` so that all results are comparable.
Data files are not committed - see data/README.md.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import PROJECT_ROOT, SEED

DATA_DIR = PROJECT_ROOT / "data" / "gender-based-violence-tweet-classification-challenge"
ID_COL, TEXT_COL, LABEL_COL = "Tweet_ID", "tweet", "type"


def load_raw(data_dir: Path = DATA_DIR):
    """Return (train, test, sample_submission) DataFrames exactly as provided."""
    train = pd.read_csv(data_dir / "Train.csv")
    test = pd.read_csv(data_dir / "Test.csv")
    sample = pd.read_csv(data_dir / "SampleSubmission.csv")
    return train, test, sample


def make_split(train: pd.DataFrame, val_size: float = 0.2, seed: int = SEED):
    """Stratified train/validation split.

    TODO (Engineer 1): after the duplicate analysis in 01_data_exploration, group
    duplicate/near-duplicate tweets so they cannot appear on both sides of the split,
    and document the final strategy in the report.
    """
    return train_test_split(
        train, test_size=val_size, stratify=train[LABEL_COL], random_state=seed
    )
