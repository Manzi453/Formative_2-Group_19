"""Dataset ingestion and the single shared train/validation split. Owner: Engineer 1.

Every model MUST use `make_split` so that all results are comparable.
Data files are not committed (they are git-ignored).
"""
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .utils import PROJECT_ROOT, SEED

DATA_ROOT = PROJECT_ROOT / "data"
DATA_DIR = DATA_ROOT / "gender-based-violence-tweet-classification-challenge"
ZIP_PATH = DATA_ROOT / "gender-based-violence-tweet-classification-challenge.zip"
ID_COL, TEXT_COL, LABEL_COL = "Tweet_ID", "tweet", "type"

_HELP = (
    "Challenge data not found. Download it from the Zindi competition page and either\n"
    f"  - place the zip at {ZIP_PATH}, or\n"
    f"  - extract Train.csv, Test.csv, SampleSubmission.csv into {DATA_DIR}\n"
    "(In Colab: upload the zip into the repo's data/ folder using the Files panel.)"
)


def ensure_data(data_dir: Path = DATA_DIR) -> Path:
    """Return the data directory, extracting the zip if only the zip is present."""
    if (data_dir / "Train.csv").exists():
        return data_dir
    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(data_dir)
        if (data_dir / "Train.csv").exists():
            return data_dir
    raise FileNotFoundError(_HELP)


def load_raw(data_dir: Path = DATA_DIR):
    """Return (train, test, sample_submission) DataFrames exactly as provided."""
    data_dir = ensure_data(data_dir)
    train = pd.read_csv(data_dir / "Train.csv")
    test = pd.read_csv(data_dir / "Test.csv")
    sample = pd.read_csv(data_dir / "SampleSubmission.csv")
    return train, test, sample


def duplicate_group_key(text: str) -> str:
    """Key used to keep duplicate tweets together: lower-cased, whitespace-collapsed text."""
    return " ".join(str(text).lower().split())


def make_split(train: pd.DataFrame, val_size: float = 0.2, seed: int = SEED):
    """Stratified, duplicate-safe train/validation split. Returns (train_df, val_df).

    Tweets with identical normalised text form one group and are never separated, so a
    duplicate cannot leak from train into validation. StratifiedGroupKFold keeps class
    proportions approximately equal; with n_splits = round(1 / val_size) the first fold
    is the validation set.
    """
    n_splits = round(1 / val_size)
    groups = train[TEXT_COL].map(duplicate_group_key)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(train, train[LABEL_COL], groups))
    return train.iloc[train_idx].reset_index(drop=True), train.iloc[val_idx].reset_index(drop=True)
