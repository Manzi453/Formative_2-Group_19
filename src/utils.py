"""Small shared helpers (seeding, paths). Owner: Engineer 1."""
import os
import random
from pathlib import Path

SEED = 42  # single project-wide seed; do not change per model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and (if installed) torch / tensorflow."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
