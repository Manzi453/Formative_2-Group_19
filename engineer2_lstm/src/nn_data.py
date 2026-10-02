"""Vocab + encoding shared by the neural models (LSTM, CNN, TCN, Transformer).
Built from training text only. Reuses the tokenizer from eda.py.
"""
from collections import Counter

import numpy as np

from .eda import tokenize

PAD, UNK = "<pad>", "<unk>"
PAD_IDX, UNK_IDX = 0, 1

DEFAULT_MAX_LEN = 64  # covers ~p99 tweet length (notebook 01)


def build_vocab(texts, min_freq: int = 2) -> dict:
    """Token -> id, built from `texts` only. Index 0/1 are reserved for PAD/UNK."""
    counts = Counter(tok for t in texts for tok in tokenize(t))
    vocab = {PAD: PAD_IDX, UNK: UNK_IDX}
    for tok, c in counts.most_common():
        if c < min_freq:
            continue
        vocab[tok] = len(vocab)
    return vocab


def encode(text: str, vocab: dict, max_len: int = DEFAULT_MAX_LEN) -> np.ndarray:
    """Tweet -> fixed-length array of vocabulary ids (truncated/right-padded)."""
    ids = [vocab.get(tok, UNK_IDX) for tok in tokenize(text)][:max_len]
    ids.extend([PAD_IDX] * (max_len - len(ids)))
    return np.asarray(ids, dtype=np.int64)


def encode_batch(texts, vocab: dict, max_len: int = DEFAULT_MAX_LEN) -> np.ndarray:
    return np.stack([encode(t, vocab, max_len) for t in texts])


def encode_labels(labels, classes: list) -> np.ndarray:
    """Class name -> integer id, using the fixed order in `classes`."""
    index = {c: i for i, c in enumerate(classes)}
    return np.asarray([index[label] for label in labels], dtype=np.int64)
