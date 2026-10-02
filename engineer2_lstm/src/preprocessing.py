"""Shared text preprocessing. Owner: Engineer 1.

All models call these functions rather than writing their own cleaning code.
Strategies map to the preprocessing experiments (A-D) logged in
results/metrics/preprocessing_experiments.csv. Nothing here uses labels.
"""
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")  # 3+ of the same char -> 2
WHITESPACE_RE = re.compile(r"\s+")


def minimal(text: str) -> str:
    """A: only strip surrounding whitespace. Case, punctuation, emojis kept."""
    return str(text).strip()


def remove_urls_mentions(text: str) -> str:
    """B: replace URLs and @mentions with placeholder tokens."""
    text = URL_RE.sub(" <url> ", str(text))
    text = MENTION_RE.sub(" <user> ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_repeats(text: str) -> str:
    """C: collapse repeated whitespace and characters repeated 3+ times."""
    text = REPEAT_CHAR_RE.sub(r"\1\1", str(text))
    return WHITESPACE_RE.sub(" ", text).strip()


def handle_hashtags(text: str) -> str:
    """D: '#word' -> 'word' (keeps the word, drops the '#')."""
    return HASHTAG_RE.sub(r"\1", str(text))


STRATEGIES = {
    "A_minimal": minimal,
    "B_no_url_mention": lambda t: remove_urls_mentions(t),
    "C_normalize_repeats": lambda t: normalize_repeats(t),
    "D_hashtag_words": lambda t: handle_hashtags(t),
}


def preprocess(text: str, strategy: str = "A_minimal") -> str:
    return STRATEGIES[strategy](text)
