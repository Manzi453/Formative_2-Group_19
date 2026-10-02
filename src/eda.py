"""Exploratory-analysis helpers used by notebooks/01_data_exploration.ipynb. Owner: Engineer 1.

Everything here is descriptive: it only reads text/labels to summarise the data and
never produces features for a model. All numbers are computed, not assumed.
"""
import re
from collections import Counter

import numpy as np
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
NUMBER_RE = re.compile(r"\d+")
REPEAT_RE = re.compile(r"(.)\1{2,}")
PUNCT_RE = re.compile(r"[^\w\s]")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")  # applied to lower-cased text
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U0001F900-\U0001F9FF]"
)
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")
HTML_ENTITY_RE = re.compile(r"&(?:amp|lt|gt|quot|#\d+);")
TYPOGRAPHIC_QUOTE_RE = re.compile("[\u2018\u2019\u201c\u201d]")
RETWEET_RE = re.compile(r"\bRT\b")

# Small hand-written lexicon of frequent Swahili / Sheng function words. It is an
# INDICATIVE probe for code-switching, not a language identifier: short words such as
# "na" or "ya" can also be English typos or names. Any conclusion must say so.
SWAHILI_PROBE = frozenset(
    "na ya wa kwa ni sana hii hiyo yake yao kuwa nini gani sasa bado kwani lakini "
    "pia tu mimi wewe yeye sisi nyinyi wao huyo hapa pale wapi mbona vipi poa".split()
)


def tokenize(text: str) -> list:
    """Simple lower-cased alphanumeric tokenizer used only for descriptive statistics."""
    return TOKEN_RE.findall(str(text).lower().replace("\u2019", "'"))


def numeric_summary(s: pd.Series, percentiles=(5, 25, 50, 75, 90, 95, 99)) -> dict:
    out = {"min": s.min(), "max": s.max(), "mean": s.mean(), "median": s.median(), "std": s.std()}
    out.update({f"p{p}": float(np.percentile(s, p)) for p in percentiles})
    return out


def dataset_overview(train: pd.DataFrame, test: pd.DataFrame, text_col, label_col, id_col) -> pd.DataFrame:
    rows = {
        "rows": (len(train), len(test)),
        "columns": (", ".join(train.columns), ", ".join(test.columns)),
        "missing values (total)": (int(train.isna().sum().sum()), int(test.isna().sum().sum())),
        "empty tweets": (int((train[text_col].str.strip() == "").sum()), int((test[text_col].str.strip() == "").sum())),
        "unique IDs": (train[id_col].nunique(), test[id_col].nunique()),
        "IDs unique?": (bool(train[id_col].is_unique), bool(test[id_col].is_unique)),
        "duplicate tweets (rows beyond first)": (int(train[text_col].duplicated().sum()), int(test[text_col].duplicated().sum())),
        "unique classes": (train[label_col].nunique(), "n/a"),
    }
    return pd.DataFrame(rows, index=["train", "test"]).T


def class_distribution(train: pd.DataFrame, label_col) -> pd.DataFrame:
    counts = train[label_col].value_counts()
    df = pd.DataFrame({"count": counts, "percent": (100 * counts / counts.sum()).round(2)})
    df["imbalance_vs_largest"] = (counts.max() / counts).round(1)
    return df


def duplicate_report(train: pd.DataFrame, test: pd.DataFrame, text_col, label_col, key) -> dict:
    """Duplicate analysis on normalised text, incl. conflicting labels and train/test overlap."""
    k_train = train[text_col].map(key)
    k_test = test[text_col].map(key)
    dup_groups = train.groupby(k_train)[label_col].agg(["size", "nunique"])
    dup_groups = dup_groups[dup_groups["size"] > 1]
    return {
        "train rows in duplicate groups": int(dup_groups["size"].sum()),
        "duplicate groups": len(dup_groups),
        "groups with conflicting labels": int((dup_groups["nunique"] > 1).sum()),
        "test tweets also present in train": int(k_test.isin(set(k_train)).sum()),
        "duplicate tweets within test (beyond first)": int(k_test.duplicated().sum()),
    }


def length_features(df: pd.DataFrame, text_col) -> pd.DataFrame:
    text = df[text_col].astype(str)
    return pd.DataFrame({
        "chars": text.str.len(),
        "words": text.str.split().str.len(),
        "tokens": text.map(lambda t: len(tokenize(t))),
    }, index=df.index)


def noise_features(df: pd.DataFrame, text_col) -> pd.DataFrame:
    """Per-tweet counts of social-media noise phenomena."""
    text = df[text_col].astype(str)
    lower = text.str.lower()
    return pd.DataFrame({
        "urls": text.map(lambda t: len(URL_RE.findall(t))),
        "mentions": text.map(lambda t: len(MENTION_RE.findall(t))),
        "hashtags": text.map(lambda t: len(HASHTAG_RE.findall(t))),
        "emojis": text.map(lambda t: len(EMOJI_RE.findall(t))),
        "numbers": text.map(lambda t: len(NUMBER_RE.findall(t))),
        "repeated_chars": text.map(lambda t: len(REPEAT_RE.findall(t))),
        "punctuation": text.map(lambda t: len(PUNCT_RE.findall(t))),
        "html_entities": text.map(lambda t: len(HTML_ENTITY_RE.findall(t))),
        "typographic_quotes": text.map(lambda t: len(TYPOGRAPHIC_QUOTE_RE.findall(t))),
        "retweet_marker": text.map(lambda t: len(RETWEET_RE.findall(t))),
        "non_ascii": text.map(lambda t: len(NON_ASCII_RE.findall(t))),
        "all_caps_words": text.map(lambda t: sum(w.isupper() and len(w) > 1 for w in t.split())),
        "swahili_probe_words": lower.map(lambda t: sum(w in SWAHILI_PROBE for w in tokenize(t))),
    }, index=df.index)


def noise_prevalence(feats: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Percentage of tweets containing >= 1 occurrence of each feature, overall and per class."""
    present = (feats > 0)
    out = present.groupby(labels.values).mean().mul(100).round(2).T
    out.insert(0, "all", (present.mean() * 100).round(2))
    return out


def non_ascii_breakdown(texts, top_n=12) -> pd.DataFrame:
    """Most frequent non-ASCII characters (with Unicode names) - characters only, no tweet text."""
    import unicodedata
    c = Counter(ch for t in texts for ch in str(t) if ord(ch) > 127)
    return pd.DataFrame(
        [{"char": repr(ch), "codepoint": f"U+{ord(ch):04X}", "name": unicodedata.name(ch, "?"), "count": n}
         for ch, n in c.most_common(top_n)]
    ).assign(distinct_non_ascii_chars=len(c))


def token_counts(texts) -> Counter:
    c = Counter()
    for t in texts:
        c.update(tokenize(t))
    return c


def vocab_stats(train_counts: Counter, test_counts: Counter) -> dict:
    total = sum(train_counts.values())
    hapax = sum(1 for v in train_counts.values() if v == 1)
    test_total = sum(test_counts.values())
    oov = sum(v for w, v in test_counts.items() if w not in train_counts)
    return {
        "train tokens": total,
        "train vocabulary size": len(train_counts),
        "hapax (freq = 1)": hapax,
        "hapax % of vocabulary": round(100 * hapax / len(train_counts), 2),
        "words with freq >= 5": sum(1 for v in train_counts.values() if v >= 5),
        "test vocabulary size": len(test_counts),
        "test OOV token rate % (vs train vocab)": round(100 * oov / test_total, 2),
        "test OOV types % (vs train vocab)": round(100 * sum(1 for w in test_counts if w not in train_counts) / len(test_counts), 2),
    }


def distinctive_tokens(df: pd.DataFrame, text_col, label_col, top_n=10, min_count=10) -> pd.DataFrame:
    """Top tokens per class by smoothed log-odds vs all other classes (descriptive only)."""
    per_class = {c: token_counts(g[text_col]) for c, g in df.groupby(label_col)}
    total = sum(per_class.values(), Counter())
    rows = []
    for c, cnt in per_class.items():
        rest = total - cnt
        n_c, n_r = sum(cnt.values()), sum(rest.values())
        scores = {
            w: np.log((cnt[w] + 0.5) / (n_c + 0.5)) - np.log((rest[w] + 0.5) / (n_r + 0.5))
            for w in cnt if cnt[w] >= min_count
        }
        for rank, (w, s) in enumerate(sorted(scores.items(), key=lambda kv: -kv[1])[:top_n], 1):
            rows.append({"class": c, "rank": rank, "token": w, "class_count": cnt[w], "log_odds": round(s, 2)})
    return pd.DataFrame(rows)
