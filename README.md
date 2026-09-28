# Formative_2-Group_19

**Formative Assignment 2 — Research-Informed Sequential Models for NLP · Group 19**

## Challenge

**Gender-Based Violence Tweet Classification Challenge 2025** — classifying real tweets related to gender-based violence into predefined GBV categories.

This is a sequential modelling problem: tweets are short, noisy sequences of words/subwords with spelling variation, slang, hashtags, and code-switching, so the project investigates how well sequence-aware models can learn the patterns that separate GBV categories.

**Research question:** *How effectively can sequential modelling approaches be used to address GBV tweet classification, and what evidence supports the strengths and limitations of the selected approaches?*

## Approach

We compare **five substantially different approaches** (hyperparameter variants are experiments, not separate models):

| # | Model | Sequential mechanism | Owner |
|---|---|---|---|
| 1 | TF-IDF + Logistic Regression | none (bag of n-grams) - classical baseline | Engineer 1 |
| 2 | LSTM | recurrence / gated memory | Engineer 2 |
| 3 | Text CNN | local n-gram-like convolutions | Engineer 3 |
| 4 | TCN | causal, dilated, residual convolutions | Engineer 3 (or reassigned) |
| 5 | Transformer | self-attention | Engineer 4 |

Every member trains at least one model and can explain the whole pipeline. All models share one
stratified train/validation split (seed 42), one preprocessing interface (`src/preprocessing.py`)
and one evaluation protocol (`src/evaluation.py`). Primary metric is **macro-F1**; accuracy is also
reported for comparability with the original challenge. Every run - including failures - is logged in
`results/metrics/experiments.csv`.

## Repository Structure

```
README.md
requirements.txt
data/            Dataset instructions (data files are git-ignored) - see data/README.md
notebooks/       Colab-ready notebooks, run in numeric order
  01_data_exploration  02_baseline_tfidf  03_lstm  04_cnn  05_tcn  06_transformer  07_model_comparison
src/             Shared code: data_loader, preprocessing, evaluation, plotting, experiments, utils
models/          Saved artifacts per model: baseline/ lstm/ cnn/ tcn/ transformer/ (checkpoints git-ignored)
results/         metrics/ (experiments.csv, preprocessing_experiments.csv), figures/, predictions/, errors/
report/          Final report (PDF)
presentation/    demo_outline.md
docs/            team_workflow.md, contribution_tracker.md
```

## Setup

Runs in **Google Colab**: clone the repo, `pip install -r requirements.txt`, place the Zindi data as
described in `data/README.md`, then run notebooks 01 -> 07 in order. Exact package and Python versions
will be recorded after the first full Colab run.

Collaboration rules and branching: see [docs/team_workflow.md](docs/team_workflow.md).

## Links

- **Demo video (7–10 min):** _TBD_
- **Contribution tracker:** [docs/contribution_tracker.md](docs/contribution_tracker.md)
- **Final report (PDF):** _TBD_

## Group Members

_TBD_
