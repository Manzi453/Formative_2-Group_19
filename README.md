# Formative_2-Group_19

**Formative Assignment 2 — Research-Informed Sequential Models for NLP · Group 19**

## Challenge

**Gender-Based Violence Tweet Classification Challenge 2025** — classifying real tweets related to gender-based violence into predefined GBV categories.

This is a sequential modelling problem: tweets are short, noisy sequences of words/subwords with spelling variation, slang, hashtags, and code-switching, so the project investigates how well sequence-aware models can learn the patterns that separate GBV categories.

**Research question:** *How effectively can sequential modelling approaches be used to address GBV tweet classification, and what evidence supports the strengths and limitations of the selected approaches?*

## Approach

We compare **five substantially different approaches**, selected based on exploratory data analysis and a review of related work:

- **3 neural sequential architectures** (e.g., LSTM / GRU / Transformer-based — final selection justified in the report)
- **2 baseline/comparison approaches** (e.g., classical ML over sequence features)

Each model is trained, evaluated with task-appropriate metrics (with justification), and analysed through error analysis. Experiments are tracked systematically, and every group member trains at least one model.

## Repository Structure

```
data/        Datasets and data preparation code
notebooks/   Exploratory analysis and model experiments (Colab-ready)
src/         Shared code: preprocessing, features, training, evaluation
reports/     Report drafts and figures
```

## Setup

The project runs end-to-end in **Google Colab**. Setup instructions will be added as notebooks are completed.

## Links

- **Demo video (7–10 min):** _TBD_
- **Contribution tracker:** _TBD_
- **Final report (PDF):** _TBD_

## Group Members

_TBD_
