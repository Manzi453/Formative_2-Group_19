# Data

The dataset is the **Gender-Based Violence Tweet Classification Challenge 2025** (Zindi).
Data files are **not committed** (see `.gitignore`); each member downloads them from the
competition page and extracts them here:

```
data/gender-based-violence-tweet-classification-challenge/
├── Train.csv              Tweet_ID, tweet, type
├── Test.csv               Tweet_ID, tweet
├── SampleSubmission.csv   Tweet_ID, type
└── StarterNotebook.ipynb
```

Five classes: sexual violence, emotional violence, harmful traditional practices,
physical violence, economic violence.

**Sensitive content:** tweets may contain graphic descriptions of abuse. Quote them sparingly
in the report/presentation, and never include usernames.

Only this dataset is used for modelling. Any external data or pretrained model must be
approved and documented (see the report's Methodology section).

In Colab, load via `src/data_loader.py` (`load_raw()`), which uses paths relative to the repo root.
