# Formative_2-Group_19 - Engineer 2 - LSTM

Your slice of the Formative 2 repo - your notebook(s), the shared code, the data, and (only
where unavoidable) the upstream file your notebook needs to run.

## Your notebook

`03_lstm.ipynb` - macro-F1 0.8754, ~1.5 min on an Apple Silicon Mac, a bit longer on CPU.

Behind the TF-IDF baseline and CNN - recurrence just converges slower on this task.

**One file isn't yours:** `results/metrics/preprocessing_experiments.csv` and the
`baseline_tfidf_logreg_v1` row in `experiments.csv` came from Engineer 1's notebook 02 (already
merged into `main`) - your notebook reads them to pick the text-cleaning strategy, but they're
not something you generated. Don't worry about them; they'll already be on `main` for real.

## Setup

**macOS**
```bash
cd engineer2_lstm
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Apple Silicon auto-uses the GPU (MPS) for the neural models, no setup needed. Intel Macs run on
CPU - slower but fine.

**Windows**
```powershell
cd engineer2_lstm
py -3 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks the activate script: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
then retry. Runs on CPU by default (no MPS on Windows) - works fine, just slower.

**Run it:** activate the venv, `jupyter notebook`, open your notebook(s), Run All. Data's
already in `data/` - no Zindi download needed.

## Pushing your branch

Everything in `results/`, `models/` here is **only your own output** - safe to commit and push
as-is, nothing here belongs to someone else's notebook. The `baseline_tfidf_logreg_v1` row in `experiments.csv` and all of `preprocessing_experiments.csv` are Engineer 1's, carried over only so your notebook runs - they'll already be on `main`.

`src/` is the shared pipeline (split, preprocessing, the model code, the training loop) - same
copy everyone has. If you need to change it, flag it with the team first.
