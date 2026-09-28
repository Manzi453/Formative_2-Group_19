# Team workflow

## Branches
`main` (stable, submission) ← `develop` (integration) ← feature branches:

| Branch | Owner |
|---|---|
| `feature/data-pipeline` | Engineer 1 |
| `feature/lstm` | Engineer 2 |
| `feature/cnn`, `feature/tcn` | Engineer 3 |
| `feature/transformer` | Engineer 4 |
| `feature/evaluation` | Engineer 4 (+ Eng. 1) |
| `feature/report` | all |

Open a pull request into `develop`; at least one other member reviews it.

## Rules
- Engineer 1 owns `src/data_loader.py` and `src/preprocessing.py`. Others use them unchanged unless a documented research reason exists.
- One split (`make_split`, seed 42) and one evaluation protocol (`src/evaluation.py`) for every model.
- Log every run, including failures, with `src.experiments.log_experiment`.
- Never write a metric, citation or dataset statistic that was not measured/verified. Use `NOT YET MEASURED` / `CITATION REQUIRED`.
- No credentials, no dataset copies, no large checkpoints in git.
