"""Shared plotting helpers so all figures look consistent. Owner: Engineer 4.

Every figure needs a title, axis labels, a legend where required, and a figure
number used in the report. Save to results/figures/.
"""
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import RESULTS_DIR

FIG_DIR = RESULTS_DIR / "figures"


def save_fig(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def plot_confusion_matrix(cm, labels, title: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax, cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    return fig


def plot_learning_curves(history: dict, title: str):
    """`history` maps series name (e.g. 'train_loss', 'val_loss') -> list of values per epoch."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, values in history.items():
        ax.plot(range(1, len(values) + 1), values, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    return fig
