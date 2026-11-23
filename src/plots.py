"""Chart helpers (saved as PNG files). Uses a non-interactive backend so the
scripts also run on servers; in a notebook, call plt.show() or view the files."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

STAY, LEAVE, NEUTRAL = "#4C78A8", "#E45756", "#8A8F98"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(y, path):
    counts = y.map({0: "Stayed (No)", 1: "Left (Yes)"}).value_counts()
    counts = counts.reindex(["Stayed (No)", "Left (Yes)"])
    fig, ax = plt.subplots(figsize=(5, 3.6))
    bars = ax.bar(counts.index, counts.values, color=[STAY, LEAVE])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v / counts.sum():.0%})",
                ha="center", va="bottom")
    ax.set_ylabel("Customers")
    ax.set_title("Churn vs non-churn (imbalanced)")
    ax.set_ylim(0, counts.max() * 1.2)
    _save(fig, path)


def plot_churn_rate_by(df, column, path, bins=None, title=None):
    """Bar chart of churn rate per category (or per numeric bin)."""
    data = df[[column, "churn"]].copy()
    if bins is not None:
        labels = [f"{a:g}-{b:g}" for a, b in zip(bins[:-1], bins[1:])]
        data[column] = pd.cut(data[column], bins=bins, labels=labels,
                              include_lowest=True)
    rate = data.groupby(column, observed=True)["churn"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(6, 3.8))
    labels = [str(i) for i in rate.index]
    bars = ax.bar(labels, rate["mean"] * 100, color=LEAVE, alpha=0.85)
    for b, (m, n) in zip(bars, zip(rate["mean"], rate["size"])):
        ax.text(b.get_x() + b.get_width() / 2, m * 100, f"{m:.0%}\n(n={n})",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(df["churn"].mean() * 100, color=NEUTRAL, ls="--", lw=1,
               label="Overall churn rate")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title(title or f"Churn rate by {column}")
    ax.set_ylim(0, max(rate["mean"].max() * 100 * 1.25, 5))
    ax.legend(frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=20 if bins is not None else 0)
    _save(fig, path)


def plot_confusion_matrix(cm, path, title="Confusion matrix"):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    names = ["Stayed", "Left"]
    ax.set_xticks([0, 1], [f"Predicted\n{n}" for n in names])
    ax.set_yticks([0, 1], [f"Actual\n{n}" for n in names])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title)
    _save(fig, path)


def plot_roc_curves(curves, path):
    """curves: {model name: (fpr, tpr, auc)}"""
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], ls="--", color=NEUTRAL, label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves (test set)")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, path)


def plot_feature_importance(names, values, path, title, top=12, signed=True):
    s = pd.Series(values, index=names)
    s = s.reindex(s.abs().sort_values(ascending=False).index)[:top][::-1]
    colors = [LEAVE if v > 0 else STAY for v in s] if signed else LEAVE
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(s.index, s.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Raises churn risk  →" if signed else "Importance")
    if signed:
        ax.set_xlabel("← lowers churn risk | raises churn risk →")
    _save(fig, path)


def plot_threshold_tradeoff(grid, precision, recall, f1, chosen, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, precision, label="Precision", lw=2, color=STAY)
    ax.plot(grid, recall, label="Recall", lw=2, color=LEAVE)
    ax.plot(grid, f1, label="F1-score", lw=2, color="#54A24B")
    ax.axvline(chosen, color=NEUTRAL, ls="--", label=f"Chosen threshold ({chosen:.2f})")
    ax.set_xlabel("Decision threshold")
    ax.set_title("Threshold trade-off (cross-validated, training data)")
    ax.legend(frameon=False)
    _save(fig, path)
