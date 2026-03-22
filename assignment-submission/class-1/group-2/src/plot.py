"""Confusion matrix figure for emotion evaluation (saved PNG, no display required)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# english axis labels for slides (same class order as src.vocab_builder.DEFAULT_LABEL_ORDER)
EMOTION_LABELS_EN = ("Sadness", "Joy", "Love", "Anger", "Fear", "Surprise")


# function to map class ids to english names for plots/reports; unknown ids fall back to id2label
def emotion_display_names_en(labels_order: list[int], id2label: dict[int, str]) -> list[str]:
    out: list[str] = []
    for i in labels_order:
        if 0 <= i < len(EMOTION_LABELS_EN):
            out.append(EMOTION_LABELS_EN[i])
        else:
            out.append(str(id2label.get(i, i)))
    return out


# function to save a confusion matrix heatmap for integer class ids and id2label names
def save_confusion_matrix_png(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    id2label: dict[int, str],
    out_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """
    y_true / y_pred: integer class ids (same convention as training CSV labels).
    id2label: maps class id -> name from checkpoint (used only as fallback for unknown ids).
    tick labels use english emotion names for the standard six-class setup.
    """
    labels_order = sorted(id2label.keys())
    if not labels_order:
        raise ValueError("id2label is empty")

    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    tick_names = emotion_display_names_en(labels_order, id2label)

    fig, ax = plt.subplots(figsize=(max(8, len(labels_order)), max(6, len(labels_order))))
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=tick_names,
        yticklabels=tick_names,
        ax=ax,
    )
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
