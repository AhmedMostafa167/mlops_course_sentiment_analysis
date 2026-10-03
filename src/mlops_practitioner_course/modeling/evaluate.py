"""Metrics and ROC plots for binary sentiment predictions."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationReport:
    n_samples: int
    threshold: float
    accuracy: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]  # rows = true label, cols = predicted label
    positive_rate: float  # share of samples predicted positive

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def evaluate_predictions(
    y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5
) -> EvaluationReport:
    """Score positive-class probabilities against true 0/1 labels."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    if y_true.shape != probs.shape:
        raise ValueError(f"y_true {y_true.shape} and probs {probs.shape} must have the same shape")

    y_pred = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return EvaluationReport(
        n_samples=len(y_true),
        threshold=threshold,
        accuracy=float(accuracy_score(y_true, y_pred)),
        roc_auc=float(roc_auc_score(y_true, probs)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        positive_rate=float(y_pred.mean()),
    )


def plot_roc(y_true: np.ndarray, probs: np.ndarray, title: str, path: str | Path) -> Path:
    """Save a ROC curve image. Uses the Figure API, so it needs no display (servers, CI)."""
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)

    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(loc="lower right")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    logger.info("Saved ROC curve to %s", path)
    return path
