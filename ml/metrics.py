"""Evaluation metrics shared by every classifier."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.config import AUC_AVERAGE, AUC_MULTI_CLASS, AVERAGE


def _positive_class_scores(estimator: Any, X: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Return an (n_samples, n_classes) score matrix for multi-class AUC.

    Prefers ``predict_proba``; falls back to ``decision_function``.
    """
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(X)
        return np.asarray(scores, dtype=float)

    if hasattr(estimator, "decision_function"):
        decision = np.asarray(estimator.decision_function(X), dtype=float)
        if decision.ndim == 1:
            return np.column_stack([-decision, decision])
        return decision

    raise AttributeError(
        f"{type(estimator).__name__} must expose predict_proba or decision_function "
        "to compute AUC."
    )


def compute_classification_metrics(
    estimator: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    """
    Compute the six assignment-required metrics on a held-out set.

    Precision / Recall / F1 use weighted averaging.
    AUC uses one-vs-rest with weighted averaging.
    """
    y_pred = estimator.predict(X)
    n_classes = len(class_names)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=AVERAGE, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average=AVERAGE, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average=AVERAGE, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    y_score = _positive_class_scores(estimator, X, n_classes)
    if y_score.shape[1] != n_classes:
        raise ValueError(
            f"Score matrix has {y_score.shape[1]} columns; expected {n_classes}."
        )

    metrics["auc"] = float(
        roc_auc_score(
            y_true,
            y_score,
            multi_class=AUC_MULTI_CLASS,
            average=AUC_AVERAGE,
            labels=np.arange(n_classes),
        )
    )
    return metrics
