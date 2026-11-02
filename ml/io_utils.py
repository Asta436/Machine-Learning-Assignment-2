"""Persistence helpers for models and metric reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml.config import METRICS_CSV, MODEL_DIR, REPORTS_DIR


METRIC_COLUMNS = [
    "model_name",
    "accuracy",
    "auc",
    "precision",
    "recall",
    "f1",
    "mcc",
    "artifact_path",
]


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_joblib(obj: Any, path: Path) -> Path:
    ensure_directory(path.parent)
    joblib.dump(obj, path)
    return path


def load_joblib(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


def model_artifact_path(slug: str) -> Path:
    """Return ``model/<slug>.joblib``."""
    return MODEL_DIR / f"{slug}.joblib"


def _empty_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=METRIC_COLUMNS)


def load_metrics_table(path: Path = METRICS_CSV) -> pd.DataFrame:
    if not path.is_file():
        return _empty_metrics_frame()
    frame = pd.read_csv(path)
    for column in METRIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[METRIC_COLUMNS]


def upsert_metrics_row(row: dict[str, Any], path: Path = METRICS_CSV) -> pd.DataFrame:
    """Insert or replace a model's metrics row and rewrite the CSV."""
    ensure_directory(REPORTS_DIR)
    frame = load_metrics_table(path)
    frame = frame[frame["model_name"] != row["model_name"]]
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)

    # Keep a stable display order when known models are present.
    preferred = [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Support Vector Machine",
    ]
    frame["model_name"] = pd.Categorical(
        frame["model_name"],
        categories=preferred + sorted(
            set(frame["model_name"].astype(str)) - set(preferred)
        ),
        ordered=True,
    )
    frame = frame.sort_values("model_name").reset_index(drop=True)
    frame["model_name"] = frame["model_name"].astype(str)
    frame.to_csv(path, index=False)
    return frame


def format_metrics_block(row: dict[str, Any]) -> str:
    return (
        f"Model     : {row['model_name']}\n"
        f"Accuracy  : {row['accuracy']:.4f}\n"
        f"AUC       : {row['auc']:.4f}\n"
        f"Precision : {row['precision']:.4f}\n"
        f"Recall    : {row['recall']:.4f}\n"
        f"F1        : {row['f1']:.4f}\n"
        f"MCC       : {row['mcc']:.4f}\n"
        f"Artifact  : {row['artifact_path']}"
    )
