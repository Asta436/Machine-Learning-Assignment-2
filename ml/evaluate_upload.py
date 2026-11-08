"""Evaluate saved models on an uploaded / local test CSV (no retraining)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from ml.config import MODEL_DIR, PREPROCESSOR_PATH, TARGET_COLUMN
from ml.io_utils import load_joblib
from ml.metrics import compute_classification_metrics
from ml.preprocess import PreprocessorArtifacts, encode_labels, transform_features

MODEL_CATALOG: dict[str, str] = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    n_rows: int
    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    class_names: list[str]
    report_frame: pd.DataFrame
    confusion: np.ndarray


def load_preprocessor(path: Path = PREPROCESSOR_PATH) -> PreprocessorArtifacts:
    return load_joblib(path)


def load_model_artifact(slug: str, model_dir: Path = MODEL_DIR) -> dict[str, Any]:
    return load_joblib(model_dir / f"{slug}.joblib")


def validate_uploaded_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Ensure the uploaded CSV has the target and all required features."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(
            f"Uploaded CSV must include the target column '{TARGET_COLUMN}'."
        )

    missing = [col for col in feature_columns if col not in frame.columns]
    if missing:
        raise ValueError(
            "Uploaded CSV is missing required feature columns: "
            + ", ".join(missing)
        )

    cleaned = frame[feature_columns + [TARGET_COLUMN]].copy()
    if cleaned[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column '{TARGET_COLUMN}' contains missing values.")
    if cleaned[feature_columns].isna().any().any():
        bad = cleaned[feature_columns].columns[
            cleaned[feature_columns].isna().any()
        ].tolist()
        raise ValueError(f"Feature columns contain missing values: {bad}")
    return cleaned


def prepare_xy(
    frame: pd.DataFrame,
    preprocessor: PreprocessorArtifacts,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    cleaned = validate_uploaded_frame(frame, preprocessor.feature_columns)
    X = transform_features(cleaned[preprocessor.feature_columns], preprocessor)
    y = encode_labels(cleaned[TARGET_COLUMN], preprocessor)
    return X, y, cleaned


def evaluate_model_on_frame(
    frame: pd.DataFrame,
    model_name: str,
    preprocessor: PreprocessorArtifacts | None = None,
) -> EvaluationResult:
    """Score one saved model on a labeled CSV without refitting."""
    if model_name not in MODEL_CATALOG:
        raise ValueError(f"Unknown model: {model_name}")

    if preprocessor is None:
        preprocessor = load_preprocessor()

    artifact = load_model_artifact(MODEL_CATALOG[model_name])
    estimator = artifact["estimator"]
    class_names = list(artifact.get("class_names", preprocessor.label_encoder.classes_))

    X, y_true, _ = prepare_xy(frame, preprocessor)
    y_pred = estimator.predict(X)
    metrics = compute_classification_metrics(estimator, X, y_true, class_names)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).transpose()

    conf = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    return EvaluationResult(
        model_name=model_name,
        n_rows=len(y_true),
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        report_frame=report_frame,
        confusion=conf,
    )


def compare_all_models(
    frame: pd.DataFrame,
    preprocessor: PreprocessorArtifacts | None = None,
) -> pd.DataFrame:
    """Return a metrics comparison table for all five saved models."""
    if preprocessor is None:
        preprocessor = load_preprocessor()

    rows: list[dict[str, Any]] = []
    for model_name in MODEL_CATALOG:
        result = evaluate_model_on_frame(frame, model_name, preprocessor)
        rows.append(
            {
                "Model": model_name,
                "Accuracy": round(result.metrics["accuracy"], 4),
                "AUC": round(result.metrics["auc"], 4),
                "Precision": round(result.metrics["precision"], 4),
                "Recall": round(result.metrics["recall"], 4),
                "F1": round(result.metrics["f1"], 4),
                "MCC": round(result.metrics["mcc"], 4),
            }
        )
    return pd.DataFrame(rows)
