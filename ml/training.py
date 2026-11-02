"""Shared training workflow used by each model script."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ml.config import PREPROCESSOR_PATH
from ml.io_utils import (
    format_metrics_block,
    load_joblib,
    model_artifact_path,
    save_joblib,
    upsert_metrics_row,
)
from ml.metrics import compute_classification_metrics
from ml.preprocess import PreparedData, build_and_persist_preprocessor


def load_or_build_prepared_data() -> PreparedData:
    """Reuse the saved preprocessor when available for consistent transforms."""
    if PREPROCESSOR_PATH.is_file():
        from ml.data import load_dataset_bundle
        from ml.preprocess import (
            PreprocessorArtifacts,
            encode_labels,
            transform_features,
        )

        bundle = load_dataset_bundle()
        artifacts: PreprocessorArtifacts = load_joblib(PREPROCESSOR_PATH)
        return PreparedData(
            feature_columns=artifacts.feature_columns,
            class_names=list(artifacts.label_encoder.classes_),
            X_train=transform_features(bundle.X_train, artifacts),
            y_train=encode_labels(bundle.y_train, artifacts),
            X_test=transform_features(bundle.X_test, artifacts),
            y_test=encode_labels(bundle.y_test, artifacts),
            preprocessor=artifacts,
        )
    return build_and_persist_preprocessor()


def run_model_training(
    *,
    model_name: str,
    artifact_slug: str,
    estimator_factory: Callable[[], Any],
) -> dict[str, Any]:
    """
    Fit one classifier on the shared prepared split, evaluate, and persist.

    Parameters
    ----------
    model_name:
        Display name stored in the metrics table / README.
    artifact_slug:
        Filename stem under ``model/`` (e.g. ``logistic_regression``).
    estimator_factory:
        Zero-arg callable that returns a fresh, unfitted estimator.
    """
    prepared = load_or_build_prepared_data()
    estimator = estimator_factory()
    estimator.fit(prepared.X_train, prepared.y_train)

    metrics = compute_classification_metrics(
        estimator,
        prepared.X_test,
        prepared.y_test,
        prepared.class_names,
    )

    artifact_path = model_artifact_path(artifact_slug)
    save_joblib(
        {
            "model_name": model_name,
            "artifact_slug": artifact_slug,
            "estimator": estimator,
            "feature_columns": prepared.feature_columns,
            "class_names": prepared.class_names,
            "metrics": metrics,
        },
        artifact_path,
    )

    row = {
        "model_name": model_name,
        "accuracy": round(metrics["accuracy"], 6),
        "auc": round(metrics["auc"], 6),
        "precision": round(metrics["precision"], 6),
        "recall": round(metrics["recall"], 6),
        "f1": round(metrics["f1"], 6),
        "mcc": round(metrics["mcc"], 6),
        "artifact_path": str(Path(artifact_path).as_posix()),
    }
    upsert_metrics_row(row)
    print(format_metrics_block(row))
    return row
