"""Train and evaluate a K-Nearest Neighbors classifier on the shared split."""

from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier

from ml.training import run_model_training

MODEL_NAME = "KNN"
ARTIFACT_SLUG = "knn"


def build_estimator() -> KNeighborsClassifier:
    """
    Distance-based classifier; relies on the shared StandardScaler.

    ``n_neighbors=7`` matches the number of fault classes and provides a
    stable default without nested hyper-parameter search.
    """
    return KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        metric="minkowski",
        p=2,
    )


def main() -> None:
    run_model_training(
        model_name=MODEL_NAME,
        artifact_slug=ARTIFACT_SLUG,
        estimator_factory=build_estimator,
    )


if __name__ == "__main__":
    main()
