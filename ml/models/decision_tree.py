"""Train and evaluate a Decision Tree classifier on the shared split."""

from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier

from ml.config import RANDOM_STATE
from ml.training import run_model_training

MODEL_NAME = "Decision Tree"
ARTIFACT_SLUG = "decision_tree"


def build_estimator() -> DecisionTreeClassifier:
    """
    Regularized decision tree to limit overfitting on 27 numeric features.

    Scaling is not required for trees; we still use the shared scaled matrix
    so every model sees an identical feature space.
    """
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
    )


def main() -> None:
    run_model_training(
        model_name=MODEL_NAME,
        artifact_slug=ARTIFACT_SLUG,
        estimator_factory=build_estimator,
    )


if __name__ == "__main__":
    main()
