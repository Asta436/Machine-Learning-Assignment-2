"""Train and evaluate a Support Vector Machine on the shared split."""

from __future__ import annotations

from sklearn.svm import SVC

from ml.config import RANDOM_STATE
from ml.training import run_model_training

MODEL_NAME = "Support Vector Machine"
ARTIFACT_SLUG = "svm"


def build_estimator() -> SVC:
    """
    RBF-kernel SVM on standardized features.

    ``probability=True`` enables predict_proba for a consistent OvR AUC with
    the other classifiers (at the cost of a longer fit).
    """
    return SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
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
