"""Train and evaluate Logistic Regression on the shared Steel Plates split."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from ml.config import RANDOM_STATE
from ml.training import run_model_training

MODEL_NAME = "Logistic Regression"
ARTIFACT_SLUG = "logistic_regression"


def build_estimator() -> LogisticRegression:
    """
    Multinomial logistic regression with a stable solver for 7 classes.

    Features are already standardized by the shared preprocessor.
    """
    return LogisticRegression(
        solver="lbfgs",
        multi_class="auto",
        max_iter=2000,
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
