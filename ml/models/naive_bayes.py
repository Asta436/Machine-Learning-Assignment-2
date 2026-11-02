"""Train and evaluate Gaussian Naive Bayes on the shared split."""

from __future__ import annotations

from sklearn.naive_bayes import GaussianNB

from ml.training import run_model_training

MODEL_NAME = "Naive Bayes"
ARTIFACT_SLUG = "naive_bayes"


def build_estimator() -> GaussianNB:
    """
    GaussianNB is appropriate for continuous numeric steel-plate features.

    MultinomialNB would require non-negative count-like inputs, so it is not
    used for this dataset.
    """
    return GaussianNB(var_smoothing=1e-9)


def main() -> None:
    run_model_training(
        model_name=MODEL_NAME,
        artifact_slug=ARTIFACT_SLUG,
        estimator_factory=build_estimator,
    )


if __name__ == "__main__":
    main()
