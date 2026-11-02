"""Convenience entrypoint to train all five classifiers sequentially."""

from __future__ import annotations

from ml.models.decision_tree import main as train_decision_tree
from ml.models.knn import main as train_knn
from ml.models.logistic_regression import main as train_logistic_regression
from ml.models.naive_bayes import main as train_naive_bayes
from ml.models.svm import main as train_svm


def main() -> None:
    print("=== Logistic Regression ===")
    train_logistic_regression()
    print("\n=== Decision Tree ===")
    train_decision_tree()
    print("\n=== KNN ===")
    train_knn()
    print("\n=== Naive Bayes ===")
    train_naive_bayes()
    print("\n=== Support Vector Machine ===")
    train_svm()
    print("\nAll models trained. See reports/metrics.csv")


if __name__ == "__main__":
    main()
