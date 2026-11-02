"""Fit and persist the shared train-only preprocessor."""

from __future__ import annotations

import pandas as pd

from ml.config import METRICS_CSV, MODEL_DIR, PREPROCESSOR_PATH, REPORTS_DIR
from ml.io_utils import METRIC_COLUMNS, ensure_directory
from ml.preprocess import build_and_persist_preprocessor


def main() -> None:
    ensure_directory(MODEL_DIR)
    ensure_directory(REPORTS_DIR)

    prepared = build_and_persist_preprocessor()

    if not METRICS_CSV.is_file():
        pd.DataFrame(columns=METRIC_COLUMNS).to_csv(METRICS_CSV, index=False)

    print(f"Preprocessor saved to: {PREPROCESSOR_PATH}")
    print(f"Features ({len(prepared.feature_columns)}): {prepared.feature_columns}")
    print(f"Classes  ({len(prepared.class_names)}): {list(prepared.class_names)}")
    print(f"Train shape: {prepared.X_train.shape} | Test shape: {prepared.X_test.shape}")
    print(f"Metrics table ready at: {METRICS_CSV}")


if __name__ == "__main__":
    main()
