"""Dataset loading and schema validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml.config import TARGET_COLUMN, TEST_CSV, TRAIN_CSV


@dataclass(frozen=True)
class DatasetBundle:
    """Train/test matrices aligned on a shared feature schema."""

    feature_columns: list[str]
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required data file not found: {path}. Run `python prepare_data.py` first."
        )
    return path


def _split_features_target(frame: pd.DataFrame, source: Path) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from {source}")

    features = frame.drop(columns=[TARGET_COLUMN])
    target = frame[TARGET_COLUMN].astype(str)

    if features.empty:
        raise ValueError(f"No feature columns found in {source}")
    if target.isna().any():
        raise ValueError(f"Target column contains missing values in {source}")
    if features.isna().any().any():
        missing = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Missing feature values in {source}: {missing}")

    return features, target


def load_dataset_bundle(
    train_path: Path = TRAIN_CSV,
    test_path: Path = TEST_CSV,
) -> DatasetBundle:
    """
    Load the stratified train/test CSVs and enforce a consistent schema.

    The test set must contain exactly the same feature columns as the train set
    (order is normalized to the training schema).
    """
    train_path = _require_file(train_path)
    test_path = _require_file(test_path)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = _split_features_target(train_df, train_path)
    X_test, y_test = _split_features_target(test_df, test_path)

    train_cols = list(X_train.columns)
    test_cols = set(X_test.columns)

    missing = [col for col in train_cols if col not in test_cols]
    extra = sorted(test_cols.difference(train_cols))
    if missing or extra:
        raise ValueError(
            "Train/test feature schema mismatch. "
            f"Missing in test: {missing}; unexpected in test: {extra}"
        )

    X_test = X_test[train_cols]

    return DatasetBundle(
        feature_columns=train_cols,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
