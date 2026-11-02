"""Feature preprocessing fitted on training data only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ml.config import MODEL_DIR, PREPROCESSOR_PATH
from ml.data import DatasetBundle, load_dataset_bundle
from ml.io_utils import ensure_directory, save_joblib


@dataclass
class PreprocessorArtifacts:
    """Scaler + label encoder persisted for reuse by every classifier."""

    feature_columns: list[str]
    scaler: StandardScaler
    label_encoder: LabelEncoder


@dataclass
class PreparedData:
    """Numeric matrices ready for model fitting and evaluation."""

    feature_columns: list[str]
    class_names: list[str]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    preprocessor: PreprocessorArtifacts


def fit_preprocessor(bundle: DatasetBundle) -> PreprocessorArtifacts:
    """Fit StandardScaler and LabelEncoder on the training split only."""
    scaler = StandardScaler()
    scaler.fit(bundle.X_train.to_numpy(dtype=float))

    label_encoder = LabelEncoder()
    label_encoder.fit(bundle.y_train)

    return PreprocessorArtifacts(
        feature_columns=list(bundle.feature_columns),
        scaler=scaler,
        label_encoder=label_encoder,
    )


def transform_features(
    frame: pd.DataFrame,
    artifacts: PreprocessorArtifacts,
) -> np.ndarray:
    ordered = frame[artifacts.feature_columns]
    return artifacts.scaler.transform(ordered.to_numpy(dtype=float))


def encode_labels(series: pd.Series, artifacts: PreprocessorArtifacts) -> np.ndarray:
    return artifacts.label_encoder.transform(series.astype(str))


def prepare_modeling_data(bundle: DatasetBundle | None = None) -> PreparedData:
    """Load data (if needed), fit preprocessor, and return train/test arrays."""
    if bundle is None:
        bundle = load_dataset_bundle()

    artifacts = fit_preprocessor(bundle)
    prepared = PreparedData(
        feature_columns=artifacts.feature_columns,
        class_names=list(artifacts.label_encoder.classes_),
        X_train=transform_features(bundle.X_train, artifacts),
        y_train=encode_labels(bundle.y_train, artifacts),
        X_test=transform_features(bundle.X_test, artifacts),
        y_test=encode_labels(bundle.y_test, artifacts),
        preprocessor=artifacts,
    )
    return prepared


def save_preprocessor(artifacts: PreprocessorArtifacts, path=PREPROCESSOR_PATH) -> None:
    ensure_directory(MODEL_DIR)
    save_joblib(artifacts, path)


def build_and_persist_preprocessor() -> PreparedData:
    """Fit preprocessor, save it under model/, and return prepared arrays."""
    prepared = prepare_modeling_data()
    save_preprocessor(prepared.preprocessor)
    return prepared
