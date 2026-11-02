"""Project-wide configuration for training and evaluation."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
REPORTS_DIR = ROOT / "reports"

TRAIN_CSV = DATA_DIR / "train_data.csv"
TEST_CSV = ROOT / "test_data.csv"
FULL_CSV = DATA_DIR / "steel_plates_faults.csv"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
METRICS_CSV = REPORTS_DIR / "metrics.csv"

TARGET_COLUMN = "Fault_Type"
RANDOM_STATE = 42

# Multi-class metric policy (applied consistently to every model)
AVERAGE = "weighted"
AUC_MULTI_CLASS = "ovr"
AUC_AVERAGE = "weighted"
