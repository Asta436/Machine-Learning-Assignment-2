"""
Download the UCI Steel Plates Faults dataset, build modeling CSVs,
and create a stratified train/test split.

Source: https://archive.ics.uci.edu/dataset/198/steel+plates+faults
"""

from __future__ import annotations

import zipfile
from io import StringIO
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip"
FAULT_COLS = [
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw_frame() -> pd.DataFrame:
    zip_path, _ = urlretrieve(UCI_ZIP_URL)
    with zipfile.ZipFile(zip_path) as archive:
        columns = [
            line.strip()
            for line in archive.read("Faults27x7_var").decode("utf-8").splitlines()
            if line.strip()
        ]
        body = archive.read("Faults.NNA").decode("utf-8")

    frame = pd.read_csv(StringIO(body), sep=r"\s+", header=None)
    if frame.shape[1] != len(columns):
        raise ValueError(
            f"Expected {len(columns)} columns, found {frame.shape[1]} in Faults.NNA"
        )
    frame.columns = columns
    return frame


def fault_type_from_indicators(row: pd.Series) -> str:
    active = [name for name in FAULT_COLS if row[name] == 1]
    if len(active) != 1:
        raise ValueError(f"Expected exactly one fault label, found {active}")
    return active[0]


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    raw = load_raw_frame()
    feature_cols = [col for col in raw.columns if col not in FAULT_COLS]
    modeled = raw[feature_cols].copy()
    modeled["Fault_Type"] = raw.apply(fault_type_from_indicators, axis=1)

    full_path = DATA_DIR / "steel_plates_faults.csv"
    modeled.to_csv(full_path, index=False)

    train_df, test_df = train_test_split(
        modeled,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=modeled["Fault_Type"],
    )
    train_df.to_csv(DATA_DIR / "train_data.csv", index=False)
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    print(f"Full dataset : {full_path} -> {modeled.shape}")
    print(f"Train split  : {DATA_DIR / 'train_data.csv'} -> {train_df.shape}")
    print(f"Test split   : {ROOT / 'test_data.csv'} -> {test_df.shape}")
    print("\nClass distribution (full):")
    print(modeled["Fault_Type"].value_counts().to_string())


if __name__ == "__main__":
    main()
