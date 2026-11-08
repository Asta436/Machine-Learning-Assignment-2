"""
Streamlit app: evaluate saved Steel Plates Fault classifiers on uploaded test CSV.

Required features:
- CSV upload (features + target)
- Model selection dropdown
- Evaluation metrics
- Confusion matrix / classification report
Models are loaded from disk; nothing is retrained in the app.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from ml.config import METRICS_CSV, TARGET_COLUMN, TEST_CSV
from ml.evaluate_upload import (
    MODEL_CATALOG,
    compare_all_models,
    evaluate_model_on_frame,
    load_preprocessor,
)

st.set_page_config(
    page_title="Steel Plates Fault Classifier",
    page_icon=None,
    layout="wide",
)

PAGES = [
    "1. Evaluate one model",
    "2. Compare all models",
    "3. Data preview",
    "4. About",
]

GITHUB_URL = "https://github.com/Asta436/Machine-Learning-Assignment-2"
DATASET_URL = "https://archive.ics.uci.edu/dataset/198/steel+plates+faults"


@st.cache_resource
def get_preprocessor():
    return load_preprocessor()


@st.cache_data
def load_default_test_csv() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV)


@st.cache_data
def load_training_metrics_table() -> pd.DataFrame | None:
    if not METRICS_CSV.is_file():
        return None
    frame = pd.read_csv(METRICS_CSV)
    keep = ["model_name", "accuracy", "auc", "precision", "recall", "f1", "mcc"]
    present = [col for col in keep if col in frame.columns]
    return frame[present].round(4)


def metrics_to_frame(metrics: dict[str, float]) -> pd.DataFrame:
    order = ["accuracy", "auc", "precision", "recall", "f1", "mcc"]
    labels = {
        "accuracy": "Accuracy",
        "auc": "AUC",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "mcc": "MCC",
    }
    return pd.DataFrame(
        {
            "Metric": [labels[name] for name in order],
            "Value": [round(metrics[name], 4) for name in order],
        }
    )


def plot_confusion_matrix(matrix, class_names: list[str]):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig


def sidebar_controls() -> tuple[str, pd.DataFrame | None]:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", PAGES, index=0)

    st.sidebar.divider()
    st.sidebar.subheader("Test data")
    st.sidebar.caption(
        "Upload a CSV with all feature columns and the target "
        f"`{TARGET_COLUMN}`. Models are not retrained."
    )

    uploaded = st.sidebar.file_uploader("Upload test CSV", type=["csv"])
    use_default = st.sidebar.checkbox(
        "Use bundled test_data.csv",
        value=uploaded is None,
        help="Repo held-out test set (389 rows).",
    )

    frame: pd.DataFrame | None = None
    if uploaded is not None:
        frame = pd.read_csv(uploaded)
        st.sidebar.success(f"Uploaded: {uploaded.name} ({len(frame)} rows)")
    elif use_default and TEST_CSV.is_file():
        frame = load_default_test_csv()
        st.sidebar.info(f"Using bundled test_data.csv ({len(frame)} rows)")
    else:
        st.sidebar.warning("Provide a CSV upload or enable the bundled test file.")

    return page, frame


def page_evaluate_one(frame: pd.DataFrame) -> None:
    st.header("Evaluate one model")
    st.write(
        "Select a saved classifier. Metrics are computed on the current test CSV "
        "using the frozen preprocessor and model artifacts."
    )

    model_name = st.selectbox("Model", list(MODEL_CATALOG.keys()), index=4)

    if st.button("Run evaluation", type="primary"):
        try:
            preprocessor = get_preprocessor()
            result = evaluate_model_on_frame(frame, model_name, preprocessor)
        except Exception as exc:  # noqa: BLE001 - show clean UI error
            st.error(str(exc))
            return

        st.subheader(f"Results — {result.model_name}")
        st.caption(f"Rows evaluated: {result.n_rows}")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{result.metrics['accuracy']:.4f}")
        m2.metric("AUC", f"{result.metrics['auc']:.4f}")
        m3.metric("Precision", f"{result.metrics['precision']:.4f}")
        m4.metric("Recall", f"{result.metrics['recall']:.4f}")
        m5.metric("F1", f"{result.metrics['f1']:.4f}")
        m6.metric("MCC", f"{result.metrics['mcc']:.4f}")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Metrics table")
            st.dataframe(
                metrics_to_frame(result.metrics),
                hide_index=True,
                use_container_width=True,
            )
        with right:
            st.markdown("#### Classification report")
            st.dataframe(result.report_frame.round(3), use_container_width=True)

        st.markdown("#### Confusion matrix")
        st.pyplot(
            plot_confusion_matrix(result.confusion, result.class_names),
            clear_figure=True,
        )


def page_compare_all(frame: pd.DataFrame) -> None:
    st.header("Compare all models")
    st.write(
        "Runs the same uploaded/bundled test CSV through all five saved models "
        "and shows a side-by-side metric table."
    )

    if st.button("Compare all models", type="primary"):
        try:
            preprocessor = get_preprocessor()
            table = compare_all_models(frame, preprocessor)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return

        st.subheader("Comparison on current test data")
        st.dataframe(table, hide_index=True, use_container_width=True)

        best_row = table.sort_values(
            by=["MCC", "F1", "Accuracy"],
            ascending=False,
        ).iloc[0]
        st.success(
            f"Best on this CSV by MCC/F1/Accuracy: **{best_row['Model']}** "
            f"(MCC={best_row['MCC']:.4f}, F1={best_row['F1']:.4f}, "
            f"Accuracy={best_row['Accuracy']:.4f})"
        )

        st.markdown("#### Accuracy / F1 / MCC by model")
        chart_df = table.set_index("Model")[["Accuracy", "F1", "MCC"]]
        st.bar_chart(chart_df)


def page_data_preview(frame: pd.DataFrame) -> None:
    st.header("Data preview")
    st.write("Inspect the active test CSV before running evaluation.")

    try:
        preprocessor = get_preprocessor()
        expected_features = list(preprocessor.feature_columns)
    except Exception:  # noqa: BLE001
        preprocessor = None
        expected_features = []

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(frame))
    c2.metric("Columns", frame.shape[1])
    c3.metric("Has target", "Yes" if TARGET_COLUMN in frame.columns else "No")
    missing_total = int(frame.isna().sum().sum())
    c4.metric("Missing values", missing_total)

    st.markdown("#### Schema check vs trained preprocessor")
    if expected_features:
        missing_cols = [col for col in expected_features if col not in frame.columns]
        extra_cols = [
            col
            for col in frame.columns
            if col not in expected_features and col != TARGET_COLUMN
        ]
        s1, s2, s3 = st.columns(3)
        s1.write(f"Expected features: **{len(expected_features)}**")
        s2.write(f"Missing required: **{len(missing_cols)}**")
        s3.write(f"Unexpected extras: **{len(extra_cols)}**")
        if missing_cols:
            st.error("Missing required feature columns:\n\n- " + "\n- ".join(missing_cols))
        elif TARGET_COLUMN not in frame.columns:
            st.error(f"Target column `{TARGET_COLUMN}` is missing.")
        else:
            st.success("Schema looks compatible with the saved models.")
        if extra_cols:
            st.warning("Extra columns will be ignored: " + ", ".join(extra_cols))
    else:
        st.info("Preprocessor artifact not available for schema validation.")

    if TARGET_COLUMN in frame.columns:
        st.markdown("#### Class distribution")
        counts = (
            frame[TARGET_COLUMN]
            .value_counts()
            .rename_axis("Fault_Type")
            .reset_index(name="Count")
        )
        counts["Percent"] = (100 * counts["Count"] / counts["Count"].sum()).round(2)
        left, right = st.columns([1, 1])
        with left:
            st.dataframe(counts, hide_index=True, use_container_width=True)
        with right:
            st.bar_chart(counts.set_index("Fault_Type")["Count"])

    st.markdown("#### Feature summary (numeric)")
    feature_cols = [
        col for col in frame.columns if col != TARGET_COLUMN and pd.api.types.is_numeric_dtype(frame[col])
    ]
    if feature_cols:
        summary = frame[feature_cols].describe().T[
            ["count", "mean", "std", "min", "50%", "max"]
        ].round(3)
        st.dataframe(summary, use_container_width=True)
    else:
        st.info("No numeric feature columns found to summarize.")

    st.markdown("#### Sample rows")
    st.dataframe(frame.head(20), hide_index=True, use_container_width=True)

    with st.expander("Full column list"):
        st.code("\n".join(frame.columns.tolist()))

    if expected_features:
        with st.expander("Expected feature order (from preprocessor)"):
            st.code("\n".join(expected_features))


def page_about() -> None:
    st.header("About")

    st.markdown(
        f"""
### Problem
Multi-class classification of **steel plate surface faults** from 27 geometric /
luminosity / process features into 7 fault types (`Fault_Type`).

### Links
- Dataset: [{DATASET_URL}]({DATASET_URL})
- GitHub: [{GITHUB_URL}]({GITHUB_URL})
- Bundled test file: `{Path(TEST_CSV).name}`
"""
    )

    st.markdown("### Models available in this app")
    st.markdown(
        """
| # | Model | Artifact |
|---|-------|----------|
| 1 | Logistic Regression | `model/logistic_regression.joblib` |
| 2 | Decision Tree | `model/decision_tree.joblib` |
| 3 | K-Nearest Neighbors | `model/knn.joblib` |
| 4 | Gaussian Naive Bayes | `model/naive_bayes.joblib` |
| 5 | Support Vector Machine | `model/svm.joblib` |
"""
    )

    st.markdown("### Evaluation protocol")
    st.markdown(
        """
- Models are **loaded only** — this app never retrains.
- Features are transformed with the frozen train-fit `StandardScaler`
  (`model/preprocessor.joblib`).
- Reported metrics: Accuracy, AUC (OvR, weighted), Precision / Recall / F1
  (weighted), MCC.
- Weighted Recall equals Accuracy by definition.
"""
    )

    st.markdown("### Reference metrics (repo held-out test set)")
    metrics_table = load_training_metrics_table()
    if metrics_table is not None and not metrics_table.empty:
        st.dataframe(metrics_table, hide_index=True, use_container_width=True)
        st.caption("Source: `reports/metrics.csv` from offline training.")
    else:
        st.info("Training metrics file not found.")

    st.markdown("### How to use")
    st.markdown(
        """
1. Keep **Use bundled test_data.csv** checked, or upload your own labeled test CSV.  
2. **Data preview** — confirm schema / class balance.  
3. **Evaluate one model** — dropdown + metrics + confusion matrix / report.  
4. **Compare all models** — same CSV scored by all five classifiers.  
"""
    )

    st.markdown("### Required CSV format")
    st.markdown(
        f"""
Must include:
- Target column: `{TARGET_COLUMN}`
- All 27 feature columns used during training

Example classes: `Pastry`, `Z_Scratch`, `K_Scatch`, `Stains`, `Dirtiness`,
`Bumps`, `Other_Faults`.
"""
    )


def main() -> None:
    st.title("Steel Plates Fault Classification")

    page, frame = sidebar_controls()

    if page == "4. About":
        page_about()
        return

    if frame is None:
        st.warning("Load test data from the sidebar to continue.")
        page_about()
        return

    if page == "1. Evaluate one model":
        page_evaluate_one(frame)
    elif page == "2. Compare all models":
        page_compare_all(frame)
    elif page == "3. Data preview":
        page_data_preview(frame)


if __name__ == "__main__":
    main()
