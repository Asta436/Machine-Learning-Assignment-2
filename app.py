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

from ml.config import TARGET_COLUMN, TEST_CSV
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


@st.cache_resource
def get_preprocessor():
    return load_preprocessor()


@st.cache_data
def load_default_test_csv() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV)


def metrics_to_frame(metrics: dict[str, float]) -> pd.DataFrame:
    order = ["accuracy", "auc", "precision", "recall", "f1", "mcc"]
    return pd.DataFrame(
        {
            "Metric": [name.upper() if name == "auc" else name.title() for name in order],
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
            st.dataframe(metrics_to_frame(result.metrics), hide_index=True, width="stretch")
        with right:
            st.markdown("#### Classification report")
            st.dataframe(result.report_frame.round(3), width="stretch")

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
        st.dataframe(table, hide_index=True, width="stretch")

        best_row = table.sort_values(
            by=["MCC", "F1", "Accuracy"],
            ascending=False,
        ).iloc[0]
        st.success(
            f"Best on this CSV by MCC/F1/Accuracy: **{best_row['Model']}** "
            f"(MCC={best_row['MCC']:.4f}, F1={best_row['F1']:.4f}, "
            f"Accuracy={best_row['Accuracy']:.4f})"
        )

        st.markdown("#### Accuracy by model")
        chart_df = table.set_index("Model")[["Accuracy", "F1", "MCC"]]
        st.bar_chart(chart_df)


def page_data_preview(frame: pd.DataFrame) -> None:
    st.header("Data preview")
    st.write("Quick look at the active test CSV before evaluation.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(frame))
    c2.metric("Columns", frame.shape[1])
    c3.metric("Has target", "Yes" if TARGET_COLUMN in frame.columns else "No")

    if TARGET_COLUMN in frame.columns:
        st.markdown("#### Class distribution")
        counts = frame[TARGET_COLUMN].value_counts().rename_axis("Fault_Type").reset_index(name="Count")
        st.dataframe(counts, hide_index=True, width="stretch")

    st.markdown("#### Sample rows")
    st.dataframe(frame.head(20), hide_index=True, width="stretch")

    with st.expander("Column list"):
        st.code("\n".join(frame.columns.tolist()))


def page_about() -> None:
    st.header("About")
    st.markdown(
        """
**Assignment:** Machine Learning Assignment 2 — Steel Plates Fault classification  

**Dataset:** [UCI Steel Plates Faults](https://archive.ics.uci.edu/dataset/198/steel+plates+faults)  

**Models (saved `.joblib`, no retraining in this app):**
1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors  
4. Gaussian Naive Bayes  
5. Support Vector Machine  

**How to use**
1. Keep **Use bundled test_data.csv** checked, or upload your own test CSV.  
2. Open **Evaluate one model** → pick a model → **Run evaluation**.  
3. Open **Compare all models** to see every classifier on the same CSV.  

**CSV format:** all 27 feature columns + `Fault_Type` target column.
"""
    )
    st.caption(f"Bundled test file path: `{Path(TEST_CSV).name}`")


def main() -> None:
    st.title("Steel Plates Fault Classification")
    st.caption("Interactive evaluation of five saved classifiers — no retraining.")

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
