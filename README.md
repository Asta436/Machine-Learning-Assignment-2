# Machine Learning Assignment 2

End-to-end classification pipeline with five classical ML models and a Streamlit demo app.

**Course:** M.Tech (AIML) — Machine Learning (AIMLZG565)  
**Marks:** 15 | **Deadline:** 08-Nov-2026

---

## a. Problem statement

Predict the **fault category** of a steel plate from geometric and luminosity measurements.

This is a **multi-class classification** problem. We train and compare five classifiers on the same dataset and held-out test split, then expose the saved models through an interactive Streamlit application for evaluation on uploaded test CSV files (without retraining).

---

## b. Dataset description

| Item | Details |
|------|---------|
| Dataset | UCI **Steel Plates Faults** |
| Source | https://archive.ics.uci.edu/dataset/198/steel+plates+faults |
| Task type | Multi-class classification |
| Instances | 1,941 (≥ 500 required) |
| Input features | 27 numeric process / geometry attributes (≥ 12 required) |
| Target | One of 7 fault types (Pastry, Z-Scratch, K-Scratch, Stains, Dirtiness, Bumps, Other Faults) |

**Preprocessing (to be documented after training):**

- Stratified train/test split
- Feature scaling where required (Logistic Regression, KNN, SVM)
- Saved preprocessor artifact under `model/`

**Test data for experiments:** `test_data.csv` (added in the data phase).

---

## c. GitHub Repository Link

> _Update after the repository is finalized / made public for evaluation._

`https://github.com/<username>/Machine-Learning-Assignment-2`

**Live Streamlit App:** _to be added after Streamlit Community Cloud deployment_

---

## d. Models used

All five models are trained on the **same** dataset and evaluated with:

Accuracy | AUC | Precision | Recall | F1 | Matthews Correlation Coefficient (MCC)

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | — | — | — | — | — | — |
| Decision Tree | — | — | — | — | — | — |
| KNN | — | — | — | — | — | — |
| Naive Bayes | — | — | — | — | — | — |
| Support Vector Machine | — | — | — | — | — | — |

_Metrics will be filled after model training (weighted averaging for multi-class Precision/Recall/F1; OvR AUC)._

### Observations on model performance

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | _Pending training_ |
| Decision Tree | _Pending training_ |
| KNN | _Pending training_ |
| Naive Bayes | _Pending training_ |
| Support Vector Machine | _Pending training_ |
| **Overall winner for this dataset?** | _Pending training_ |

---

## Project structure

```text
Machine-Learning-Assignment-2/
├── app.py                 # Streamlit application (Phase 3)
├── train_models.py        # Training + evaluation script (Phase 2)
├── requirements.txt       # Python dependencies (Phase 4)
├── README.md
├── test_data.csv          # Held-out test set with target (Phase 1)
├── data/                  # Full dataset
└── model/                 # Saved .joblib models + preprocessor
```

---

## How to run locally (placeholder)

```bash
# After requirements.txt is added
pip install -r requirements.txt
streamlit run app.py
```

---

## Status

- [x] Phase 0 — Project structure and README skeleton
- [ ] Phase 1 — Dataset download, citation, `test_data.csv`
- [ ] Phase 2 — Train 5 models, compute metrics, save artifacts
- [ ] Phase 3 — Streamlit app (upload, dropdown, metrics, confusion matrix)
- [ ] Phase 4 — Requirements freeze + Streamlit Cloud deploy
- [ ] Phase 5 — BITS Virtual Lab screenshot + submission PDF
