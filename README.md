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
| Source URL | https://archive.ics.uci.edu/dataset/198/steel+plates+faults |
| Direct download | https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip |
| Task type | Multi-class classification |
| Instances | 1,941 (≥ 500 required) |
| Input features | 27 numeric process / geometry attributes (≥ 12 required) |
| Target column | `Fault_Type` — one of 7 fault types |
| Citation | Dataset donated to the UCI Machine Learning Repository (Dataset ID 198) |

### Target classes (full dataset)

| Fault type | Count |
|------------|------:|
| Other_Faults | 673 |
| Bumps | 402 |
| K_Scatch | 391 |
| Z_Scratch | 190 |
| Pastry | 158 |
| Stains | 72 |
| Dirtiness | 55 |

### Feature groups (27 inputs)

- Geometry / location: `X_Minimum`, `X_Maximum`, `Y_Minimum`, `Y_Maximum`, `Pixels_Areas`, `X_Perimeter`, `Y_Perimeter`, …
- Luminosity: `Sum_of_Luminosity`, `Minimum_of_Luminosity`, `Maximum_of_Luminosity`, `Luminosity_Index`
- Process / steel type: `Length_of_Conveyer`, `TypeOfSteel_A300`, `TypeOfSteel_A400`, `Steel_Plate_Thickness`
- Shape / index features: `Edges_Index`, `Empty_Index`, `Square_Index`, `Orientation_Index`, `SigmoidOfAreas`, and related log/index columns

### Preprocessing for this assignment

- Converted the original 7 one-hot fault indicator columns into a single `Fault_Type` label
- Stratified train/test split (**80% / 20%**, `random_state=42`)
- Held-out test set saved as `test_data.csv` (389 rows; includes all features + target)
- Full cleaned dataset: `data/steel_plates_faults.csv`
- Matching train split: `data/train_data.csv`
- Reproduce with: `python prepare_data.py`
- Feature scaling (for LR / KNN / SVM) will be fit on the training split only during model training

---

## c. GitHub Repository Link

https://github.com/Asta436/Machine-Learning-Assignment-2

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
├── app.py                 # Streamlit application (Step 3)
├── prepare_data.py        # Download UCI data + stratified split
├── train_models.py        # Training + evaluation script (Step 2)
├── requirements.txt       # Python dependencies (Step 4)
├── README.md
├── test_data.csv          # Held-out test set with target
├── data/
│   ├── steel_plates_faults.csv
│   └── train_data.csv
└── model/                 # Saved .joblib models + preprocessor
```

---

## How to run locally (placeholder)

```bash
# After requirements.txt is added
pip install -r requirements.txt
python prepare_data.py
streamlit run app.py
```

---

## Status

- [x] Step 0 — Project structure and README skeleton
- [x] Step 1 — Dataset download, citation, `test_data.csv`
- [ ] Step 2 — Train 5 models, compute metrics, save artifacts
- [ ] Step 3 — Streamlit app (upload, dropdown, metrics, confusion matrix)
- [ ] Step 4 — Requirements freeze + Streamlit Cloud deploy
- [ ] Step 5 — BITS Virtual Lab screenshot + submission PDF
