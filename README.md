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
- `StandardScaler` + `LabelEncoder` fitted **only on the training split** and saved as `model/preprocessor.joblib`
- Evaluation protocol: weighted Precision / Recall / F1; one-vs-rest weighted AUC; multiclass MCC
- Note: **weighted Recall equals Accuracy** by definition, so those two columns match in the comparison table (not a reporting error)

---

## c. GitHub Repository Link

https://github.com/Asta436/Machine-Learning-Assignment-2

**Live Streamlit App:** _to be added after Streamlit Community Cloud deployment_

---

## d. Models used

All five models are trained on the **same** dataset and evaluated on the **same** held-out `test_data.csv` (no retraining on test).

Accuracy | AUC | Precision | Recall | F1 | Matthews Correlation Coefficient (MCC)

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.7275 | 0.9066 | 0.7329 | 0.7275 | 0.7277 | 0.6487 |
| Decision Tree | 0.7481 | 0.8811 | 0.7506 | 0.7481 | 0.7464 | 0.6723 |
| KNN | 0.7558 | 0.9038 | 0.7521 | 0.7558 | 0.7522 | 0.6884 |
| Naive Bayes | 0.6093 | 0.8840 | 0.6700 | 0.6093 | 0.5883 | 0.5484 |
| Support Vector Machine | 0.7635 | 0.9298 | 0.7672 | 0.7635 | 0.7644 | 0.6949 |

Source of truth for these numbers: `reports/metrics.csv`.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Strong probabilistic baseline (AUC 0.91). Linear decision boundaries are competitive but slightly behind non-linear neighbors/SVM on accuracy and MCC. |
| Decision Tree | Higher accuracy than LR with interpretable splits, but the lowest AUC among tree-capable models — depth limits reduce overfit while also limiting ranking quality. |
| KNN | Best accuracy among classical distance/linear baselines after scaling (`k=7`, distance weights). Sensitive to feature scale; benefits clearly from the shared StandardScaler. |
| Naive Bayes | Weakest accuracy / F1 / MCC. Feature independence is a poor fit for correlated geometry and luminosity measurements, even though OvR AUC remains decent. |
| Support Vector Machine | Best overall: highest Accuracy, AUC, Precision, Recall, F1, and MCC. RBF kernel captures non-linear fault boundaries well on standardized inputs. |
| **Overall winner for this dataset?** | **Support Vector Machine (RBF)** — leads on every reported metric on the held-out test set. |

### How to retrain

```bash
python fit_preprocessor.py
python -m ml.models.logistic_regression
python -m ml.models.decision_tree
python -m ml.models.knn
python -m ml.models.naive_bayes
python -m ml.models.svm
# or all at once:
python train_models.py
```

Saved artifacts: `model/*.joblib` (preprocessor + five classifiers).

---

## Project structure

```text
Machine-Learning-Assignment-2/
├── app.py                 # Streamlit application (Step 3)
├── prepare_data.py        # Download UCI data + stratified split
├── fit_preprocessor.py    # Fit train-only scaler / label encoder
├── train_models.py        # Train all five classifiers
├── requirements.txt       # Python dependencies (Step 4)
├── README.md
├── test_data.csv          # Held-out test set with target
├── reports/metrics.csv    # Evaluation comparison table
├── ml/                    # Shared training / metrics code
├── data/
│   ├── steel_plates_faults.csv
│   └── train_data.csv
└── model/                 # Saved .joblib models + preprocessor
```

---

## How to run locally

```bash
pip install -r requirements.txt
python prepare_data.py
python fit_preprocessor.py
python train_models.py
streamlit run app.py
```

### Streamlit app pages

1. **Evaluate one model** — upload/bundled CSV + model dropdown → metrics, classification report, confusion matrix  
2. **Compare all models** — same CSV scored by all five saved models (no retraining)  
3. **Data preview** — shape, class counts, sample rows  
4. **About** — dataset / usage notes  

---

## Status

- [x] Step 0 — Project structure and README skeleton
- [x] Step 1 — Dataset download, citation, `test_data.csv`
- [x] Step 2 — Train 5 models, compute metrics, save artifacts
- [x] Step 3 — Streamlit app (upload, dropdown, metrics, confusion matrix)
- [ ] Step 4 — Requirements freeze + Streamlit Cloud deploy
- [ ] Step 5 — BITS Virtual Lab screenshot + submission PDF
