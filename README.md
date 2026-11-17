# Heart Disease Prediction – Machine Learning Mini Project

An end-to-end machine learning project that predicts the presence of heart disease from clinical measurements, with a Streamlit web app.

**Author:** Adarsh Trivedi & Sajal Khatri
**Live app:** https://heart-disease-prediction-application-as.streamlit.app

> ⚠️ **Disclaimer:** This is an educational prototype built for a college mini project. It is **not** a diagnostic tool and must not be used for medical decisions.

## Problem Statement
- **Prediction target:** presence of heart disease (`0` = no disease, `1` = disease)
- **Inputs:** 13 clinical features (age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST depression, ST slope, number of major vessels, thalassemia)
- **Task type:** binary classification
- **Output meaning:** the app shows a lower or higher predicted risk with the model's probability

## Dataset
- **Source:** UCI Machine Learning Repository – Heart Disease (Cleveland subset): https://archive.ics.uci.edu/dataset/45/heart+disease
- **Records:** 303 originally; 6 rows with missing values (`ca`: 4, `thal`: 2) were dropped, leaving 297
- **Target:** the original values 0–4 were converted to binary (0 = no disease, 1–4 = disease)
- **Class distribution after cleaning:** 160 no disease (54%), 137 disease (46%)
- **Feature types:** numeric (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) and categorical codes (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`)
- **Duplicates:** none found
- `data/dataset.csv` is the **cleaned** version (missing rows removed, target binarised), not the raw UCI file.

## Method
1. **EDA** (`notebooks/analysis.ipynb`): missing values, duplicates, class balance, feature-vs-target charts, correlation heatmap
2. **Preprocessing** inside a scikit-learn `Pipeline`: `StandardScaler` for numeric features, `OneHotEncoder` for categorical features
3. **Train/test split:** 80/20, stratified, `random_state=42` (237 train / 60 test)
4. **Model comparison** with 5-fold cross-validation on the training set only: Logistic Regression, K-Nearest Neighbours, Random Forest
5. **Tuning:** `GridSearchCV` (`cv=5`, scoring = F1) on the best model by mean CV F1
6. **Final evaluation** once on the held-out test set
7. **Saved model:** the complete fitted pipeline in `model.pkl` (pickle)

## Results

**5-fold CV on the training set**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.835 | 0.848 | 0.787 | 0.811 | 0.898 |
| K-Nearest Neighbours | 0.818 | 0.802 | 0.796 | 0.797 | 0.876 |
| Random Forest | 0.814 | 0.797 | 0.797 | 0.794 | 0.897 |

The models performed similarly. **Logistic Regression** was selected because it scored highest on mean accuracy, F1 and ROC-AUC while being the simplest and most interpretable. GridSearchCV tested `C` in {0.01, 0.1, 1, 10, 100} and selected **C = 0.1**.

**Final test set (60 records)**

| Metric | Value |
|---|---|
| Accuracy | 0.85 |
| ROC-AUC | 0.94 |
| Precision (disease) | 0.88 |
| Recall (disease) | 0.79 |
| F1 (disease) | 0.83 |

Confusion matrix (rows = actual, columns = predicted): `[[29, 3], [6, 22]]`

## Limitations
- Small dataset (297 records) from a single source; the test set has only 60 records, so the results are noisy.
- The model missed 6 of 28 disease cases in the test set (recall 0.79), which matters for a health task.
- The data is mostly male (about 68%) and comes from one location, so the model may not generalise to other populations.
- Not clinically validated; for educational use only.

## Project Structure
```
ML_Mini_Project/
├── data/dataset.csv          # cleaned dataset
├── notebooks/analysis.ipynb  # EDA
├── train_model.py            # comparison, tuning, final test, saves model.pkl
├── app.py                    # Streamlit app
├── model.pkl                 # saved fitted pipeline
├── requirements.txt
└── README.md
```

## How to Run Locally
```bash
pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```
`train_model.py` regenerates `model.pkl`. The app loads this saved pipeline and does not retrain on each prediction.

## Input Encoding (UCI codes)
- `cp`: 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic
- `restecg`: 0 normal, 1 ST-T abnormality, 2 left ventricular hypertrophy
- `slope`: 1 upsloping, 2 flat, 3 downsloping
- `thal`: 3 normal, 6 fixed defect, 7 reversible defect

## Tools
Python, pandas, scikit-learn, Streamlit, Git/GitHub.