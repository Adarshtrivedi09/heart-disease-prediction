"""
Heart Disease Prediction Mini Project - Model Training Script
Author: Adarsh Trivedi
Description: Compares three models with 5-fold cross-validation (training data only),
tunes the best one with GridSearchCV, evaluates once on the untouched test set,
and saves the final fitted Pipeline with pickle.
Dataset: UCI Heart Disease (Cleveland) - https://archive.ics.uci.edu/dataset/45/heart+disease
"""

import os
import pickle

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/processed.cleveland.data")
CSV_PATH = "data/dataset.csv"
RANDOM_STATE = 42


# ---------------------------------------------------------------- 1. DATA
def load_data():
    """Load the cleaned CSV; if missing, download from UCI, clean and save it."""
    os.makedirs("data", exist_ok=True)
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)

    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(DATA_URL, names=column_names, na_values='?')
    df = df.dropna()                                   # only a few rows have '?'
    df['target'] = (df['target'] > 0).astype(int)      # 0 = no disease, 1-4 -> 1 = disease
    df.to_csv(CSV_PATH, index=False)
    return df


df = load_data()
X = df.drop(columns=['target'])
y = df['target']

# Test set is kept aside and used ONLY once, at the very end.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Training records: {X_train.shape[0]} | Test records: {X_test.shape[0]}")

# ------------------------------------------------------- 2. PREPROCESSING
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numeric_features = [c for c in X.columns if c not in categorical_features]


def build_pipeline(model):
    """Fresh preprocessing + model, so every model gets its own untouched copy."""
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ])
    return Pipeline([('preprocessor', preprocessor), ('classifier', model)])


# ------------------------------------- 3. MODEL COMPARISON (CV, train only)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "K-Nearest Neighbours": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
}

scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_summary = {}

print("\n--- 5-Fold CV comparison (training data only) ---")
for name, model in models.items():
    result = cross_validate(build_pipeline(model), X_train, y_train, cv=5, scoring=scoring)
    cv_summary[name] = {m: result[f'test_{m}'].mean() for m in scoring}
    print(f"{name:22s} " + " | ".join(f"{m}: {v:.3f}" for m, v in cv_summary[name].items()))

# Pick the model to tune using mean CV F1 (balances precision and recall).
best_name = max(cv_summary, key=lambda n: cv_summary[n]['f1'])
print(f"\nSelected for tuning (highest mean CV F1): {best_name}")

# ------------------------------------------ 4. HYPERPARAMETER TUNING
param_grids = {
    "Logistic Regression": {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
    },
    "K-Nearest Neighbours": {
        'classifier__n_neighbors': [3, 5, 7, 9, 11],
        'classifier__weights': ['uniform', 'distance'],
    },
    "Random Forest": {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__max_depth': [None, 5, 10],
        'classifier__min_samples_split': [2, 5],
    },
}

grid_search = GridSearchCV(
    estimator=build_pipeline(models[best_name]),
    param_grid=param_grids[best_name],
    cv=5,
    scoring='f1',
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best mean CV F1: {grid_search.best_score_:.3f}")

# ---------------------------------------- 5. FINAL TEST (used only once)
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

print("\n--- Final Test Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")
print("\nConfusion Matrix (rows = actual, cols = predicted):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------------------------------- 6. SAVE PIPELINE
with open('model.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)

print("Pipeline saved to model.pkl")
print(f"Trained with scikit-learn {sklearn.__version__} -> pin this in requirements.txt")