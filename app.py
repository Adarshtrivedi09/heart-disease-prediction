"""
Streamlit Interface for Heart Disease Prediction Mini Project
Input codes follow the UCI Heart Disease (Cleveland) dataset used for training.
"""

import pickle

import pandas as pd
import streamlit as st


@st.cache_resource
def load_saved_model():
    """Load the saved Pipeline once (no retraining on any click)."""
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)


try:
    model = load_saved_model()
except FileNotFoundError:
    st.error("model.pkl not found. Run `python train_model.py` first.")
    st.stop()

st.title("🩺 Heart Disease Risk Assessment Tool")
st.write("An academic machine learning prototype that estimates heart disease risk "
         "from clinical measurements.")

st.sidebar.header("Patient Inputs")


def label_select(title, mapping):
    """Selectbox that shows text but returns the numeric code the model was trained on."""
    return st.sidebar.selectbox(title, list(mapping.keys()), format_func=lambda k: mapping[k])


def get_user_inputs():
    age = st.sidebar.slider("Age (years)", 29, 77, 54)
    sex = label_select("Sex", {1: "Male", 0: "Female"})
    cp = label_select("Chest Pain Type", {
        1: "Typical angina", 2: "Atypical angina",
        3: "Non-anginal pain", 4: "Asymptomatic"})
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 94, 200, 130)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 126, 564, 246)
    fbs = label_select("Fasting Blood Sugar > 120 mg/dl", {0: "No", 1: "Yes"})
    restecg = label_select("Resting ECG Result", {
        0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"})
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 71, 202, 150)
    exang = label_select("Exercise Induced Angina", {0: "No", 1: "Yes"})
    oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.2, 1.0, 0.1)
    slope = label_select("Slope of Peak Exercise ST Segment", {
        1: "Upsloping", 2: "Flat", 3: "Downsloping"})
    ca = label_select("Major Vessels Coloured by Fluoroscopy", {
        0: "0", 1: "1", 2: "2", 3: "3"})
    thal = label_select("Thalassemia", {
        3: "Normal", 6: "Fixed defect", 7: "Reversible defect"})

    # Column names and order must match the training data.
    return pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal,
    }])


input_df = get_user_inputs()

st.subheader("Submitted Parameters")
st.dataframe(input_df)

if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"Higher risk of heart disease predicted "
                     f"(model probability: {probability:.1%})")
        else:
            st.success(f"Lower risk of heart disease predicted "
                       f"(model probability of disease: {probability:.1%})")
    except Exception as err:
        st.error(f"Prediction failed: {err}")

st.markdown("---")
st.markdown("### ⚠️ Limitations")
st.warning(
    "This is an educational machine learning prototype trained on a small dataset "
    "(about 300 patients). It is NOT a diagnostic tool and must not be used for "
    "medical decisions. Consult a qualified doctor for any health concern."
)