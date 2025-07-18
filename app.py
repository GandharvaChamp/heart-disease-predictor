import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and other components
model = joblib.load('logisticRegression.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

# Set page config
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# App title
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("🔎 **Fill out the details below to assess your risk for heart disease.**")

st.subheader("👤 Demographic Information")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=40, step=1)
with col2:
    sex = st.radio("Sex", ['M', 'F'], horizontal=True)

# --- Symptoms and History ---
st.subheader("🫀 Medical & Symptom Details")
col3, col4 = st.columns(2)
with col3:
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dL", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
    exercise_angina = st.radio("Exercise-Induced Angina", ["Y", "N"], horizontal=True)
with col4:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# --- Vitals ---
st.subheader("💉 Vital Signs")
col5, col6 = st.columns(2)
with col5:
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
with col6:
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)

# --- Prediction ---
if st.button("🧠 Predict Risk"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Create and align DataFrame
    input_df = pd.DataFrame([raw_input])
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[columns]

    # Scale and predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    # Display result
    st.markdown("---")
    if prediction == 1:
        st.error("⚠️ **High Risk** of Heart Disease. Please consult a doctor.")
    else:
        st.success("✅ **Low Risk** of Heart Disease. Keep maintaining a healthy lifestyle!")
