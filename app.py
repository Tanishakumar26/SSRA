import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
preprocessor = joblib.load("ssra_preprocessor.pkl")
model = joblib.load("ssra_xgb_model.pkl")

st.title("🩺 SSRA - Surgical Risk Prediction")

# Input form
with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 120, 50)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)
        hb = st.number_input("Hemoglobin (Hb)", 5.0, 20.0, 13.5, step=0.1)
        creatinine = st.number_input("Creatinine", 0.3, 3.0, 1.0, step=0.1)
        asa = st.selectbox("ASA Score", [1, 2, 3, 4, 5])
    with col2:
        surgery_duration = st.number_input("Surgery Duration (minutes)", 1, 600, 90)
        sex = st.selectbox("Sex", ["M", "F"])
        surgery_type = st.text_input("Surgery Type", "Knee Replacement")
        smoking = st.selectbox("Smoking", [0, 1])
        diabetes = st.selectbox("Diabetes", [0, 1])
        hypertension = st.selectbox("Hypertension", [0, 1])
        cardiac_history = st.selectbox("Cardiac History", [0, 1])
        emergency = st.selectbox("Emergency Surgery", [0, 1])
    
    submitted = st.form_submit_button("Predict Risk")

if submitted:
    # Build DataFrame in same format as training
    input_df = pd.DataFrame([{
        "Surgery_Duration": surgery_duration,
        "Age": age,
        "BMI": bmi,
        "Hb": hb,
        "Creatinine": creatinine,
        "ASA": asa,
        "Smoking": smoking,
        "Cardiac_History": cardiac_history,
        "Diabetes": diabetes,
        "Hypertension": hypertension,
        "Emergency": emergency,
        "Sex": sex,
        "Surgery_Type": surgery_type
    }])

    # Preprocess input
    X_input = preprocessor.transform(input_df)

    # Predict
    proba = model.predict_proba(X_input)[0]
    pred = int(np.argmax(proba))

    risk_map = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}

    st.subheader("Prediction Result")
    st.success(f"Predicted Surgical Risk: {risk_map[pred]}")

    st.write("Class probabilities:")
    st.table({
        "Low Risk (0)": [round(proba[0], 3)],
        "Moderate Risk (1)": [round(proba[1], 3)],
        "High Risk (2)": [round(proba[2], 3)]
    })