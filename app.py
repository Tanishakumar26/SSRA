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
    # --- Load pre-op recommender ---
try:
    from preop_recommender import load_kb, generate_recommendations
    print("✅ preop_recommender imported OK")
    KB = load_kb("knowledge_base.json")
    print(f"✅ Knowledge base loaded: {len(KB)} rules")
except Exception as e:
    st.error("Failed to import or load Pre-Op recommender. Check console logs.")
    print("❌ Error while importing/loading KB:", e)
    KB = []

# Prepare patient and risk dicts (you can reuse your form variables)
patient_data = {
    "demographics": {"age": age, "sex": sex, "bmi": bmi},
    "comorbidities": {
        "diabetes": bool(diabetes),
        "hypertension": bool(hypertension)
    },
    "medications": {
        "warfarin": False,
        "antiplatelet": False,
        "insulin": bool(diabetes)
    },
    "labs": {"hb": hb, "creatinine": creatinine, "hba1c": None},
    "lifestyle": {"smoker": bool(smoking), "alcohol_units_week": 0},
    "surgery": {
        "type": str(surgery_type).lower(),
        "subtype": "",
        "urgency": "emergency" if emergency == 1 else "elective"
    }
}

risk_output = {
    "predicted": ("low" if pred == 0 else "moderate" if pred == 1 else "high"),
    "domain": "general",
    "confidence": float(max(proba))
}

preop_out = generate_recommendations(patient_data, risk_output, KB)

st.markdown("---")
st.subheader("🩺 Pre-Operative Care Recommendations")

# Show doctor recommendations
if preop_out["doctor_recommendations_detailed"]:
    st.write("**Clinician Recommendations:**")
    for r in preop_out["doctor_recommendations_detailed"]:
        st.write(f"- {r['text']}")
        if r["notes"]:
            st.caption(f"💡 {r['notes']}")
else:
    st.info("No clinician recommendations triggered for this patient.")

# Show patient checklist
if preop_out["patient_checklist"]:
    st.write("**Patient Checklist:**")
    for item in preop_out["patient_checklist"]:
        st.markdown(f"- {item}")
        # ---------- Streamlit monitoring UI: Step 3/4/5 ----------
from postop_monitoring import detect_complications_from_vitals_and_notes, create_alert, update_recovery_plan
import streamlit as st
import json, os

# where alerts/bundles will be stored
ALERT_DIR = "preop_alerts"
os.makedirs(ALERT_DIR, exist_ok=True)

st.markdown("## Real-time Post-op Monitoring & Alerts (Steps 3–5)")

# Simulated input: in real app, you should fetch live vitals for the patient (time-series)
# For demo, allow user to paste JSON or use the synthetic CSV row loaded earlier
vitals_json = st.text_area("Paste recent vitals time-series JSON (list of records with timestamp, hr, temp, spo2, ddimer)", height=150)
nurse_notes = st.text_area("Nurse notes (free text)", value="", height=100)

if st.button("Run complication detector"):
    try:
        vitals = json.loads(vitals_json) if vitals_json.strip() else []
    except Exception as e:
        st.error(f"Invalid vitals JSON: {e}")
        vitals = []

    # call detector
    detect_out = detect_complications_from_vitals_and_notes(vitals=vitals, notes=nurse_notes, extra_labs={})
    st.write("### Detection output")
    st.json(detect_out)

    # create alerts for flags meeting severity threshold
    alerts = []
    for flag, details in detect_out["flags"].items():
        # you can tune which severities cause immediate alerts; here we alert on moderate+ or score>=0.5
        if details.get("score", 0) >= 0.5:
            alert = create_alert(patient_id="demo_patient_001", flag_key=flag, flag_details=details)
            alerts.append(alert)
            # persist alert
            alert_path = os.path.join(ALERT_DIR, f"{alert['alert_id']}.json")
            with open(alert_path, "w") as f:
                json.dump(alert, f, indent=2)
    if alerts:
        st.success(f"{len(alerts)} alert(s) generated; saved to {ALERT_DIR}")
        for a in alerts:
            st.metric(label=f"ALERT — {a['flag'].upper()}", value=a["severity"])
            st.write("Recommended action:", a["recommended_action"])
            st.write("Evidence:", a["evidence"])
            # optional clinician action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Acknowledge {a['alert_id']}"):
                    st.info(f"Acknowledged {a['alert_id']}")
            with col2:
                if st.button(f"Escalate {a['alert_id']}"):
                    st.warning(f"Escalated {a['alert_id']} to on-call team (simulation)")
    else:
        st.info("No immediate alerts generated.")

    # Step 5 — update recovery plan (you can pass existing plan if you have one)
    existing_plan = {}  # in real app, load current care plan for patient
    updated_plan = update_recovery_plan(existing_plan, detect_out["flags"], risk_output, surgery_type=patient_data.get("surgery", {}).get("type",""))
    st.write("### Suggested Recovery Plan Updates (Day-wise)")
    for day, items in sorted(updated_plan.items()):
        if items:
            st.write(f"**{day}**")
            for it in items:
                st.markdown(f"- {it}")

    # option to save updated plan (clinician acceptance)
    if st.button("Save updated recovery plan (simulate)"):
        save_path = os.path.join(ALERT_DIR, f"recovery_plan_{int(datetime.datetime.utcnow().timestamp())}.json")
        with open(save_path, "w") as f:
            json.dump({"patient_id":"demo_patient_001","plan":updated_plan,"detected":detect_out}, f, indent=2)
        st.success(f"Saved updated plan to {save_path}")
