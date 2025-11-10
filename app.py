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
    # --- Load pre-op recommender & Postop monitoring (inside submitted block) ---
    import json, os, datetime
    # Try import (will show UI error if missing)
    try:
        from preop_recommender import load_kb, generate_recommendations
        print("✅ preop_recommender imported OK")
        KB = load_kb("knowledge_base.json")
        print(f"✅ Knowledge base loaded: {len(KB)} rules")
    except Exception as e:
        st.error("Failed to import or load Pre-Op recommender. Check console logs.")
        print("❌ Error while importing/loading KB:", e)
        KB = []

    # === Robust prediction safety & risk_output build ===
    # (ensures pred/proba exist even if model prediction failed)
    try:
        # proba and pred already computed above in normal flow; if not, try to compute again
        if 'proba' not in globals() or 'pred' not in globals():
            if 'model' in globals() and 'X_input' in locals():
                proba = model.predict_proba(X_input)[0]
                pred = int(np.argmax(proba))
    except Exception as e:
        print("Warning: prediction compute failed:", e)
        proba = proba if 'proba' in globals() and proba is not None else [0.0, 0.0, 0.0]
        pred = pred if 'pred' in globals() and pred is not None else 0

    # Final defensive defaults
    if 'proba' not in globals() or proba is None:
        proba = [0.0, 0.0, 0.0]
    if 'pred' not in globals() or pred is None:
        pred = 0

    _pred_label = "low" if int(pred) == 0 else ("moderate" if int(pred) == 1 else "high")
    risk_output = {
        "predicted": _pred_label,
        "domain": "general",
        "confidence": float(max(proba)) if hasattr(proba, "__iter__") else float(proba)
    }

    # Build patient_data (use same inputs from the form)
    patient_data = {
        "demographics": {"age": int(age), "sex": str(sex), "bmi": float(bmi)},
        "comorbidities": {
            "diabetes": bool(diabetes),
            "hypertension": bool(hypertension),
            "pci_stent": False
        },
        "medications": {
            "warfarin": False,
            "antiplatelet": False,
            "insulin": bool(diabetes)
        },
        "labs": {"hba1c": None, "hb": float(hb), "creatinine": float(creatinine)},
        "lifestyle": {"smoker": bool(smoking), "alcohol_units_week": 0},
        "surgery": {
            "type": str(surgery_type).lower() if surgery_type else "",
            "subtype": "",
            "urgency": "emergency" if int(emergency) == 1 else "elective"
        }
    }

    # Run recommender (safe even if KB is empty)
    try:
        preop_out = generate_recommendations(patient_data, risk_output, KB)
    except Exception as e:
        st.error("Pre-Op recommender failed (see logs).")
        print("Pre-Op recommender exception:", e)
        preop_out = {
            "doctor_recommendations": [],
            "doctor_recommendations_detailed": [],
            "patient_checklist": [],
            "matched_rule_ids": [],
            "flat_context": {}
        }

    # Display Pre-Op output
    st.markdown("---")
    st.subheader("🩺 Pre-Operative Care Recommendations")

    if preop_out.get("doctor_recommendations_detailed"):
        st.write("**Clinician Recommendations:**")
        for r in preop_out["doctor_recommendations_detailed"]:
            st.write(f"- {r['text']}")
            if r.get("notes"):
                st.caption(f"💡 {r['notes']}")
    else:
        st.info("No clinician recommendations triggered for this patient.")

    if preop_out.get("patient_checklist"):
        st.write("**Patient Checklist:**")
        for item in preop_out["patient_checklist"]:
            st.markdown(f"- {item}")

    # ---------- Streamlit monitoring UI: Step 3/4/5 ----------
    from postop_monitoring import detect_complications_from_vitals_and_notes, create_alert, update_recovery_plan

    # where alerts/bundles will be stored
    ALERT_DIR = "preop_alerts"
    os.makedirs(ALERT_DIR, exist_ok=True)

    st.markdown("## Real-time Post-op Monitoring & Alerts (Steps 3–5)")

   # ---------- Monitoring UI as its own form (recommended) ----------
import json, os
from postop_monitoring import detect_complications_from_vitals_and_notes, create_alert, update_recovery_plan

ALERT_DIR = "preop_alerts"
os.makedirs(ALERT_DIR, exist_ok=True)

st.markdown("## Real-time Post-op Monitoring & Alerts (Steps 3–5)")

with st.form("monitor_form"):
    vitals_json = st.text_area(
        "Paste recent vitals time-series JSON (list of records with timestamp, hr, temp, spo2, ddimer)",
        height=150
    )
    nurse_notes = st.text_area("Nurse notes (free text)", value="", height=100)
    run_monitor = st.form_submit_button("Run complication detector")

if run_monitor:
    # parse vitals
    try:
        vitals = json.loads(vitals_json) if vitals_json.strip() else []
    except Exception as e:
        st.error(f"Invalid vitals JSON: {e}")
        vitals = []

    # call detector
    detect_out = detect_complications_from_vitals_and_notes(vitals=vitals, notes=nurse_notes, extra_labs={})
  # --- Clean, clinician-facing output ---
flags = detect_out.get("flags", {})
summary = detect_out.get("summary", "No summary available")

st.markdown("### 🩺 Monitoring Summary")

if not flags:
    st.success("✅ No complications detected. Continue routine monitoring.")
else:
    # Count the number of triggered complications
    st.warning(f"⚠️ {len(flags)} potential complication(s) detected. Review required.")
    for flag, details in flags.items():
        sev = details.get("severity", "moderate").capitalize()
        action = details.get("recommended_action", "Review patient condition.")
        evidence = details.get("evidence", [])
        st.markdown(f"**{flag.upper()} ({sev})** — {action}")
        if evidence:
            with st.expander("View supporting evidence"):
                for e in evidence:
                    st.write(f"• {e}")
                    # Suggested next steps
if not flags:
    st.info("Continue with standard post-op care. No alerts at this time.")
else:
    st.markdown("### 🧾 Recommended Actions")
    for flag, details in flags.items():
        if "infection" in flag:
            st.write("- Start empirical antibiotics and send wound swab for culture.")
        elif "dvt" in flag:
            st.write("- Perform venous Doppler and initiate DVT prophylaxis.")
        elif "respiratory" in flag:
            st.write("- Start oxygen support and chest physiotherapy.")
        elif "bleeding" in flag:
            st.write("- Assess surgical site and order hemoglobin test.")

    # create alerts for flags meeting severity threshold
    alerts = []
    for flag, details in detect_out.get("flags", {}).items():
        if details.get("score", 0) >= 0.5:
            alert = create_alert(patient_id="demo_patient_001", flag_key=flag, flag_details=details)
            alerts.append(alert)
            alert_path = os.path.join(ALERT_DIR, f"{alert['alert_id']}.json")
            with open(alert_path, "w") as f:
                json.dump(alert, f, indent=2)

    if alerts:
        st.success(f"{len(alerts)} alert(s) generated; saved to {ALERT_DIR}")
        for a in alerts:
            st.metric(label=f"ALERT — {a['flag'].upper()}", value=a["severity"])
            st.write("Recommended action:", a["recommended_action"])
            st.write("Evidence:", a["evidence"])
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Acknowledge {a['alert_id']}"):
                    st.info(f"Acknowledged {a['alert_id']}")
            with col2:
                if st.button(f"Escalate {a['alert_id']}"):
                    st.warning(f"Escalated {a['alert_id']} to on-call team (simulation)")
    else:
        st.info("No immediate alerts generated.")

    # Step 5 — update recovery plan
    existing_plan = {}
    updated_plan = update_recovery_plan(existing_plan, detect_out.get("flags", {}), risk_output, surgery_type=patient_data.get("surgery", {}).get("type",""))
    st.write("### Suggested Recovery Plan Updates (Day-wise)")
    for day, items in sorted(updated_plan.items()):
        if items:
            st.write(f"**{day}**")
            for it in items:
                st.markdown(f"- {it}")

    if st.button("Save updated recovery plan (simulate)"):
        save_path = os.path.join(ALERT_DIR, f"recovery_plan_{int(datetime.datetime.utcnow().timestamp())}.json")
        with open(save_path, "w") as f:
            json.dump({"patient_id":"demo_patient_001","plan":updated_plan,"detected":detect_out}, f, indent=2)
        st.success(f"Saved updated plan to {save_path}")
