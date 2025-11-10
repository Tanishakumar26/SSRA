# app.py - SSRA (combined risk prediction + preop recommender + monitoring)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import re
import datetime

# Local modules
from postop_monitoring import detect_complications_from_vitals_and_notes, create_alert, update_recovery_plan

# Load artifacts (make sure filenames match repo)
preprocessor = joblib.load("ssra_preprocessor.pkl")
model = joblib.load("ssra_xgb_model.pkl")

# Initialize session_state keys (optional nice-to-have)
if "patient_data" not in st.session_state:
    st.session_state["patient_data"] = None
if "risk_output" not in st.session_state:
    st.session_state["risk_output"] = None

st.set_page_config(page_title="SSRA — Surgical Risk & Preop", layout="wide")
st.title("🩺 SSRA - Surgical Risk Prediction & Pre-op Recommendations")

# -------------------------
# Patient input form (Module 1)
# -------------------------
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

# -------------------------
# If user submitted the risk form
# -------------------------
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

    # Preprocess input (may raise if preprocessor mismatch)
    try:
        X_input = preprocessor.transform(input_df)
    except Exception as e:
        st.error(f"Preprocessor transform failed: {e}")
        st.stop()

    # Predict
    try:
        proba = model.predict_proba(X_input)[0]
        pred = int(np.argmax(proba))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        proba = [0.0, 0.0, 0.0]
        pred = 0

    risk_map = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}

    st.subheader("Prediction Result")
    st.success(f"Predicted Surgical Risk: {risk_map[pred]}")

    st.write("Class probabilities:")
    st.table({
        "Low Risk (0)": [round(proba[0], 3)],
        "Moderate Risk (1)": [round(proba[1], 3)],
        "High Risk (2)": [round(proba[2], 3)]
    })

    # Build canonical patient_data & risk_output
    try:
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
    except Exception as e:
        st.error(f"Failed to build patient_data: {e}")
        patient_data = {
            "demographics": {"age": None, "sex": "", "bmi": None},
            "comorbidities": {}, "medications": {}, "labs": {}, "lifestyle": {}, "surgery": {}
        }

    _pred_label = "low" if int(pred) == 0 else ("moderate" if int(pred) == 1 else "high")
    risk_output = {
        "predicted": _pred_label,
        "domain": "general",
        "confidence": float(max(proba)) if hasattr(proba, "__iter__") else float(proba)
    }

    # Persist to session_state so monitoring and other modules can reuse
    st.session_state["patient_data"] = patient_data
    st.session_state["risk_output"] = risk_output

    # --- Load pre-op recommender (safe) ---
    try:
        from preop_recommender import load_kb, generate_recommendations
        KB = load_kb("knowledge_base.json")
    except Exception as e:
        st.error("Failed to import or load Pre-Op recommender. Check console logs.")
        print("❌ Error while importing/loading KB:", e)
        KB = []

    # Run recommender (safe)
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

# -------------------------
# Monitoring UI (Module 2) — works whether or not Predict Risk just ran
# -------------------------
ALERT_DIR = "preop_alerts"
os.makedirs(ALERT_DIR, exist_ok=True)

st.markdown("## Real-time Post-op Monitoring & Alerts (Steps 3–5)")

with st.form("monitor_form"):
    vitals_json = st.text_area(
        "Paste recent vitals time-series JSON (list of records with timestamp, hr, temp, spo2, ddimer) OR paste free text like 'HR 105, Temp 100.8, D-dimer 1.2'",
        height=150
    )
    nurse_notes = st.text_area("Nurse notes (free text)", value="", height=100)
    run_monitor = st.form_submit_button("Run complication detector")

def _parse_vitals_input(text: str):
    """Try parse JSON; else extract vitals from free text using regex. Return list of dicts."""
    if not text or not text.strip():
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        txt = text.lower()
        parsed = {"timestamp": "manual_entry"}
        patterns = {
            "hr": r"hr[:\s]*([0-9]{2,3})",
            "temp": r"temp(?:erature)?[:\s]*([0-9]{2,3}\.?[0-9]*)",
            "spo2": r"spo2[:\s]*([0-9]{2,3})",
            "ddimer": r"d[-\s]*dimer[:\s]*([0-9.]+)"
        }
        for key, pat in patterns.items():
            m = re.search(pat, txt)
            if m:
                try:
                    parsed[key] = float(m.group(1))
                except Exception:
                    pass
        return [parsed] if len(parsed) > 1 else []

if run_monitor:
    try:
        # Parse vitals (JSON or free text)
        vitals = _parse_vitals_input(vitals_json)

        # Determine patient_data and risk_output (prefer session_state)
        if "patient_data" in st.session_state and st.session_state["patient_data"]:
            patient_data = st.session_state["patient_data"]
        else:
            # if module-1 not run, try to build minimal patient_data from any local variables
            try:
                patient_data = {
                    "demographics": {
                        "age": int(age) if 'age' in locals() else None,
                        "sex": str(sex) if 'sex' in locals() else "",
                        "bmi": float(bmi) if 'bmi' in locals() else None
                    },
                    "comorbidities": {
                        "diabetes": bool(diabetes) if 'diabetes' in locals() else False,
                        "hypertension": bool(hypertension) if 'hypertension' in locals() else False
                    },
                    "medications": {
                        "warfarin": False,
                        "antiplatelet": False,
                        "insulin": bool(diabetes) if 'diabetes' in locals() else False
                    },
                    "labs": {
                        "hba1c": None,
                        "hb": float(hb) if 'hb' in locals() else None,
                        "creatinine": float(creatinine) if 'creatinine' in locals() else None
                    },
                    "lifestyle": {
                        "smoker": bool(smoking) if 'smoking' in locals() else False,
                        "alcohol_units_week": 0
                    },
                    "surgery": {
                        "type": str(surgery_type).lower() if 'surgery_type' in locals() else "",
                        "subtype": "",
                        "urgency": "emergency" if ('emergency' in locals() and int(emergency) == 1) else "elective"
                    }
                }
            except Exception:
                patient_data = {"demographics": {"age": None, "sex": "", "bmi": None}, "comorbidities": {}, "medications": {}, "labs": {}, "lifestyle": {}, "surgery": {}}

        if "risk_output" in st.session_state and st.session_state["risk_output"]:
            risk_output = st.session_state["risk_output"]
        else:
            # fallback: try to infer from locals or default to low
            try:
                if 'pred' in locals():
                    _pred_label = "low" if int(pred) == 0 else ("moderate" if int(pred) == 1 else "high")
                elif 'proba' in locals():
                    _pred_label = "low" if int(np.argmax(proba)) == 0 else ("moderate" if int(np.argmax(proba)) == 1 else "high")
                else:
                    _pred_label = "low"
                risk_output = {"predicted": _pred_label, "domain": "general", "confidence": float(max(proba)) if 'proba' in locals() else 0.0}
            except Exception:
                risk_output = {"predicted": "low", "domain": "general", "confidence": 0.0}

        # Call detector
        detect_out = detect_complications_from_vitals_and_notes(vitals=vitals, notes=nurse_notes, extra_labs={})

        # Build alerts and persist
        alerts = []
        for flag, details in detect_out.get("flags", {}).items():
            if details.get("score", 0) >= 0.5:
                alert = create_alert(patient_id="demo_patient_001", flag_key=flag, flag_details=details)
                alerts.append(alert)
                alert_path = os.path.join(ALERT_DIR, f"{alert['alert_id']}.json")
                with open(alert_path, "w", encoding="utf-8") as f:
                    json.dump(alert, f, indent=2)

        # Simplified clinician UI (no raw JSON dump)
        st.markdown("### 🩺 Monitoring Summary")
        flags = detect_out.get("flags", {})

        if not flags:
            st.success("✅ No complications detected. Continue routine monitoring.")
            st.info("Continue with standard post-op care. No alerts at this time.")
        else:
            st.warning(f"⚠️ {len(flags)} potential complication(s) detected. Review required.")
            for flag, details in flags.items():
                sev = details.get("severity", "moderate").capitalize()
                # detector may include recommended_action (but if not, choose a generic text)
                action_text = details.get("recommended_action", "Review patient condition.")
                # evidence may be a list of strings
                evidence = details.get("evidence", [])
                # show concise line
                st.markdown(f"**{flag.upper()} ({sev})** — {action_text}")
                if evidence:
                    with st.expander("💡 View supporting evidence"):
                        for e in evidence:
                            st.write(f"• {e}")

        # Show alerts summary persisted
        if alerts:
            st.success(f"{len(alerts)} alert(s) generated; saved to {ALERT_DIR}")
            for a in alerts:
                st.metric(label=f"ALERT — {a['flag'].upper()}", value=a.get("severity", ""))
                st.write(a.get("recommended_action", "See evidence"))
                with st.expander(f"Evidence for {a['flag']}"):
                    for ev in a.get("evidence", []):
                        st.write(f"• {ev}")
                st.caption(f"Alert ID: {a['alert_id']} — generated at {a.get('timestamp')}")

        # Update recovery plan (uses patient_data and risk_output)
        existing_plan = {}  # in prod, load current plan for patient
        try:
            surg_type = ""
            if isinstance(patient_data, dict):
                surg_type = patient_data.get("surgery", {}).get("type", "")
            updated_plan = update_recovery_plan(existing_plan, detect_out.get("flags", {}), risk_output, surgery_type=surg_type)
        except Exception as e:
            st.error(f"Error updating recovery plan: {e}")
            print("update_recovery_plan error:", e)
            updated_plan = {}

        if updated_plan:
            st.write("### Suggested Recovery Plan Updates (Day-wise)")
            for day, items in sorted(updated_plan.items()):
                if items:
                    st.write(f"**{day}**")
                    for it in items:
                        st.markdown(f"- {it}")

            # Save option (clinician chooses to persist)
            if st.button("Save updated recovery plan (simulate)"):
                save_path = os.path.join(ALERT_DIR, f"recovery_plan_{int(datetime.datetime.utcnow().timestamp())}.json")
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump({"patient_id": "demo_patient_001", "plan": updated_plan, "detected": detect_out}, f, indent=2)
                st.success(f"Saved updated plan to {save_path}")

    except Exception as e:
        st.error(f"⚠️ Error while running detector: {e}")
        print("Detector error:", e)
