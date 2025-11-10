# app.py - SSRA (risk prediction + preop recommender + monitoring + dashboard)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import re
import datetime
from typing import Dict, Any, List

# Local detector/recommender modules
# postop_monitoring must exist in repo
from postop_monitoring import detect_complications_from_vitals_and_notes, create_alert, update_recovery_plan

# ---------------- Config ----------------
PATIENT_DB_PATH = "patients_db.json"
ALERT_DIR = "preop_alerts"
os.makedirs(ALERT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PATIENT_DB_PATH) or ".", exist_ok=True)

# ---------------- Utility: patient DB helpers ----------------
def load_patient_db() -> List[Dict[str,Any]]:
    if not os.path.exists(PATIENT_DB_PATH):
        return []
    try:
        with open(PATIENT_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_patient_db(db: List[Dict[str,Any]]):
    with open(PATIENT_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, default=str)

def make_patient_id() -> str:
    stamp = int(datetime.datetime.utcnow().timestamp() * 1000)
    return f"patient_{stamp}"

def add_patient_record(patient_id: str,
                       patient_data: Dict[str,Any],
                       risk_output: Dict[str,Any],
                       preop_out: Dict[str,Any] = None,
                       alerts: List[Dict[str,Any]] = None,
                       recovery_plan: Dict[str,Any] = None):
    db = load_patient_db()
    rec = next((r for r in db if r.get("patient_id") == patient_id), None)
    now = datetime.datetime.utcnow().isoformat()
    event = {
        "timestamp": now,
        "patient_data": patient_data,
        "risk_output": risk_output,
        "preop_out": preop_out or {},
        "alerts": alerts or [],
        "recovery_plan": recovery_plan or {}
    }
    if rec is None:
        db.append({
            "patient_id": patient_id,
            "created_at": now,
            "last_updated": now,
            "events": [event]
        })
    else:
        rec.setdefault("events", []).append(event)
        rec["last_updated"] = now
    save_patient_db(db)
    return True

def update_patient_with_monitoring(patient_id: str, alerts: List[Dict[str,Any]], recovery_plan: Dict[str,Any], detect_out: Dict[str,Any]=None):
    db = load_patient_db()
    rec = next((r for r in db if r.get("patient_id") == patient_id), None)
    now = datetime.datetime.utcnow().isoformat()
    event = {
        "timestamp": now,
        "monitoring": {
            "alerts": alerts or [],
            "recovery_plan": recovery_plan or {},
            "detector_output": detect_out or {}
        }
    }
    if rec is None:
        db.append({
            "patient_id": patient_id,
            "created_at": now,
            "last_updated": now,
            "events": [event]
        })
    else:
        rec.setdefault("events", []).append(event)
        rec["last_updated"] = now
    save_patient_db(db)
    return True

# ---------------- UI helpers ----------------
def patient_view_ui(patient_id: str):
    """
    Show full patient record (events). Called when a patient is selected in dashboard.
    """
    db = load_patient_db()
    patient_rec = next((x for x in db if x["patient_id"] == patient_id), None)
    if not patient_rec:
        st.error("Selected patient not found.")
        return

    st.markdown(f"## Patient: `{patient_rec['patient_id']}`")
    st.write(f"Created: {patient_rec.get('created_at')}  —  Last updated: {patient_rec.get('last_updated')}")
    st.markdown("---")

    for ev in patient_rec.get("events", []):
        st.markdown(f"**Event @ {ev.get('timestamp', '')}**")
        if ev.get("risk_output"):
            ro = ev["risk_output"]
            st.write("**Risk Output**:")
            st.write(f"- Predicted: **{ro.get('predicted', '')}**  (confidence: {ro.get('confidence')})")
        if ev.get("patient_data"):
            pdemo = ev["patient_data"].get("demographics", {})
            st.write("**Demographics**:", f"age={pdemo.get('age')}, sex={pdemo.get('sex')}, bmi={pdemo.get('bmi')}")
            st.write("**Surgery**:", ev["patient_data"].get("surgery", {}))
        if ev.get("preop_out"):
            po = ev["preop_out"]
            if po.get("doctor_recommendations_detailed"):
                st.write("**Clinician recommendations (pre-op):**")
                for r in po.get("doctor_recommendations_detailed", []):
                    st.write(f"- {r.get('text')}")
            if po.get("patient_checklist"):
                st.write("**Patient checklist (pre-op):**")
                for it in po.get("patient_checklist", []):
                    st.write(f"- {it}")
        if ev.get("alerts"):
            st.write("**Alerts (on save):**")
            for a in ev.get("alerts", []):
                st.write(f"- {a.get('flag', '').upper()} ({a.get('severity', '')}) — {a.get('recommended_action', '')}")
                if a.get("evidence"):
                    with st.expander("Evidence"):
                        for e in a["evidence"]:
                            st.write(f"• {e}")
        if ev.get("monitoring"):
            mon = ev["monitoring"]
            if mon.get("alerts"):
                st.write("**Monitoring Alerts**:")
                for a in mon["alerts"]:
                    st.write(f"- {a.get('flag', '').upper()} ({a.get('severity')}) — {a.get('recommended_action')}")
            if mon.get("recovery_plan"):
                st.write("**Recovery plan snapshot:**")
                for d in sorted(mon["recovery_plan"].keys()):
                    st.write(f"- {d}:")
                    for item in mon["recovery_plan"][d]:
                        st.write(f"    - {item}")
            if mon.get("detector_output"):
                with st.expander("View raw detector output (toggle)"):
                    st.json(mon["detector_output"])

    st.markdown("---")
    st.download_button(
        label="📥 Download full patient record (JSON)",
        data=json.dumps(patient_rec, indent=2, default=str),
        file_name=f"{patient_rec['patient_id']}.json",
        mime="application/json"
    )

def patient_dashboard_ui():
    """
    Clean dashboard with search/filter and "Create new patient" button.
    """
    st.markdown("---")
    st.header("📋 Patient Dashboard")

    db = load_patient_db() or []

    # Top row: create new patient + search
    col_left, col_right = st.columns([2, 3])
    with col_left:
        st.markdown("**Create new patient**")
        new_label = st.text_input("Optional short label", value="", placeholder="e.g. John Doe (ward 5)", key="db_new_label")
        if st.button("➕ Create new patient"):
            pid = make_patient_id()
            now = datetime.datetime.utcnow().isoformat()
            skeleton_patient_data = {
                "demographics": {"age": None, "sex": "", "bmi": None, "label": new_label or ""},
                "comorbidities": {},
                "medications": {},
                "labs": {},
                "lifestyle": {},
                "surgery": {}
            }
            skeleton_risk_output = {"predicted": "unknown", "domain": "general", "confidence": 0.0}
            try:
                add_patient_record(patient_id=pid, patient_data=skeleton_patient_data, risk_output=skeleton_risk_output, preop_out={}, alerts=[], recovery_plan={})
                st.success(f"Created patient: {pid}")
                st.session_state["_selected_patient"] = pid
                st.session_state["last_patient_id"] = pid
                # re-run so that selection opens
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to create patient: {e}")
                return

    with col_right:
        st.markdown("**Search / filter patients**")
        search_term = st.text_input("Search by patient id or label (substring)", value="", placeholder="type to filter...", key="db_search")

    if not db:
        st.info("No patient records yet. Create one using the controls above or run 'Predict Risk' (which saves events).")
        return

    # Apply filter
    def matches_filter(rec, term):
        if not term:
            return True
        term = term.strip().lower()
        if term in rec.get("patient_id", "").lower():
            return True
        # check label in last event
        for ev in rec.get("events", []):
            pdemo = ev.get("patient_data", {}).get("demographics", {})
            label = pdemo.get("label", "") if isinstance(pdemo, dict) else ""
            if label and term in str(label).lower():
                return True
        return False

    filtered = [r for r in db if matches_filter(r, search_term)]
    if not filtered:
        st.info("No matching patients for that search.")
        return

    db_sorted = sorted(filtered, key=lambda r: r.get("last_updated", ""), reverse=True)

    # Header row
    cols = st.columns([3, 1, 1])
    cols[0].markdown("**Patient ID**")
    cols[1].markdown("**Created**")
    cols[2].markdown("**Last updated**")

    for r in db_sorted:
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            label = ""
            if r.get("events"):
                last_ev = r["events"][-1]
                pdemo = last_ev.get("patient_data", {}).get("demographics", {})
                label = pdemo.get("label", "") if isinstance(pdemo, dict) else ""
            display_text = r["patient_id"] + (f" — {label}" if label else "")
            if st.button(display_text, key=f"open_{r['patient_id']}"):
                st.session_state["_selected_patient"] = r["patient_id"]
                st.session_state["last_patient_id"] = r["patient_id"]
                st.experimental_rerun()
        with c2:
            c2.write(r.get("created_at", ""))
        with c3:
            c3.write(r.get("last_updated", ""))

    # Show patient view if selected
    selected = st.session_state.get("_selected_patient", None)
    if selected:
        patient_view_ui(selected)

# ---------------- App init ----------------
st.set_page_config(page_title="SSRA — Surgical Risk & Preop", layout="wide")
st.title("🩺 SSRA - Surgical Risk Prediction & Pre-op Recommendations")

# session state keys
if "patient_data" not in st.session_state:
    st.session_state["patient_data"] = None
if "risk_output" not in st.session_state:
    st.session_state["risk_output"] = None
if "last_patient_id" not in st.session_state:
    st.session_state["last_patient_id"] = None
if "_selected_patient" not in st.session_state:
    st.session_state["_selected_patient"] = None

# Sidebar toggle for dashboard
if st.sidebar.checkbox("Show Patient Dashboard"):
    patient_dashboard_ui()

# ---------------- Load ML artifacts ----------------
try:
    preprocessor = joblib.load("ssra_preprocessor.pkl")
except Exception as e:
    st.warning(f"Preprocessor not loaded: {e}")
    preprocessor = None

try:
    model = joblib.load("ssra_xgb_model.pkl")
except Exception as e:
    st.warning(f"Model not loaded: {e}")
    model = None

# ---------------- Patient input form (Module 1) ----------------
with st.form("patient_form"):
    st.markdown("### Patient / Surgical details (Module 1)")
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

# ---------------- Handle Predict Risk submission ----------------
if submitted:
    # Build DataFrame
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

    if preprocessor is None or model is None:
        st.error("Model or preprocessor not loaded; cannot run prediction.")
    else:
        try:
            X_input = preprocessor.transform(input_df)
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
                "comorbidities": {"diabetes": bool(diabetes), "hypertension": bool(hypertension)},
                "medications": {"warfarin": False, "antiplatelet": False, "insulin": bool(diabetes)},
                "labs": {"hba1c": None, "hb": float(hb), "creatinine": float(creatinine)},
                "lifestyle": {"smoker": bool(smoking), "alcohol_units_week": 0},
                "surgery": {"type": str(surgery_type).lower() if surgery_type else "", "subtype": "", "urgency": "emergency" if int(emergency) == 1 else "elective"}
            }
        except Exception:
            patient_data = {"demographics": {"age": None, "sex": "", "bmi": None}, "comorbidities": {}, "medications": {}, "labs": {}, "lifestyle": {}, "surgery": {}}

        _pred_label = "low" if int(pred) == 0 else ("moderate" if int(pred) == 1 else "high")
        risk_output = {"predicted": _pred_label, "domain": "general", "confidence": float(max(proba)) if hasattr(proba, "__iter__") else float(proba)}

        # persist in session_state
        st.session_state["patient_data"] = patient_data
        st.session_state["risk_output"] = risk_output

        # Try to load KB and run preop recommender (optional)
        try:
            from preop_recommender import load_kb, generate_recommendations
            KB = load_kb("knowledge_base.json")
        except Exception:
            KB = []

        try:
            preop_out = generate_recommendations(patient_data, risk_output, KB) if KB else {}
        except Exception:
            preop_out = {}

        st.markdown("---")
        st.subheader("🩺 Pre-Operative Care Recommendations")
        if preop_out and preop_out.get("doctor_recommendations_detailed"):
            st.write("**Clinician Recommendations:**")
            for r in preop_out["doctor_recommendations_detailed"]:
                st.write(f"- {r['text']}")
                if r.get("notes"):
                    st.caption(f"💡 {r['notes']}")
        else:
            st.info("No clinician recommendations triggered for this patient.")

        if preop_out and preop_out.get("patient_checklist"):
            st.write("**Patient Checklist:**")
            for item in preop_out["patient_checklist"]:
                st.markdown(f"- {item}")

        # Persist the patient event to DB (new patient if not present)
        patient_id = st.session_state.get("last_patient_id") or make_patient_id()
        st.session_state["last_patient_id"] = patient_id
        try:
            add_patient_record(patient_id=patient_id, patient_data=patient_data, risk_output=risk_output, preop_out=preop_out or {}, alerts=[], recovery_plan={})
            st.info(f"Saved patient event to DB: {patient_id}")
        except Exception as e:
            st.error(f"Failed to save patient to DB: {e}")

# ---------------- Monitoring UI (Module 2/3) ----------------
st.markdown("## Real-time Post-op Monitoring & Alerts (Steps 3–5)")

with st.form("monitor_form"):
    vitals_json = st.text_area(
        "Paste recent vitals time-series JSON (list of records with timestamp, hr, temp, spo2, ddimer) OR paste free text like 'HR 105, Temp 100.8, D-dimer 1.2'",
        height=150
    )
    nurse_notes = st.text_area("Nurse notes (free text)", value="", height=100)
    run_monitor = st.form_submit_button("Run complication detector")

def _parse_vitals_input(text: str):
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
        vitals = _parse_vitals_input(vitals_json)

        # determine patient_data & risk_output (prefer session)
        if st.session_state.get("patient_data"):
            patient_data = st.session_state["patient_data"]
        else:
            # minimal fallback
            patient_data = {"demographics": {"age": None, "sex": "", "bmi": None}, "surgery": {}}

        if st.session_state.get("risk_output"):
            risk_output = st.session_state["risk_output"]
        else:
            risk_output = {"predicted": "low", "domain": "general", "confidence": 0.0}

        # run detector
        detect_out = detect_complications_from_vitals_and_notes(vitals=vitals, notes=nurse_notes, extra_labs={})

        # build alerts and persist locally
        alerts = []
        for flag, details in detect_out.get("flags", {}).items():
            if details.get("score", 0) >= 0.5:
                alert = create_alert(patient_id=st.session_state.get("last_patient_id") or "demo_patient_001", flag_key=flag, flag_details=details)
                alerts.append(alert)
                alert_path = os.path.join(ALERT_DIR, f"{alert['alert_id']}.json")
                with open(alert_path, "w", encoding="utf-8") as f:
                    json.dump(alert, f, indent=2)

        # simplified UI
        st.markdown("### 🩺 Monitoring Summary")
        flags = detect_out.get("flags", {})
        if not flags:
            st.success("✅ No complications detected. Continue routine monitoring.")
            st.info("Continue with standard post-op care. No alerts at this time.")
        else:
            st.warning(f"⚠️ {len(flags)} potential complication(s) detected. Review required.")
            for flag, details in flags.items():
                sev = details.get("severity", "moderate").capitalize()
                action_text = details.get("recommended_action", "Review patient condition.")
                evidence = details.get("evidence", [])
                st.markdown(f"**{flag.upper()} ({sev})** — {action_text}")
                if evidence:
                    with st.expander("💡 View supporting evidence"):
                        for e in evidence:
                            st.write(f"• {e}")

        if alerts:
            st.success(f"{len(alerts)} alert(s) generated; saved to {ALERT_DIR}")
            for a in alerts:
                st.metric(label=f"ALERT — {a['flag'].upper()}", value=a.get("severity", ""))
                st.write(a.get("recommended_action", "See evidence"))
                with st.expander(f"Evidence for {a['flag']}"):
                    for ev in a.get("evidence", []):
                        st.write(f"• {ev}")
                st.caption(f"Alert ID: {a['alert_id']} — generated at {a.get('timestamp')}")

        # update recovery plan
        existing_plan = {}
        try:
            surg_type = patient_data.get("surgery", {}).get("type", "") if isinstance(patient_data, dict) else ""
            updated_plan = update_recovery_plan(existing_plan, detect_out.get("flags", {}), risk_output, surgery_type=surg_type)
        except Exception as e:
            st.error(f"Error updating recovery plan: {e}")
            updated_plan = {}

        if updated_plan:
            st.write("### Suggested Recovery Plan Updates (Day-wise)")
            for day, items in sorted(updated_plan.items()):
                if items:
                    st.write(f"**{day}**")
                    for it in items:
                        st.markdown(f"- {it}")

            if st.button("Save updated recovery plan (simulate)"):
                save_path = os.path.join(ALERT_DIR, f"recovery_plan_{int(datetime.datetime.utcnow().timestamp())}.json")
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump({"patient_id": st.session_state.get("last_patient_id") or "demo_patient_001", "plan": updated_plan, "detected": detect_out}, f, indent=2)
                st.success(f"Saved updated plan to {save_path}")

        # append monitoring event to patient DB
        try:
            patient_id = st.session_state.get("last_patient_id") or make_patient_id()
            st.session_state["last_patient_id"] = patient_id
            update_patient_with_monitoring(patient_id=patient_id, alerts=alerts, recovery_plan=updated_plan, detect_out=detect_out)
            st.info(f"Monitoring event appended to patient {patient_id}")
        except Exception as e:
            st.warning("Failed to append monitoring event to DB.")
            print("update_patient_with_monitoring error:", e)

    except Exception as e:
        st.error(f"⚠️ Error while running detector: {e}")
        print("Detector error:", e)
