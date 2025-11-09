# postop_monitoring.py
from typing import Dict, List, Any, Tuple
import datetime
import re

# ---------- Utilities ----------
def now_iso():
    return datetime.datetime.utcnow().isoformat()

# simple keyword sets for nurse-note NLP (can expand later)
COMPLICATION_KEYWORDS = {
    "dvt": ["swelling", "redness", "calf pain", "leg swelling", "unilateral swelling", "tenderness in leg"],
    "infection": ["fever", "redness", "pus", "drainage", "wound infection", "cellulitis"],
    "bleeding": ["bleeding", "oozing", "hematoma", "excessive bleeding"],
    "respiratory": ["shortness of breath", "desaturation", "desat", "low spo2", "low oxygen", "respiratory distress", "coughing"],
    "sepsis": ["fever", "hypotension", "tachycardia", "confused", "lethargy"]
}

# thresholds (tunable)
THRESHOLDS = {
    "fever_temp": 38.0,        # degrees Celsius
    "tachycardia_hr": 100,     # bpm
    "hypoxia_spo2": 92,        # %
    "elevated_ddimer": 0.5     # units depend on lab; set as example
}

# ---------- Time-series helper ----------
def compute_recent_stats(vitals: List[Dict[str,Any]], window_minutes: int = 60) -> Dict[str,Any]:
    """
    vitals: list of dicts: {"timestamp": iso, "hr": int, "temp": float, "spo2": int, "ddimer": float (opt)}
    returns simple aggregates over last window_minutes (max, mean, last)
    """
    now = datetime.datetime.utcnow()
    window = datetime.timedelta(minutes=window_minutes)
    vals = {"hr": [], "temp": [], "spo2": [], "ddimer": []}
    for rec in vitals:
        try:
            t = datetime.datetime.fromisoformat(rec["timestamp"])
        except Exception:
            # assume rec["timestamp"] already datetime
            t = rec["timestamp"]
        if now - t <= window:
            if rec.get("hr") is not None:
                vals["hr"].append(rec.get("hr"))
            if rec.get("temp") is not None:
                vals["temp"].append(rec.get("temp"))
            if rec.get("spo2") is not None:
                vals["spo2"].append(rec.get("spo2"))
            if rec.get("ddimer") is not None:
                vals["ddimer"].append(rec.get("ddimer"))
    agg = {}
    for k, arr in vals.items():
        if arr:
            agg[f"{k}_last"] = arr[-1]
            agg[f"{k}_max"] = max(arr)
            agg[f"{k}_mean"] = sum(arr)/len(arr)
        else:
            agg[f"{k}_last"] = None
            agg[f"{k}_max"] = None
            agg[f"{k}_mean"] = None
    return agg

# ---------- Simple NLP for nurse notes ----------
def extract_keywords_from_notes(notes: str) -> Dict[str,int]:
    """
    returns counts for each complication keyword bucket
    """
    if not notes:
        return {}
    text = notes.lower()
    hits = {}
    for comp, keywords in COMPLICATION_KEYWORDS.items():
        count = 0
        for kw in keywords:
            # simple phrase search
            if kw in text:
                count += len(re.findall(re.escape(kw), text))
        if count:
            hits[comp] = count
    return hits

# ---------- Complication detection logic ----------
def detect_complications_from_vitals_and_notes(vitals: List[Dict[str,Any]], notes: str, extra_labs: Dict[str,Any]=None) -> Dict[str,Any]:
    """
    Returns a dict:
      {
        "flags": {"dvt": {"score":0.8, "evidence":[...], "severity":"moderate"}, ...},
        "summary": "DVT suspected based on D-dimer + unilateral leg swelling"
      }
    """
    agg = compute_recent_stats(vitals, window_minutes=180)  # 3-hour window by default
    kw_hits = extract_keywords_from_notes(notes)
    flags = {}
    # DVT heuristic: nurse note keyword OR elevated d-dimer OR unilateral swelling mention + calf pain + HR change
    dvt_score = 0.0
    evidence = []
    dd = (extra_labs or {}).get("ddimer") or agg.get("ddimer_last")
    if dd is not None and dd > THRESHOLDS["elevated_ddimer"]:
        dvt_score += 0.5
        evidence.append(f"d-dimer {dd} > {THRESHOLDS['elevated_ddimer']}")
    if "dvt" in kw_hits or "swelling" in kw_hits:
        dvt_score += 0.4
        evidence.append("nurse note mentions swelling / DVT keywords")
    if agg.get("hr_last") is not None and agg["hr_last"] > THRESHOLDS["tachycardia_hr"]:
        dvt_score += 0.05
        evidence.append(f"HR {agg['hr_last']} > {THRESHOLDS['tachycardia_hr']}")
    if dvt_score >= 0.5:
        flags["dvt"] = {"score": round(dvt_score,2), "evidence": evidence, "severity": "high" if dvt_score>0.75 else "moderate"}

    # Infection / sepsis heuristic
    inf_score = 0.0
    inf_evidence = []
    if agg.get("temp_max") and agg["temp_max"] >= THRESHOLDS["fever_temp"]:
        inf_score += 0.5
        inf_evidence.append(f"Temp {agg['temp_max']}°C >= {THRESHOLDS['fever_temp']}")
    if agg.get("hr_max") and agg["hr_max"] >= THRESHOLDS["tachycardia_hr"]:
        inf_score += 0.2
        inf_evidence.append(f"HR {agg['hr_max']} >= {THRESHOLDS['tachycardia_hr']}")
    if "infection" in kw_hits:
        inf_score += 0.4
        inf_evidence.append("nurse note indicates infection keywords")
    if inf_score >= 0.5:
        flags["infection"] = {"score": round(inf_score,2), "evidence": inf_evidence, "severity": "high" if inf_score>0.8 else "moderate"}

    # Respiratory / hypoxia
    resp_score = 0.0
    resp_evidence = []
    if agg.get("spo2_last") is not None and agg["spo2_last"] < THRESHOLDS["hypoxia_spo2"]:
        resp_score += 0.6
        resp_evidence.append(f"SpO2 {agg['spo2_last']}% < {THRESHOLDS['hypoxia_spo2']}%")
    if "respiratory" in kw_hits:
        resp_score += 0.3
        resp_evidence.append("nurse note respiratory keywords")
    if resp_score >= 0.5:
        flags["respiratory"] = {"score": round(resp_score,2), "evidence": resp_evidence, "severity": "high" if resp_score>0.8 else "moderate"}

    # bleeding
    bleed_score = 0.0
    bleed_evidence = []
    if "bleeding" in kw_hits:
        bleed_score += 0.7
        bleed_evidence.append("nurse note mentions bleeding/hematoma")
    if bleed_score >= 0.5:
        flags["bleeding"] = {"score": round(bleed_score,2), "evidence": bleed_evidence, "severity": "high" if bleed_score>0.8 else "moderate"}

    # if none flagged, return low-risk summary
    summary = "No obvious complication flags."
    if flags:
        summary = "; ".join([f"{k.upper()} suspected (score {v['score']})" for k,v in flags.items()])

    return {
        "flags": flags,
        "summary": summary,
        "agg": agg,
        "keywords": kw_hits,
        "timestamp": now_iso()
    }

# ---------- Alert creation ----------
def create_alert(patient_id: str, flag_key: str, flag_details: Dict[str,Any]) -> Dict[str,Any]:
    """
    Creates an alert dict (to be saved to a DB/file or emitted to UI).
    """
    severity = flag_details.get("severity", "moderate")
    text = f"ALERT: {flag_key.upper()} suspected for patient {patient_id} (severity={severity})"
    recs = {
        "dvt": "Start compression therapy and consult vascular surgery/cardiology. Consider duplex ultrasound and therapeutic anticoagulation as per protocol.",
        "infection": "Obtain cultures, start empirical antibiotics per local policy, consider blood tests (CBC, CRP).",
        "respiratory": "Assess airway and oxygen; give supplemental O2 and escalate to ICU if necessary. Consider chest x-ray.",
        "bleeding": "Check vitals, stop anticoagulants if safe, consider surgical review and transfusion."
    }
    action = recs.get(flag_key, "Clinical review required.")
    alert = {
        "alert_id": f"{patient_id}_{flag_key}_{int(datetime.datetime.utcnow().timestamp())}",
        "patient_id": patient_id,
        "flag": flag_key,
        "score": flag_details.get("score"),
        "severity": severity,
        "evidence": flag_details.get("evidence", []),
        "recommended_action": action,
        "timestamp": now_iso()
    }
    return alert

# ---------- Recovery plan update rules ----------
def update_recovery_plan(existing_plan: Dict[str,Any], flags: Dict[str,Any], risk_profile: Dict[str,Any], surgery_type: str) -> Dict[str,Any]:
    """
    existing_plan: dict with keys like "day_1": ["walk 5 min"], etc.
    flags: output['flags'] from detect_complications...
    returns updated_plan with suggested modifications (does not overwrite clinicians' choices)
    """
    plan = dict(existing_plan or {})
    # default: ensure baseline days exist
    for d in range(1, 15):
        plan.setdefault(f"day_{d}", [])

    # examples of changes:
    if "dvt" in flags:
        # postpone standing/weight bear therapy for 48-72h and add anticoagulation recommendation
        plan["day_1"].append("Postpone standing/weight-bearing physio for 48-72 hours; begin mechanical compression.")
        plan["day_1"].append("Discuss therapeutic anticoagulation pending imaging and consultant review.")
        # add increased monitoring
        plan["day_0"] = plan.get("day_0", []) + ["Increase limb observations (color, swelling, pain) every 4 hours"]
    if "infection" in flags:
        plan["day_0"] = plan.get("day_0", []) + ["Obtain cultures, start empirical antibiotics per policy", "Increase wound checks twice daily"]
        plan["day_1"].append("Reassess infection markers and wound status; consider infectious diseases input")
    if "respiratory" in flags:
        plan["day_0"] = plan.get("day_0", []) + ["Supplemental oxygen as required; repeat SpO2 monitoring hourly"]
        plan["day_1"].append("Early chest physiotherapy and consider escalation to HDU if desaturation persists")
    if "bleeding" in flags:
        plan["day_0"] = plan.get("day_0", []) + ["Surgical review, crossmatch, check Hb q6h", "Hold anticoagulants until reviewed"]
    # If high baseline risk, add conservative steps
    if (risk_profile or {}).get("predicted") == "high":
        plan["day_0"].append("Enhanced monitoring: Telemetry, vital checks q4h, strict input/output charting")

    # deduplicate list items
    for k, v in plan.items():
        seen = set()
        new = []
        for item in v:
            if item not in seen:
                new.append(item)
                seen.add(item)
        plan[k] = new

    return plan
