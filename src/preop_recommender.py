"""
preop_recommender.py

Usage:
    from preop_recommender import load_kb, generate_recommendations
    kb = load_kb("knowledge_base.json")
    out = generate_recommendations(patient_dict, risk_dict, kb)

Returns a dict with:
    - doctor_recommendations: [ "text1", "text2", ... ] (strings)
    - patient_checklist: [ "textA", "textB", ... ] (strings)
    - doctor_recommendations_detailed: [ {rule_id, text, notes, priority}, ... ]
    - patient_checklist_detailed: [ {rule_id, text, notes, priority}, ... ]
    - matched_rule_ids: [ "rule_x", ... ]
    - flat_context: { "demographics_age": 62, "labs_hba1c": 9.2, "risk_predicted": "high", ... }
    - kb_version: optional, if present in KB root object (or 'unknown')
"""

import json
import copy
from typing import Any, Dict, List

# -------------------------
# Loader
# -------------------------
def load_kb(path: str = "knowledge_base.json") -> List[Dict[str, Any]]:
    """Load KB from a JSON file. The KB is expected to be a list of rule objects."""
    with open(path, "r", encoding="utf-8") as f:
        kb = json.load(f)
    # If KB is a dict with metadata and "rules" key, normalize
    if isinstance(kb, dict) and "rules" in kb:
        rules = kb["rules"]
        # attach version metadata if present
        for r in rules:
            r.setdefault("priority", r.get("priority", 999))
        rules_meta = rules
        return rules_meta
    if isinstance(kb, list):
        return kb
    raise ValueError("knowledge_base.json must be a list of rule objects or an object with 'rules' key")

# -------------------------
# Helpers
# -------------------------
def _safe_get(d: Dict, *keys, default=None):
    """Safe nested get for dictionaries."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is default:
            return default
    return cur

def _to_number(val):
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None

def _compare_value(value, op_dict: Dict[str, Any]) -> bool:
    """
    Supports op_dict with keys: gt, gte, lt, lte, eq, neq
    """
    if value is None:
        return False
    v = _to_number(value) if not isinstance(value, str) or op_dict.get("eq") is None else value
    # numeric comparisons
    if "gt" in op_dict:
        try:
            if not (float(v) > float(op_dict["gt"])):
                return False
        except Exception:
            return False
    if "gte" in op_dict:
        try:
            if not (float(v) >= float(op_dict["gte"])):
                return False
        except Exception:
            return False
    if "lt" in op_dict:
        try:
            if not (float(v) < float(op_dict["lt"])):
                return False
        except Exception:
            return False
    if "lte" in op_dict:
        try:
            if not (float(v) <= float(op_dict["lte"])):
                return False
        except Exception:
            return False
    if "eq" in op_dict:
        # support numeric or string equality
        try:
            if isinstance(v, (int, float)) or isinstance(op_dict["eq"], (int, float)):
                if not float(v) == float(op_dict["eq"]):
                    return False
            else:
                if str(v) != str(op_dict["eq"]):
                    return False
        except Exception:
            return False
    if "neq" in op_dict:
        try:
            if isinstance(v, (int, float)) or isinstance(op_dict["neq"], (int, float)):
                if float(v) == float(op_dict["neq"]):
                    return False
            else:
                if str(v) == str(op_dict["neq"]):
                    return False
        except Exception:
            return False
    return True

def _flatten_context(patient: Dict[str, Any], risk: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    for cat, vals in patient.items():
        if isinstance(vals, dict):
            for k, v in vals.items():
                flat[f"{cat}_{k}"] = v
        else:
            flat[cat] = vals
    for k, v in (risk or {}).items():
        flat[f"risk_{k}"] = v
    return flat

def _render_template(text: str, context: Dict[str, Any]) -> str:
    """
    Simple and safe templating using str.format_map with missing keys replaced by empty string.
    Context keys should be strings (e.g., 'demographics_age', 'labs_hba1c', 'surgery_type').
    """
    if not text:
        return text
    class SafeDict(dict):
        def __missing__(self, key):
            return ""
    try:
        return text.format_map(SafeDict(context))
    except Exception:
        # fallback: return original text if templating fails
        return text

# -------------------------
# Rule evaluation
# -------------------------
def _rule_applies(rule: Dict[str, Any], patient: Dict[str, Any], risk: Dict[str, Any]) -> bool:
    """
    Evaluate whether a rule applies to the patient + risk.
    Rule structure assumptions:
      - rule may contain "surgery_types": list OR special value ["all"]
      - rule.condition may contain:
           - "urgency": "elective"/"urgent"/"emergency"
           - "comorbidity": single string (key in patient['comorbidities'])
           - "medication": single string (key in patient['medications'])
           - "labs": { labname: {gt:..., lt:..., eq:...}, ... }
           - "demographics": { key: {gt/lt/eq...}, ... }
           - "risk_level": e.g., "high"
           - "surgery_subtype": list or string (matches patient['surgery']['subtype'])
           - "surgery_types_sub": list of subtypes (legacy support)
    """
    # surgery_types matching
    surgery_types = rule.get("surgery_types", [])
    if surgery_types:
        # allow ["all"] to mean match everything
        if "all" not in surgery_types:
            patient_surg_type = (patient.get("surgery", {}) or {}).get("type", "")
            if patient_surg_type not in surgery_types:
                return False

    cond = rule.get("condition", {}) or {}

    # urgency
    urgency_needed = cond.get("urgency")
    if urgency_needed:
        if (patient.get("surgery", {}) or {}).get("urgency") != urgency_needed:
            return False

    # comorbidity
    comorb = cond.get("comorbidity")
    if comorb:
        if not (patient.get("comorbidities", {}) or {}).get(comorb, False):
            return False

    # medication
    med = cond.get("medication")
    if med:
        if not (patient.get("medications", {}) or {}).get(med, False):
            return False

    # labs thresholds
    labs_cond = cond.get("labs", {})
    for lab_name, ops in (labs_cond or {}).items():
        val = (patient.get("labs", {}) or {}).get(lab_name)
        if val is None:
            # if lab missing, treat as not matching
            return False
        if not _compare_value(val, ops):
            return False

    # demographics thresholds (e.g., bmi, age)
    dem_cond = cond.get("demographics", {})
    for key, ops in (dem_cond or {}).items():
        val = (patient.get("demographics", {}) or {}).get(key)
        if val is None:
            return False
        if not _compare_value(val, ops):
            return False

    # risk level
    risk_level = cond.get("risk_level")
    if risk_level:
        if (risk or {}).get("predicted") != risk_level:
            return False

    # surgery_subtype / surgery_types_sub (support either key)
    surgery_subtype_cond = cond.get("surgery_subtype") or cond.get("surgery_types_sub")
    if surgery_subtype_cond:
        patient_sub = (patient.get("surgery", {}) or {}).get("subtype", "")
        # allow list or single value
        if isinstance(surgery_subtype_cond, list):
            if patient_sub not in surgery_subtype_cond:
                return False
        else:
            if patient_sub != surgery_subtype_cond:
                return False

    return True

# -------------------------
# Public API
# -------------------------
def generate_recommendations(patient: Dict[str, Any], risk: Dict[str, Any], kb: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main entry point.
    patient: normalized patient dict (use your normalize_patient before calling).
    risk: risk dict produced by Module-1, must have key 'predicted' like 'low'|'moderate'|'high'.
    kb: list of rule dicts from load_kb().
    """
    # Defensive copies
    patient = copy.deepcopy(patient or {})
    risk = copy.deepcopy(risk or {})
    kb = list(kb or [])

    # Build flat context for templating and debugging
    flat = _flatten_context(patient, risk)
    # add some canonical keys expected in templates
    flat.setdefault("surgery_type", (patient.get("surgery", {}) or {}).get("type", ""))
    flat.setdefault("surgery_subtype", (patient.get("surgery", {}) or {}).get("subtype", ""))
    # demographics and labs also available as demographics_age, labs_hba1c etc (already in flat)
    # ensure string keys for templating
    flat_str = {k: ("" if v is None else v) for k, v in flat.items()}

    # Sort KB by priority (lower numbers = higher priority). Missing priority -> 999
    sorted_kb = sorted(kb, key=lambda r: r.get("priority", 999))

    doctor_recs = []
    patient_checklist = []
    doctor_recs_detailed = []
    patient_checklist_detailed = []
    matched_rule_ids = []

    for rule in sorted_kb:
        try:
            # support rules where surgery_types == ["all"] or absent
            # evaluate rule
            if _rule_applies(rule, patient, risk):
                rid = rule.get("id") or rule.get("name") or f"rule_{len(matched_rule_ids)+1}"
                if rid in matched_rule_ids:
                    # avoid duplicate multiple matches of same rule
                    continue
                matched_rule_ids.append(rid)

                # Render templated recommendation text (safe)
                doc_template = rule.get("recommendation_doctor", "")
                pat_template = rule.get("recommendation_patient", "")

                # Provide a richer templating context: include hospital-friendly keys
                templating_context = dict(flat_str)
                # also include nested keys like demographics_age => demographics_age already provided
                # add surgery_type formatted string
                templating_context["surgery_type"] = templating_context.get("surgery_type", "")
                templating_context["surgery_subtype"] = templating_context.get("surgery_subtype", "")

                doc_text = _render_template(doc_template, templating_context).strip()
                pat_text = _render_template(pat_template, templating_context).strip()

                # deduplicate textual recommendations
                if doc_text and doc_text not in doctor_recs:
                    doctor_recs.append(doc_text)
                    doctor_recs_detailed.append({
                        "rule_id": rid,
                        "text": doc_text,
                        "notes": rule.get("notes", ""),
                        "priority": rule.get("priority", 999)
                    })

                if pat_text and pat_text not in patient_checklist:
                    patient_checklist.append(pat_text)
                    patient_checklist_detailed.append({
                        "rule_id": rid,
                        "text": pat_text,
                        "notes": rule.get("notes", ""),
                        "priority": rule.get("priority", 999)
                    })
        except Exception:
            # If any rule throws, skip it (robustness)
            continue

    result = {
        "doctor_recommendations": doctor_recs,
        "patient_checklist": patient_checklist,
        "doctor_recommendations_detailed": doctor_recs_detailed,
        "patient_checklist_detailed": patient_checklist_detailed,
        "matched_rule_ids": matched_rule_ids,
        "flat_context": flat,
    }

    # If KB object had top-level metadata like version, include it if present
    kb_version = None
    if isinstance(kb, list) and len(kb) > 0 and isinstance(kb[0], dict):
        # nothing to do: user can pass metadata separately
        pass
    result["kb_version"] = kb_version or "unknown"

    return result

# -------------------------
# If run as script, quick local smoke test
# -------------------------
if __name__ == "__main__":
    # Quick demo - only for local testing
    import pprint
    demo_patient = {
        "demographics": {"age": 62, "sex": "M", "bmi": 32.0},
        "comorbidities": {"diabetes": True, "hypertension": True},
        "medications": {"warfarin": False, "antiplatelet": False, "insulin": True},
        "labs": {"hba1c": 9.2, "hb": 11.0, "creatinine": 1.0},
        "lifestyle": {"smoker": True, "alcohol_units_week": 2},
        "surgery": {"type": "orthopedic", "subtype": "hip", "urgency": "elective"}
    }
    demo_risk = {"predicted": "high", "domain": "cardiac", "confidence": 0.87}

    try:
        kb = load_kb("knowledge_base.json")
    except Exception as e:
        print("Failed to load knowledge_base.json:", e)
        kb = []

    out = generate_recommendations(demo_patient, demo_risk, kb)
    pprint.pprint(out)
