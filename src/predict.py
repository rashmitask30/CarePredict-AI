"""
Prediction helper used by the app. Loads saved artifacts once, then scores one encounter.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

NUMERIC = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
           "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]

GUIDANCE = {
    "High": ("Higher-than-typical risk of 30-day readmission compared with other diabetic encounters. "
             "Consider prioritising for follow-up: early post-discharge contact, medication review, "
             "and confirmation of a follow-up appointment."),
    "Medium": ("Moderate risk. Standard discharge planning with a clear follow-up plan; "
               "consider a reminder call if resources allow."),
    "Low": ("Lower-than-typical risk. Routine discharge process is likely sufficient; "
            "clinicians may still escalate based on their own judgment."),
}


class Predictor:
    def __init__(self):
        self.meta = json.loads((MODELS / "metadata.json").read_text())
        self.pre = joblib.load(MODELS / "preprocessor.joblib")
        self.model = joblib.load(MODELS / "best_model.joblib")
        self.cal = joblib.load(MODELS / "calibrator.joblib")
        self.cols = self.meta["feature_columns"]
        self.defaults = self.meta["defaults"]
        self.cut = self.meta["risk_cutoffs"]
        self.names = list(self.pre.get_feature_names_out())

    def _pretty(self, name):
        name = name.split("__", 1)[1]
        for c in sorted(self.cols, key=len, reverse=True):
            if name == c:
                return c, None
            if name.startswith(c + "_"):
                return c, name[len(c) + 1:]
        return name, None

    def _top_factors(self, A, entered, k=4):
        """Real model contributions (XGBoost SHAP-style values), grouped by original feature."""
        try:
            contrib = self.model.get_booster().predict(xgb.DMatrix(A), pred_contribs=True)[0][:-1]
        except Exception:
            return []
        agg, detail = {}, {}
        for v, n in zip(contrib, self.names):
            col, val = self._pretty(n)
            agg[col] = agg.get(col, 0) + float(v)
            if val is not None and abs(v) > abs(detail.get(col, (0, ""))[0]):
                detail[col] = (float(v), val)
        agg = {c: v for c, v in agg.items() if c in entered}   # only fields the user entered
        top = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:k]
        return [{"feature": c, "value": detail.get(c, (0, None))[1],
                 "direction": "raises" if v > 0 else "lowers"} for c, v in top]

    def predict(self, inputs: dict):
        row = dict(self.defaults)
        row.update({k: v for k, v in inputs.items() if v is not None})
        for c in NUMERIC:
            row[c] = float(row[c])
        df = pd.DataFrame([row])[self.cols]
        A = self.pre.transform(df)
        score = float(self.model.predict_proba(A)[:, 1][0])
        tier = "High" if score >= self.cut["high"] else "Medium" if score >= self.cut["medium"] else "Low"
        calibrated = float(self.cal.predict([score])[0])
        return {
            "risk_score": round(score, 4),
            "estimated_30d_readmission_probability": round(calibrated, 4),
            "risk_category": tier,
            "guidance": GUIDANCE[tier],
            "top_factors": self._top_factors(A, set(inputs)),
            "model": self.meta["best_model"],
        }
