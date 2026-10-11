"""
Quick quality-control checks. Run from the project root:   python src/verify_project.py
Every line should say PASS.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd

from predict import Predictor
from preprocessing import load_raw_data, prepare_dataset
from train_models import SEED, split_by_patient

ROOT = Path(__file__).resolve().parent.parent
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(("PASS  " if cond else "FAIL  ") + name)


# 1. raw data unchanged (checksum recorded the first time this script runs)
sums_file = ROOT / "reports" / "raw_data_checksums.json"
sums = {f: hashlib.md5((ROOT / "data" / "raw" / f).read_bytes()).hexdigest()
        for f in ["diabetic_data.csv", "IDS_mapping.csv"]}
if not sums_file.exists():
    sums_file.write_text(json.dumps(sums, indent=2))
    print("INFO  raw checksums recorded for future checks")
check("raw files unchanged since first check", json.loads(sums_file.read_text()) == sums)

# 2. data preparation facts
df = load_raw_data(ROOT / "data" / "raw" / "diabetic_data.csv")
X, y, groups = prepare_dataset(df)
check("100,114 encounters after expired-disposition exclusion", len(X) == 100114)
check("positive rate is 11.34%", round(y.mean() * 100, 2) == 11.34)
check("0 missing values after preprocessing", X.isna().sum().sum() == 0)
check("44 prepared features", X.shape[1] == 44)
check("no identifiers / target in features",
      not {"patient_nbr", "encounter_id", "readmitted", "target"} & set(X.columns))

# 3. no patient leakage across train / validation / test
tv, te = split_by_patient(X, y, groups, 0.20, SEED)
tr_rel, va_rel = split_by_patient(X.iloc[tv], y.iloc[tv], groups.iloc[tv], 0.20, SEED)
g_tr, g_va, g_te = (set(groups.iloc[tv].iloc[tr_rel]), set(groups.iloc[tv].iloc[va_rel]), set(groups.iloc[te]))
check("no patient in more than one split", not (g_tr & g_va) and not (g_tr & g_te) and not (g_va & g_te))
check("test split = 20,075 encounters / 14,088 patients (D13)", len(te) == 20075 and len(g_te) == 14088)

# 4. preprocessing leakage: saved preprocessor was fit on train only (n_samples_seen not stored,
#    so verify its scaler mean equals the TRAIN-ONLY mean, not the full-data mean)
pre = joblib.load(ROOT / "models" / "preprocessor.joblib")
train_mean = X.iloc[tv].iloc[tr_rel]["time_in_hospital"].mean()
full_mean = X["time_in_hospital"].mean()
scaler_mean = pre.named_transformers_["numeric"].mean_[0]
check("preprocessor scaler matches TRAIN-only statistics", abs(scaler_mean - train_mean) < 1e-9 and abs(scaler_mean - full_mean) > 1e-9)

# 5. artifacts load and score a brand-new sample
pred = Predictor()
sample = {"age": "[60-70)", "time_in_hospital": 4, "num_lab_procedures": 40, "num_procedures": 1,
          "num_medications": 15, "number_outpatient": 0, "number_emergency": 1, "number_inpatient": 1,
          "number_diagnoses": 7, "admission_type_id": 1, "discharge_disposition_id": 1,
          "admission_source_id": 7, "diag_1_category": "Circulatory", "diag_2_category": "Endocrine_Metabolic",
          "diag_3_category": "Unknown", "A1Cresult": "Not measured", "max_glu_serum": "Not measured",
          "insulin": "Steady", "change": "No", "diabetesMed": "Yes"}
out = pred.predict(sample)
check("model + preprocessor + calibrator load; new sample scored", 0 <= out["risk_score"] <= 1 and out["risk_category"] in ("Low", "Medium", "High"))
check("race and gender are not model inputs (audit only)", set(pred.meta["excluded_from_model_inputs"]) == {"race", "gender"})

# 6. fairness outputs exist
check("fairness tables exist", (ROOT / "reports" / "fairness_by_group.csv").exists())

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED - send me the FAIL lines")
