"""
Train Logistic Regression, Random Forest and XGBoost; pick the best on the
VALIDATION set; evaluate on the untouched TEST set once; save artifacts.

Run from the project root:   python src/train_models.py
Optional ablation (validation only):  python src/train_models.py --ablation
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from preprocessing import create_preprocessor, load_raw_data, prepare_dataset

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# D8 (proposed): the "High risk" group = the top 20% highest scores on the
# validation set. Medium = next 30%. Low = the lowest 50%.
HIGH_TOP_FRACTION = 0.20
MEDIUM_TOP_FRACTION = 0.50

# D7 (proposed): race and gender are NOT model inputs; they are audit-only.
SENSITIVE_EXCLUDED = ("race", "gender")
AUDIT_COLUMNS = ["race", "gender", "age"]


def make_models(pos_weight):
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight,
            eval_metric="auc", tree_method="hist", n_jobs=-1,
            random_state=SEED),
    }


def split_by_patient(X, y, groups, test_size, seed):
    """Patient-grouped split -> (first_idx, second_idx) as positional indexes."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(X, y, groups))


def metrics_at(y_true, score, threshold):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, score),
        "pr_auc": average_precision_score(y_true, score),
        "flagged_fraction": float(pred.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run_ablation(X_tr, y_tr, X_val, y_val, pos_weight):
    """D7f: compare with/without race+gender using VALIDATION only."""
    print("\nAblation (validation only, XGBoost): effect of race + gender as inputs")
    for label, exclude in [("with race+gender   ", ()), ("without race+gender", SENSITIVE_EXCLUDED)]:
        pre = create_preprocessor(exclude=exclude)
        a, b = pre.fit_transform(X_tr), pre.transform(X_val)
        m = make_models(pos_weight)["XGBoost"].fit(a, y_tr)
        s = m.predict_proba(b)[:, 1]
        print(f"  {label}: ROC-AUC={roc_auc_score(y_val, s):.4f}  PR-AUC={average_precision_score(y_val, s):.4f}")


def main(ablation=False):
    t0 = time.time()
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    print("1. Loading + preparing data (raw file is only read, never changed)")
    df = load_raw_data(ROOT / "data" / "raw" / "diabetic_data.csv")
    X, y, groups = prepare_dataset(df)
    assert X.isna().sum().sum() == 0, "unexpected missing values"
    for forbidden in ("patient_nbr", "encounter_id", "readmitted", "target"):
        assert forbidden not in X.columns, f"{forbidden} must not be a feature"

    print("2. Splits (patient-grouped): TEST 20% (D13), then VALIDATION 20% of the rest")
    trainval_idx, test_idx = split_by_patient(X, y, groups, 0.20, SEED)
    X_tv, y_tv, g_tv = X.iloc[trainval_idx], y.iloc[trainval_idx], groups.iloc[trainval_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    tr_rel, val_rel = split_by_patient(X_tv, y_tv, g_tv, 0.20, SEED)
    X_tr, y_tr = X_tv.iloc[tr_rel], y_tv.iloc[tr_rel]
    X_val, y_val = X_tv.iloc[val_rel], y_tv.iloc[val_rel]

    p_tr, p_val, p_te = set(g_tv.iloc[tr_rel]), set(g_tv.iloc[val_rel]), set(groups.iloc[test_idx])
    assert not (p_tr & p_val) and not (p_tr & p_te) and not (p_val & p_te), "PATIENT LEAKAGE"
    split_info = {
        "train_encounters": len(X_tr), "val_encounters": len(X_val), "test_encounters": len(X_test),
        "train_patients": len(p_tr), "val_patients": len(p_val), "test_patients": len(p_te),
        "train_pos_rate": float(y_tr.mean()), "val_pos_rate": float(y_val.mean()),
        "test_pos_rate": float(y_test.mean()),
    }
    print("  ", json.dumps(split_info))

    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())

    if ablation:
        run_ablation(X_tr, y_tr, X_val, y_val, pos_weight)
        return

    print("3. Preprocessor: fit on TRAIN only; transform validation and test")
    preprocessor = create_preprocessor(exclude=SENSITIVE_EXCLUDED)
    A_tr = preprocessor.fit_transform(X_tr)
    A_val = preprocessor.transform(X_val)
    A_test = preprocessor.transform(X_test)
    print(f"   processed columns: {A_tr.shape[1]}")

    print("4. Training models (fixed hyper-parameters, no test-set tuning)")
    trained, val_scores, val_rows = {}, {}, []
    for name, model in make_models(pos_weight).items():
        s = time.time()
        model.fit(A_tr, y_tr)
        trained[name] = model
        val_scores[name] = model.predict_proba(A_val)[:, 1]
        auc = roc_auc_score(y_val, val_scores[name])
        pr = average_precision_score(y_val, val_scores[name])
        val_rows.append({"model": name, "val_roc_auc": auc, "val_pr_auc": pr})
        print(f"   {name:20s} {time.time()-s:5.1f}s  val ROC-AUC={auc:.4f}  PR-AUC={pr:.4f}")

    best_name = max(val_rows, key=lambda r: r["val_roc_auc"])["model"]
    print(f"   -> best on VALIDATION ROC-AUC: {best_name}")

    print("5. Thresholds from VALIDATION scores (top-20% = flagged High risk)")
    cutoffs = {}
    for name, s in val_scores.items():
        cutoffs[name] = {
            "high": float(np.quantile(s, 1 - HIGH_TOP_FRACTION)),
            "medium": float(np.quantile(s, 1 - MEDIUM_TOP_FRACTION)),
        }

    print("6. FINAL evaluation on the untouched TEST set (used once)")
    results, test_scores = [], {}
    for name, model in trained.items():
        s = model.predict_proba(A_test)[:, 1]
        test_scores[name] = s
        m = metrics_at(y_test, s, cutoffs[name]["high"])
        m05 = metrics_at(y_test, s, 0.5)
        results.append({"model": name,
                        **{f"{k}": v for k, v in m.items()},
                        "recall_at_0.5": m05["recall"], "precision_at_0.5": m05["precision"],
                        "accuracy_at_0.5": m05["accuracy"]})
    res_df = pd.DataFrame(results)
    res_df.to_csv(REPORTS_DIR / "model_comparison_test.csv", index=False)
    pd.DataFrame(val_rows).to_csv(REPORTS_DIR / "model_comparison_validation.csv", index=False)
    show = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "flagged_fraction"]
    print(res_df[show].round(4).to_string(index=False))
    print(f"   (positive rate in test = {y_test.mean():.4f}; a random flag of 20% would give precision ~{y_test.mean():.3f}, recall 0.20)")

    print("7. Risk tiers, calibration and observed rates (validation)")
    best_val = val_scores[best_name]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(best_val, y_val)
    cut = cutoffs[best_name]
    tier = np.where(best_val >= cut["high"], "High", np.where(best_val >= cut["medium"], "Medium", "Low"))
    tier_rates = {t: {"n": int((tier == t).sum()), "observed_readmit_rate_30d": float(y_val[tier == t].mean())}
                  for t in ["Low", "Medium", "High"]}
    print("  ", json.dumps(tier_rates))

    print("8. Saving artifacts and test predictions (for the fairness analysis)")
    defaults = {}
    for c in X_tr.columns:
        defaults[c] = (float(X_tr[c].median()) if pd.api.types.is_numeric_dtype(X_tr[c]) and c in
                       ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
                        "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
                       else X_tr[c].mode().iloc[0])
        if hasattr(defaults[c], "item"):
            defaults[c] = defaults[c].item()
    options = {c: sorted(map(lambda v: v.item() if hasattr(v, "item") else v, X_tr[c].unique()))
               for c in X_tr.columns if c not in ["time_in_hospital", "num_lab_procedures", "num_procedures",
                                                   "num_medications", "number_outpatient", "number_emergency",
                                                   "number_inpatient", "number_diagnoses"]}
    metadata = {
        "best_model": best_name,
        "feature_columns": list(X.columns),
        "excluded_from_model_inputs": list(SENSITIVE_EXCLUDED),
        "defaults": defaults,
        "categorical_options": options,
        "risk_cutoffs": cut,
        "risk_policy": "High = top 20% of validation scores; Medium = next 30%; Low = lowest 50%",
        "tier_rates_validation": tier_rates,
        "test_metrics_best_model": res_df[res_df.model == best_name].iloc[0].to_dict(),
        "split_info": split_info,
        "seed": SEED,
    }
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(trained[best_name], MODELS_DIR / "best_model.joblib")
    joblib.dump(calibrator, MODELS_DIR / "calibrator.joblib")
    for name, model in trained.items():
        joblib.dump(model, MODELS_DIR / f"model_{name.lower().replace(' ', '_')}.joblib")
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    audit = X_test[AUDIT_COLUMNS].copy()
    audit["y_true"] = y_test.values
    for name, s in test_scores.items():
        audit[f"score_{name}"] = s
        audit[f"flag_{name}"] = (s >= cutoffs[name]["high"]).astype(int)
    audit.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)
    (REPORTS_DIR / "cutoffs.json").write_text(json.dumps(cutoffs, indent=2))

    print(f"Done in {time.time()-t0:.0f}s. Artifacts in models/, tables in reports/.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true")
    main(ablation=ap.parse_args().ablation)
