"""
Fairness audit of the best model on the TEST set, by age, gender and race.
Reads reports/test_predictions.csv (written by train_models.py).

Run from the project root:   python src/fairness_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
MIN_GROUP_N = 500        # groups smaller than this are listed but excluded from disparity gaps
MIN_EACH_CLASS = 30      # need this many positives AND negatives to trust recall/FPR/AUC


def group_table(df, col, score_col, flag_col):
    rows = []
    for g, d in df.groupby(col):
        y, flag = d["y_true"], d[flag_col]
        n, pos, neg = len(d), int(y.sum()), int((1 - y).sum())
        tp = int(((flag == 1) & (y == 1)).sum())
        fp = int(((flag == 1) & (y == 0)).sum())
        reliable = n >= MIN_GROUP_N and pos >= MIN_EACH_CLASS and neg >= MIN_EACH_CLASS
        rows.append({
            "attribute": col, "group": g, "n": n, "positives": pos,
            "base_rate": pos / n,
            "selection_rate": flag.mean(),                       # share flagged High risk
            "precision": tp / (tp + fp) if (tp + fp) else np.nan,
            "recall_TPR": tp / pos if pos else np.nan,
            "FPR": fp / neg if neg else np.nan,
            "FNR": 1 - tp / pos if pos else np.nan,
            "roc_auc": roc_auc_score(y, d[score_col]) if pos >= MIN_EACH_CLASS and neg >= MIN_EACH_CLASS else np.nan,
            "reliable": reliable,
        })
    return pd.DataFrame(rows)


def main():
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    best = meta["best_model"]
    df = pd.read_csv(REPORTS / "test_predictions.csv")
    score_col, flag_col = f"score_{best}", f"flag_{best}"
    print(f"Fairness audit of {best} on the TEST set (n={len(df)}). 'Flagged' = High-risk tier.\n")

    tables, summary = [], []
    for col in ["age", "gender", "race"]:
        t = group_table(df, col, score_col, flag_col)
        tables.append(t)
        ok = t[t.reliable]
        print(f"=== {col} ===")
        print(t.drop(columns=["attribute"]).round(3).to_string(index=False))
        if len(ok) >= 2:
            row = {"attribute": col, "groups_compared": len(ok), "excluded_small_groups": int((~t.reliable).sum())}
            for m in ["recall_TPR", "FPR", "selection_rate", "precision"]:
                row[f"{m}_gap_max_minus_min"] = ok[m].max() - ok[m].min()
            row["selection_rate_ratio_min_over_max"] = ok["selection_rate"].min() / ok["selection_rate"].max()
            row["recall_ratio_min_over_max"] = ok["recall_TPR"].min() / ok["recall_TPR"].max()
            row["highest_recall_group"], row["lowest_recall_group"] = ok.loc[ok.recall_TPR.idxmax(), "group"], ok.loc[ok.recall_TPR.idxmin(), "group"]
            summary.append(row)
        print()

    full = pd.concat(tables)
    full.to_csv(REPORTS / "fairness_by_group.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(REPORTS / "fairness_disparity_summary.csv", index=False)
    print("=== Disparity summary (only groups with n>=%d and >=%d pos/neg) ===" % (MIN_GROUP_N, MIN_EACH_CLASS))
    print(summ.round(3).T.to_string(header=False))
    print("\nNote: age/gender/race are NOT model inputs for race/gender (audit only); age IS an input.")
    print("Small groups are shown but excluded from gaps; their metrics are too noisy to interpret.")


if __name__ == "__main__":
    main()
