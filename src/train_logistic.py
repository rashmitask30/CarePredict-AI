"""
First baseline: Logistic Regression on the project's preprocessed data.

Run from the project root:   python src/train_logistic.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit

from preprocessing import create_preprocessor, load_raw_data, prepare_dataset

print("1. Loading and preparing data...", flush=True)
df = load_raw_data()
X, y, groups = prepare_dataset(df)
print(f"   X={X.shape}, positive rate={y.mean():.4f}", flush=True)

print("2. Patient-grouped train/test split...", flush=True)
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("3. Preprocessing (fit on TRAIN only)...", flush=True)
preprocessor = create_preprocessor()
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)
print(f"   processed train={X_train_p.shape}, test={X_test_p.shape}", flush=True)

print("4. Training Logistic Regression...", flush=True)
start = time.time()
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train_p, y_train)
print(f"   trained in {time.time() - start:.1f} seconds", flush=True)

print("5. Evaluating on the test set...", flush=True)
proba = model.predict_proba(X_test_p)[:, 1]
pred = (proba >= 0.5).astype(int)
print(f"   Accuracy : {accuracy_score(y_test, pred):.4f}")
print(f"   Precision: {precision_score(y_test, pred):.4f}")
print(f"   Recall   : {recall_score(y_test, pred):.4f}")
print(f"   ROC-AUC  : {roc_auc_score(y_test, proba):.4f}")
