# CarePredict-AI: Hospital Readmission Risk Prediction

A hackathon project that uses predictive analytics to estimate the risk of
hospital readmission for diabetic patient encounters, to support (not replace)
clinical judgment in discharge planning and follow-up.

## Dataset
UCI Diabetes 130-US Hospitals for Years 1999-2008
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

The raw data is stored in `data/raw/` and is never modified.

## Project structure
- `data/raw/`        original dataset (read-only)
- `data/processed/`  cleaned and engineered data
- `notebooks/`       exploration and learning
- `src/`             reusable Python code
- `models/`          saved model and settings
- `reports/figures/` charts for documentation
- `app/`             FastAPI backend and HTML/CSS/JS frontend
- `docs/`            documentation deliverables
- `DECISIONS.md`     running log of project decisions

## Setup (Windows, PowerShell)
    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

## How to run (from the project root, in PowerShell with .venv active)
    python -m pip install -r requirements.txt
    python src/train_models.py          # trains 3 models, saves models/ and reports/ (about 1-2 min)
    python src/fairness_analysis.py     # fairness audit by age, gender, race
    python src/verify_project.py        # quality-control checks (all should PASS)
    python -m uvicorn app.main:app --port 8000   # then open http://127.0.0.1:8000

## What was built
- `src/preprocessing.py`  cleaning, diagnosis categories, ColumnTransformer (fit on train only)
- `src/train_models.py`   Logistic Regression, Random Forest, XGBoost; patient-grouped train/validation/test
- `src/fairness_analysis.py`  group-wise recall/precision/FPR/FNR/selection rate and disparity gaps
- `src/predict.py`, `app/`  FastAPI backend + plain HTML/CSS/JS frontend
- `DECISIONS.md`  decision log (see the Reconciliation section)
- `docs/`  presentation outline and limitations

## Method summary
Target: readmitted within 30 days (1) vs not (0), positive rate 11.34%.
Splits grouped by patient (no patient in two sets). Preprocessing is fit on the training set only.
Race and gender are not model inputs (audit only). The test set is used once, for final reporting.
Risk tiers come from validation-score ranks (High = top 20%), not from clinical thresholds.

## Disclaimer
This tool is for decision support only. It is a prototype trained on historical 1999-2008 data.
It is not a clinical diagnosis and does not replace clinical judgment.
