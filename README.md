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

## Status
Stage 0: project setup (in progress)

## Disclaimer
This tool is for decision support only. It does not replace clinical judgment.