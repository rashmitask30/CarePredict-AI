"""
FastAPI backend.  Run from the project root:  uvicorn app.main:app --port 8000
"""
import csv
import io
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from predict import Predictor

app = FastAPI(title="CarePredict-AI", description="Decision-support prototype. Not a clinical diagnosis.")
predictor = Predictor()
STATIC = Path(__file__).resolve().parent / "static"

DISCLAIMER = ("CarePredict-AI is a decision-support prototype trained on historical 1999-2008 US hospital data. "
              "It is NOT a clinical diagnosis and does NOT replace clinician judgment.")


def load_id_labels():
    """Parse the three sections of IDS_mapping.csv into {column: {id: label}}."""
    labels, current = {}, None
    text = (ROOT / "data" / "raw" / "IDS_mapping.csv").read_text(encoding="utf-8-sig")
    for rec in csv.reader(io.StringIO(text)):
        if not rec or not rec[0].strip():
            continue
        if rec[0].endswith("_id") and rec[1].strip() == "description":
            current = rec[0]; labels[current] = {}
        elif current:
            labels[current][rec[0].strip()] = rec[1].strip()
    return labels


ID_LABELS = load_id_labels()


class Encounter(BaseModel):
    age: str
    time_in_hospital: int = Field(ge=1, le=14)
    num_lab_procedures: int = Field(ge=0, le=150)
    num_procedures: int = Field(ge=0, le=10)
    num_medications: int = Field(ge=0, le=100)
    number_outpatient: int = Field(ge=0, le=100)
    number_emergency: int = Field(ge=0, le=100)
    number_inpatient: int = Field(ge=0, le=50)
    number_diagnoses: int = Field(ge=1, le=20)
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    diag_1_category: str
    diag_2_category: str
    diag_3_category: str
    A1Cresult: str
    max_glu_serum: str
    insulin: str
    change: str
    diabetesMed: str


@app.get("/api/health")
def health():
    return {"status": "ok", "model": predictor.meta["best_model"]}


@app.get("/api/options")
def options():
    opts = predictor.meta["categorical_options"]
    out = {k: opts[k] for k in ["age", "diag_1_category", "A1Cresult", "max_glu_serum", "insulin", "change", "diabetesMed"]}
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        out[col] = [{"value": v, "label": ID_LABELS.get(col, {}).get(str(v), f"Code {v}")} for v in opts[col]]
    return {"options": out, "defaults": predictor.defaults, "disclaimer": DISCLAIMER,
            "risk_policy": predictor.meta["risk_policy"], "tier_rates_validation": predictor.meta["tier_rates_validation"]}


@app.post("/api/predict")
def predict(e: Encounter):
    opts = predictor.meta["categorical_options"]
    for field in ["age", "diag_1_category", "diag_2_category", "diag_3_category", "A1Cresult", "max_glu_serum",
                  "insulin", "change", "diabetesMed", "admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        ref = opts["diag_1_category" if field.startswith("diag_") else field]
        if getattr(e, field) not in ref:
            raise HTTPException(422, f"Unsupported value for {field}: {getattr(e, field)}")
    result = predictor.predict(e.model_dump())
    result["disclaimer"] = DISCLAIMER
    return result


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
