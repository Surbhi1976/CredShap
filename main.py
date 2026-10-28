from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
state = {}


LABELS = {
    "person_age": "Age",
    "person_income": "Annual income",
    "person_home_ownership": "Home ownership",
    "person_emp_length": "Years employed",
    "loan_intent": "Loan purpose",
    "loan_grade": "Loan grade",
    "loan_amnt": "Loan amount",
    "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan as share of income",
    "cb_person_default_on_file": "Previous default",
    "cb_person_cred_hist_length": "Credit history length",
}

# Value ranges seen in training data; inputs outside them are extrapolation
TRAIN_RANGES = {
    "person_age": (20, 94),
    "person_income": (4000, 2039784),
    "person_emp_length": (0, 41),
    "loan_amnt": (500, 35000),
    "loan_int_rate": (5.42, 23.22),
    "cb_person_cred_hist_length": (2, 30),
}


def explain(df: pd.DataFrame, top_n: int = 5):
    """Per-feature contributions (XGBoost tree contributions averaged over the
    5 calibration folds), with one-hot columns grouped back to the original field."""
    try:
        total, names = [], None
        for cc in state["model"].calibrated_classifiers_:
            pipe = cc.estimator
            pre = pipe.named_steps["preprocessor"]
            booster = pipe.named_steps["classifier"].get_booster()
            contrib = booster.predict(xgb.DMatrix(pre.transform(df)), pred_contribs=True)[0][:-1]
            total.append(contrib)
            names = pre.get_feature_names_out()
        mean = np.mean(total, axis=0)
        grouped = {}
        for name, val in zip(names, mean):
            col = name.split("__", 1)[1]
            for orig in LABELS:
                if col == orig or col.startswith(orig + "_"):
                    grouped[orig] = grouped.get(orig, 0.0) + float(val)
                    break
        ranked = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
        return [{"feature": LABELS[k], "impact": round(v, 3)} for k, v in ranked]
    except Exception:
        return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(BASE_DIR / "credit_risk_model.pkl")
    state["threshold"] = float(joblib.load(BASE_DIR / "best_threshold.pkl"))
    yield
    state.clear()


app = FastAPI(title="Credit Risk API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class LoanApplication(BaseModel):
    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(gt=0)
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: float = Field(ge=0, le=60)
    loan_intent: Literal["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(gt=0)
    loan_int_rate: float = Field(ge=0, le=40)
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(ge=0, le=60)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/model-info")
def model_info():
    return json.loads((BASE_DIR / "metrics.json").read_text())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: LoanApplication):
    row = data.model_dump()
    # Derived server-side so it can never disagree with income and amount
    row["loan_percent_income"] = round(row["loan_amnt"] / row["person_income"], 2)

    # The model predicts P(default), where class 1 = default
    p_default = float(state["model"].predict_proba(pd.DataFrame([row]))[:, 1][0])
    threshold = state["threshold"]
    is_high_risk = p_default >= threshold

    if is_high_risk:
        band = "High"
    elif p_default >= 0.30:
        band = "Medium"
    else:
        band = "Low"

    warnings = [
        f"{LABELS[k]} is outside the range the model was trained on ({lo:g} to {hi:g}), so treat this score with extra caution."
        for k, (lo, hi) in TRAIN_RANGES.items()
        if not lo <= row[k] <= hi
    ]

    return {
        "drivers": explain(pd.DataFrame([row])),
        "warnings": warnings,
        "probability": p_default,
        "threshold": threshold,
        "risk_band": band,
        "decision": "Rejected" if is_high_risk else "Approved",
        "loan_percent_income": row["loan_percent_income"],
    }
