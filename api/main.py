"""thin on-demand scoring API (E4). feature-vector-in -> calibrated P + EV decision.

uses the SAME cost math as the batch scorer (src.score), read from src.config, so the API and the
nightly job cannot disagree. this is the on-demand single-customer path (a demo / CSM lookup); the
batch contact list is the actual product. the model is loaded from a local export (src.export_model)
so serving has no MLflow-registry / sqlite dependency — falls back to the registry only for local dev.
"""
import os
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src import config

MODEL_PATH = os.environ.get("MODEL_PATH", str(config.REPO_ROOT / "api" / "model"))
VALUE = config.ARPU * config.HORIZON_MONTHS
BREAK_EVEN = config.OFFER / (config.SAVE_RATE * VALUE)

app = FastAPI(title="churn-base12 scorer", version="1.0")
_model = None


def get_model():
    global _model
    if _model is None:
        if Path(MODEL_PATH).exists():
            _model = mlflow.pyfunc.load_model(MODEL_PATH)            # exported path (container)
        else:
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)      # local-dev fallback
            _model = mlflow.pyfunc.load_model(f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}")
    return _model


# dtypes mirror the model signature (built by features.sql): 8 integer feats, 4 double feats.
class Features(BaseModel):
    is_auto_renew: int
    payment_plan_days: int
    actual_amount_paid: int
    plan_list_price: int
    discount: int
    has_activity_60d: int
    recency_days: int
    active_days_30: int
    secs_30: float
    unq_30: float
    completion_ratio: float
    activity_trend: float


class Score(BaseModel):
    p_churn: float
    contact: bool
    expected_value: float
    break_even: float


@app.get("/health")
def health():
    try:
        get_model()
        return {"status": "ok", "model": f"{config.MODEL_NAME}@{config.MODEL_ALIAS}",
                "break_even": round(BREAK_EVEN, 3)}
    except Exception as e:
        return {"status": "model_not_loaded", "detail": str(e)}


@app.post("/score", response_model=Score)
def score(f: Features):
    # pinned feature order; cast to the model's OWN signature dtypes (not assumed ones) before
    # predict — the signature was inferred from a training sample and can pin a narrower dtype
    # (e.g. int32) on a column than a fresh request naturally produces (int64); enforcing against
    # the signature here means any such pandas/duckdb dtype artifact never reaches the user as a 500.
    X = pd.DataFrame([f.model_dump()])[config.BASE12]
    sig = get_model().metadata.get_input_schema()
    if sig is not None:
        for col, dtype in zip(sig.input_names(), sig.pandas_types()):
            X[col] = X[col].astype(dtype)
    p = float(get_model().predict(X)[0])
    ev = config.SAVE_RATE * VALUE * p - config.OFFER          # contact iff this clears 0
    return Score(p_churn=p, contact=bool(ev >= 0), expected_value=ev, break_even=BREAK_EVEN)
