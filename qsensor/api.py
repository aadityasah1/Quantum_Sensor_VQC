"""FastAPI service for the quantum sensor fault classifier.

Endpoints:
    GET  /health        liveness + loaded model metadata
    POST /predict       classify a sensor signal  (?model=vqc|classical)
    GET  /sample        generate a demo signal to try /predict with
    GET  /docs          interactive OpenAPI UI (auto)

Run locally:
    uvicorn qsensor.api:app --reload
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import List, Optional

import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

from . import __version__
from .signals import generate_one, N_FEATURES
from .inference import Predictor

_state: dict = {"predictor": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["predictor"] = Predictor()
    except Exception as exc:                      # start anyway; /health reports it
        _state["error"] = str(exc)
    yield


app = FastAPI(
    title="Quantum Sensor Fault Classifier",
    version=__version__,
    description="Classifies a vibration-sensor signal as `normal` or `fault` "
                "using a 4-qubit variational quantum circuit (with a classical "
                "RBF-SVM fallback).",
    lifespan=lifespan,
)


class SignalIn(BaseModel):
    signal: List[float] = Field(..., description=f"{N_FEATURES} raw sensor samples",
                                min_length=1)
    model_config = {"json_schema_extra": {"examples": [
        {"signal": [0.97, 0.41, 0.05, -0.22, -0.88, -0.41, 0.18, 0.12]}]}}


class PredictionOut(BaseModel):
    label: int
    prediction: str
    confidence: float
    model: str
    z_expectation: Optional[float] = None


def _predictor() -> Predictor:
    if _state["predictor"] is None:
        raise HTTPException(503, f"model not loaded: {_state['error']}. "
                                 "Train first: python -m qsensor.training")
    return _state["predictor"]


@app.get("/", include_in_schema=False)
def index():
    """Minimalist web UI."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    p = _state["predictor"]
    return {
        "status": "ok" if p is not None else "model_not_loaded",
        "service": "quantum-sensor-fault-classifier",
        "version": __version__,
        "model_loaded": p is not None,
        "metadata": p.meta if p is not None else None,
        "error": _state["error"],
    }


@app.post("/predict", response_model=PredictionOut)
def predict(payload: SignalIn,
            model: str = Query("vqc", pattern="^(vqc|classical)$",
                               description="which model to use")):
    try:
        result = _predictor().predict(payload.signal, model=model)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return PredictionOut(**result)


@app.get("/sample")
def sample(fault: bool = Query(False, description="generate a fault signal?"),
           seed: Optional[int] = Query(None)):
    """A ready-to-use signal you can paste straight into POST /predict."""
    return {"signal": generate_one(fault=fault, seed=seed).round(6).tolist(),
            "true_label": int(fault),
            "hint": "POST this 'signal' array to /predict"}
