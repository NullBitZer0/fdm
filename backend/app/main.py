"""FastAPI service for credit card fraud scoring.

Stage 9 requirements, and where each one lives:
  load model + preprocessing   model_service.load_bundle, cached at startup
  accept and validate input    schemas.Transaction (pydantic)
  same preprocessing as train  model_service.ModelBundle.build_frame
  generate predictions         ModelBundle.predict
  meaningful results           schemas.Prediction (probability, verdict, reasons)
  handle invalid input         422 from pydantic, 503 if the model is unavailable
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .model_service import CATEGORICAL, explain, load_bundle, risk_band
from .schemas import (BatchRequest, BatchResponse, ErrorResponse, HealthResponse,
                      ModelInfo, Prediction, Transaction)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fraud-api")

CORS_ORIGINS = os.environ.get(
    "FDM_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once, before the first request, and report loudly if absent."""
    try:
        bundle = load_bundle()
        _state["bundle"] = bundle
        logger.info("loaded %s (threshold %.4f, %d features)", bundle.name,
                    bundle.threshold, len(bundle.preprocessor.feature_names))
    except FileNotFoundError as error:
        logger.error("%s", error)
    yield
    _state.clear()


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=("Scores a single transaction for fraud risk. Built on the "
                 "tuned ensemble selected in Stage 7."),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in CORS_ORIGINS if origin.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_bundle():
    """Dependency: refuse to serve rather than answer with a wrong default."""
    bundle = _state.get("bundle")
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded. Run scripts/tune_models.py first.",
        )
    return bundle


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"service": app.title, "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    bundle = _state.get("bundle")
    return HealthResponse(
        status="ok" if bundle else "degraded",
        model_loaded=bundle is not None,
        model_name=bundle.name if bundle else None,
    )


@app.get("/api/model", response_model=ModelInfo, tags=["model"])
def model_info(bundle=Depends(get_bundle)) -> ModelInfo:
    """What is serving traffic, and how good is it."""
    meta = bundle.metadata
    return ModelInfo(
        name=bundle.name,
        threshold=bundle.threshold,
        validation_pr_auc=meta.get("validation_pr_auc", 0.0),
        test_pr_auc=meta.get("test_pr_auc"),
        test_roc_auc=meta.get("test_roc_auc"),
        test_precision=meta.get("test_precision"),
        test_recall=meta.get("test_recall"),
        test_f1=meta.get("test_f1"),
        n_features=len(bundle.preprocessor.feature_names),
        n_train_rows=meta.get("n_train_rows", 0),
        trained_on=meta.get("trained_on", "unknown"),
        features=bundle.preprocessor.feature_names,
        top_features=bundle.top_features(15),
    )


@app.get("/api/categories", tags=["reference"])
def categories(bundle=Depends(get_bundle)) -> dict:
    """Valid merchant categories and states, so the frontend can populate selects.

    Sourced from the fitted encoder rather than hardcoded, which means the list
    cannot drift away from what the model actually learned.
    """
    # The encoder was fitted on CATEGORICAL in that exact order, so category i of
    # encoder.categories_ belongs to CATEGORICAL[i].
    categories = getattr(bundle.preprocessor.encoder, "categories_", [])
    return {column: sorted(str(value) for value in categories[index])
            for index, column in enumerate(CATEGORICAL) if index < len(categories)}


@app.post("/api/predict", response_model=Prediction, tags=["prediction"],
          responses={422: {"model": ErrorResponse},
                     503: {"model": ErrorResponse}})
def predict_one(transaction: Transaction,
                 bundle=Depends(get_bundle)) -> Prediction:
    """Score one transaction."""
    records = [transaction.model_dump()]
    frame = bundle.build_frame(records)
    probability = float(bundle.predict(records)[0])

    return _to_prediction(frame, probability, bundle.threshold, explain)[0]


@app.post("/api/predict/batch", response_model=BatchResponse, tags=["prediction"],
          responses={422: {"model": ErrorResponse},
                     503: {"model": ErrorResponse}})
def predict_batch(request: BatchRequest,
                  bundle=Depends(get_bundle)) -> BatchResponse:
    """Score up to 1000 transactions in one call."""
    start = time.perf_counter()

    records = [transaction.model_dump() for transaction in request.transactions]
    frame = bundle.build_frame(records)
    probabilities = bundle.predict(records)

    predictions = _to_prediction(frame, probabilities, bundle.threshold, explain)
    flagged = sum(1 for p in predictions if p.flagged_for_review)

    return BatchResponse(
        model=bundle.name,
        threshold=bundle.threshold,
        fraud_rate_in_batch=round(flagged / len(predictions), 4),
        predictions=predictions,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )


@app.exception_handler(Exception)
async def unhandled_error(request: Request, error: Exception) -> JSONResponse:
    """Never leak a stack trace or an internal path to a client."""
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Check the server logs."},
    )


def _to_prediction(frame, probabilities, threshold: float, explain_fn) -> list[Prediction]:
    # predict_one passes a bare float; normalise so both callers share one path.
    scores = np.atleast_1d(np.asarray(probabilities, dtype=float))
    reasons = explain_fn(frame, scores)
    out: list[Prediction] = []
    for probability, notes in zip(scores, reasons):
        probability = float(probability)
        flagged = probability >= threshold
        out.append(Prediction(
            fraud_probability=round(probability, 6),
            prediction="fraud" if flagged else "legitimate",
            flagged_for_review=flagged,
            risk_band=risk_band(probability, threshold),
            reasons=notes,
        ))
    return out
