"""Request and response schemas.

Validation lives here rather than in the model code so that a bad request is
rejected with a clear message before any expensive work happens, and so the
OpenAPI docs the frontend is generated from are accurate.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Transaction(BaseModel):
    """One raw transaction, exactly as it appears in the source CSV."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trans_date_trans_time": "2020-06-21 23:44:07",
                "amt": 1024.55,
                "cc_num": 371449635398431,
                "merchant": "fraud_Rippin, Kub and Mann",
                "category": "gas_transport",
                "gender": "F",
                "state": "NC",
                "lat": 36.0788,
                "long": -81.1781,
                "city_pop": 3495,
                "dob": "1988-03-09",
                "merch_lat": 36.0113,
                "merch_long": -82.0483,
            }
        }
    )

    trans_date_trans_time: datetime = Field(
        ..., description="When the transaction happened (ISO 8601)")
    amt: float = Field(..., gt=0, le=100_000,
                       description="Amount in dollars, must be greater than 0")
    cc_num: int = Field(..., description="Card number")
    merchant: str = Field(..., min_length=1, max_length=200,
                          description="Merchant name")
    category: str = Field(..., min_length=1, max_length=64,
                          description="Merchant category")
    gender: Literal["F", "M"] = Field(..., description="Cardholder gender")
    state: str = Field(..., min_length=2, max_length=2,
                       description="Two-letter US state code")
    lat: float = Field(..., ge=-90, le=90, description="Cardholder latitude")
    long: float = Field(..., ge=-180, le=180, description="Cardholder longitude")
    city_pop: int = Field(..., ge=0, le=10_000_000,
                          description="Cardholder city population")
    dob: datetime = Field(..., description="Cardholder date of birth")
    merch_lat: float = Field(..., ge=-90, le=90, description="Merchant latitude")
    merch_long: float = Field(..., ge=-180, le=180, description="Merchant longitude")

    @field_validator("dob")
    @classmethod
    def dob_must_be_in_the_past(cls, value: datetime) -> datetime:
        if value > datetime.now():
            raise ValueError("date of birth cannot be in the future")
        return value

    @field_validator("trans_date_trans_time")
    @classmethod
    def transaction_cannot_be_in_the_future(cls, value: datetime) -> datetime:
        if value > datetime.now():
            raise ValueError("transaction time cannot be in the future")
        return value


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(
        ..., min_length=1, max_length=1000,
        description="Between 1 and 1000 transactions to score in one call")


class Prediction(BaseModel):
    """One scored transaction."""

    fraud_probability: float = Field(
        ..., ge=0, le=1,
        description="Model's estimate that this transaction is fraudulent")
    prediction: Literal["fraud", "legitimate"] = Field(
        ..., description="Verdict after applying the decision threshold")
    flagged_for_review: bool = Field(
        ..., description="True when the score clears the review threshold")
    risk_band: Literal["low", "medium", "high"] = Field(
        ..., description="Coarse band around the score, for display only")
    reasons: list[str] = Field(
        default_factory=list,
        description="Which inputs pushed this transaction towards fraud")


class BatchResponse(BaseModel):
    model: str = Field(..., description="Name of the model that produced these scores")
    threshold: float = Field(
        ..., description="Decision threshold, chosen on validation data")
    fraud_rate_in_batch: float = Field(
        ..., description="Share of this batch that was flagged")
    predictions: list[Prediction]
    latency_ms: float


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class ModelInfo(BaseModel):
    name: str
    threshold: float
    validation_pr_auc: float
    test_pr_auc: float | None
    test_roc_auc: float | None
    test_precision: float | None
    test_recall: float | None
    test_f1: float | None
    n_features: int
    n_train_rows: int
    trained_on: str
    features: list[str]
    top_features: list[FeatureImportance]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str | None


class ErrorResponse(BaseModel):
    detail: str
