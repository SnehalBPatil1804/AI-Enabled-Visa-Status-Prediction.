from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.predict import VisaInput, predict_all  # noqa: E402

app = FastAPI(title="Visa AI API", version="2.0.0")


class EstimateRequest(BaseModel):
    visa_type: str
    applicant_country: str
    processing_office: str
    submission_month: int = Field(ge=1, le=12)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/estimate")
def estimate(request: EstimateRequest) -> dict:
    return predict_all(
        VisaInput(
            visa_type=request.visa_type,
            applicant_country=request.applicant_country,
            processing_office=request.processing_office,
            submission_month=request.submission_month,
        )
    )