"""Load models and return predictions with professional LCA-style labels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import json
import joblib
import pandas as pd

from app.config import (  # noqa: E402
    CLASSIFICATION_MODEL,
    ENV_DATA_FILE,
    LEGACY_CLASSIFICATION,
    LEGACY_REGRESSION,
    MODEL_METRICS_JSON,
    REGRESSION_MODEL,
    REPORTS_DIR,
    SEGMENT_STATS_JSON,
)


@dataclass
class VisaInput:
    visa_type: str
    applicant_country: str
    processing_office: str
    submission_month: int


def _load_segment_stats() -> dict:
    if SEGMENT_STATS_JSON.exists():
        return json.loads(SEGMENT_STATS_JSON.read_text(encoding="utf-8"))
    return {}


def _lookup_avg(stats: dict, key: str, bucket: str, fallback: float) -> float:
    m = stats.get(key) or {}
    v = m.get(str(bucket).strip().upper()) if isinstance(m, dict) else None
    if v is None and isinstance(m, dict):
        v = m.get(bucket)
    return float(v) if v is not None else float(fallback)


def _resolve_regression_path() -> Path:
    if REGRESSION_MODEL.exists():
        return REGRESSION_MODEL
    if LEGACY_REGRESSION.exists():
        return LEGACY_REGRESSION
    return REGRESSION_MODEL


def _resolve_classification_path() -> Path:
    if CLASSIFICATION_MODEL.exists():
        return CLASSIFICATION_MODEL
    if LEGACY_CLASSIFICATION.exists():
        return LEGACY_CLASSIFICATION
    return CLASSIFICATION_MODEL


def _load_residual_std(default_std: float = 30.0) -> float:
    if not MODEL_METRICS_JSON.exists():
        return default_std
    data = json.loads(MODEL_METRICS_JSON.read_text(encoding="utf-8"))
    return float(data.get("regression", {}).get("residual_std", default_std))


def _display_sigma(raw_sigma: float, predicted: float) -> float:
    """Cap interval width for UX (raw residual std can be very large on noisy data)."""
    cap = min(120.0, max(12.0, abs(predicted) * 0.25))
    return min(raw_sigma, cap)


def build_feature_row(data: VisaInput, stats: dict) -> pd.DataFrame:
    quarter = ((data.submission_month - 1) // 3) + 1
    peak = int(data.submission_month in [6, 7, 8, 9, 12])
    gm = float(stats.get("global_median_days", 35.0))
    country_key = str(data.applicant_country).strip().upper()
    office_key = str(data.processing_office).strip().upper()
    c_avg = _lookup_avg(stats, "median_by_country", country_key, gm)
    o_avg = _lookup_avg(stats, "median_by_office", office_key, gm)
    return pd.DataFrame(
        [
            {
                "visa_type": data.visa_type,
                "applicant_country": data.applicant_country,
                "processing_office": data.processing_office,
                "submission_month": data.submission_month,
                "submission_quarter": quarter,
                "is_peak_season": peak,
                "country_avg_days": c_avg,
                "office_avg_days": o_avg,
            }
        ]
    )


def predict_all(data: VisaInput) -> dict:
    reg_path = _resolve_regression_path()
    if not reg_path.exists():
        raise FileNotFoundError(
            "Regression model not found. Run: python -m src.train  (from project root)"
        )

    stats = _load_segment_stats()
    row = build_feature_row(data, stats)
    model = joblib.load(reg_path)
    pred = float(model.predict(row)[0])
    raw_sigma = _load_residual_std()
    sigma = _display_sigma(raw_sigma, pred)
    low = max(1, int(round(pred - sigma)))
    high = max(low + 1, int(round(pred + sigma)))

    result: dict = {
        "predicted_days": round(pred, 2),
        "estimated_range_days": [low, high],
        "estimated_range": f"{low}-{high} days",
        "interval_note": "Range uses capped residual spread for readability; see README for details.",
    }

    cls_path = _resolve_classification_path()
    if cls_path.exists():
        clf = joblib.load(cls_path)
        if hasattr(clf, "predict_proba"):
            prob_cert = float(clf.predict_proba(row)[0][1])
        else:
            prob_cert = float(clf.predict(row)[0])
        cert = prob_cert >= 0.5
        result["case_status"] = "Certified" if cert else "Denied"
        result["status_confidence"] = round(max(prob_cert, 1.0 - prob_cert), 4)
        result["status_probability_certified"] = round(prob_cert, 4)
    else:
        result["case_status"] = "Unknown"
        result["status_confidence"] = None
        result["note"] = "Train with labeled CASE_STATUS to enable status prediction."

    return result


def data_file_hint() -> str:
    if ENV_DATA_FILE:
        return ENV_DATA_FILE
    raw = ROOT / "data" / "raw"
    for name in ("Uncleaned_data.xlsx", "LCA_Disclosure_Data_FY2026_Q1.xlsx", "visa_applications.csv"):
        p = raw / name
        if p.exists():
            return str(p)
    return str(raw / "place_your_dataset_here.xlsx")