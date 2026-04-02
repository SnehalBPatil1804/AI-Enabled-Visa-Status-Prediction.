"""Central paths and settings. All paths are relative to project root."""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = project_root()

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
NOTEBOOKS_DIR = ROOT / "notebooks"
SCREENSHOTS_DIR = ROOT / "screenshots"

# Optional override: set VISA_DATA_FILE to full path of xlsx/csv
ENV_DATA_FILE = os.environ.get("VISA_DATA_FILE", "").strip()

PROCESSED_CSV = DATA_PROCESSED / "visa_processed.csv"
SEGMENT_STATS_JSON = REPORTS_DIR / "segment_stats.json"
MODEL_METRICS_JSON = REPORTS_DIR / "model_metrics.json"

REGRESSION_MODEL = MODELS_DIR / "regression_model.joblib"
CLASSIFICATION_MODEL = MODELS_DIR / "classification_model.joblib"

# Backward compatibility
LEGACY_REGRESSION = MODELS_DIR / "best_processing_time_model.joblib"
LEGACY_CLASSIFICATION = MODELS_DIR / "best_status_model.joblib"