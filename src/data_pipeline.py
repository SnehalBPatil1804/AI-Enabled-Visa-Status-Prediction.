from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "visa_processed.csv"

# Priority: env VISA_DATA_FILE -> data/raw/*.xlsx/csv -> synthetic
ENV_DATA = os.environ.get("VISA_DATA_FILE", "").strip()


def _list_raw_candidates() -> list[Path]:
    names = [
        "Uncleaned_data.xlsx",
        "LCA_Disclosure_Data_FY2026_Q1.xlsx",
        "visa_applications.csv",
    ]
    return [RAW_DIR / n for n in names if (RAW_DIR / n).exists()]


def generate_synthetic_data(n_rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    visa_types = ["Tourist", "Student", "Work", "Business"]
    countries = ["India", "Nepal", "Brazil", "Nigeria", "Germany", "Japan"]
    offices = ["Delhi", "Mumbai", "Berlin", "Tokyo", "SaoPaulo"]

    submission_dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 365, size=n_rows), unit="D"
    )
    month = submission_dates.month

    seasonal = np.where((month >= 6) & (month <= 9), 8, 0) + np.where(month == 12, 6, 0)
    base = rng.normal(loc=28, scale=6, size=n_rows)
    office_load = rng.choice([0, 4, 7, 2, 5], size=n_rows)
    visa_factor = rng.choice([2, 5, 9, 4], size=n_rows)
    country_factor = rng.choice([1, 3, 2, 4, 1, 2], size=n_rows)

    processing_days = np.clip(base + seasonal + office_load + visa_factor + country_factor, 7, 120).round().astype(
        int
    )
    decision_dates = submission_dates + pd.to_timedelta(processing_days, unit="D")

    status = np.where(processing_days < 45, "Certified", "Denied")

    return pd.DataFrame(
        {
            "application_id": np.arange(1, n_rows + 1),
            "submission_date": submission_dates,
            "decision_date": decision_dates,
            "visa_type": rng.choice(visa_types, size=n_rows),
            "applicant_country": rng.choice(countries, size=n_rows),
            "processing_office": rng.choice(offices, size=n_rows),
            "status": status,
        }
    )


def load_raw_data() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ENV_DATA:
        path = Path(ENV_DATA)
        if not path.exists():
            raise FileNotFoundError(f"VISA_DATA_FILE not found: {path}")
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        return pd.read_csv(path)

    candidates = _list_raw_candidates()
    if candidates:
        path = candidates[0]
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        return pd.read_csv(path)

    csv_path = RAW_DIR / "visa_applications.csv"
    df = generate_synthetic_data()
    df.to_csv(csv_path, index=False)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "RECEIVED_DATE" in df.columns and "DECISION_DATE" in df.columns:
        df = df.rename(
            columns={
                "RECEIVED_DATE": "submission_date",
                "DECISION_DATE": "decision_date",
                "VISA_CLASS": "visa_type",
                "EMPLOYER_COUNTRY": "applicant_country",
                "WORKSITE_STATE": "processing_office",
                "CASE_STATUS": "status",
            }
        )

    if len(df) > 200000:
        df = df.sample(n=200000, random_state=42)

    df["submission_date"] = pd.to_datetime(df["submission_date"], errors="coerce")
    df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")
    df = df.dropna(subset=["submission_date", "decision_date", "visa_type", "applicant_country", "processing_office"])

    df["processing_days"] = (df["decision_date"] - df["submission_date"]).dt.days
    df = df[df["processing_days"] >= 0].copy()

    df["submission_month"] = df["submission_date"].dt.month
    df["submission_quarter"] = df["submission_date"].dt.quarter
    df["is_peak_season"] = df["submission_month"].isin([6, 7, 8, 9, 12]).astype(int)

    df["country_avg_days"] = df.groupby("applicant_country")["processing_days"].transform("mean")
    df["office_avg_days"] = df.groupby("processing_office")["processing_days"].transform("mean")

    if "status" in df.columns:
        status_norm = df["status"].astype(str).str.strip().str.upper()
        certified = {"CERTIFIED", "CERTIFIED-WITHDRAWN", "APPROVED"}
        denied = {"DENIED", "REJECTED"}
        df["status_binary"] = np.where(
            status_norm.isin(certified),
            1,
            np.where(status_norm.isin(denied), 0, np.nan),
        )

    return df


def run_pipeline() -> pd.DataFrame:
    df_raw = load_raw_data()
    df_processed = preprocess(df_raw)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(PROCESSED_PATH, index=False)
    return df_processed


if __name__ == "__main__":
    data = run_pipeline()
    print(f"Processed rows: {len(data)}")
    print(f"Saved to: {PROCESSED_PATH}")