from __future__ import annotations

import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_pipeline import run_pipeline  # noqa: E402

REPORTS_DIR = ROOT / "reports"


def run_eda() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = run_pipeline()
    sns.set_style("whitegrid")

    plt.figure(figsize=(8, 4))
    sns.histplot(df["processing_days"], bins=30, kde=True)
    plt.title("Processing Time Distribution")
    plt.xlabel("Processing Days")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "processing_time_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df, x="visa_type", y="processing_days")
    plt.title("Processing Days by Visa Type")
    plt.xlabel("Visa Type")
    plt.ylabel("Processing Days")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "processing_by_visa_type.png")
    plt.close()

    monthly = df.groupby("submission_month", as_index=False)["processing_days"].mean()
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=monthly, x="submission_month", y="processing_days", marker="o")
    plt.title("Monthly Average Processing Time")
    plt.xlabel("Submission Month")
    plt.ylabel("Avg Processing Days")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "monthly_trend.png")
    plt.close()

    trend_table = (
        df.groupby(["processing_office", "visa_type"], as_index=False)["processing_days"]
        .mean()
        .sort_values("processing_days", ascending=False)
    )
    trend_table.to_csv(REPORTS_DIR / "trend_summary.csv", index=False)


if __name__ == "__main__":
    run_eda()
    print("EDA outputs saved in reports/")