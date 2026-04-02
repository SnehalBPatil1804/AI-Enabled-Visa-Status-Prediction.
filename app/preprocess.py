"""Preprocessing entrypoint — delegates to src/data_pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_pipeline import run_pipeline  # noqa: E402

PROCESSED_CSV = ROOT / "data" / "processed" / "visa_processed.csv"


def run() -> Path:
    run_pipeline()
    return PROCESSED_CSV


if __name__ == "__main__":
    run()
    print(f"Processed: {PROCESSED_CSV}")