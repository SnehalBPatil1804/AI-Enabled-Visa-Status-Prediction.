"""
Run full pipeline: preprocess (via train), EDA, train models.
Usage (from project root): python run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    run([sys.executable, str(ROOT / "src" / "eda.py")])
    run([sys.executable, str(ROOT / "src" / "train.py")])


if __name__ == "__main__":
    main()