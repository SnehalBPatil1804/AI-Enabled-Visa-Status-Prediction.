from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_trend_summary(reports_dir: Path) -> pd.DataFrame | None:
    p = reports_dir / "trend_summary.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def monthly_trend_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        df,
        x="submission_month",
        y="processing_days",
        markers=True,
        title="Average processing time by submission month",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def distribution_figure_from_processed(processed_csv: Path) -> go.Figure | None:
    if not processed_csv.exists():
        return None
    df = pd.read_csv(processed_csv, usecols=["processing_days"], nrows=50000)
    fig = px.histogram(df, x="processing_days", nbins=40, opacity=0.85)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        title="Historical processing time (days)",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig