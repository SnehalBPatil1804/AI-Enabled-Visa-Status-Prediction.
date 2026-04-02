"""
AI Enabled Visa Status Prediction & Processing Time Estimator — Streamlit UI.
Run from project root: streamlit run app/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import pandas as pd
import streamlit as st

from app.config import DATA_PROCESSED, REPORTS_DIR
from app.predict import VisaInput, data_file_hint, predict_all
from app.utils import distribution_figure_from_processed, load_trend_summary, monthly_trend_chart, read_json

st.set_page_config(
    page_title="Visa AI Estimator",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #312e81 100%);
    padding: 2.2rem 2rem;
    border-radius: 20px;
    color: #f8fafc;
    margin-bottom: 1.5rem;
    box-shadow: 0 20px 50px rgba(15, 23, 42, 0.35);
    animation: fadeIn 0.8s ease-out;
  }
  .hero h1 { margin: 0 0 0.5rem 0; font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em; }
  .hero p { margin: 0; opacity: 0.92; font-size: 1.05rem; }
  .card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(148,163,184,0.25);
    border-radius: 16px;
    padding: 1.25rem 1.35rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
  }
  .badge {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.02em;
  }
  .badge-cert { background: linear-gradient(90deg,#22c55e,#4ade80); color:#052e16; }
  .badge-den { background: linear-gradient(90deg,#f97316,#fb923c); color:#431407; }
  @keyframes fadeIn { from { opacity:0; transform: translateY(8px);} to { opacity:1; transform: none;} }
  .stButton>button {
    background: linear-gradient(90deg,#6366f1,#8b5cf6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.4rem;
    font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
  }
  .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 10px 25px rgba(99,102,241,0.35); }
  div[data-testid="stMetric"] { background: rgba(15,23,42,0.5); border-radius: 12px; padding: 0.75rem; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

hero = """
<div class="hero">
  <h1>AI Enabled Visa Status & Processing Time</h1>
  <p>Data-driven estimates from historical LCA-style records — transparency for applicants and reviewers.</p>
</div>
"""
st.markdown(hero, unsafe_allow_html=True)

col_left, col_right = st.columns([1.05, 1.0], gap="large")

with col_left:
    st.markdown("### Application details")
    with st.container():
        visa_type = st.text_input("Visa class (e.g. H-1B)", value="H-1B", help="As reported in disclosure data (VISA_CLASS).")
        country = st.text_input("Employer country", value="USA", help="Mapped from EMPLOYER_COUNTRY in source data.")
        office = st.text_input("Worksite state / region", value="CA", help="Mapped from WORKSITE_STATE.")
        month = st.slider("Submission month", 1, 12, 7)

    submitted = st.button("Run AI estimation", type="primary", use_container_width=True)

with col_right:
    st.markdown("### Insights from your dataset")
    trend_path = DATA_PROCESSED / "visa_processed.csv"
    monthly = None
    if trend_path.exists():
        df_hist = pd.read_csv(trend_path, nrows=80000)
        if "submission_month" in df_hist.columns and "processing_days" in df_hist.columns:
            monthly = df_hist.groupby("submission_month", as_index=False)["processing_days"].mean()
    if monthly is not None and len(monthly) > 0:
        fig_m = monthly_trend_chart(monthly)
        st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.info("Train the pipeline to populate trend charts.")

    processed_csv = DATA_PROCESSED / "visa_processed.csv"
    fig_h = distribution_figure_from_processed(processed_csv)
    if fig_h is not None:
        st.plotly_chart(fig_h, use_container_width=True)

metrics_json = read_json(REPORTS_DIR / "model_metrics.json")
with st.sidebar:
    st.markdown("### Project health")
    st.caption("Data file hint")
    st.code(data_file_hint(), language="text")
    if metrics_json.get("regression", {}).get("best_model"):
        st.success(f"Regression model: **{metrics_json['regression']['best_model']}**")
    if metrics_json.get("classification", {}).get("best_model"):
        st.success(f"Classification model: **{metrics_json['classification']['best_model']}**")
    st.markdown("---")
    st.markdown("**Quick start**")
    st.markdown(
        """
1. `pip install -r requirements.txt`  
2. `python -m src.train`  
3. `streamlit run app/main.py`
        """
    )

result_container = st.container()
if submitted:
    with st.spinner("Running models — applying preprocessing & segment statistics..."):
        time.sleep(0.35)
        try:
            out = predict_all(
                VisaInput(
                    visa_type=visa_type.strip(),
                    applicant_country=country.strip(),
                    processing_office=office.strip(),
                    submission_month=int(month),
                )
            )
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    with result_container:
        st.markdown("### Results")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Predicted processing (days)", f"{out['predicted_days']}")
        with c2:
            st.metric("Estimated range", out.get("estimated_range", "—"))
        with c3:
            conf = out.get("status_confidence")
            st.metric("Status confidence", f"{conf}" if conf is not None else "—")

        status = out.get("case_status", "Unknown")
        if status == "Certified":
            st.markdown(
                f'<span class="badge badge-cert">Case status: {status}</span>',
                unsafe_allow_html=True,
            )
        elif status == "Denied":
            st.markdown(
                f'<span class="badge badge-den">Case status: {status}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.info(f"Case status: **{status}**")

        st.caption(out.get("interval_note", ""))
        if "note" in out:
            st.warning(out["note"])

        st.markdown("#### What this means")
        st.write(
            "The **processing-time** estimate uses gradient-boosted regression on historical features. "
            "The **Certified / Denied** prediction reflects a balanced classifier on case outcomes where labels exist. "
            "Always treat outputs as decision-support, not legal advice."
        )

st.markdown("---")
st.caption("Infosys Internship 6.0 · Visa Analytics Demo")