"""
Internship project — training script for:
  1) Regression: estimated processing time (days)
  2) Classification: Certified (1) vs Denied (0) under SEVERE CLASS IMBALANCE

Typical LCA data has far more Certified than Denied rows. Accuracy alone is misleading.
We combine:
  - class_weight='balanced' (sklearn re-weights loss by inverse frequency)
  - optional TRAINING-ONLY undersampling of the majority class (test set stays real)
  - metrics suited to imbalance: balanced accuracy, average precision (PR-AUC), per-class recall

Environment (optional):
  VISA_CLF_UNDERSAMPLE=1     — cap majority samples to at most RATIO × minority count (training only)
  VISA_CLF_MAJORITY_RATIO=20 — default ratio (e.g. 20 → at most 20×392 ≈ 7840 certified rows for training)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_pipeline import run_pipeline  # noqa: E402

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Denied = 0, Certified = 1 (matches status_binary in data_pipeline)
LABEL_DENIED = 0
LABEL_CERTIFIED = 1


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_preprocessor() -> ColumnTransformer:
    categorical = ["visa_type", "applicant_country", "processing_office"]
    numeric = ["submission_month", "submission_quarter", "is_peak_season", "country_avg_days", "office_avg_days"]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", "passthrough", numeric),
        ]
    )


def build_segment_stats(df: pd.DataFrame) -> dict:
    gm = float(df["processing_days"].median())
    by_c = df.groupby("applicant_country")["processing_days"].median()
    by_o = df.groupby("processing_office")["processing_days"].median()
    return {
        "global_median_days": gm,
        "median_by_country": {str(k).strip().upper(): float(v) for k, v in by_c.items()},
        "median_by_office": {str(k).strip().upper(): float(v) for k, v in by_o.items()},
    }


def _undersample_majority_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    majority_ratio: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Reduce majority-class rows in TRAINING only so the classifier sees a less skewed distribution.
    Test set must remain untouched (real distribution for honest evaluation).
    """
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    idx_min = np.where(y_train.values == LABEL_DENIED)[0]
    idx_maj = np.where(y_train.values == LABEL_CERTIFIED)[0]
    n_min = len(idx_min)
    n_maj = len(idx_maj)
    if n_min < 5 or n_maj == 0:
        return X_train, y_train

    cap = min(n_maj, majority_ratio * n_min)
    if cap >= n_maj:
        return X_train, y_train

    rng = np.random.default_rng(random_state)
    picked_maj = rng.choice(idx_maj, size=cap, replace=False)
    keep = np.sort(np.concatenate([idx_min, picked_maj]))
    return X_train.iloc[keep].reset_index(drop=True), y_train.iloc[keep].reset_index(drop=True)


def classification_metrics_report(y_true: np.ndarray, y_pred: np.ndarray, y_prob_certified: np.ndarray) -> dict:
    """Metrics appropriate for imbalanced binary problems (internship report)."""
    # Per-class precision/recall/F1 (0=Denied, 1=Certified)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[LABEL_DENIED, LABEL_CERTIFIED], zero_division=0)

    out: dict = {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob_certified)),
        "average_precision": float(average_precision_score(y_true, y_prob_certified)),
        "precision_denied_class0": float(p[0]),
        "recall_denied_class0": float(r[0]),
        "f1_denied_class0": float(f[0]),
        "precision_certified_class1": float(p[1]),
        "recall_certified_class1": float(r[1]),
        "f1_certified_class1": float(f[1]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[LABEL_DENIED, LABEL_CERTIFIED]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[LABEL_DENIED, LABEL_CERTIFIED],
            target_names=["Denied", "Certified"],
            zero_division=0,
        ),
    }
    return out


def train() -> dict:
    df = run_pipeline()
    stats = build_segment_stats(df)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "segment_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    feature_cols = [
        "visa_type",
        "applicant_country",
        "processing_office",
        "submission_month",
        "submission_quarter",
        "is_peak_season",
        "country_avg_days",
        "office_avg_days",
    ]
    X = df[feature_cols]
    y = df["processing_days"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_defs = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    reg_scores = {}
    best_name = None
    best_mae = float("inf")
    best_pipeline = None
    best_residual_std = None

    for name, model in model_defs.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", model)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": rmse(y_test, preds),
            "r2": float(r2_score(y_test, preds)),
        }
        reg_scores[name] = metrics

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_name = name
            best_pipeline = pipeline
            best_residual_std = float(np.std(y_test - preds))

    classification_metadata: dict = {}
    if "status_binary" in df.columns:
        cls_df = df.dropna(subset=["status_binary"]).copy()
        if cls_df["status_binary"].nunique() >= 2 and len(cls_df) > 100:
            Xc = cls_df[feature_cols]
            yc = cls_df["status_binary"].astype(int)

            Xc_train, Xc_test, yc_train, yc_test = train_test_split(
                Xc, yc, test_size=0.2, random_state=42, stratify=yc
            )

            use_undersample = os.environ.get("VISA_CLF_UNDERSAMPLE", "").strip() in ("1", "true", "True", "yes")
            majority_ratio = int(os.environ.get("VISA_CLF_MAJORITY_RATIO", "20"))

            Xc_fit, yc_fit = Xc_train, yc_train
            undersample_note = "disabled (set VISA_CLF_UNDERSAMPLE=1 to enable training-only majority cap)"
            if use_undersample:
                Xc_fit, yc_fit = _undersample_majority_train(Xc_train, yc_train, majority_ratio=majority_ratio)
                undersample_note = (
                    f"training rows capped: majority at most {majority_ratio}× minority count; "
                    f"train shape after cap {len(yc_fit)} (test set unchanged)"
                )

            cls_models = {
                "logistic_balanced": LogisticRegression(
                    solver="liblinear",
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
                "random_forest_balanced": RandomForestClassifier(
                    n_estimators=250,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            }

            cls_scores: dict = {}
            best_cls_name = None
            best_cls_score = -1.0
            best_cls_pipeline = None

            # Select best model by macro F1 (fairer than default F1 when classes skewed)
            for name, model in cls_models.items():
                cls_pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", model)])
                cls_pipeline.fit(Xc_fit, yc_fit)
                yc_pred = cls_pipeline.predict(Xc_test)
                yc_prob = cls_pipeline.predict_proba(Xc_test)[:, 1]

                cls_metrics = classification_metrics_report(
                    yc_test.values,
                    yc_pred,
                    yc_prob,
                )
                cls_scores[name] = cls_metrics

                score = cls_metrics["f1_macro"]
                if score > best_cls_score:
                    best_cls_score = score
                    best_cls_name = name
                    best_cls_pipeline = cls_pipeline

            if best_cls_pipeline is not None:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                joblib.dump(best_cls_pipeline, MODELS_DIR / "classification_model.joblib")
                joblib.dump(best_cls_pipeline, MODELS_DIR / "best_status_model.joblib")

                n_denied = int((yc == LABEL_DENIED).sum())
                n_cert = int((yc == LABEL_CERTIFIED).sum())
                imbalance_ratio = round(n_cert / max(n_denied, 1), 2)

                classification_metadata = {
                    "internship_note": (
                        "LCA-style data is often highly imbalanced (many Certified, few Denied). "
                        "We use class_weight='balanced' and report macro-F1, balanced accuracy, "
                        "average precision (PR-AUC), and per-class recall — not accuracy alone."
                    ),
                    "best_model": best_cls_name,
                    "model_selection_metric": "f1_macro (higher is better across both classes)",
                    "metrics": cls_scores,
                    "class_distribution_full_dataset": {
                        "denied": n_denied,
                        "certified": n_cert,
                        "approx_ratio_certified_to_denied": imbalance_ratio,
                    },
                    "balancing_strategy": {
                        "sklearn_class_weight": "balanced",
                        "training_undersample_majority": undersample_note,
                    },
                    "label_map": {"0": "Denied", "1": "Certified"},
                }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODELS_DIR / "regression_model.joblib")
    joblib.dump(best_pipeline, MODELS_DIR / "best_processing_time_model.joblib")

    metadata = {
        "regression": {
            "best_model": best_name,
            "metrics": reg_scores,
            "residual_std": best_residual_std,
        },
        "classification": classification_metadata,
    }
    (REPORTS_DIR / "model_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    sample_cols = [
        c
        for c in ["visa_type", "applicant_country", "processing_office", "submission_month", "processing_days"]
        if c in df.columns
    ]
    if sample_cols:
        (ROOT / "data").mkdir(parents=True, exist_ok=True)
        df[sample_cols].head(25).to_csv(ROOT / "data" / "sample_input.csv", index=False)

    return metadata


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))