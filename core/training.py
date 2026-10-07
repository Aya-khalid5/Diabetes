"""Training + evaluation, identical to main.py / diabetes.ipynb.

Used by:
  * train_model.py            (one-off script that writes model/ files)
  * core.predictor (fallback) (re-train in memory if the saved model cannot be loaded)

The model, split and hyper-parameters are exactly those of the original project.
This module only ADDS the bookkeeping the app needs (metadata + evaluation numbers).
"""
from __future__ import annotations

import platform
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from . import config
from .compact_forest import CompactForest

# (derived label column, numeric source column) pairs present in Clean_diabetes_data.csv
CATEGORY_SOURCES = {
    "General_Health": "GenHlth",
    "Gender": "Sex",
    "Age_Category": "Age",
    "Education_Level": "Education",
    "Income_Level": "Income",
}


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Same preparation as main.py: drop targets, one-hot any categorical column."""
    missing = [c for c in config.TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Target column(s) missing from the dataset: {missing}")
    X_raw = df.drop(columns=config.TARGET_COLUMNS)
    y = df[config.TARGET]
    X = pd.get_dummies(X_raw)
    return X, y


def derive_category_maps(df: pd.DataFrame) -> dict[str, Any]:
    """Learn, FROM THE DATA, how each derived label column relates to its numeric code.

    The chatbot supplies the numeric codes; the model was trained on the one-hot
    version of the label columns. Each label is verified to be a deterministic
    function of the code, so the one-hot columns can be rebuilt at prediction time
    without guessing anything.
    """
    out: dict[str, Any] = {}
    for label_col, code_col in CATEGORY_SOURCES.items():
        nun = df.groupby(code_col)[label_col].nunique()
        if not (nun == 1).all():
            raise ValueError(f"{label_col} is not a deterministic function of {code_col}")
        mapping = df.groupby(code_col)[label_col].first()
        out[label_col] = {
            "source": code_col,
            "type": "map",
            "map": {str(int(k)): str(v) for k, v in mapping.items()},
        }

    # BMI_Category is derived from BMI with the standard 18.5 / 25 / 30 cut-offs.
    # Verify that against the data instead of assuming it.
    bmi = df.groupby("BMI_Category")["BMI"].agg(["min", "max"])
    assert bmi.loc["Underweight", "max"] < 18.5 <= bmi.loc["Normal", "min"]
    assert bmi.loc["Normal", "max"] < 25 <= bmi.loc["Overweight", "min"]
    assert bmi.loc["Overweight", "max"] < 30 <= bmi.loc["Obese", "min"]
    out["BMI_Category"] = {
        "source": "BMI",
        "type": "thresholds",
        "bins": [18.5, 25, 30],
        "labels": ["Underweight", "Normal", "Overweight", "Obese"],
    }
    return out


def _thin(x: np.ndarray, y: np.ndarray, n: int = 120) -> tuple[list[float], list[float]]:
    """Down-sample a curve for compact JSON storage (keeps first and last point)."""
    if len(x) <= n:
        idx = np.arange(len(x))
    else:
        idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return [round(float(v), 5) for v in x[idx]], [round(float(v), 5) for v in y[idx]]


def verify_compact(model, compact: CompactForest, X: pd.DataFrame, chunk: int = 4000) -> dict[str, Any]:
    """Prove the compact forest is a lossless copy: compare with scikit-learn row by row."""
    model.n_jobs = 1
    max_diff, mismatches = 0.0, 0
    for i in range(0, len(X), chunk):
        part = X.iloc[i : i + chunk].astype(float)
        max_diff = max(max_diff, float(np.abs(model.predict_proba(part) - compact.predict_proba(part)).max()))
        mismatches += int((model.predict(part) != compact.predict(part)).sum())
    if mismatches or max_diff > 1e-9:
        raise RuntimeError(
            f"Compact forest differs from scikit-learn (label mismatches={mismatches}, max diff={max_diff})"
        )
    return {"rows_checked": int(len(X)), "label_mismatches": mismatches, "max_abs_proba_diff": max_diff}


def train_and_evaluate(
    df: pd.DataFrame, log=print
) -> tuple[RandomForestClassifier, CompactForest, dict[str, Any]]:
    X, y = prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    log(f"train shape: {X_train.shape} | test shape: {X_test.shape}")

    model = RandomForestClassifier(**config.RF_PARAMS)
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - t0
    log(f"fit done in {fit_seconds:.1f}s")

    # ---------------- evaluation on the held-out 20 % ----------------
    classes = [int(c) for c in model.classes_]
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    y_bin = label_binarize(y_test, classes=classes)
    roc: dict[str, Any] = {}
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        fx, ty = _thin(fpr, tpr)
        roc[str(c)] = {
            "auc": round(float(roc_auc_score(y_bin[:, i], proba[:, i])), 4),
            "fpr": fx,
            "tpr": ty,
        }
    auc_macro = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))

    importance = sorted(
        ({"feature": f, "importance": float(v)} for f, v in zip(X.columns, model.feature_importances_)),
        key=lambda d: d["importance"],
        reverse=True,
    )

    # ---------------- lossless compact copy for deployment ----------------
    compact = CompactForest.from_sklearn(model)
    verification = verify_compact(model, compact, X_test)
    log(f"compact forest verified on {verification['rows_checked']:,} test rows: "
        f"{verification['label_mismatches']} label mismatches")

    # ---------------- bookkeeping ----------------
    label_names = df.groupby(config.TARGET)["Diabetes_Status"].first()
    class_labels = {str(int(k)): str(v) for k, v in label_names.items()}

    def counts(s: pd.Series) -> dict[str, int]:
        return {str(int(k)): int(v) for k, v in s.value_counts().sort_index().items()}

    metadata: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fit_seconds": round(fit_seconds, 1),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "dataset": {
            "file": config.DATA_PATH.name,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "input_columns": int(df.shape[1] - len(config.TARGET_COLUMNS)),
            "model_features": int(X.shape[1]),
            "target": config.TARGET,
            "class_counts": counts(y),
        },
        "split": {
            "test_size": config.TEST_SIZE,
            "random_state": config.RANDOM_STATE,
            "stratified": True,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "class_counts_train": counts(y_train),
            "class_counts_test": counts(y_test),
        },
        "model": {
            "type": "RandomForestClassifier",
            "params": {k: (v if v is not None else None) for k, v in config.RF_PARAMS.items()},
        },
        "compact_verification": verification,
        "feature_columns": list(X.columns),
        "classes": classes,
        "class_labels": class_labels,
        "derived_features": derive_category_maps(df),
        "metrics": {
            "accuracy": acc,
            "roc_auc_ovr_macro": round(auc_macro, 4),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "roc": roc,
            "feature_importance": importance,
        },
    }
    log(f"test accuracy: {acc:.4f}")
    return model, compact, metadata
