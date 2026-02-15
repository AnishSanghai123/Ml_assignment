import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
)

import matplotlib.pyplot as plt
# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="ML Model Comparator", layout="wide")

MODEL_DIR = Path("models")

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logreg.pkl",
    "Decision Tree": MODEL_DIR / "dtree.pkl",
    "KNN (k=5)": MODEL_DIR / "knn.pkl",
    "Gaussian Naive Bayes": MODEL_DIR / "gnb.pkl",
    "Random Forest": MODEL_DIR / "rf.pkl",
    "XGBoost": MODEL_DIR / "xgb.pkl",
}

METRICS_CSV = Path("metrics.csv")
FEATURES_JSON = Path("feature_columns.json")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_feature_columns() -> list[str]:
    if not FEATURES_JSON.exists():
        raise FileNotFoundError(f"Missing {FEATURES_JSON}. Please add it to your repo.")
    with open(FEATURES_JSON, "r") as f:
        return json.load(f)


@st.cache_data
def load_metrics_table() -> pd.DataFrame:
    if not METRICS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(METRICS_CSV)


@st.cache_resource
def load_model(model_name: str):
    path = MODEL_FILES[model_name]
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    return joblib.load(path)


def validate_and_prepare_input(df: pd.DataFrame, required_features: list[str]) -> pd.DataFrame:
    missing = [c for c in required_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # keep required, ignore extras, reorder
    X = df[required_features].copy()

    # numeric conversion
    for c in required_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    if X.isna().any().any():
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"Non-numeric or missing values detected in columns: {bad_cols}. "
            "Please ensure all required feature columns are numeric and contain no blanks."
        )
    return X


def get_scores(model, X: pd.DataFrame):
    y_pred = model.predict(X)

    y_score = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            y_score = proba[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X)

    return y_pred, y_score


def plot_confusion_matrix(cm, labels=("0", "1"), title="Confusion Matrix"):
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    st.pyplot(fig)


def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1])
    ax.set_title(f"ROC Curve (AUC = {auc_val:.4f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    st.pyplot(fig)


def plot_pr_curve(y_true, y_score):
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    fig, ax = plt.subplots()
    ax.plot(rec, prec)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    st.pyplot(fig)


def normalize_binary_labels(series: pd.Series, positive_label_hint: str = "1") -> pd.Series:
    """
    Converts common binary label formats into 0/1.
    Supports numeric (0/1), and strings like:
      - "benign"/"malignant"
      - "B"/"M"
      - "yes"/"no", "true"/"false"
    """
    s = series.copy()

    # If already numeric 0/1
    if pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(s, errors="coerce")
        return s

    # Normalize strings
    s = s.astype(str).str.strip().str.lower()

    mapping = {
        "1": 1, "0": 0,
        "true": 1, "false": 0,
        "yes": 1, "no": 0,
        "y": 1, "n": 0,
        "benign": 1, "malignant": 0,
        "b": 1, "m": 0,
        "positive": 1, "negative": 0,
    }

    # Try direct map
    mapped = s.map(mapping)

    # If mapping didn't work for most values, fallback:
    # treat whatever user says is "positive label" as 1, rest as 0
    if mapped.isna().mean() > 0.5:
        pos = str(positive_label_hint).strip().lower()
        mapped = (s == pos).astype(int)

    return mapped


# -----------------------------
# UI
# -----------------------------
st.title("📌 ML Classification Models - Streamlit App")

required_features = load_feature_columns()
metrics_df = load_metrics_table()

with st.sidebar:
    st.header("Controls")
    selected_model_name = st.selectbox("Select Model", list(MODEL_FILES.keys()))

    st.subheader("CSV / Labels")
    target_col_guess = st.text_input(
        "Target column name (for confusion matrix/report)",
        value="target",
        help="If your uploaded CSV includes ground-truth labels, enter the column name here.",
    )
    positive_label_hint = st.text_input(
        "Positive label value (only if labels are text)",
        value="benign",
        help="Examples: benign / 1 / yes / true. Used only if label mapping is unclear.",
    )

    max_rows = st.number_input(
        "Max rows to use (free tier friendly)",
        min_value=10,
        max_value=5000,
        value=1000,
        step=50,
    )
    show_extra_plots = st.checkbox("Show ROC & PR plots (if possible)", value=True)

st.subheader("1) Upload Test Dataset (CSV)")
# --- CSV Template download ---
with st.expander("📥 Download CSV Template (recommended)"):
    st.write("Use this template to avoid column mismatch. Fill numeric feature values. Optionally include a target column.")
    include_target = st.checkbox("Include target column in template", value=True)

    template_cols = required_features + ([target_col_guess] if include_target else [])
    template_df = pd.DataFrame(columns=template_cols)

    st.download_button(
        label="⬇️ Download Template CSV",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="input_template.csv",
        mime="text/csv",
    )

uploaded = st.file_uploader("Upload CSV", type=["csv"])

# --- Metrics display ---
st.subheader("2) Model Evaluation Metrics (from saved results)")
if metrics_df.empty:
    st.info("metrics.csv not found in /model. Add it to show evaluation metrics here.")
else:
    row = metrics_df[metrics_df["Model"] == selected_model_name]
    if row.empty:
        st.warning(f"No row found in metrics.csv for model '{selected_model_name}'. Showing full table.")
        st.dataframe(metrics_df)
    else:
        r = row.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{r['Accuracy']:.4f}")
        c1.metric("Precision", f"{r['Precision']:.4f}")
        c2.metric("Recall", f"{r['Recall']:.4f}")
        c2.metric("F1", f"{r['F1']:.4f}")
        c3.metric("AUC", f"{r['AUC']:.4f}")
        c3.metric("MCC", f"{r['MCC']:.4f}")

        with st.expander("Show full metrics table"):
            st.dataframe(metrics_df)

# --- Prediction section ---
st.subheader("3) Run Predictions")

if uploaded is None:
    st.info("Upload a CSV to run predictions.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if df.empty:
    st.error("Uploaded CSV is empty.")
    st.stop()

if len(df) > max_rows:
    st.warning(f"Uploaded {len(df)} rows. Using first {max_rows} rows to keep the app responsive.")
    df = df.head(int(max_rows))

st.write("Preview of uploaded data:")
st.dataframe(df.head())

# Check for extra cols info (optional)
extra_cols = [c for c in df.columns if c not in required_features and c != target_col_guess]
if extra_cols:
    st.info(f"Extra columns detected and ignored for prediction: {extra_cols}")

try:
    X_input = validate_and_prepare_input(df, required_features)
except ValueError as e:
    st.error(str(e))
    st.stop()

try:
    model = load_model(selected_model_name)
except Exception as e:
    st.error(str(e))
    st.stop()

run = st.button("✅ Run Prediction")
if not run:
    st.stop()

y_pred, y_score = get_scores(model, X_input)

out = df.copy()
out["prediction"] = y_pred
if y_score is not None:
    out["probability_positive_class"] = y_score

st.success("Predictions generated!")
st.dataframe(out.head())

st.download_button(
    label="⬇️ Download Predictions CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv",
)

# --- Confusion matrix / report ---
st.subheader("4) Confusion Matrix / Classification Report")

if target_col_guess not in df.columns:
    st.warning(
        f"No ground-truth label column named '{target_col_guess}' found in uploaded CSV. "
        "Upload test data with labels to generate confusion matrix / classification report."
    )
else:
    y_true_raw = df[target_col_guess]
    y_true = normalize_binary_labels(y_true_raw, positive_label_hint=positive_label_hint)

    if pd.isna(y_true).any():
        st.error(
            f"Target column '{target_col_guess}' contains missing/unknown label values. "
            "Please clean the labels or use the template."
        )
        st.stop()

    cm = confusion_matrix(y_true, y_pred)
    colA, colB = st.columns(2)

    with colA:
        plot_confusion_matrix(cm, labels=("0", "1"))

    with colB:
        st.text("Classification Report")
        st.code(classification_report(y_true, y_pred, digits=4))

    # Optional plots
    if show_extra_plots:
        if y_score is None:
            st.info("ROC/PR plots require probability or decision scores. This model did not provide them.")
        else:
            st.subheader("Optional Plots")
            c1, c2 = st.columns(2)
            with c1:
                plot_roc_curve(y_true, y_score)
            with c2:
                plot_pr_curve(y_true, y_score)

