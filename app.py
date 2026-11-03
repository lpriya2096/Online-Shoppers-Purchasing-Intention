import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="Online Shopping Purchase Prediction",
                   page_icon="🛒", layout="wide")

TARGET = "Revenue"
EXPECTED_TEST_ROWS = 2466
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "K-Nearest Neighbors (KNN)": "model/knn.joblib",
    "Gaussian Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine (SVM)": "model/svm.joblib",
}


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)


def parse_target(series: pd.Series):
    """Accept True/False or 1/0 labels; return an int series (or None if invalid)."""
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0})
    return None if mapped.isna().any() else mapped.astype(int)


def evaluate(model, X, y):
    """Run a SAVED model on X (no retraining) and return predictions + metrics."""
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, list(model.classes_).index(1)]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, prob) if y.nunique() == 2 else float("nan"),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1 Score": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return pred, metrics, (int(tn), int(fp), int(fn), int(tp))


def formula_table(m, cm):
    """Metric | formula | numbers plugged in | value (computed from the confusion matrix)."""
    tn, fp, fn, tp = cm
    return pd.DataFrame([
        ["Accuracy", "(TP + TN) / (TP + TN + FP + FN)",
         f"({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn})", m["Accuracy"]],
        ["AUC", "Area under the ROC curve (TPR vs FPR over all thresholds)",
         "computed from predicted probabilities", m["AUC"]],
        ["Precision", "TP / (TP + FP)", f"{tp} / ({tp} + {fp})", m["Precision"]],
        ["Recall", "TP / (TP + FN)", f"{tp} / ({tp} + {fn})", m["Recall"]],
        ["F1 Score", "2 × Precision × Recall / (Precision + Recall)",
         f"2 × {m['Precision']:.4f} × {m['Recall']:.4f} / ({m['Precision']:.4f} + {m['Recall']:.4f})",
         m["F1 Score"]],
        ["MCC", "(TP×TN − FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))",
         f"({tp}×{tn} − {fp}×{fn}) / √(({tp}+{fp})({tp}+{fn})({tn}+{fp})({tn}+{fn}))",
         m["MCC"]],
    ], columns=["Metric", "Formula", "Calculation", "Value"]).assign(
        Value=lambda d: d["Value"].map(lambda v: f"{v:.4f}"))


# ------------------------------ UI ------------------------------
st.title("🛒 Online Shopping Purchase Prediction")
st.caption("Evaluate five pre-trained classifiers on an uploaded labelled CSV. "
           "Models are loaded from disk; nothing is retrained.")

left, right = st.columns([1, 1.3], gap="medium")
with left:
    selected = st.selectbox("Select a classification model", list(MODEL_FILES))
with right:
    upload = st.file_uploader("Upload test CSV (must include the Revenue column)", type=["csv"])

if upload is None:
    st.info("Upload **test_data.csv** (the held-out test set) to see the results.")
    st.stop()

try:
    df = pd.read_csv(upload)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

if df.empty or TARGET not in df.columns:
    st.error(f"The CSV must contain data rows and a '{TARGET}' column.")
    st.stop()

y = parse_target(df[TARGET])
if y is None:
    st.error("Revenue must contain only True/False or 1/0 values, with no blanks.")
    st.stop()
X = df.drop(columns=[TARGET])

if len(df) != EXPECTED_TEST_ROWS:
    st.warning(f"This file has {len(df):,} rows. The held-out test set has "
               f"{EXPECTED_TEST_ROWS:,}. Results on data the models were trained on "
               "will look better than they really are, so use test_data.csv for honest numbers.")

path = Path(MODEL_FILES[selected])
if not path.is_file():
    st.error(f"Saved model not found: {path}")
    st.stop()

try:
    model = load_model(str(path))
    expected = list(getattr(model, "feature_names_in_", []))
    if expected:
        missing = [c for c in expected if c not in X.columns]
        if missing:
            st.error("Missing required feature columns: " + ", ".join(missing))
            st.stop()
        X = X[expected]
    pred, metrics, cm = evaluate(model, X, y)
except Exception as exc:
    st.error(f"Model loading or prediction failed: {exc}")
    st.stop()

st.subheader(f"Evaluation metrics: {selected}")
for col, (name, value) in zip(st.columns(6), metrics.items()):
    col.metric(name, f"{value:.4f}")
st.caption(f"Rows evaluated: {len(df):,}  ·  Features used: {X.shape[1]}")

c1, c2 = st.columns([1, 1.6], gap="large")
with c1:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=[0, 1]),
                           display_labels=["No Purchase", "Purchase"]
                           ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    tn, fp, fn, tp = cm
    st.caption(f"TN = {tn} · FP = {fp} · FN = {fn} · TP = {tp}")
with c2:
    st.subheader("How each metric is calculated")
    st.table(formula_table(metrics, cm))
    st.caption("TP = purchases correctly predicted · TN = non-purchases correctly predicted · "
               "FP = predicted purchase but none happened · FN = missed purchases")


with st.expander("Compare all five models on this file"):
    if st.button("Run comparison"):
        rows = []
        for name, p in MODEL_FILES.items():
            try:
                _, m, _ = evaluate(load_model(p), X, y)
                rows.append({"Model": name, **{k: round(v, 4) for k, v in m.items()}})
            except Exception as exc:
                rows.append({"Model": name, "Accuracy": f"failed: {exc}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)









            rows.append({"Model": name, "Accuracy": f"failed: {exc}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
