from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

st.set_page_config(page_title="Online Shopping Purchase Prediction", page_icon="🛒", layout="wide")
st.title("🛒 Online Shopping Purchase Prediction")
st.caption("Evaluate five saved classification models using a labeled test CSV. Models are not retrained.")

BASE = Path(__file__).resolve().parent
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

left, right = st.columns(2)
with left:
    choice = st.selectbox("Choose a model", list(MODEL_FILES))
with right:
    uploaded = st.file_uploader("Upload test CSV (including Revenue)", type="csv")

if uploaded is None:
    st.info("Upload test_data.csv (2,466 held-out rows) to display evaluation metrics and confusion matrix.")
    st.stop()

try:
    data = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

if "Revenue" not in data.columns:
    st.error("Missing target column: Revenue")
    st.stop()

raw_target = data["Revenue"].astype("string").str.strip().str.lower()
y_true = raw_target.map({"true": 1, "false": 0, "1": 1, "0": 0})
if y_true.isna().any():
    st.error("Revenue must contain only True/False or 1/0, with no missing values.")
    st.stop()
y_true = y_true.astype(int)
if y_true.nunique() != 2:
    st.error("The test CSV must include both purchase and non-purchase records to calculate AUC.")
    st.stop()

X = data.drop(columns=["Revenue"])
model_path = BASE / "model" / MODEL_FILES[choice]
if not model_path.is_file():
    st.error(f"Saved model not found: model/{MODEL_FILES[choice]}. Check your GitHub repository.")
    st.stop()

try:
    model = joblib.load(model_path)
    expected = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
    if expected is not None:
        missing = [name for name in expected if name not in X.columns]
        if missing:
            st.error("Missing required input columns: " + ", ".join(missing))
            st.stop()
        X = X[expected]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, list(model.classes_).index(1)]
except Exception as exc:
    st.error(f"Model loading or prediction failed: {type(exc).__name__}: {exc}")
    st.stop()

if len(data) != 2466:
    st.warning(f"Uploaded {len(data):,} rows. Use the 2,466-row held-out test_data.csv for your assignment's final results.")

metrics = {
    "Accuracy": accuracy_score(y_true, predictions),
    "AUC": roc_auc_score(y_true, probabilities),
    "Precision": precision_score(y_true, predictions, zero_division=0),
    "Recall": recall_score(y_true, predictions, zero_division=0),
    "F1 Score": f1_score(y_true, predictions, zero_division=0),
    "MCC": matthews_corrcoef(y_true, predictions),
}
st.subheader("Evaluation metrics")
for col, (name, score) in zip(st.columns(6), metrics.items()):
    col.metric(name, f"{score:.4f}")

st.subheader("Confusion matrix")
cm = confusion_matrix(y_true, predictions, labels=[0, 1])
fig, ax = plt.subplots(figsize=(4.2, 3.3))
ConfusionMatrixDisplay(cm, display_labels=["No Purchase", "Purchase"]).plot(
    ax=ax, cmap="Blues", colorbar=False, values_format="d"
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
fig.tight_layout()
chart_col, _ = st.columns([1, 1])
with chart_col:
    st.pyplot(fig, use_container_width=True)
plt.close(fig)
tn, fp, fn, tp = cm.ravel()
st.caption(f"TN: {tn:,}  |  FP: {fp:,}  |  FN: {fn:,}  |  TP: {tp:,}")

with st.expander("Metric formulas"):
    st.latex(r"Accuracy = \frac{TP+TN}{TP+TN+FP+FN}")
    st.latex(r"Precision = \frac{TP}{TP+FP}")
    st.latex(r"Recall = \frac{TP}{TP+FN}")
    st.latex(r"F1 = \frac{2\,Precision\,Recall}{Precision+Recall}")
    st.latex(r"MCC = \frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}")
    st.markdown("**AUC:** Area under the ROC curve, computed from prediction probabilities across thresholds.")
