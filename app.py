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

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib",
}

st.title("Online Shoppers Purchase Prediction")

# 1. Model selection dropdown
model_name = st.selectbox("Select a model", list(MODEL_FILES))

# 2. Dataset upload (CSV)
uploaded = st.file_uploader("Upload test data (CSV with Revenue column)", type="csv")

if uploaded is None:
    st.info("Please upload test_data.csv to see the results.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
    if "Revenue" not in df.columns:
        st.error("The CSV must contain a 'Revenue' column.")
        st.stop()

    y_true = df["Revenue"].astype(str).str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0}).astype(int)
    X = df.drop(columns=["Revenue"])

    # Load the saved model (no retraining)
    model = joblib.load(MODEL_FILES[model_name])
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
except Exception as e:
    st.error(f"Something went wrong: {e}")
    st.stop()

# 3. Evaluation metrics
st.subheader("Evaluation Metrics")
metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC": roc_auc_score(y_true, y_prob),
    "Precision": precision_score(y_true, y_pred, zero_division=0),
    "Recall": recall_score(y_true, y_pred, zero_division=0),
    "F1": f1_score(y_true, y_pred, zero_division=0),
    "MCC": matthews_corrcoef(y_true, y_pred),
}
for col, (name, value) in zip(st.columns(6), metrics.items()):
    col.metric(name, f"{value:.4f}")

# 4. Confusion matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(4, 3.5))
ConfusionMatrixDisplay(
    confusion_matrix(y_true, y_pred, labels=[0, 1]),
    display_labels=["No Purchase", "Purchase"],
).plot(ax=ax, cmap="Blues", colorbar=False)
st.pyplot(fig)
