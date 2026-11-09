
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "outputs" / "random_forest.pkl"
SCALER_PATH = ROOT / "outputs" / "scaler.pkl"
NAME_PATH = ROOT / "outputs" / "model_name.pkl"

SCALE_COLUMNS = ["Time", "Amount"]

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")

st.caption(
    "Academic demonstration using an anonymized public transaction dataset."
)

# STEP 1: Check saved files

if not MODEL_PATH.exists():
    st.error("Random Forest model not found in outputs folder.")
    st.stop()

if not SCALER_PATH.exists():
    st.error("Scaler file not found in outputs folder.")
    st.stop()


# STEP 2: Load model and scaler

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# STEP 3: Get training feature names

if hasattr(model, "feature_names_in_"):
    features = list(model.feature_names_in_)

else:
    st.error(
        "Model feature names are unavailable. "
        "Please check the training configuration."
    )
    st.stop()


# STEP 4: Validate scaler configuration

if not all(col in features for col in SCALE_COLUMNS):
    st.error("Time or Amount column missing from model features.")
    st.stop()

if scaler.n_features_in_ != len(SCALE_COLUMNS):
    st.error("Scaler configuration does not match training setup.")
    st.stop()

if hasattr(scaler, "feature_names_in_"):
    if list(scaler.feature_names_in_) != SCALE_COLUMNS:
        st.error("Scaler feature order does not match training setup.")
        st.stop()


# STEP 5: Model information

model_name = "Random Forest"

if NAME_PATH.exists():
    try:
        model_name = joblib.load(NAME_PATH)
    except Exception:
        pass

st.info(f"Active Model: {model_name}")

st.write(
    "Upload a CSV containing the same feature columns "
    "as the training dataset. The Class column is optional."
)


# STEP 6: Upload CSV

uploaded = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)


if uploaded is not None:

    try:

        data = pd.read_csv(uploaded)

        if data.empty:
            st.error("Uploaded CSV is empty.")
            st.stop()

        # Check required features

        missing = [
            col for col in features
            if col not in data.columns
        ]

        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        # Select features in exact training order

        X = data[features].copy()

        # Convert values to numeric

        X = X.apply(pd.to_numeric, errors="coerce")

        if X.isnull().any().any():
            st.error(
                "Missing or non-numeric values found "
                "in the feature columns."
            )
            st.stop()

        # STEP 7: Scale ONLY Time and Amount

        X.loc[:, SCALE_COLUMNS] = scaler.transform(
            X[SCALE_COLUMNS]
        )

        # STEP 8: Predict

        predictions = model.predict(X)

        # STEP 9: Prepare results

        results = data.copy()

        results["Prediction"] = predictions

        results["Status"] = results["Prediction"].map({
            0: "Legitimate",
            1: "Potential Fraud"
        })

        total = len(results)

        flagged = int((predictions == 1).sum())

        legitimate = total - flagged

        # STEP 10: Summary

        st.subheader("Prediction Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Transactions Analysed",
            f"{total:,}"
        )

        c2.metric(
            "Potential Fraud",
            f"{flagged:,}"
        )

        c3.metric(
            "Likely Legitimate",
            f"{legitimate:,}"
        )

        # STEP 11: Results

        st.subheader("Prediction Results")

        st.dataframe(
            results,
            use_container_width=True
        )

        # STEP 12: Download results

        csv_data = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Results as CSV",
            data=csv_data,
            file_name="fraud_detection_results.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")


# STEP 13: Disclaimer

st.warning(
    "This is an educational risk-scoring demonstration, "
    "not a production banking system."
)
