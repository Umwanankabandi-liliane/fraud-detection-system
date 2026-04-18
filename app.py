import streamlit as st
import pandas as pd
import pickle

# Load model and feature columns
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

st.set_page_config(page_title="Fraud Detection System", layout="centered")

# Header
st.title("Transaction Fraud Detection System")
st.markdown("### Real-time Fraud Monitoring System")
st.caption("Powered by Machine Learning")

st.write("Enter key transaction details below to assess fraud risk.")

# IMPORTANT: Only show key features
important_features = ["Time", "Amount", "V14", "V17", "V12", "V10"]

input_data = {}

st.subheader("Transaction Input")

# Show only important inputs
for col in important_features:
    if col == "Time":
        input_data[col] = st.number_input("Transaction Time", min_value=0.0, value=0.0)
    elif col == "Amount":
        input_data[col] = st.number_input("Transaction Amount", min_value=0.0, value=0.0)
    else:
        input_data[col] = st.number_input(col, value=0.0)

# Fill remaining features automatically with 0
for col in feature_columns:
    if col not in input_data:
        input_data[col] = 0.0

# Prediction
if st.button("Check Transaction"):
    input_df = pd.DataFrame([input_data])

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Fraud Risk Analysis")
    st.write(f"Risk Score: **{probability:.4f}**")

    # Risk levels
    if probability > 0.7:
        st.error("HIGH RISK: Fraud Likely")
    elif probability > 0.3:
        st.warning("MEDIUM RISK: Needs Review")
    else:
        st.success("LOW RISK: Normal Transaction")