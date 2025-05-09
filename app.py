import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved models and encoders
rf_model = joblib.load("rf_model.pkl")
le_gender = joblib.load("le_gender.pkl")
le_device = joblib.load("le_device.pkl")
le_time = joblib.load("le_time.pkl")
le_category = joblib.load("le_category.pkl")
le_adpos = joblib.load("le_adpos.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Privacy-First Ad Click Predictor")
st.title(" Privacy-First Ad Click Predictor")
st.markdown("Predict user ad click behavior using only first-party, privacy-compliant data.")

# --- Input fields ---
age = st.slider("Age", 18, 70, 30)
gender = st.selectbox("Gender", le_gender.classes_)
device = st.selectbox("Device Type", le_device.classes_)
time_of_day = st.selectbox("Time of Day", le_time.classes_)
browsing_history = st.selectbox("Browsing Category", le_category.classes_)
ad_position = st.selectbox("Ad Position on Screen", le_adpos.classes_)

# --- Prepare input for prediction ---
columns_order = ['age', 'gender', 'device_type', 'time_of_day', 'browsing_history', 'ad_position']
input_values = [
    age,
    le_gender.transform([gender])[0],
    le_device.transform([device])[0],
    le_time.transform([time_of_day])[0],
    le_category.transform([browsing_history])[0],
    le_adpos.transform([ad_position])[0]
]
input_df = pd.DataFrame([input_values], columns=columns_order)

# Scale numeric column
input_df["age"] = scaler.transform(input_df[["age"]])

# --- Predict ---
prediction = rf_model.predict(input_df)[0]

# --- Segment Logic ---
def assign_segment(row):
    if prediction == 1:
        if row["device_type"] == 0:
            return "Engaged Desktop Shopper"
        elif row["browsing_history"] == 2:
            return "Fashion Clicker"
        elif row["time_of_day"] == 3:
            return "Evening Engager"
        else:
            return "High Intent User"
    else:
        return "Low Engagement User"

segment = assign_segment(input_df.iloc[0])

# --- Strategies ---
strategies = {
    "Engaged Desktop Shopper": "Use web banners and desktop-targeted comparison ads during afternoon browsing.",
    "Fashion Clicker": "Show carousel ads featuring fashion influencers on mobile platforms.",
    "Evening Engager": "Push flash sales and personalized ads between 7PM–11PM.",
    "High Intent User": "Use multi-channel retargeting campaigns (email, web, mobile).",
    "Low Engagement User": "Avoid paid ads. Use passive channels like loyalty emails or push notifications."
}

# --- Display Results ---
st.subheader(" Prediction")
st.success(" Will Click the Ad" if prediction == 1 else " Will Not Click the Ad")
st.write("**User Segment:**", segment)

st.subheader(" Recommended Campaign Strategy")
st.info(strategies[segment])

st.markdown("---")
st.caption(" This tool uses only anonymized, first-party data and complies with GDPR & CCPA.")
