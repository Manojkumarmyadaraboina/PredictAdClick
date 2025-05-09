import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and encoders
rf_model = joblib.load("rf_model.pkl")
le_gender = joblib.load("le_gender.pkl")
le_device = joblib.load("le_device.pkl")
le_time = joblib.load("le_time.pkl")
le_category = joblib.load("le_category.pkl")
le_adpos = joblib.load("le_adpos.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Ad Click Predictor")
st.title("Privacy-First Ad Click Predictor")
st.write("This app predicts whether a user will click on an ad using only anonymized, first-party data.")

# Input fields
age = st.slider("Age", 18, 70, 30)
gender = st.selectbox("Gender", le_gender.classes_)
device = st.selectbox("Device Type", le_device.classes_)
time_of_day = st.selectbox("Time of Day", le_time.classes_)
browsing_history = st.selectbox("Browsing Category", le_category.classes_)
ad_position = st.selectbox("Ad Position", le_adpos.classes_)

# Encode and scale input
input_values = {
    'age': scaler.transform([[age]])[0][0],
    'gender': le_gender.transform([gender])[0],
    'device_type': le_device.transform([device])[0],
    'time_of_day': le_time.transform([time_of_day])[0],
    'browsing_history': le_category.transform([browsing_history])[0],
    'ad_position': le_adpos.transform([ad_position])[0]
}

# Match training feature order
input_df = pd.DataFrame([[input_values[col] for col in feature_names]], columns=feature_names)

# Prediction
prediction = rf_model.predict(input_df)[0]

# Assign segment
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

# Strategy suggestions
strategies = {
    "Engaged Desktop Shopper": "Use web banners and desktop comparison ads in the afternoon.",
    "Fashion Clicker": "Show fashion influencer ads on social media platforms.",
    "Evening Engager": "Promote flash sales between 7 PM to 11 PM.",
    "High Intent User": "Use multi-channel retargeting (email, web, mobile).",
    "Low Engagement User": "Use passive campaigns like loyalty emails or app notifications."
}

# Output
st.subheader("Prediction Result")
st.write("Click Prediction:", "Will Click" if prediction == 1 else "Will Not Click")
st.write("User Segment:", segment)

st.subheader("Recommended Campaign Strategy")
st.write(strategies[segment])

st.markdown("---")
st.caption("This app uses only anonymized first-party data. No personal information is stored or used.")
