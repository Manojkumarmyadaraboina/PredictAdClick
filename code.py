import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("xgboost_ad_click_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Ad Click Predictor", layout="centered")

st.title(" Ad Click Prediction App")
st.write("Enter user details to predict whether they will click the ad.")

# Input form
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    device_type = st.selectbox("Device Type", ["Mobile", "Tablet", "Desktop"])
    ad_position = st.selectbox("Ad Position", ["Top", "Bottom", "Side"])
    browsing_history = st.selectbox("Browsing History Score", list(range(6)))  # assuming scores from 0 to 5
    time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

    submit = st.form_submit_button("Predict")

if submit:
    # Convert categorical inputs to numeric values (adjust mappings based on your encoding)
    gender_map = {"Male": 1, "Female": 0}
    device_map = {"Mobile": 0, "Tablet": 1, "Desktop": 2}
    position_map = {"Top": 2, "Bottom": 0, "Side": 1}
    time_map = {"Morning": 1, "Afternoon": 0, "Evening": 2, "Night": 3}

    input_data = pd.DataFrame([[
        age,
        gender_map[gender],
        device_map[device_type],
        position_map[ad_position],
        browsing_history,
        time_map[time_of_day]
    ]], columns=['age', 'gender', 'device_type', 'ad_position', 'browsing_history', 'time_of_day'])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][prediction]

    st.subheader(" Prediction Result")
    st.write("Prediction:", " Will Click Ad" if prediction == 1 else " Will Not Click Ad")
    st.write("Confidence:", f"{prob*100:.2f}%")