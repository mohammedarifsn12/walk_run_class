import streamlit as st
import numpy as np
import joblib

# Load the trained Random Forest model
model = joblib.load("random_forest_best.pkl")

# Streamlit app title
st.title("Walk-Run Classification App")

# Input fields for wrist sensor data
st.header("Enter Sensor Data:")
wrist = st.selectbox("Wrist (1 for worn, 0 for not worn)", [0, 1])
acceleration_x = st.number_input("Acceleration X", value=0.0)
acceleration_y = st.number_input("Acceleration Y", value=0.0)
acceleration_z = st.number_input("Acceleration Z", value=0.0)
gyro_x = st.number_input("Gyro X", value=0.0)
gyro_y = st.number_input("Gyro Y", value=0.0)
gyro_z = st.number_input("Gyro Z", value=0.0)

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_features = np.array([[wrist, acceleration_x, acceleration_y, acceleration_z, gyro_x, gyro_y, gyro_z]])
    
    # Make prediction
    prediction = model.predict(input_features)

    # Display result
    activity = "Running" if prediction[0] == 1 else "Walking"
    st.write(f"Predicted Activity: **{activity}**")

# Run the Streamlit app using: streamlit run app.py
