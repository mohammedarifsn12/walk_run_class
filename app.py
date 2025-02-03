import streamlit as st
import numpy as np
import joblib
import urllib.request
import os

# Set page configuration
st.set_page_config(page_title="Walk-Run Classification", layout="wide", page_icon="🏃")

# GitHub RAW URL for the model (replace with your actual repo & file name)
MODEL_URL = "https://raw.githubusercontent.com/mohammedarifsn12/walk-run-classification/main/random_forest_best.pkl"
MODEL_PATH = "./random_forest_best.pkl"

# Function to download model if not available locally
def download_model():
    if not os.path.exists(MODEL_PATH):
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download the model: {e}")

# Download the model (only if needed)
download_model()

# Load the trained Random Forest model
try:
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")

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
    try:
        prediction = model.predict(input_features)
        activity = "Running" if prediction[0] == 1 else "Walking"
        st.success(f"Predicted Activity: **{activity}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Run the Streamlit app using: streamlit run app.py

