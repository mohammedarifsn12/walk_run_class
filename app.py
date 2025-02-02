import streamlit as st
import joblib
import numpy as np

# Load the trained Random Forest model
model = joblib.load("walk_run_model.pkl")

# Streamlit app title
st.title("Walk-Run Classification App")
st.write("Predict whether the activity is Walking or Running based on sensor data.")

# Sidebar input features
st.sidebar.header("Input Features")

def user_input_features():
    feature1 = st.sidebar.slider("Acceleration X", -10.0, 10.0, 0.0)
    feature2 = st.sidebar.slider("Acceleration Y", -10.0, 10.0, 0.0)
    feature3 = st.sidebar.slider("Acceleration Z", -10.0, 10.0, 0.0)
    feature4 = st.sidebar.slider("Gyroscope X", -10.0, 10.0, 0.0)
    feature5 = st.sidebar.slider("Gyroscope Y", -10.0, 10.0, 0.0)
    feature6 = st.sidebar.slider("Gyroscope Z", -10.0, 10.0, 0.0)
    
    data = np.array([[feature1, feature2, feature3, feature4, feature5, feature6]])
    return data

# Get user input
input_data = user_input_features()

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_data)
    class_label = "Running" if prediction[0] == 1 else "Walking"
    st.success(f"Predicted Activity: **{class_label}**")
