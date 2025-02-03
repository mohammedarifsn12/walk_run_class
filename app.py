import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the trained model
model = pickle.load(open('walk_run_new.sav', 'rb'))

# Define a function to make predictions
def predict_activity(features):
    prediction = model.predict([features])
    return prediction[0]

# Streamlit UI
st.title("Walk-Run Activity Classifier")

st.write("Enter the feature values for the classifier:")

# Collect input values for the features
wrist = st.number_input("Wrist", value=0.0)
acceleration_x = st.number_input("Acceleration X", value=0.0)
acceleration_y = st.number_input("Acceleration Y", value=0.0)
acceleration_z = st.number_input("Acceleration Z", value=0.0)
gyro_x = st.number_input("Gyro X", value=0.0)
gyro_y = st.number_input("Gyro Y", value=0.0)
gyro_z = st.number_input("Gyro Z", value=0.0)

# Make prediction button
if st.button("Predict Activity"):
    features = [wrist, acceleration_x, acceleration_y, acceleration_z, gyro_x, gyro_y, gyro_z]
    result = predict_activity(features)
    
    if result == 0:
        st.write("Predicted Activity: Person is Walking")
    else:
        st.write("Predicted Activity: Person is Running")

# Optional: Add example data or more features if needed

