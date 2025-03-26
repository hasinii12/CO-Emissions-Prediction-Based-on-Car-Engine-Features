import streamlit as st
import requests
import pandas as pd
import joblib

# Load trained model
model = joblib.load("co2_emission_model.pkl")

# Streamlit UI
st.title("🚗 CO2 Emission Predictor 💨")
st.write("Enter vehicle details to predict CO2 emissions.")

# Input fields
engine_size = st.number_input("Engine Size (L)", min_value=1.0, max_value=10.0, value=2.0)
cylinders = st.number_input("Cylinders", min_value=2, max_value=16, value=4)
fuel_city = st.number_input("Fuel Consumption (City) L/100km", min_value=1.0, max_value=20.0, value=8.0)
fuel_hwy = st.number_input("Fuel Consumption (Highway) L/100km", min_value=1.0, max_value=20.0, value=6.0)
fuel_comb = st.number_input("Fuel Consumption (Combined) L/100km", min_value=1.0, max_value=20.0, value=7.0)
fuel_comb_mpg = st.number_input("Fuel Consumption (Combined) MPG", min_value=10, max_value=100, value=30)

# Categorical inputs (must match training features)
vehicle_class = st.selectbox("Vehicle Class", ["COMPACT", "SUV - SMALL", "MID-SIZE", "SUV - LARGE"])
transmission = st.selectbox("Transmission", ["AS", "M", "AV", "A"])
fuel_type = st.selectbox("Fuel Type", ["Z", "E", "N", "X", "D"])

# One-Hot Encoding (Ensure same encoding as training)
input_data = pd.DataFrame([[engine_size, cylinders, fuel_city, fuel_hwy, fuel_comb, fuel_comb_mpg, 
                            vehicle_class, transmission, fuel_type]],
                          columns=['engine_size', 'cylinders', 'fuel_consumption_city', 'fuel_consumption_hwy',
                                   'fuel_consumption_comb(l/100km)', 'fuel_consumption_comb(mpg)',
                                   'vehicle_class', 'transmission', 'fuel_type'])

# Perform One-Hot Encoding
input_data = pd.get_dummies(input_data, columns=['vehicle_class', 'transmission', 'fuel_type'])

# Ensure missing columns from training are added (with value 0)
expected_features = joblib.load("feature_columns.pkl")  # Save feature columns from training
for col in expected_features:
    if col not in input_data:
        input_data[col] = 0

# Reorder columns to match training order
input_data = input_data[expected_features]

# Make prediction
if st.button("Predict CO2 Emission"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted CO2 Emission: {prediction:.2f} g/km")
