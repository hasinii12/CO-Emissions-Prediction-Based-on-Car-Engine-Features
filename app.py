from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("co2_emission_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the CO2 Emission Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert JSON input to DataFrame
    df = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(df)[0]

    return jsonify({"co2_emission_prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
