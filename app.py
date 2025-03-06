from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # Enable CORS for all 

# Load the trained model
model = joblib.load('calorie_model.pkl')

@app.route("/")
def home():
    return "API is running!", 200

@app.route('/calculate-calories', methods=['POST'])
def calculate_calories():
    try:
        # Get data from the frontend
        data = request.json
        gender = 0 if data['gender'] == "Male" else 1
        age = float(data['age'])
        height = float(data['height'])
        weight = float(data['weight'])
        duration = float(data['duration'])
        intensity = data['intensity']

        # Estimate heart rate and body temperature
        heart_rate = estimate_heart_rate(age, intensity)
        body_temp = estimate_body_temp(duration, intensity)

        # Create DataFrame for prediction
        features = {
            'Gender': gender,
            'Age': age,
            'Height': height,
            'Weight': weight,
            'Duration': duration,
            'Heart_Rate': heart_rate,
            'Body_Temp': body_temp
        }
        input_df = pd.DataFrame([features])

        # Predict calories
        prediction = model.predict(input_df)
        result = {
            "calories_burned": float(prediction[0]),
            "average_heart_rate": heart_rate
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

def estimate_heart_rate(age, intensity):
    """Estimate heart rate based on age and intensity"""
    max_hr = 220 - age
    intensity_factors = {
        "Light": 0.50,
        "Moderate": 0.70,
        "Vigorous": 0.85
    }
    return (max_hr - 70) * intensity_factors[intensity] + 70

def estimate_body_temp(duration_min, intensity):
    """Estimate body temperature based on duration and intensity"""
    base_temp = 37.0  # Normal body temp in °C
    temp_increase = {
        "Light": 0.3,
        "Moderate": 0.6,
        "Vigorous": 1.2
    }
    return base_temp + (temp_increase[intensity] * (duration_min / 60))

# Remove app.run() as Render uses Gunicorn
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Use PORT from Render, default to 5000
    app.run(host="0.0.0.0", port=port, debug=False)

