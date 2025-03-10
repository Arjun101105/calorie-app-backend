from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)

# ✅ Fix CORS: Explicitly allow credentials & frontend domain
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Load model and feature names
model, FEATURE_COLUMNS = joblib.load("calorie_model.pkl")

def estimate_heart_rate(age, workout_type, duration):
    """Estimate Heart Rate using Karvonen formula approximation with duration impact."""
    max_hr = 220 - age  # Max Heart Rate
    resting_hr = 70  # Approximate Resting HR

    intensity_factors = {
        "Cardio": 0.85,
        "Endurance": 0.7,
        "Strength": 0.5,
        "No Workout": 0.3
    }
    intensity = intensity_factors.get(workout_type, 0.3)  # Default: No Workout

    # Adjust HR based on duration (longer workouts lower HR efficiency)
    duration_factor = max(0.5, 1 - (duration / 180))  # HR decreases over long sessions

    estimated_hr = resting_hr + (max_hr - resting_hr) * intensity * duration_factor
    return max(60, min(estimated_hr, 200))  # Keep HR in realistic range

def estimate_body_temp(duration, workout_type):
    """Estimate Body Temperature based on duration and intensity."""
    base_temp = 37.0  # Normal body temp in °C
    temp_increase_rate = {
        "Cardio": 1.5,
        "Endurance": 1.0,
        "Strength": 0.6,
        "No Workout": 0.2
    }
    temp_increase = temp_increase_rate.get(workout_type, 0.2) * (duration / 60)  # Per hour scaling

    return round(base_temp + temp_increase, 2)  # Round temp for readability

@app.route("/calculate-calories", methods=["POST"])
def calculate_calories():
    try:
        # Validate incoming request
        data = request.json
        required_fields = ["gender", "age", "height", "weight", "duration", "workout_type"]
        
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        # Convert input values
        gender = 0 if data["gender"].lower() == "male" else 1
        age = float(data["age"])
        height = float(data["height"])
        weight = float(data["weight"])
        duration = float(data["duration"])
        workout_type = data.get("workout_type", "No Workout")  # Default if missing

        # ✅ **Additional Validations**
        if not (10 <= age <= 100):
            return jsonify({"error": "Age must be between 10 and 100"}), 400
        if not (50 <= height <= 250):
            return jsonify({"error": "Height must be between 50 cm and 250 cm"}), 400
        if not (20 <= weight <= 300):
            return jsonify({"error": "Weight must be between 20 kg and 300 kg"}), 400
        if not (1 <= duration <= 300):
            return jsonify({"error": "Duration must be between 1 and 300 minutes"}), 400

        # **Estimate Heart Rate & Body Temp**
        heart_rate = estimate_heart_rate(age, workout_type, duration)
        body_temp = estimate_body_temp(duration, workout_type)

        # ✅ Ensure workout type is valid
        valid_workout_types = ["Cardio", "Endurance", "Strength", "No Workout"]
        if workout_type not in valid_workout_types:
            workout_type = "No Workout"  # Default to No Workout if invalid type is sent

        # One-hot encode workout type
        workout_encoded = {f"workout_type_{wt}": (1 if workout_type == wt else 0) for wt in valid_workout_types}

        # **Normalize Duration (Optional: If needed in training)**
        normalized_duration = duration / 180  # Normalized between 0 and 1

        # Create DataFrame with correct feature order
        features = {
            "Age": age,
            "Gender": gender,
            "Height": height,
            "Weight": weight,
            "Duration": normalized_duration,  # Use normalized duration
            "Heart_Rate": heart_rate,
            "Body_Temp": body_temp,
            **workout_encoded,  # Unpack one-hot encoded workout types
        }

        # Convert to DataFrame & Ensure Correct Column Order
        input_df = pd.DataFrame([features])

        # ✅ Add missing columns if necessary
        for col in FEATURE_COLUMNS:
            if col not in input_df.columns:
                input_df[col] = 0  # Fill missing features with 0

        input_df = input_df[FEATURE_COLUMNS]  # Ensure correct column order

        # Predict calories burnt
        prediction = model.predict(input_df)

        return jsonify({
            "calories_burned": round(float(prediction[0]), 2),
            "estimated_heart_rate": round(heart_rate, 2),
            "estimated_body_temp": round(body_temp, 2)
        }), 200

    except ValueError:
        return jsonify({"error": "Invalid input type. Ensure numerical values for age, height, weight, and duration."}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# ✅ Remove app.run() since Render uses Gunicorn
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
