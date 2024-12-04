from flask import Flask, request, jsonify
import pandas as pd
import mlflow.pyfunc
import joblib
import os

app = Flask(__name__)

# Load column structure, encoders, and scaler
CLEAN_COLS = joblib.load("clean_df_cols.pkl")
ENCODER_DIR = "encoders"
SCALER = joblib.load("StandardScaler.pkl")

# Load MLflow Model URI
MODEL_URI = "mlflow-artifacts:/1c1380b39e9248c6ad4366557dd0df2e/6abc95022e654bf0b12ad2dbd464083f/artifacts/ML_models"  # Update with your MLflow model URI
MODEL = mlflow.pyfunc.load_model(MODEL_URI)


@app.route('/predict', methods=['POST'])
def predict():
    """
    API Endpoint to predict malware based on uploaded data
    """
    try:
        # Parse input JSON
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert input to DataFrame
        df = pd.DataFrame(data)

        # Validate and preprocess data
        if not all(col in df.columns for col in CLEAN_COLS):
            return jsonify({"error": "Missing required columns"}), 400

        # Keep only relevant columns
        df = df[CLEAN_COLS]

        # Apply encoders
        for col in df.columns:
            encoder_path = os.path.join(ENCODER_DIR, f"{col}_encoder.pkl")
            if os.path.exists(encoder_path):
                encoder = joblib.load(encoder_path)
                df[col] = encoder.transform(df[col].astype(str))

        # Scale data
        scaled_data = SCALER.transform(df)

        # Predict using the MLflow model
        predictions = MODEL.predict(scaled_data)

        # Return predictions as JSON
        return jsonify({"predictions": predictions.tolist()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
