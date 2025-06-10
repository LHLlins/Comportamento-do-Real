import os
import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import List

# --- Configuration & Model Loading ---
MODEL_DIR = os.path.dirname(os.path.realpath(__file__)) # Assumes model files are in the same directory
MODEL_FILE = os.path.join(MODEL_DIR, 'model.h5')
FEATURE_SCALER_FILE = os.path.join(MODEL_DIR, 'feature_scaler.pkl')
TARGET_SCALER_FILE = os.path.join(MODEL_DIR, 'target_scaler.pkl')

# Initialize FastAPI app
app = FastAPI(title="Stock Predictor API")

# Load model and scalers on startup
model = None
feature_scaler = None
target_scaler = None

@app.on_event("startup")
async def startup_event():
    global model, feature_scaler, target_scaler
    try:
        if not os.path.exists(MODEL_FILE):
            print(f"Error: Model file not found at {MODEL_FILE}")
            raise FileNotFoundError(f"Model file not found at {MODEL_FILE}")
        if not os.path.exists(FEATURE_SCALER_FILE):
            print(f"Error: Feature scaler file not found at {FEATURE_SCALER_FILE}")
            raise FileNotFoundError(f"Feature scaler file not found at {FEATURE_SCALER_FILE}")
        if not os.path.exists(TARGET_SCALER_FILE):
            print(f"Error: Target scaler file not found at {TARGET_SCALER_FILE}")
            raise FileNotFoundError(f"Target scaler file not found at {TARGET_SCALER_FILE}")

        model = tf.keras.models.load_model(MODEL_FILE)
        feature_scaler = joblib.load(FEATURE_SCALER_FILE)
        target_scaler = joblib.load(TARGET_SCALER_FILE)
        print("Model and scalers loaded successfully.")
    except Exception as e:
        print(f"Error loading model or scalers during startup: {e}")
        # Set to None so endpoints can check and return 503
        model = None
        feature_scaler = None
        target_scaler = None

# --- Pydantic Models for Request/Response ---
# Note: The StockFeature model is not directly used in the /predict endpoint's payload,
# as the payload `data` is a list of lists of floats.
# It's included here if you plan to have other endpoints or for clarity on feature names.
class StockFeature(BaseModel):
    Open: float
    High: float
    Low: float
    Close: float
    Adj_Close: float # Adjusted Pydantic field name
    Volume: float

class PredictionInput(BaseModel):
    # Expects a list of 90 records, each record being a list of 6 floats
    # Validates that the outer list has exactly 90 elements
    # and each inner list has exactly 6 float elements.
    data: conlist(conlist(float, min_length=6, max_length=6), min_length=90, max_length=90)

class PredictionOutput(BaseModel):
    predicted_open_price: float

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Stock Predictor API. Use the /predict endpoint to make predictions."}

@app.post("/predict", response_model=PredictionOutput)
async def predict_stock_price(payload: PredictionInput):
    if not model or not feature_scaler or not target_scaler:
        # This check is important if loading failed during startup
        print("Prediction attempted but model/scalers are not available.")
        raise HTTPException(status_code=503, detail="Model or scalers not loaded. API is not available.")

    try:
        # Convert input data to NumPy array
        input_data = np.array(payload.data)

        # Pydantic already validates the shape (90x6), but an explicit check here can be useful for debugging
        # or if Pydantic validation is bypassed for some reason.
        if input_data.shape != (90, 6):
            # This case should ideally be caught by Pydantic's conlist validation
            raise HTTPException(
                status_code=400,
                detail=f"Input data shape mismatch. Expected (90, 6), got {input_data.shape}. This should have been caught by input validation."
            )

        # Normalize features
        # Ensure feature_scaler expects 6 features if input_data has 6 columns
        scaled_input_data = feature_scaler.transform(input_data)

        # Reshape for LSTM: (1, 90, 6) - predicting for one sequence
        reshaped_data = np.reshape(scaled_input_data, (1, scaled_input_data.shape[0], scaled_input_data.shape[1]))

        # Make prediction
        raw_prediction = model.predict(reshaped_data)

        # Inverse transform the prediction
        # Ensure target_scaler was fitted on a single feature if raw_prediction is (1,1)
        predicted_price = target_scaler.inverse_transform(raw_prediction)

        # The output of inverse_transform is a 2D array, e.g., [[value]]
        # Extract the single value
        final_prediction = float(predicted_price[0, 0])

        return {"predicted_open_price": final_prediction}

    except ValueError as ve:
        # This could happen if feature_scaler.transform fails due to unexpected data
        # despite Pydantic's float validation (e.g., if scaler was trained on different number of features)
        print(f"ValueError during prediction: {str(ve)}") # Log for debugging
        raise HTTPException(status_code=400, detail=f"Error processing input data: {str(ve)}")
    except Exception as e:
        # Catch-all for other unexpected errors
        print(f"Unexpected error during prediction: {e}") # Log for debugging
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    # This part is for local execution and debugging, not used by Uvicorn directly in production
    # Uvicorn will import the `app` object from this file.
    print("FastAPI app main.py created.")
    print("To run this app (example):")
    print("Ensure model.h5, feature_scaler.pkl, target_scaler.pkl are in the same directory.")
    print("Install dependencies: pip install fastapi uvicorn tensorflow scikit-learn joblib numpy")
    print("Then run in terminal: uvicorn main:app --reload --host 0.0.0.0 --port 8000")

    # Example of how to manually check model loading if needed for debugging:
    # if model and feature_scaler and target_scaler:
    #     print("Manual check: Model and scalers appear to be loaded.")
    # else:
    #     print("Manual check: Model or scalers failed to load. Check startup logs.")
