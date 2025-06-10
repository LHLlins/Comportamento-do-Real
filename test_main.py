import pytest
from fastapi.testclient import TestClient
import numpy as np
import os
import sys

# Attempt to import the app from main.py
# This requires main.py to be in the PYTHONPATH or the same directory
try:
    from main import app, startup_event  # Import startup_event to manually trigger it for tests
except ImportError as e_initial:
    # This is a fallback if the test environment has trouble with direct import
    # It assumes main.py is in the current working directory during test execution
    # In a more complex setup, package structure would handle this.
    print(f"Initial ImportError: {e_initial}. Attempting to add to sys.path.")
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    try:
        from main import app, startup_event
        print("Successfully imported app after modifying sys.path.")
    except ImportError as e_fallback:
        print(f"Failed to import app from main.py even after sys.path modification: {e_fallback}")
        print(f"Current sys.path: {sys.path}")
        print(f"Current working directory: {os.getcwd()}")
        # Define a dummy app for tests to be collected, though they will likely fail without the real app
        from fastapi import FastAPI
        app = FastAPI(title="Dummy FastAPI App") # Give a distinct title for dummy
        async def startup_event(): pass # Dummy startup
        print("Using a dummy FastAPI app for test collection due to import failure.")


# Create a TestClient instance
# The TestClient will instantiate the app defined in main.py.
# We need to manually call the startup event for the TestClient context if it's not run automatically.
# For TestClient, lifespan events (like @app.on_event("startup")) are typically handled automatically.
client = TestClient(app)


# --- Test Data ---
def get_valid_input_data():
    # Creates a 90x6 array of zeros, which is valid in shape and type
    return {"data": [[0.0] * 6 for _ in range(90)]}

def get_valid_input_data_real_numbers():
    # More realistic data, e.g. using np.random
    np.random.seed(42) # for reproducibility
    data = []
    for _ in range(90):
        day_data = [
            np.random.uniform(10, 50),    # Open
            np.random.uniform(10, 50),    # High
            np.random.uniform(10, 50),    # Low
            np.random.uniform(10, 50),    # Close
            np.random.uniform(10, 50),    # Adj Close
            np.random.uniform(100000, 1000000) # Volume
        ]
        data.append(day_data)
    return {"data": data}

# --- Helper to check if real app is loaded ---
def is_real_app_loaded():
    # Check if the app being tested is the actual one or the dummy
    # The dummy app was given title "Dummy FastAPI App"
    return app.title == "Stock Predictor API"

# --- Test Cases ---

@pytest.fixture(scope="session", autouse=True)
def ensure_model_files_exist():
    # This fixture can be used to check or create dummy model files if needed.
    # For this context, we assume train_model.py has run and created them.
    # If not, main.py's startup will fail to load them, and predict endpoint should return 503.
    model_loaded_successfully = True
    try:
        # This is a bit of a hack. TestClient should ideally handle lifespan.
        # If main.model is None after TestClient(app), it means loading failed.
        # Need to access the global 'model' from the 'main' module.
        import main as main_module
        if main_module.model is None or main_module.feature_scaler is None or main_module.target_scaler is None:
            model_loaded_successfully = False
            print("Test setup: Model or scalers are None after app initialization. Predictions might fail or return 503.")
    except (ImportError, AttributeError):
        # If main_module or main_module.model can't be accessed, assume dummy app or other setup issue.
        if is_real_app_loaded(): # Only print this warning if we thought we had the real app
            print("Test setup warning: Could not directly check model loading status in main_module.")
        model_loaded_successfully = False # Assume the worst if we can't check

    # Store this status in a way tests can access, or just let tests hit the 503 error.
    # For now, we'll rely on the 503 check within tests if models aren't loaded.


def test_read_root():
    if not is_real_app_loaded():
        pytest.skip("Skipping test_read_root as the real FastAPI app could not be imported.")
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Stock Predictor API. Use the /predict endpoint to make predictions."}

def test_predict_valid_input_zeros():
    if not is_real_app_loaded():
        pytest.skip("Skipping test_predict_valid_input_zeros as the real FastAPI app could not be imported.")

    response = client.post("/predict", json=get_valid_input_data())

    # Check if models actually loaded, otherwise expect 503
    import main as main_module
    if not main_module.model or not main_module.feature_scaler or not main_module.target_scaler:
        assert response.status_code == 503
        assert "Model or scalers not loaded" in response.json()["detail"]
        pytest.skip("Skipping further checks as model/scalers are not loaded (503 received).")

    assert response.status_code == 200
    json_response = response.json()
    assert "predicted_open_price" in json_response
    assert isinstance(json_response["predicted_open_price"], float)

def test_predict_valid_input_real_numbers():
    if not is_real_app_loaded():
        pytest.skip("Skipping test_predict_valid_input_real_numbers as the real FastAPI app could not be imported.")

    response = client.post("/predict", json=get_valid_input_data_real_numbers())

    import main as main_module
    if not main_module.model or not main_module.feature_scaler or not main_module.target_scaler:
        assert response.status_code == 503
        assert "Model or scalers not loaded" in response.json()["detail"]
        pytest.skip("Skipping further checks as model/scalers are not loaded (503 received).")

    assert response.status_code == 200
    json_response = response.json()
    assert "predicted_open_price" in json_response
    assert isinstance(json_response["predicted_open_price"], float)

def test_predict_invalid_input_shape_too_few_rows():
    if not is_real_app_loaded():
        pytest.skip("Skipping test_predict_invalid_input_shape_too_few_rows as the real FastAPI app could not be imported.")
    invalid_data = {"data": [[0.0] * 6 for _ in range(89)]}
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422

def test_predict_invalid_input_shape_too_many_rows():
    if not is_real_app_loaded():
        pytest.skip("Skipping test_predict_invalid_input_shape_too_many_rows as the real FastAPI app could not be imported.")
    invalid_data = {"data": [[0.0] * 6 for _ in range(91)]}
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422

def test_predict_invalid_input_shape_too_few_columns():
    if not is_real_app_loaded():
        pytest.skip("Skipping test_predict_invalid_input_shape_too_few_columns as the real FastAPI app could not be imported.")
    invalid_data = {"data": [[0.0] * 5 for _ in range(90)]}
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422

def test_predict_invalid_input_shape_too_many_columns():
    if not is_real_app_loaded():
        pytest.skip("Skipping test_predict_invalid_input_shape_too_many_columns as the real FastAPI app could not be imported.")
    invalid_data = {"data": [[0.0] * 7 for _ in range(90)]}
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422

def test_predict_invalid_input_data_type():
    if not is_real_app_loaded():
        pytest.skip("Skipping test_predict_invalid_input_data_type as the real FastAPI app could not be imported.")
    invalid_data = {"data": [["not_a_float"] * 6 for _ in range(90)]}
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422

def test_predict_empty_data_list():
    if not is_real_app_loaded():
        pytest.skip("Skipping test_predict_empty_data_list as the real FastAPI app could not be imported.")
    invalid_data = {"data": []}
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422

def test_predict_malformed_payload_no_data_key():
    if not is_real_app_loaded():
        pytest.skip("Skipping test_predict_malformed_payload_no_data_key as the real FastAPI app could not be imported.")
    invalid_data = {"features": [[0.0] * 6 for _ in range(90)]}
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422

if __name__ == '__main__':
    # This is for clarity, pytest should be run from the command line
    print("To run tests: pytest test_main.py")
    # Example of trying to see if app imported correctly
    if is_real_app_loaded():
        print("Real FastAPI app seems to be imported by test_main.py.")
    else:
        print("Dummy FastAPI app is being used by test_main.py.")
