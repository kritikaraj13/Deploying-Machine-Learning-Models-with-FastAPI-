# main.py - FastAPI application to serve the Iris prediction model

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 1. Create FastAPI app instance
app = FastAPI(
    title="Iris Species Predictor",
    description="A simple API to predict Iris species based on sepal and petal measurements.",
    version="1.0.0"
)

# 2. Load the pre-trained model when the app starts
#    Make sure 'iris_model.pkl' is in the same directory as this main.py file.
try:
    model = joblib.load('iris_model.pkl')
    print("Iris model loaded successfully!")
except FileNotFoundError:
    print("Error: 'iris_model.pkl' not found. Make sure it's in the same directory.")
    # You might want to exit or handle this more gracefully in a production app
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 3. Define Input Data Model using Pydantic
#    This describes the expected JSON structure for incoming requests.
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# 4. Define the mapping for numerical predictions to species names
#    This should match the order in iris.target_names
iris_species_names = {
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
}

# 5. Define an API Endpoint for Root/Home path (optional, good for health check)
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Species Predictor API! Visit /docs for details."}

# 6. Define the Prediction API Endpoint
@app.post("/predict")
def predict_species(features: IrisFeatures):
    """
    Predicts the Iris species based on sepal and petal measurements.

    - **sepal_length**: Length of the sepal in cm.
    - **sepal_width**: Width of the sepal in cm.
    - **petal_length**: Length of the petal in cm.
    - **petal_width**: Width of the petal in cm.
    """
    try:
        # Convert input features from Pydantic model to a NumPy array for the ML model
        # .values() gets the values from the features object in the order they were defined
        # reshape(1, -1) is crucial for a single sample prediction (expected by scikit-learn models)
        data_array = np.array([
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]).reshape(1, -1)

        # Make prediction using the loaded model
        prediction_id = model.predict(data_array)[0] # [0] to get the single prediction value

        # Map the numerical prediction ID to the actual species name
        predicted_species_name = iris_species_names.get(prediction_id, "Unknown Species ID")

        # Return the prediction
        return {"predicted_species": predicted_species_name, "prediction_id": int(prediction_id)}

    except Exception as e:
        # Basic error handling
        return {"error": f"An error occurred during prediction: {str(e)}"}