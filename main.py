from fastapi import FastAPI
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# Define FastAPI app
app = FastAPI()

# ✅ Add a root endpoint to avoid 404 errors
@app.get("/")
def home():
    return {"message": "Welcome to the Car Sales Price Prediction API"}

# Define API endpoint for predictions
@app.post("/predict")
def predict(annual_inc: float, company: str, model_name: str, transmission: str, color: str):
    # Encode input (Modify based on your encoding logic)
    encoders = joblib.load("label_encoders.pkl")  # Load encoders if needed
    company_encoded = encoders['Company'].transform([company])[0]
    model_encoded = encoders['Model'].transform([model_name])[0]
    transmission_encoded = encoders['Transmission'].transform([transmission])[0]
    color_encoded = encoders['Color'].transform([color])[0]

    # Create DataFrame for model input
    input_data = pd.DataFrame([[annual_inc, company_encoded, model_encoded, transmission_encoded, color_encoded]],
                              columns=['Annual Inc', 'Company', 'Model', 'Transmission', 'Color'])

    # Make prediction
    predicted_price = model.predict(input_data)[0]

    return {"Predicted Price": predicted_price}
