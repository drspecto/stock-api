# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load model and encoder
model = joblib.load("decision_tree_model.pkl")
encoder = joblib.load("company_encoder.pkl")

app = FastAPI()

# Define input structure
class StockInput(BaseModel):
    yesterday_close: float
    day_before_yesterday: float
    open_price: float
    volume: float
    company: str

@app.post("/predict")
def predict_stock(data: StockInput):
    # Encode the company name
    company_encoded = encoder.transform([data.company])[0]

    # Prepare features
    input_features = np.array([[data.yesterday_close, data.day_before_yesterday, data.open_price,data.volume, company_encoded]])

    # Predict
    prediction = model.predict(input_features)[0]
    message = "Stock likely to go UP" if prediction == 1 else "Stock likely to go DOWN"

    return {
        "prediction": int(prediction),
        "message": message
    }
