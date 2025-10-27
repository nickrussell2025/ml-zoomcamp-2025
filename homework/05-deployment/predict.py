# SYSTEM IMPORTS
import pickle
import uvicorn
import requests

# NAMED IMPORTS
from fastapi import FastAPI
from typing import Dict, Any

app = FastAPI(title="churn-prediction")

url = 'http://localhost:9696/predict'

# LOAD PIPELINE
input_file = '/code/pipeline_v2.bin'

with open(input_file, 'rb') as f_in: 
    pipeline = pickle.load(f_in)

# DATA
customer = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# MAKE PREDICTIONS
def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)

@app.post("/predict")
def predict(customer: Dict[str, Any]):
    churn = predict_single(customer)
    
    return {
        "churn_probability": churn,
        "churn": bool(churn >= 0.5)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)