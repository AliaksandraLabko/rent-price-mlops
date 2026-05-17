# serving/main.py
import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

# 1. Load these two files at startup.
preprocessor = joblib.load("data/processed/preprocessor.joblib")
model = joblib.load("models/best_model.joblib")

@app.post("/predict")
def predict_rent(data: dict):
    # 2. Convert the user-sent JSON into a DataFrame
    df = pd.DataFrame([data])
    # 3. Transform the input data using Person 2's preprocessing rules.
    X_transformed = preprocessor.transform(df)
    # 4. Predicting prices using a model trained on Person 3
    prediction = model.predict(X_transformed)
    return {"estimated_rent_price": float(prediction[0])}