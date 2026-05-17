import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "data" / "processed" / "preprocessor.joblib"

model = None
preprocessor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts when service starts"""
    global model, preprocessor
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise RuntimeError("Unable to find model or preprocessed file! Please ensure it has been run. preprocess.py 和 train.py！")
    
    print("[Lifespan] Loading preprocessor converter...")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("[Lifespan] Loading the best machine learning model...")
    model = joblib.load(MODEL_PATH)
    print("All artifacts have been loaded successfully, and the API service is ready!")
    yield
    print("Lifespan is shutting down its API service...")


app = FastAPI(title="Warsaw Apartment Rent Price Predictor", lifespan=lifespan)


class ApartmentInput(BaseModel):
    month: str
    city: str
    type: Optional[str] = None
    squareMeters: float
    rooms: float
    floor: Optional[float] = None
    floorCount: Optional[float] = None
    buildYear: Optional[float] = None
    latitude: float
    longitude: float
    centreDistance: float
    poiCount: float
    schoolDistance: float
    clinicDistance: float
    postOfficeDistance: float
    kindergartenDistance: float
    restaurantDistance: float
    collegeDistance: float
    pharmacyDistance: float
    ownership: Optional[str] = None
    buildingMaterial: Optional[str] = None
    condition: Optional[str] = None
    hasParkingSpace: str
    hasBalcony: str
    hasElevator: str
    hasSecurity: str
    hasStorageRoom: str


@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict_rent(data: ApartmentInput):
    """Main interface for predicting rent"""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # 1. Convert the received 27 features into a single-row DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # 2. The data is then sent to Person 2's preprocessor for cleaning (missing values ​​are automatically filled in by SimpleImputer).
        X_processed = preprocessor.transform(df)
        
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
            
        # 3. Predict and return
        prediction = model.predict(X_processed)
        return {
            "status": "success",
            "predicted_price_pln": float(prediction[0])
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("predict:app", host="127.0.0.1", port=8000, reload=True)