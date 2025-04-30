from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest
import time
from typing import Optional
import uvicorn

# Configure logging
logger.add("logs/app.log", rotation="500 MB", retention="10 days")

# Prometheus metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total number of predictions')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency in seconds')
ERROR_COUNTER = Counter('errors_total', 'Total number of errors', ['type'])

app = FastAPI(
    title="Predictive Maintenance API",
    description="API for predicting machine failures based on sensor data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.joblib")
DATA_PATH = os.getenv("DATA_PATH", "predictive_maintenance.csv")

# Initialize model and scaler
model = None
scaler = None

class MaintenanceData(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    type: str  # M, L, or H

    class Config:
        json_schema_extra = {
            "example": {
                "air_temperature": 25.0,
                "process_temperature": 35.0,
                "rotational_speed": 1500.0,
                "torque": 40.0,
                "tool_wear": 0.0,
                "type": "M"
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    failure_probability: float
    no_failure_probability: float
    message: str

@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        logger.info("Starting model loading process")
        
        # Train the model if it doesn't exist
        if not os.path.exists(MODEL_PATH):
            logger.info("Model not found, training new model")
            # Load and preprocess the data
            df = pd.read_csv(DATA_PATH)
            logger.info(f"Loaded data with shape: {df.shape}")

            # Check for required columns
            required_columns = ['Type', 'Target', 'UDI', 'Product ID']
            for col in required_columns:
                if col not in df.columns:
                    raise Exception(f"Missing required column: {col}")

            # Convert 'Type' to numeric using one-hot encoding
            type_dummies = pd.get_dummies(df['Type'], prefix='type')
            df = pd.concat([df, type_dummies], axis=1)

            # Drop unnecessary columns
            df = df.drop(['Type', 'UDI', 'Product ID', 'Failure Type'], axis=1)

            # If Target is already 0/1, skip mapping. Otherwise, map and drop NaN.
            if df['Target'].dtype == object:
                df['Target'] = df['Target'].map({'No': 0, 'Yes': 1})
            df = df.dropna(subset=['Target'])
            df['Target'] = df['Target'].astype(int)

            # Split features and target
            X = df.drop('Target', axis=1)
            y = df['Target']

            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train the model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)

            # Save the model and scaler
            joblib.dump(model, MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)
            logger.info("Model trained and saved successfully")
        else:
            # Load the saved model and scaler
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/")
async def root():
    return {"message": "Welcome to the Predictive Maintenance API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: MaintenanceData):
    start_time = time.time()
    try:
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'Air temperature [K]': [data.air_temperature],
            'Process temperature [K]': [data.process_temperature],
            'Rotational speed [rpm]': [data.rotational_speed],
            'Torque [Nm]': [data.torque],
            'Tool wear [min]': [data.tool_wear]
        })
        
        # Add one-hot encoded type columns
        type_dummies = pd.get_dummies(pd.Series([data.type]), prefix='type')
        input_data = pd.concat([input_data, type_dummies], axis=1)
        
        # Ensure all type columns exist
        for col in ['type_H', 'type_L', 'type_M']:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reorder columns to match training data
        input_data = input_data[['Air temperature [K]', 'Process temperature [K]', 
                               'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                               'type_H', 'type_L', 'type_M']]
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        failure_prob = probabilities[1] * 100  # Probability of failure in percent
        no_failure_prob = probabilities[0] * 100  # Probability of no failure in percent
        
        # Prepare response message
        message = "No Failure" if prediction == 0 else "Failure"
        
        # Update metrics
        PREDICTION_COUNTER.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        return PredictionResponse(
            prediction=int(prediction),
            failure_probability=round(failure_prob, 2),
            no_failure_probability=round(no_failure_prob, 2),
            message=message
        )
        
    except Exception as e:
        ERROR_COUNTER.labels(type="prediction").inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/metrics")
async def metrics():
    return generate_latest()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 