"""
FastAPI inference server for trading model.
Provides /predict endpoint for real-time trading signals.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from typing import List, Optional
import os
from datetime import datetime

# Import model components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.tcn import TCNModel, CalibratedTCN
from src.data.features import FeatureEngineer
from src.models.volatility import VolatilityEstimator
from src.trading.signals import TradingSignalGenerator, TradingConfig


# Initialize FastAPI app
app = FastAPI(
    title="Neural Trading System API",
    description="Real-time trading signal generation using TCN",
    version="1.0.0"
)

# Global state
model = None
calibrated_model = None
feature_engineer = None
signal_generator = None
vol_estimator = None
model_loaded = False
server_start_time = datetime.now()


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    features: List[List[float]]  # (sequence_length, n_features)
    current_price: float
    vol_forecast: Optional[float] = None


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    probability: float
    signal: float
    target_weight: float
    confidence: str
    uncertainty: Optional[float] = None
    timestamp: str


class HealthResponse(BaseModel):
    """Response schema for health endpoint."""
    status: str
    model_loaded: bool
    uptime_seconds: float
    timestamp: str


@app.on_event("startup")
async def load_model():
    """Load model on server startup."""
    global model, calibrated_model, feature_engineer, signal_generator, vol_estimator, model_loaded
    
    try:
        # Load model
        model_path = os.getenv("MODEL_PATH", "models/tcn_v1/final_model.pt")
        scaler_path = os.getenv("SCALER_PATH", "models/tcn_v1/scaler.pkl")
        
        print(f"Loading model from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = checkpoint['model_config']
        
        model = TCNModel(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load calibrator
        calibrator = checkpoint.get('calibrator', None)
        if calibrator:
            calibrated_model = CalibratedTCN(model, calibration_method='platt', device='cpu')
            calibrated_model.calibrator = calibrator
        else:
            calibrated_model = None
        
        # Load feature engineer with scaler
        feature_engineer = FeatureEngineer()
        if os.path.exists(scaler_path):
            feature_engineer.load_scaler(scaler_path)
        
        # Initialize signal generator
        config = TradingConfig()
        signal_generator = TradingSignalGenerator(config)
        
        # Initialize volatility estimator
        vol_estimator = VolatilityEstimator(method='dual')
        
        model_loaded = True
        print("✓ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model_loaded = False


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "service": "Neural Trading System API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - server_start_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        uptime_seconds=uptime,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate trading signal from input features.
    
    Args:
        request: PredictionRequest with features and current price
        
    Returns:
        PredictionResponse with signal and position sizing
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to tensor
        features = np.array(request.features).reshape(1, -1, 12)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Get prediction
        if calibrated_model:
            probability = calibrated_model.predict_proba(features_tensor)[0]
        else:
            with torch.no_grad():
                logit = model(features_tensor).item()
                probability = 1 / (1 + np.exp(-logit))
        
        # Estimate volatility if not provided
        if request.vol_forecast is None:
            # Use default or estimate from features
            vol_forecast = 0.02  # Default 2%
        else:
            vol_forecast = request.vol_forecast
        
        # Generate signal
        signal = signal_generator.generate_signal(probability, vol_forecast)
        
        # Compute target weight
        target_vol = 0.02
        r_t = (target_vol / vol_forecast) * signal
        target_weight = np.clip(r_t, -1.0, 1.0)
        
        # Determine confidence level
        confidence_score = abs(probability - 0.5)
        if confidence_score > 0.2:
            confidence = "high"
        elif confidence_score > 0.1:
            confidence = "medium"
        else:
            confidence = "low"
        
        return PredictionResponse(
            probability=float(probability),
            signal=float(signal),
            target_weight=float(target_weight),
            confidence=confidence,
            uncertainty=None,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint."""
    uptime = (datetime.now() - server_start_time).total_seconds()
    
    metrics = f"""
# HELP model_loaded Model loaded status
# TYPE model_loaded gauge
model_loaded {int(model_loaded)}

# HELP server_uptime_seconds Server uptime in seconds
# TYPE server_uptime_seconds counter
server_uptime_seconds {uptime}
"""
    
    return metrics


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )