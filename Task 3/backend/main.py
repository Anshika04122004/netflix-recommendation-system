"""
FastAPI application entry point for the Netflix Audience Rating Classifier.
Auspify Technologies - Machine Learning Internship | Task 3
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.schemas import ContentPredictionInput, PredictionResponse, ModelInfoResponse, HealthResponse
from backend.services.predictor import RatingPredictorService
from backend.services.analytics import AnalyticsService

app = FastAPI(
    title="Netflix Audience Rating Classifier API",
    description="Machine Learning classification pipeline predicting audience ratings for Netflix content using Gradient Boosting, Random Forest, and Voting Ensemble models.",
    version="2.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
predictor_service = RatingPredictorService(models_dir='models')
analytics_service = AnalyticsService(data_path='data/Dataset.csv', metrics_path='models/metrics.json')


@app.get("/api/health", response_model=HealthResponse)
def get_health():
    """
    Health check endpoint returning system status and model readiness.
    """
    return {
        "status": "healthy",
        "service": "netflix-audience-rating-classifier",
        "version": "1.0.0",
        "dataset_loaded": bool(analytics_service.dataset_stats),
        "model_loaded": predictor_service.is_loaded
    }


@app.get("/api/model-info")
def get_model_info():
    """
    Returns information on all trained models, hyperparameter configurations, and evaluation metrics.
    """
    if not analytics_service.metrics_data:
        analytics_service.load_data()
    return analytics_service.metrics_data


@app.get("/api/analytics")
def get_analytics():
    """
    Returns comprehensive EDA analytics, class distributions, release trends, and model comparisons.
    """
    return analytics_service.get_analytics()


@app.post("/api/predict", response_model=PredictionResponse)
def predict_rating(payload: ContentPredictionInput):
    """
    Predicts the audience rating category (e.g., TV-MA, PG-13, TV-14, R) for given Netflix content metadata.
    """
    try:
        result = predictor_service.predict(payload.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount built frontend static files if available
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist'))
if os.path.exists(frontend_dist):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
