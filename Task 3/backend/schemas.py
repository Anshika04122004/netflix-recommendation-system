"""
Pydantic schemas for the Netflix Audience Rating Classifier API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ContentPredictionInput(BaseModel):
    type: str = Field(..., example="Movie", description="Content type: Movie or TV Show")
    title: Optional[str] = Field(None, example="Inception", description="Content Title (excluded from ML features)")
    director: Optional[str] = Field("Not Given", example="Christopher Nolan", description="Director name(s)")
    cast: Optional[str] = Field("", example="Leonardo DiCaprio, Cillian Murphy", description="Comma-separated cast members")
    country: Optional[str] = Field("United States", example="United States, United Kingdom", description="Country/countries of production")
    date_added: Optional[str] = Field("2021-09-25", example="9/25/2021", description="Date content was added to Netflix")
    release_year: Optional[int] = Field(2020, example=2010, description="Year content was originally released")
    duration: Optional[str] = Field("90 min", example="148 min", description="Content duration (e.g., '148 min' or '3 Seasons')")
    listed_in: Optional[str] = Field("Documentaries", example="Action & Adventure, Sci-Fi & Fantasy", description="Comma-separated genres/categories")
    description: Optional[str] = Field("", example="A thief who steals corporate secrets...", description="Content synopsis/description")


class PredictionResponse(BaseModel):
    predicted_rating: str
    confidence: float
    model_used: str
    rating_category_info: Dict[str, str]
    probabilities: Dict[str, float]
    top_contributing_features: List[Dict[str, Any]]
    input_received: Dict[str, Any]


class ModelMetricDetail(BaseModel):
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    classification_report: Dict[str, Any]
    confusion_matrix: List[List[int]]


class ModelInfoResponse(BaseModel):
    best_model_name: str
    dataset_info: Dict[str, Any]
    models: Dict[str, Any]
    tuning_parameters: Dict[str, Any]
    feature_importances: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    dataset_loaded: bool
    model_loaded: bool
