"""
FastAPI Web Application & REST API for Netflix Content Type Prediction.
Serves interactive prediction dashboard, model metrics, visual assets, and REST endpoints.
"""

import os
import sys
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.predict import NetflixPredictor
from src.data_loader import load_dataset, clean_data, get_dataset_summary

app = FastAPI(
    title="Netflix Content Type Intelligence System",
    description="Production-grade Machine Learning API for Netflix Movie vs. TV Show Classification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
frontend_dir = os.path.join(BASE_DIR, "frontend")
reports_fig_dir = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(frontend_dir, exist_ok=True)
os.makedirs(reports_fig_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
app.mount("/figures", StaticFiles(directory=reports_fig_dir), name="figures")

# Load Predictor and Dataset Sample in memory
PREDICTOR: Optional[NetflixPredictor] = None
DATASET_SAMPLE: Optional[pd.DataFrame] = None
METRICS_DATA: Optional[Dict[str, Any]] = None


@app.on_event("startup")
def startup_event():
    global PREDICTOR, DATASET_SAMPLE, METRICS_DATA
    try:
        model_path = os.path.join(BASE_DIR, "models", "content_type_pipeline.joblib")
        if os.path.exists(model_path):
            PREDICTOR = NetflixPredictor(model_path)
            print("Successfully loaded trained ML pipeline artifact.")
        else:
            print("Warning: Model pipeline artifact not found. Please run main.py first.")
            
        dataset_path = os.path.join(BASE_DIR, "Dataset.csv")
        if os.path.exists(dataset_path):
            df = clean_data(load_dataset(dataset_path))
            DATASET_SAMPLE = df
            print(f"Loaded dataset catalog with {len(df):,} items.")

        metrics_path = os.path.join(BASE_DIR, "reports", "metrics_summary.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                METRICS_DATA = json.load(f)
    except Exception as e:
        print(f"Startup initialization note: {e}")


class PredictionRequest(BaseModel):
    title: str = Field(..., example="Stranger Things")
    director: Optional[str] = Field(default="Unknown", example="The Duffer Brothers")
    country: Optional[str] = Field(default="United States", example="United States")
    release_year: Optional[int] = Field(default=2021, example=2016)
    rating: Optional[str] = Field(default="TV-MA", example="TV-14")
    listed_in: Optional[str] = Field(default="TV Dramas, TV Sci-Fi & Fantasy", example="TV Dramas, TV Sci-Fi & Fantasy")
    date_added: Optional[str] = Field(default="9/24/2021", example="7/15/2016")


class BatchPredictionRequest(BaseModel):
    items: List[PredictionRequest]


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h2>Netflix Content Type Prediction System API is running.</h2>")


@app.post("/api/predict")
async def predict_content_type(req: PredictionRequest):
    global PREDICTOR
    if PREDICTOR is None:
        model_path = os.path.join(BASE_DIR, "models", "content_type_pipeline.joblib")
        if os.path.exists(model_path):
            PREDICTOR = NetflixPredictor(model_path)
        else:
            raise HTTPException(status_code=503, detail="Model pipeline not yet trained. Run main.py.")
            
    try:
        result = PREDICTOR.predict_single(req.dict())
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict-batch")
async def predict_batch(req: BatchPredictionRequest):
    global PREDICTOR
    if PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Model pipeline not initialized.")
    try:
        items_dict = [item.dict() for item in req.items]
        df = pd.DataFrame(items_dict)
        res_df = PREDICTOR.predict_batch(df)
        return {
            "status": "success",
            "count": len(res_df),
            "data": res_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics")
async def get_metrics():
    global METRICS_DATA
    if METRICS_DATA is None:
        metrics_path = os.path.join(BASE_DIR, "reports", "metrics_summary.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                METRICS_DATA = json.load(f)
        else:
            raise HTTPException(status_code=404, detail="Metrics summary not found.")
    return METRICS_DATA


@app.get("/api/catalog")
async def get_catalog(page: int = 1, page_size: int = 20, search: str = "", content_type: str = "ALL"):
    global DATASET_SAMPLE
    if DATASET_SAMPLE is None:
        dataset_path = os.path.join(BASE_DIR, "Dataset.csv")
        if os.path.exists(dataset_path):
            DATASET_SAMPLE = clean_data(load_dataset(dataset_path))
        else:
            raise HTTPException(status_code=404, detail="Dataset not found.")
            
    filtered = DATASET_SAMPLE.copy()
    if search:
        s_lower = search.lower()
        filtered = filtered[
            filtered['title'].str.lower().str.contains(s_lower) |
            filtered['director'].str.lower().str.contains(s_lower) |
            filtered['country'].str.lower().str.contains(s_lower) |
            filtered['listed_in'].str.lower().str.contains(s_lower)
        ]
        
    if content_type != "ALL":
        filtered = filtered[filtered['type'].str.upper() == content_type.upper()]
        
    total_count = len(filtered)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_data = filtered.iloc[start_idx:end_idx].to_dict(orient="records")
    
    return {
        "page": page,
        "page_size": page_size,
        "total_records": total_count,
        "total_pages": (total_count + page_size - 1) // page_size,
        "records": page_data
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
