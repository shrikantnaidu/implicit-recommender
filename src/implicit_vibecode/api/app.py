"""FastAPI application for serving recommendations."""
from typing import Dict, List

import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from implicit_vibecode.models.als_model import ALSModel

app = FastAPI(title="Implicit Vibecode API")

# Global model and metadata
model = None
metadata = {}


class RecommendationRequest(BaseModel):
    user_id: int
    k: int = 10


class SimilarItemsRequest(BaseModel):
    item_id: int
    k: int = 10


@app.on_event("startup")
def load_model():
    """Load the trained model and metadata."""
    global model, metadata
    
    try:
        # Load the latest model from MLflow
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("implicit-vibecode")
        
        if experiment is None:
            raise HTTPException(
                status_code=404,
                detail="No trained model found. Please train a model first."
            )
            
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        
        if not runs:
            raise HTTPException(
                status_code=404,
                detail="No trained model found. Please train a model first."
            )
            
        run_id = runs[0].info.run_id
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Load metadata (you might want to store this in MLflow as well)
        metadata = {}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Implicit Vibecode API",
        "status": "running",
        "endpoints": [
            "/recommend",
            "/similar",
            "/health"
        ]
    }


@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest) -> Dict[str, List[Dict]]:
    """Get recommendations for a user.
    
    Args:
        request: Recommendation request containing user_id and number of recommendations
        
    Returns:
        List of recommended items with scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # In a real application, you would get the user's interaction history
        # For now, we'll just return similar items to the user's most interacted with item
        user_interactions = np.zeros((1, model.item_factors.shape[0]))
        
        # Get top recommendations
        item_indices, scores = model.recommend(
            user_id=0,  # Placeholder user ID
            user_items=user_interactions,
            N=request.k,
            filter_already_liked_items=True
        )
        
        # Convert to list of dicts
        recommendations = [
            {"item_id": int(idx), "score": float(score)}
            for idx, score in zip(item_indices, scores)
        ]
        
        return {"recommendations": recommendations}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similar")
async def get_similar_items(request: SimilarItemsRequest) -> Dict[str, List[Dict]]:
    """Get similar items to the given item.
    
    Args:
        request: Similar items request containing item_id and number of similar items
        
    Returns:
        List of similar items with scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get similar items
        item_indices, scores = model.similar_items(
            item_id=request.item_id,
            N=request.k
        )
        
        # Convert to list of dicts
        similar_items = [
            {"item_id": int(idx), "score": float(score)}
            for idx, score in zip(item_indices, scores)
        ]
        
        return {"similar_items": similar_items}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "healthy"}
