"""Training pipeline for the ALS model with MLflow tracking."""
import argparse
from typing import Dict, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from implicit_vibecode.data.dataset import load_movielens_sample
from implicit_vibecode.models.als_model import ALSModel


def prepare_data() -> Tuple[csr_matrix, csr_matrix, Dict]:
    """Load and prepare the dataset.
    
    Returns:
        Tuple of (train_matrix, test_matrix, metadata)
    """
    # Load the dataset
    interaction_matrix, movies_df, ratings_df, idx_to_user, idx_to_movie = load_movielens_sample()
    
    # Split into train/test
    train_matrix, test_matrix = train_test_split(
        interaction_matrix, test_size=0.2, random_state=42
    )
    
    # Prepare metadata
    metadata = {
        "num_users": interaction_matrix.shape[0],
        "num_items": interaction_matrix.shape[1],
        "total_interactions": interaction_matrix.nnz,
        "train_interactions": train_matrix.nnz,
        "test_interactions": test_matrix.nnz,
        "sparsity": 1 - (interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1])),
        "idx_to_user": idx_to_user,
        "idx_to_movie": idx_to_movie,
        "movies_df": movies_df
    }
    
    return train_matrix, test_matrix, metadata


def train_model(params: Dict = None) -> Tuple[ALSModel, Dict]:
    """Train the ALS model with the given parameters.
    
    Args:
        params: Dictionary of model parameters
        
    Returns:
        Tuple of (trained_model, metrics)
    """
    # Default parameters
    if params is None:
        params = {
            "factors": 64,
            "regularization": 0.01,
            "iterations": 15,
            "use_gpu": False,
            "random_state": 42,
        }
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Prepare data
        train_matrix, test_matrix, metadata = prepare_data()
        
        # Log dataset metrics
        mlflow.log_metrics({
            "num_users": metadata["num_users"],
            "num_items": metadata["num_items"],
            "train_interactions": metadata["train_interactions"],
            "test_interactions": metadata["test_interactions"],
            "sparsity": metadata["sparsity"]
        })
        
        # Initialize and train model
        model = ALSModel(**params)
        model.fit(train_matrix)
        
        # Evaluate model (placeholder for actual evaluation)
        metrics = {
            "train_loss": 0.0,  # Replace with actual metrics
            "test_ndcg": 0.0,   # Replace with actual metrics
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return model, metrics


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train ALS model")
    parser.add_argument("--factors", type=int, default=64, help="Number of factors")
    parser.add_argument("--regularization", type=float, default=0.01, help="Regularization parameter")
    parser.add_argument("--iterations", type=int, default=15, help="Number of iterations")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("implicit-vibecode")
    
    # Train model
    params = {
        "factors": args.factors,
        "regularization": args.regularization,
        "iterations": args.iterations,
        "use_gpu": args.use_gpu,
        "random_state": args.seed,
    }
    
    model, metrics = train_model(params)
    print(f"Training complete. Metrics: {metrics}")


if __name__ == "__main__":
    main()
