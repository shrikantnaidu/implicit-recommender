"""Example script to train and evaluate the ALS model on the MovieLens dataset."""
import argparse
import logging

import mlflow
import numpy as np
from scipy.sparse import csr_matrix  # noqa: F401

from implicit_vibecode.data.dataset import load_movielens_sample
from implicit_vibecode.evaluation.metrics import calculate_ndcg, calculate_precision_at_k
from implicit_vibecode.models.als_model import ALSModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_and_evaluate(params=None):
    """Train and evaluate the ALS model.
    
    Args:
        params: Dictionary of model parameters
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
    
    # Set up MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("implicit-vibecode-demo")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Load data
        print("Loading data...")
        interaction_matrix, movies_df, ratings_df, idx_to_user, idx_to_movie = load_movielens_sample()
        
        # Log dataset statistics
        num_users, num_items = interaction_matrix.shape
        num_interactions = interaction_matrix.nnz
        sparsity = 1 - (num_interactions / (num_users * num_items))
        
        logger.info("Dataset stats:")
        logger.info("- Users: %s", num_users)
        logger.info("- Items: %s", num_items)
        logger.info("- Interactions: %s", num_interactions)
        logger.info("- Sparsity: %.4f", sparsity)
        
        mlflow.log_metrics({
            "num_users": num_users,
            "num_items": num_items,
            "num_interactions": num_interactions,
            "sparsity": sparsity,
        })
        
        # Split into train/test
        np.random.seed(params["random_state"])
        train_data = interaction_matrix.copy()
        test_data = interaction_matrix.copy()
        
        # For simplicity, we'll use a random 80/20 split
        # In practice, you might want to use time-based splitting
        test_set = np.random.choice(
            interaction_matrix.nnz, 
            size=int(0.2 * interaction_matrix.nnz), 
            replace=False
        )
        
        train_data = train_data.tocoo()
        test_data = test_data.tocoo()
        
        # Set test interactions to zero in train data
        train_data.data[test_set] = 0
        train_data.eliminate_zeros()
        train_data = train_data.tocsr()
        
        # Only keep test interactions that are in the test set
        test_data = test_data.tocoo()
        test_mask = np.zeros(interaction_matrix.nnz, dtype=bool)
        test_mask[test_set] = True
        test_data.data[~test_mask] = 0
        test_data.eliminate_zeros()
        test_data = test_data.tocsr()
        
        # Train model
        logger.info("\nTraining model...")
        model = ALSModel(**params)
        model.fit(train_data)
        
        # Evaluate on test set
        logger.info("\nEvaluating model...")
        all_ndcg = []
        all_precision = []
        
        # For each user with test interactions, get recommendations and compute metrics
        test_users = test_data.nonzero()[0]
        for user_id in test_users[:100]:  # Limit to first 100 users for speed
            # Get test items for this user
            test_items = test_data[user_id].indices
            
            if len(test_items) == 0:
                continue
                
            # Get recommendations
            recommended_items, _ = model.recommend(
                user_id=0,  # Placeholder user ID
                user_items=train_data[user_id],
                N=100,  # Get top 100 recommendations
                filter_already_liked_items=True
            )
            
            if len(recommended_items) == 0:
                continue
                
            # Create relevance vectors
            true_relevance = np.zeros(100)
            pred_relevance = np.zeros(100)
            
            # Set true relevance for test items
            for i, item in enumerate(recommended_items):
                if item in test_items:
                    true_relevance[i] = 1
                pred_relevance[i] = 1.0 - (i * 0.01)  # Simulate decreasing relevance
            
            # Calculate metrics
            ndcg = calculate_ndcg(true_relevance, pred_relevance, k=10)
            precision = calculate_precision_at_k(true_relevance, pred_relevance, k=10)
            
            all_ndcg.append(ndcg)
            all_precision.append(precision)
        
        # Calculate average metrics
        avg_ndcg = np.mean(all_ndcg) if all_ndcg else 0
        avg_precision = np.mean(all_precision) if all_precision else 0
        
        logger.info("\nEvaluation results:")
        logger.info("- NDCG@10: %.4f", avg_ndcg)
        logger.info("- Precision@10: %.4f", avg_precision)
        
        # Log metrics
        mlflow.log_metrics({
            "ndcg@10": avg_ndcg,
            "precision@10": avg_precision,
        })
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log sample recommendations
        logger.info("\nSample recommendations for user 0:")
        recommended_items, scores = model.recommend(
            user_id=0,
            user_items=train_data[0],
            N=5,
            filter_already_liked_items=True
        )
        
        logger.info("\nTop 5 recommendations:")
        for i, (item_id, score) in enumerate(zip(recommended_items, scores), 1):
            movie_id = idx_to_movie[item_id]
            movie_title = movies_df[movies_df["movieId"] == movie_id]["title"].values[0]
            logger.info("%d. %s (score: %.4f)", i, movie_title, score)
        
        return model, {"ndcg@10": avg_ndcg, "precision@10": avg_precision}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ALS model on MovieLens data")
    parser.add_argument("--factors", type=int, default=64, help="Number of factors")
    parser.add_argument("--regularization", type=float, default=0.01, help="Regularization parameter")
    parser.add_argument("--iterations", type=int, default=15, help="Number of iterations")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    params = {
        "factors": args.factors,
        "regularization": args.regularization,
        "iterations": args.iterations,
        "use_gpu": args.use_gpu,
        "random_state": args.seed,
    }
    
    train_and_evaluate(params)
