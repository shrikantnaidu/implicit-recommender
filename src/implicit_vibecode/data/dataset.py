"""Dataset loading and preprocessing utilities for MovieLens dataset."""
import os
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


def load_movielens_sample(data_dir: str = "data") -> Tuple[csr_matrix, pd.DataFrame, pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """Load the MovieLens sample dataset.
    
    Downloads the dataset if not already present and loads it into memory.
    
    Args:
        data_dir: Directory to store/load the dataset files
        
    Returns:
        A tuple containing:
            - interaction_matrix: Sparse matrix of user-movie interactions
            - movies_df: DataFrame containing movie metadata
            - ratings_df: DataFrame containing raw rating data
            - idx_to_user: Mapping from matrix index to user ID
            - idx_to_movie: Mapping from matrix index to movie ID
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Define file paths
    ratings_path = os.path.join(data_dir, "ratings.csv")
    movies_path = os.path.join(data_dir, "movies.csv")
    
    # Download the dataset if it doesn't exist
    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        _download_movielens_data(data_dir)
    
    # Load the data
    try:
        print("Loading MovieLens dataset...")
        ratings_df = pd.read_csv(ratings_path)
        movies_df = pd.read_csv(movies_path)
        
        # Basic data validation
        required_columns = {
            'ratings': ['userId', 'movieId', 'rating'],
            'movies': ['movieId', 'title', 'genres']
        }
        
        for col in required_columns['ratings']:
            if col not in ratings_df.columns:
                raise ValueError(f"Missing required column '{col}' in ratings data")
                
        for col in required_columns['movies']:
            if col not in movies_df.columns:
                raise ValueError(f"Missing required column '{col}' in movies data")
        
        print(f"Loaded {len(ratings_df)} ratings from {len(ratings_df['userId'].unique())} users "
              f"on {len(ratings_df['movieId'].unique())} movies")
        
        # Create user and movie mappings
        user_ids = {user_id: idx for idx, user_id in enumerate(ratings_df['userId'].unique())}
        movie_ids = {movie_id: idx for idx, movie_id in enumerate(ratings_df['movieId'].unique())}
        
        # Create interaction matrix (implicit feedback: 1 if user rated the movie, 0 otherwise)
        rows = [user_ids[user_id] for user_id in ratings_df['userId']]
        cols = [movie_ids[movie_id] for movie_id in ratings_df['movieId']]
        # Using rating values as weights (normalized to 0-1)
        values = [min(float(r), 5.0) / 5.0 for r in ratings_df['rating']]
        
        interaction_matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(user_ids), len(movie_ids)),
            dtype=np.float32
        )
        
        # Create reverse mappings
        idx_to_user = {idx: user_id for user_id, idx in user_ids.items()}
        idx_to_movie = {idx: movie_id for movie_id, idx in movie_ids.items()}
        
        return interaction_matrix, movies_df, ratings_df, idx_to_user, idx_to_movie
        
    except Exception as e:
        print(f"Error loading MovieLens dataset: {str(e)}")
        raise


def _download_movielens_data(data_dir: str) -> None:
    """Download and extract the MovieLens dataset."""
    import urllib.request
    import zipfile
    import shutil
    
    print("Downloading MovieLens dataset...")
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zip_path = os.path.join(data_dir, "ml-latest-small.zip")
    
    try:
        # Download the file
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract the dataset
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Move files to the main data directory
        src_dir = os.path.join(data_dir, "ml-latest-small")
        for file in ["ratings.csv", "movies.csv"]:
            src_file = os.path.join(src_dir, file)
            if os.path.exists(src_file):
                shutil.move(src_file, data_dir)
        
        # Clean up
        shutil.rmtree(src_dir, ignore_errors=True)
        os.remove(zip_path)
        
        print("Successfully downloaded and extracted MovieLens dataset")
        
    except Exception as e:
        print(f"Error downloading MovieLens dataset: {str(e)}")
        # Clean up partially downloaded files
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise
