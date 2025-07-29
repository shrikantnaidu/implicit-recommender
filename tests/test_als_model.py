"""Tests for the ALS model."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from implicit_vibecode.models.als_model import ALSModel


def test_als_model_initialization():
    """Test that the ALS model initializes correctly."""
    model = ALSModel(factors=32, regularization=0.01, iterations=10)
    
    assert model is not None
    assert model.model.factors == 32
    assert model.model.regularization == 0.01
    assert model.model.iterations == 10


def test_als_model_fit():
    """Test that the ALS model can be fitted."""
    # Create a small synthetic dataset
    data = np.array([
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1],
    ], dtype=np.float32)
    
    sparse_data = csr_matrix(data)
    
    # Initialize and fit model
    model = ALSModel(factors=2, iterations=5)
    model.fit(sparse_data)
    
    # Check that user and item factors were learned
    assert model.user_factors is not None
    assert model.item_factors is not None
    assert model.user_factors.shape == (3, 2)  # 3 users, 2 factors
    assert model.item_factors.shape == (5, 2)  # 5 items, 2 factors


def test_als_model_recommend():
    """Test that the ALS model can generate recommendations."""
    # Create a small synthetic dataset
    data = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
    ], dtype=np.float32)
    
    sparse_data = csr_matrix(data)
    
    # Initialize and fit model
    model = ALSModel(factors=2, iterations=5)
    model.fit(sparse_data)
    
    # Get recommendations for user 0
    user_id = 0
    item_indices, scores = model.recommend(
        user_id=user_id,
        user_items=sparse_data[user_id],
        N=2,
        filter_already_liked_items=True
    )
    
    # Check that we get the expected number of recommendations
    assert len(item_indices) == 2
    assert len(scores) == 2
    
    # Check that scores are in descending order
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))


def test_als_model_similar_items():
    """Test that the ALS model can find similar items."""
    # Create a small synthetic dataset
    data = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
    ], dtype=np.float32)
    
    sparse_data = csr_matrix(data)
    
    # Initialize and fit model
    model = ALSModel(factors=2, iterations=5)
    model.fit(sparse_data)
    
    # Get similar items for item 0
    item_id = 0
    item_indices, scores = model.similar_items(item_id=item_id, N=2)
    
    # Check that we get the expected number of similar items
    assert len(item_indices) == 2
    assert len(scores) == 2
    
    # Check that the item itself is not in the results
    assert item_id not in item_indices
    
    # Check that scores are in descending order
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
