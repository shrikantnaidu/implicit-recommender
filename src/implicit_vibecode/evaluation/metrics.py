"""Evaluation metrics for recommender systems."""
from typing import List, Optional, Union

import numpy as np
from scipy.sparse import csr_matrix


def calculate_ndcg(
    true_relevance: Union[np.ndarray, List[float]],
    predicted_relevance: Union[np.ndarray, List[float]],
    k: Optional[int] = None,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG).

    Args:
        true_relevance: True relevance scores (binary or graded)
        predicted_relevance: Predicted relevance scores
        k: Number of top results to consider. If None, use all.

    Returns:
        NDCG score
    """
    true_relevance = np.asarray(true_relevance).flatten()
    predicted_relevance = np.asarray(predicted_relevance).flatten()
    
    if k is not None:
        k = min(k, len(true_relevance))
        top_k_indices = np.argsort(predicted_relevance)[-k:][::-1]
        true_relevance = true_relevance[top_k_indices]
        predicted_relevance = predicted_relevance[top_k_indices]
    
    # Calculate DCG
    ranks = np.arange(2, len(true_relevance) + 2)
    dcg = np.sum((2 ** true_relevance - 1) / np.log2(ranks))
    
    # Calculate ideal DCG
    ideal_relevance = np.sort(true_relevance)[::-1]
    idcg = np.sum((2 ** ideal_relevance - 1) / np.log2(ranks))
    
    return dcg / idcg if idcg > 0 else 0.0


def calculate_precision_at_k(
    true_relevance: Union[np.ndarray, List[int]],
    predicted_relevance: Union[np.ndarray, List[float]],
    k: int,
    threshold: float = 0.5,
) -> float:
    """Calculate Precision@K.

    Args:
        true_relevance: True relevance scores (binary or graded)
        predicted_relevance: Predicted relevance scores
        k: Number of top results to consider
        threshold: Threshold for considering an item relevant

    Returns:
        Precision@K score
    """
    true_relevance = np.asarray(true_relevance).flatten()
    predicted_relevance = np.asarray(predicted_relevance).flatten()
    
    if k > len(true_relevance):
        k = len(true_relevance)
    
    # Get top K predictions
    top_k_indices = np.argsort(predicted_relevance)[-k:][::-1]
    top_k_relevance = true_relevance[top_k_indices]
    
    # Calculate precision
    relevant = np.sum(top_k_relevance >= threshold)
    return relevant / k


def calculate_map(
    true_relevance: Union[np.ndarray, List[int]],
    predicted_relevance: Union[np.ndarray, List[float]],
    k: Optional[int] = None,
    threshold: float = 0.5,
) -> float:
    """Calculate Mean Average Precision (MAP).

    Args:
        true_relevance: True relevance scores (binary or graded)
        predicted_relevance: Predicted relevance scores
        k: Number of top results to consider. If None, use all.
        threshold: Threshold for considering an item relevant

    Returns:
        MAP score
    """
    true_relevance = np.asarray(true_relevance).flatten()
    predicted_relevance = np.asarray(predicted_relevance).flatten()
    
    if k is not None and k < len(true_relevance):
        top_k_indices = np.argsort(predicted_relevance)[-k:][::-1]
        true_relevance = true_relevance[top_k_indices]
        predicted_relevance = predicted_relevance[top_k_indices]
    
    # Sort by predicted relevance
    sort_indices = np.argsort(predicted_relevance)[::-1]
    true_relevance = true_relevance[sort_indices]
    
    # Calculate precision at each position
    relevant = (true_relevance >= threshold).astype(float)
    cumsum = np.cumsum(relevant)
    precision_at_k = cumsum / (np.arange(len(true_relevance)) + 1)
    
    # Calculate average precision
    avg_precision = np.sum(precision_at_k * relevant) / max(1, np.sum(relevant))
    return avg_precision
