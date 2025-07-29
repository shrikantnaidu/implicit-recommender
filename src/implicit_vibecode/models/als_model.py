"""ALS (Alternating Least Squares) model for implicit feedback."""
from typing import Dict, Optional, Tuple

import implicit
import numpy as np
from scipy.sparse import csr_matrix


class ALSModel:
    """ALS model wrapper for implicit feedback recommendations."""

    def __init__(
        self,
        factors: int = 100,
        regularization: float = 0.01,
        iterations: int = 15,
        use_gpu: bool = False,
        random_state: Optional[int] = None,
    ):
        """Initialize ALS model.

        Args:
            factors: Number of latent factors
            regularization: Regularization parameter
            iterations: Number of ALS iterations
            use_gpu: Whether to use GPU acceleration
            random_state: Random seed for reproducibility
        """
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_gpu=use_gpu,
            random_state=random_state,
        )
        self.user_factors = None
        self.item_factors = None

    def fit(
        self, user_items: csr_matrix, user_weights: Optional[csr_matrix] = None, **kwargs
    ) -> "ALSModel":
        """Fit the ALS model.

        Args:
            user_items: Sparse matrix of user-item interactions
            user_weights: Optional weights for user-item interactions
            **kwargs: Additional arguments to pass to the underlying model

        Returns:
            self: The fitted model
        """
        self.model.fit(user_items, user_weights=user_weights, **kwargs)
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors
        return self

    def recommend(
        self,
        user_id: int,
        user_items: csr_matrix,
        N: int = 10,
        filter_already_liked_items: bool = True,
        filter_items: Optional[np.ndarray] = None,
        recalculate_user: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate recommendations for a user.

        Args:
            user_id: User ID to generate recommendations for
            user_items: Sparse matrix of user-item interactions
            N: Number of recommendations to return
            filter_already_liked_items: Whether to filter out items the user has already interacted with
            filter_items: Array of item IDs to filter from recommendations
            recalculate_user: Whether to recalculate the user's factors

        Returns:
            Tuple of (item_ids, scores) for the top N recommendations
        """
        return self.model.recommend(
            user_id,
            user_items,
            N=N,
            filter_already_liked_items=filter_already_liked_items,
            filter_items=filter_items,
            recalculate_user=recalculate_user,
        )

    def similar_items(
        self, item_id: int, N: int = 10, item_users: Optional[csr_matrix] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find similar items to the given item.

        Args:
            item_id: Item ID to find similar items for
            N: Number of similar items to return
            item_users: Sparse matrix of item-user interactions (optional)

        Returns:
            Tuple of (item_ids, scores) for the top N similar items
        """
        return self.model.similar_items(item_id, N=N, item_users=item_users)

    def get_params(self) -> Dict:
        """Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        return {
            "factors": self.model.factors,
            "regularization": self.model.regularization,
            "iterations": self.model.iterations,
            "use_gpu": self.model.use_gpu,
            "random_state": self.model.random_state,
        }
