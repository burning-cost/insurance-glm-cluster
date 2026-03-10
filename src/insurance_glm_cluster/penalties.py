"""
R2VF penalty construction and split-coding for GLM factor clustering.

Implements the two-step R2VF algorithm from Ben Dror (arXiv:2503.01521, 2025):

  Step 1 — Ranking: fit a Ridge GLM on all factor dummies simultaneously to
  produce a multivariate-adjusted ordering for each nominal factor.

  Step 2 — Fusion: re-encode nominals as ordinal using Step 1 ranking, then
  apply standard fused lasso via split-coding on all factors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

from .utils import (
    build_split_coding_matrix,
    apply_split_coding,
    levels_to_onehot,
    exposure_adjusted_target,
    bin_numeric,
)


class R2VFRanker:
    """
    Step 1 of R2VF: rank nominal factor levels via Ridge regression.

    Fits a regularised GLM on all factor dummies simultaneously, then uses the
    resulting coefficients to impose a data-driven ordering on each nominal
    factor. The ordering determines which levels are adjacent in Step 2.

    Parameters
    ----------
    alpha : float
        Ridge regularisation strength. The paper recommends alpha=2.
    random_state : int
        Random seed for reproducibility.

    Notes
    -----
    We use Ridge (L2) rather than Lasso for Step 1 because L2 preserves the
    full coefficient ranking — Lasso would zero out many levels, collapsing
    the ordering before Step 2 runs.
    """

    def __init__(self, alpha: float = 2.0, random_state: int = 42) -> None:
        self.alpha = alpha
        self.random_state = random_state
        self._rankings: dict[str, list] = {}
        self._coefficients: dict[str, NDArray[np.float64]] = {}

    def fit(
        self,
        X_dummies: NDArray[np.float64],
        y: NDArray[np.float64],
        factor_slices: dict[str, slice],
        sample_weight: NDArray[np.float64] | None = None,
    ) -> "R2VFRanker":
        """
        Fit Ridge regression and extract per-factor level rankings.

        Parameters
        ----------
        X_dummies : NDArray[np.float64]
            Full dummy-encoded design matrix of shape (n_samples, total_levels).
        y : NDArray[np.float64]
            Response variable (exposure-adjusted for Poisson).
        factor_slices : dict[str, slice]
            Maps each factor name to the column slice in X_dummies.
        sample_weight : NDArray[np.float64], optional
            Sample weights (use exposure for Poisson frequency models).

        Returns
        -------
        self
        """
        ridge = Ridge(
            alpha=self.alpha,
            fit_intercept=True,
        )
        ridge.fit(X_dummies, y, sample_weight=sample_weight)
        coef = ridge.coef_

        for factor, sl in factor_slices.items():
            factor_coef = coef[sl]
            # Rank levels by coefficient (ascending) — this is the fused lasso
            # ordering. Levels with similar coefficients will be adjacent.
            ranked_order = np.argsort(factor_coef)
            self._rankings[factor] = ranked_order.tolist()
            self._coefficients[factor] = factor_coef

        return self

    def get_ranking(self, factor: str) -> list[int]:
        """
        Return the ranked level indices for a factor (sorted by coefficient).

        Parameters
        ----------
        factor : str
            Factor name.

        Returns
        -------
        list[int]
            Indices that sort the levels by Ridge coefficient, ascending.
        """
        if factor not in self._rankings:
            raise ValueError(f"Factor '{factor}' not in fitted rankings.")
        return self._rankings[factor]

    def get_coefficients(self, factor: str) -> NDArray[np.float64]:
        """
        Return Ridge coefficients for each level of a factor.

        Parameters
        ----------
        factor : str
            Factor name.

        Returns
        -------
        NDArray[np.float64]
            Coefficient for each level, in original level order.
        """
        if factor not in self._coefficients:
            raise ValueError(f"Factor '{factor}' not in fitted coefficients.")
        return self._coefficients[factor]


def build_r2vf_design_matrix(
    factor_data: dict[str, NDArray[np.float64]],
    factor_categories: dict[str, list],
    nominal_rankings: dict[str, list] | None = None,
) -> tuple[NDArray[np.float64], dict[str, slice], dict[str, list]]:
    """
    Build the split-coded design matrix for the R2VF Step 2 fused lasso.

    For each factor:
    - Ordinal factors: levels already ordered, apply split-coding directly.
    - Nominal factors: reorder levels by Step 1 ranking, then split-code.

    Parameters
    ----------
    factor_data : dict[str, NDArray[np.float64]]
        Maps factor name to one-hot encoded matrix (n_samples, n_levels).
    factor_categories : dict[str, list]
        Maps factor name to ordered list of level labels.
    nominal_rankings : dict[str, list], optional
        Maps nominal factor name to ranked level indices from R2VFRanker.
        If None, treats all factors as ordinal.

    Returns
    -------
    NDArray[np.float64]
        Split-coded design matrix of shape (n_samples, total_levels).
    dict[str, slice]
        Maps factor name to column slice in the output matrix.
    dict[str, list]
        Maps factor name to the reordered level labels (for result lookup).
    """
    if nominal_rankings is None:
        nominal_rankings = {}

    blocks: list[NDArray[np.float64]] = []
    slices: dict[str, slice] = {}
    reordered_categories: dict[str, list] = {}
    col_offset = 0

    for factor, onehot in factor_data.items():
        categories = factor_categories[factor]

        if factor in nominal_rankings:
            # Reorder columns by Step 1 ranking
            ranking = nominal_rankings[factor]
            onehot_reordered = onehot[:, ranking]
            reordered_cats = [categories[i] for i in ranking]
        else:
            onehot_reordered = onehot
            reordered_cats = categories

        # Apply split-coding transformation
        n_levels = onehot_reordered.shape[1]
        if n_levels > 1:
            X_delta = apply_split_coding(onehot_reordered)
        else:
            X_delta = onehot_reordered.copy()

        blocks.append(X_delta)
        end = col_offset + n_levels
        slices[factor] = slice(col_offset, end)
        reordered_categories[factor] = reordered_cats
        col_offset = end

    if blocks:
        X_combined = np.hstack(blocks)
    else:
        X_combined = np.empty((0, 0), dtype=np.float64)

    return X_combined, slices, reordered_categories


def fit_fused_lasso(
    X_delta: NDArray[np.float64],
    y: NDArray[np.float64],
    lam: float,
    sample_weight: NDArray[np.float64] | None = None,
    max_iter: int = 10_000,
) -> NDArray[np.float64]:
    """
    Fit a Lasso on the split-coded design matrix to achieve fused lasso.

    Parameters
    ----------
    X_delta : NDArray[np.float64]
        Split-coded design matrix (n_samples, n_delta_cols).
    y : NDArray[np.float64]
        Response variable (exposure-adjusted for Poisson frequency).
    lam : float
        L1 penalty strength (lambda).
    sample_weight : NDArray[np.float64], optional
        Sample weights.
    max_iter : int
        Maximum iterations for coordinate descent.

    Returns
    -------
    NDArray[np.float64]
        Fitted delta coefficients of length n_delta_cols.

    Notes
    -----
    The Lasso objective here is:
        (1/2n) * ‖y - X_delta @ δ‖² + λ ‖δ‖₁

    Because X_delta encodes the split-coding transformation, L1 on δ is
    equivalent to a fused lasso penalty on adjacent factor level differences.
    """
    lasso = Lasso(
        alpha=lam,
        fit_intercept=True,
        max_iter=max_iter,
        random_state=42,
    )
    lasso.fit(X_delta, y, sample_weight=sample_weight)
    return lasso.coef_


def lambda_grid(
    X_delta: NDArray[np.float64],
    y: NDArray[np.float64],
    n_points: int = 50,
    lam_min_ratio: float = 0.001,
    sample_weight: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """
    Compute a log-spaced lambda grid from lambda_max down to lambda_min.

    lambda_max is the smallest lambda that zeroes all delta coefficients,
    computed analytically as max(|X.T @ y|) / n.

    Parameters
    ----------
    X_delta : NDArray[np.float64]
        Split-coded design matrix.
    y : NDArray[np.float64]
        Response variable.
    n_points : int
        Number of lambda values in the grid.
    lam_min_ratio : float
        Ratio of lambda_min to lambda_max.
    sample_weight : NDArray[np.float64], optional
        Sample weights for weighted X.T @ y computation.

    Returns
    -------
    NDArray[np.float64]
        Array of lambda values, decreasing.
    """
    n = X_delta.shape[0]
    if sample_weight is not None:
        w = sample_weight / sample_weight.sum() * n
        y_centered = y - np.average(y, weights=sample_weight)
        Xty = np.abs(X_delta.T @ (w * y_centered))
    else:
        y_centered = y - y.mean()
        Xty = np.abs(X_delta.T @ y_centered)

    lam_max = Xty.max() / n
    lam_max = max(lam_max, 1e-4)  # guard against degenerate cases
    lam_min = lam_max * lam_min_ratio
    return np.exp(np.linspace(np.log(lam_max), np.log(lam_min), n_points))
