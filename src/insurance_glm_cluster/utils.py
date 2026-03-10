"""
Utility functions for split-coding, adjacency, and bin construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def build_split_coding_matrix(n_levels: int) -> NDArray[np.float64]:
    """
    Build the split-coding transformation matrix for fused lasso.

    The split-coding trick converts a standard fused lasso (penalty on
    differences β_i - β_{i-1}) into a standard L1 problem. Define
    δ_i = β_i - β_{i-1} for i >= 1, and δ_0 = β_0. Then β_i = Σ_{s≤i} δ_s.

    This function returns the matrix T such that β = T @ δ, i.e. a lower
    triangular matrix of ones.

    Parameters
    ----------
    n_levels : int
        Number of factor levels.

    Returns
    -------
    NDArray[np.float64]
        Lower triangular matrix of shape (n_levels, n_levels).

    Examples
    --------
    >>> T = build_split_coding_matrix(3)
    >>> T
    array([[1., 0., 0.],
           [1., 1., 0.],
           [1., 1., 1.]])
    """
    return np.tril(np.ones((n_levels, n_levels), dtype=np.float64))


def apply_split_coding(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Transform a design matrix from β-space to δ-space using split coding.

    If X has columns [x_0, x_1, ..., x_{K-1}] corresponding to level
    indicators, the transformed matrix X_delta = X @ T^{-T} such that fitting
    with L1 penalty on the transformed coefficients δ achieves the fused lasso
    on the original β.

    In practice, for the split-coded design matrix, each row is transformed by
    cumulative summation: X_delta[:, i] = sum of X[:, i:] along columns,
    which is equivalent to X @ T^{-T} where T is lower triangular ones.

    Parameters
    ----------
    X : NDArray[np.float64]
        Design matrix of shape (n_samples, n_levels). Each row sums to at most 1
        (one-hot encoded factor levels).

    Returns
    -------
    NDArray[np.float64]
        Transformed design matrix of shape (n_samples, n_levels).
    """
    # X_delta[:, i] = sum(X[:, i:], axis=1)
    # This means: column i of X_delta equals "is level >= i"
    return np.cumsum(X[:, ::-1], axis=1)[:, ::-1].astype(np.float64)


def levels_to_onehot(
    series: pd.Series,
    categories: list | None = None,
) -> tuple[NDArray[np.float64], list]:
    """
    Convert a categorical Series to a one-hot encoded matrix.

    Parameters
    ----------
    series : pd.Series
        Categorical series to encode.
    categories : list, optional
        Ordered list of categories. If None, uses sorted unique values.

    Returns
    -------
    NDArray[np.float64]
        One-hot matrix of shape (n_samples, n_categories).
    list
        Ordered list of category labels used.
    """
    if categories is None:
        categories = sorted(series.dropna().unique().tolist())
    cat_type = pd.CategoricalDtype(categories=categories, ordered=True)
    cat = series.astype(cat_type)
    codes = cat.cat.codes.values  # -1 for NaN
    n = len(series)
    k = len(categories)
    X = np.zeros((n, k), dtype=np.float64)
    mask = codes >= 0
    X[mask, codes[mask]] = 1.0
    return X, categories


def delta_to_beta(delta: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert delta coefficients back to beta (level) coefficients.

    β_i = Σ_{s≤i} δ_s, i.e. cumulative sum of deltas.

    Parameters
    ----------
    delta : NDArray[np.float64]
        Array of delta coefficients of length K.

    Returns
    -------
    NDArray[np.float64]
        Array of beta coefficients of length K.
    """
    return np.cumsum(delta)


def bin_numeric(
    series: pd.Series,
    n_bins: int,
    strategy: str = "quantile",
) -> tuple[pd.Series, list]:
    """
    Bin a numeric series into discrete bins for ordinal treatment.

    Parameters
    ----------
    series : pd.Series
        Numeric series to bin.
    n_bins : int
        Maximum number of bins.
    strategy : str
        'quantile' or 'uniform'. Quantile binning is preferred for skewed
        insurance variables (e.g. vehicle age, engine size).

    Returns
    -------
    pd.Series
        Integer bin codes (0-indexed), same length as input.
    list
        Sorted list of unique bin labels used.
    """
    n_unique = series.nunique()
    if n_unique <= n_bins:
        # Enough unique values — use rank encoding directly
        unique_sorted = sorted(series.dropna().unique().tolist())
        mapping = {v: i for i, v in enumerate(unique_sorted)}
        binned = series.map(mapping)
        return binned.fillna(-1).astype(int), list(range(len(unique_sorted)))

    if strategy == "quantile":
        quantiles = np.linspace(0, 100, n_bins + 1)
        breakpoints = np.unique(np.percentile(series.dropna(), quantiles))
        labels = list(range(len(breakpoints) - 1))
        binned = pd.cut(
            series,
            bins=breakpoints,
            labels=labels,
            include_lowest=True,
        )
    else:
        min_val, max_val = series.min(), series.max()
        breakpoints = np.linspace(min_val, max_val, n_bins + 1)
        labels = list(range(n_bins))
        binned = pd.cut(
            series,
            bins=breakpoints,
            labels=labels,
            include_lowest=True,
        )

    binned = pd.to_numeric(binned, errors="coerce").fillna(-1).astype(int)
    unique_bins = sorted(binned[binned >= 0].unique().tolist())
    return binned, unique_bins


def groups_from_delta(
    delta: NDArray[np.float64],
    tol: float = 1e-8,
) -> NDArray[np.int64]:
    """
    Derive merged group assignments from delta coefficients.

    When δ_i ≈ 0, levels i-1 and i belong to the same merged group.

    Parameters
    ----------
    delta : NDArray[np.float64]
        Delta coefficients of length K.
    tol : float
        Absolute tolerance for treating a delta as zero.

    Returns
    -------
    NDArray[np.int64]
        Integer group labels of length K. Levels in the same group share
        the same integer.
    """
    groups = np.zeros(len(delta), dtype=np.int64)
    current_group = 0
    for i in range(1, len(delta)):
        if abs(delta[i]) > tol:
            current_group += 1
        groups[i] = current_group
    return groups


def exposure_adjusted_target(
    y: NDArray[np.float64],
    exposure: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute exposure-adjusted target and weights for sklearn compatibility.

    Since sklearn does not support GLM offsets, we adjust the target as
    y_adj = y / exposure and use exposure as sample_weight. This gives the
    correct Poisson deviance for frequency models.

    Parameters
    ----------
    y : NDArray[np.float64]
        Observed response (e.g. claim counts).
    exposure : NDArray[np.float64]
        Exposure (e.g. earned years).

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        (y_adjusted, sample_weights)
    """
    exposure = np.asarray(exposure, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    safe_exp = np.where(exposure > 0, exposure, 1.0)
    y_adj = y / safe_exp
    return y_adj, exposure
