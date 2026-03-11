"""
Monotonicity and minimum exposure constraints for merged factor groups.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def enforce_min_exposure(
    groups: NDArray[np.int64],
    exposures: NDArray[np.float64],
    min_exposure: float,
    coefficients: NDArray[np.float64],
) -> NDArray[np.int64]:
    """
    Merge groups whose total exposure falls below the minimum threshold.

    Uses a greedy absorption strategy: find the smallest-exposure group,
    merge it into the neighbouring group (by coefficient proximity), and
    repeat until all groups satisfy the minimum exposure constraint.

    Parameters
    ----------
    groups : NDArray[np.int64]
        Group assignment for each level, length K. Groups are integer codes.
    exposures : NDArray[np.float64]
        Exposure for each level, length K.
    min_exposure : float
        Minimum required exposure per merged group.
    coefficients : NDArray[np.float64]
        Merged group coefficient for each level, length K.

    Returns
    -------
    NDArray[np.int64]
        Updated group assignments after absorption. Group codes are
        renumbered to be contiguous from 0.
    """
    groups = groups.copy().astype(np.int64)
    exposures = np.asarray(exposures, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)

    max_iters = len(np.unique(groups)) + 1
    for _ in range(max_iters):
        unique_groups = np.unique(groups)
        if len(unique_groups) <= 1:
            break

        # Compute per-group exposure
        group_exposure = {
            g: float(exposures[groups == g].sum()) for g in unique_groups
        }
        group_coef = {
            g: float(np.mean(coefficients[groups == g])) for g in unique_groups
        }

        # Find the group with minimum exposure
        min_group = min(group_exposure, key=lambda g: group_exposure[g])
        if group_exposure[min_group] >= min_exposure:
            break  # All groups satisfy the constraint

        # Find nearest neighbour by coefficient value
        min_coef = group_coef[min_group]
        other_groups = [g for g in unique_groups if g != min_group]
        nearest = min(
            other_groups,
            key=lambda g: abs(group_coef[g] - min_coef),
        )

        # Absorb min_group into nearest
        groups[groups == min_group] = nearest

    # Renumber groups to be contiguous from 0
    unique_final = np.unique(groups)
    remap = {old: new for new, old in enumerate(unique_final)}
    groups = np.vectorize(remap.__getitem__)(groups).astype(np.int64)

    return groups


def enforce_min_claims(
    groups: NDArray[np.int64],
    claim_counts: NDArray[np.float64],
    min_claims: int,
    coefficients: NDArray[np.float64],
) -> NDArray[np.int64]:
    """
    Merge groups whose total claim count falls below the minimum threshold.

    Same greedy absorption as enforce_min_exposure, but uses claim counts.

    Parameters
    ----------
    groups : NDArray[np.int64]
        Group assignment for each level.
    claim_counts : NDArray[np.float64]
        Claim count for each level.
    min_claims : int
        Minimum claims per merged group.
    coefficients : NDArray[np.float64]
        Merged group coefficient for each level.

    Returns
    -------
    NDArray[np.int64]
        Updated group assignments, renumbered from 0.
    """
    return enforce_min_exposure(
        groups,
        claim_counts.astype(np.float64),
        float(min_claims),
        coefficients,
    )


def _pav_increasing(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Pool adjacent violators for monotone increasing constraint.

    Pure NumPy implementation that works with any scipy version.

    Parameters
    ----------
    values : NDArray[np.float64]
        Input values to project onto the increasing cone.

    Returns
    -------
    NDArray[np.float64]
        Monotone increasing values of the same length.
    """
    n = len(values)
    result = values.copy()

    # Build blocks iteratively
    i = 0
    blocks: list[list[float]] = []
    block_means: list[float] = []

    for v in result:
        block = [v]
        mean = v
        while block_means and block_means[-1] > mean:
            # Merge with previous block
            prev = blocks.pop()
            block = prev + block
            mean = sum(block) / len(block)
            block_means.pop()
        blocks.append(block)
        block_means.append(mean)

    # Flatten
    idx = 0
    for block, mean in zip(blocks, block_means):
        for _ in block:
            result[idx] = mean
            idx += 1

    return result


def enforce_monotonicity(
    group_coefficients: pd.Series,
    direction: str = "increasing",
) -> pd.Series:
    """
    Project group coefficients onto the monotone cone.

    Uses scipy's isotonic_regression (scipy >= 1.12) when available,
    falling back to a pure NumPy pool adjacent violators implementation
    for older scipy versions.

    Parameters
    ----------
    group_coefficients : pd.Series
        Coefficients indexed by group code (integer). Groups should be in
        the natural ordering (0, 1, 2, ...).
    direction : str
        'increasing' or 'decreasing'.

    Returns
    -------
    pd.Series
        Monotone coefficients with the same index as the input.

    Raises
    ------
    ValueError
        If direction is not 'increasing' or 'decreasing'.
    """
    if direction not in ("increasing", "decreasing"):
        raise ValueError(
            f"direction must be 'increasing' or 'decreasing', got '{direction}'."
        )

    sorted_idx = group_coefficients.sort_index()
    values = sorted_idx.values.astype(np.float64)

    if direction == "decreasing":
        values = -values

    # Try scipy >= 1.12 first
    try:
        from scipy.optimize import isotonic_regression
        result = isotonic_regression(values, increasing=True)
        if hasattr(result, "x"):
            monotone_values = result.x
        else:
            monotone_values = np.asarray(result)
    except ImportError:
        # Fallback: pure NumPy PAV
        monotone_values = _pav_increasing(values)

    if direction == "decreasing":
        monotone_values = -monotone_values

    return pd.Series(monotone_values, index=sorted_idx.index)


def check_monotonicity(
    group_coefficients: pd.Series,
    direction: str = "increasing",
    tol: float = 1e-6,
) -> tuple[bool, list[int]]:
    """
    Check whether group coefficients satisfy a monotonicity constraint.

    Parameters
    ----------
    group_coefficients : pd.Series
        Coefficients indexed by group code.
    direction : str
        'increasing' or 'decreasing'.
    tol : float
        Tolerance for numerical noise.

    Returns
    -------
    bool
        True if monotone.
    list[int]
        Indices of violating pairs (group codes where the constraint is broken).

    Raises
    ------
    ValueError
        If direction is not 'increasing' or 'decreasing'.
    """
    if direction not in ("increasing", "decreasing"):
        raise ValueError(
            f"direction must be 'increasing' or 'decreasing', got '{direction}'."
        )

    sorted_coef = group_coefficients.sort_index()
    values = sorted_coef.values
    indices = sorted_coef.index.tolist()
    violations: list[int] = []

    for i in range(len(values) - 1):
        if direction == "increasing" and values[i] > values[i + 1] + tol:
            violations.append(indices[i])
        elif direction == "decreasing" and values[i] < values[i + 1] - tol:
            violations.append(indices[i])

    return len(violations) == 0, violations
