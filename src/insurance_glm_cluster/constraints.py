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


def enforce_monotonicity(
    group_coefficients: pd.Series,
    direction: str = "increasing",
) -> pd.Series:
    """
    Project group coefficients onto the monotone cone.

    Uses the pool adjacent violators (PAV) algorithm via scipy's
    isotonic_regression (requires scipy >= 1.12).

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
    ImportError
        If scipy < 1.12 (isotonic_regression not available).
    ValueError
        If direction is not 'increasing' or 'decreasing'.
    """
    try:
        from scipy.optimize import isotonic_regression
    except ImportError as exc:
        raise ImportError(
            "scipy >= 1.12 is required for monotonicity enforcement. "
            "Current scipy version does not have isotonic_regression."
        ) from exc

    if direction not in ("increasing", "decreasing"):
        raise ValueError(
            f"direction must be 'increasing' or 'decreasing', got '{direction}'."
        )

    sorted_idx = group_coefficients.sort_index()
    values = sorted_idx.values.astype(np.float64)

    if direction == "increasing":
        result = isotonic_regression(values, increasing=True)
    else:
        result = isotonic_regression(values, increasing=False)

    # isotonic_regression returns a namedtuple-like; extract .x in scipy >= 1.12
    if hasattr(result, "x"):
        monotone_values = result.x
    else:
        monotone_values = np.asarray(result)

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
    """
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
