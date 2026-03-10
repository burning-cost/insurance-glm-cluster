"""
Tests for constraints.py: min_exposure, min_claims, and monotonicity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_glm_cluster.constraints import (
    enforce_min_exposure,
    enforce_min_claims,
    enforce_monotonicity,
    check_monotonicity,
)


class TestEnforceMinExposure:
    def test_no_merging_needed(self):
        """All groups satisfy min_exposure — no merging should occur."""
        groups = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        exposures = np.array([10.0, 10.0, 15.0, 15.0, 20.0, 20.0])
        coef = np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9])
        result = enforce_min_exposure(groups, exposures, min_exposure=5.0, coefficients=coef)
        assert len(np.unique(result)) == 3

    def test_small_group_absorbed(self):
        """Group with exposure=1 should be absorbed into a neighbour."""
        groups = np.array([0, 1, 1, 2, 2], dtype=np.int64)
        exposures = np.array([1.0, 20.0, 20.0, 25.0, 25.0])
        coef = np.array([0.0, 0.5, 0.5, 0.9, 0.9])
        result = enforce_min_exposure(groups, exposures, min_exposure=5.0, coefficients=coef)
        assert len(np.unique(result)) == 2

    def test_output_renumbered_from_zero(self):
        """Result group codes should be contiguous from 0."""
        groups = np.array([0, 0, 3, 3, 7, 7], dtype=np.int64)
        exposures = np.array([10.0, 10.0, 12.0, 12.0, 15.0, 15.0])
        coef = np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9])
        result = enforce_min_exposure(groups, exposures, min_exposure=5.0, coefficients=coef)
        unique = np.unique(result)
        assert unique[0] == 0
        np.testing.assert_array_equal(unique, np.arange(len(unique)))

    def test_single_group_never_absorbed(self):
        """If only one group remains, stop — don't create empty groups."""
        groups = np.array([0, 1], dtype=np.int64)
        exposures = np.array([1.0, 1.0])
        coef = np.array([0.0, 1.0])
        result = enforce_min_exposure(groups, exposures, min_exposure=100.0, coefficients=coef)
        # After absorption, only 1 group can remain at most
        assert len(np.unique(result)) == 1

    def test_absorbed_into_nearest_by_coefficient(self):
        """Small group at coefficient 0.4 should merge with group at 0.5 (not 0.0)."""
        # Group 0: coef=0.0, exposure=50
        # Group 1: coef=0.4, exposure=2 (small — should merge with group 2)
        # Group 2: coef=0.5, exposure=50
        groups = np.array([0, 0, 1, 2, 2], dtype=np.int64)
        exposures = np.array([25.0, 25.0, 2.0, 25.0, 25.0])
        coef = np.array([0.0, 0.0, 0.4, 0.5, 0.5])
        result = enforce_min_exposure(groups, exposures, min_exposure=5.0, coefficients=coef)
        # The merged group for level 2 should equal levels 3 or 4 (group 2 by coef proximity)
        assert result[2] == result[3] or result[2] == result[4]

    def test_output_dtype(self):
        groups = np.array([0, 0, 1, 1], dtype=np.int64)
        exposures = np.array([10.0, 10.0, 10.0, 10.0])
        coef = np.array([0.0, 0.0, 0.5, 0.5])
        result = enforce_min_exposure(groups, exposures, min_exposure=1.0, coefficients=coef)
        assert result.dtype == np.int64


class TestEnforceMinClaims:
    def test_same_logic_as_exposure(self):
        """min_claims should behave identically to min_exposure with counts."""
        groups = np.array([0, 0, 1, 1], dtype=np.int64)
        claims = np.array([2.0, 3.0, 100.0, 100.0])
        coef = np.array([0.1, 0.1, 0.5, 0.5])
        result = enforce_min_claims(groups, claims, min_claims=10, coefficients=coef)
        # Group 0 has 5 claims, below min of 10 → should be absorbed
        assert len(np.unique(result)) == 1

    def test_sufficient_claims_no_change(self):
        groups = np.array([0, 0, 1, 1], dtype=np.int64)
        claims = np.array([15.0, 15.0, 20.0, 20.0])
        coef = np.array([0.1, 0.1, 0.5, 0.5])
        result = enforce_min_claims(groups, claims, min_claims=10, coefficients=coef)
        assert len(np.unique(result)) == 2


class TestEnforceMonotonicity:
    def test_already_increasing(self):
        coef = pd.Series({0: 0.1, 1: 0.3, 2: 0.7})
        result = enforce_monotonicity(coef, direction="increasing")
        np.testing.assert_array_almost_equal(
            result.sort_index().values, coef.sort_index().values
        )

    def test_already_decreasing(self):
        coef = pd.Series({0: 0.9, 1: 0.4, 2: 0.1})
        result = enforce_monotonicity(coef, direction="decreasing")
        # Should be unchanged
        np.testing.assert_array_almost_equal(
            result.sort_index().values, coef.sort_index().values
        )

    def test_violation_corrected_increasing(self):
        """Non-monotone input should be projected onto increasing cone."""
        coef = pd.Series({0: 0.1, 1: 0.9, 2: 0.3})  # 0.9 then 0.3 violates
        result = enforce_monotonicity(coef, direction="increasing")
        vals = result.sort_index().values
        assert np.all(np.diff(vals) >= -1e-10), f"Not increasing: {vals}"

    def test_violation_corrected_decreasing(self):
        coef = pd.Series({0: 0.9, 1: 0.1, 2: 0.5})  # 0.1 then 0.5 violates
        result = enforce_monotonicity(coef, direction="decreasing")
        vals = result.sort_index().values
        assert np.all(np.diff(vals) <= 1e-10), f"Not decreasing: {vals}"

    def test_invalid_direction_raises(self):
        coef = pd.Series({0: 0.1, 1: 0.2})
        with pytest.raises(ValueError, match="direction must be"):
            enforce_monotonicity(coef, direction="sideways")

    def test_index_preserved(self):
        coef = pd.Series({5: 0.3, 10: 0.1, 15: 0.8})
        result = enforce_monotonicity(coef, direction="increasing")
        assert set(result.index) == {5, 10, 15}

    def test_pav_gives_optimal_result(self):
        """
        For [3, 1, 2] increasing, the PAV solution should be [2, 2, 2].
        The pool adjacent violators algorithm merges 3 and 1 (mean=2),
        then checks 2 and 2 — no violation.
        """
        coef = pd.Series({0: 3.0, 1: 1.0, 2: 2.0})
        result = enforce_monotonicity(coef, direction="increasing")
        vals = result.sort_index().values
        np.testing.assert_array_almost_equal(vals, [2.0, 2.0, 2.0])


class TestCheckMonotonicity:
    def test_increasing_satisfied(self):
        coef = pd.Series({0: 0.1, 1: 0.4, 2: 0.7})
        ok, violations = check_monotonicity(coef, "increasing")
        assert ok
        assert violations == []

    def test_increasing_violated(self):
        coef = pd.Series({0: 0.1, 1: 0.9, 2: 0.3})
        ok, violations = check_monotonicity(coef, "increasing")
        assert not ok
        assert len(violations) > 0

    def test_decreasing_satisfied(self):
        coef = pd.Series({0: 0.9, 1: 0.4, 2: 0.1})
        ok, violations = check_monotonicity(coef, "decreasing")
        assert ok
        assert violations == []

    def test_decreasing_violated(self):
        coef = pd.Series({0: 0.9, 1: 0.1, 2: 0.5})
        ok, violations = check_monotonicity(coef, "decreasing")
        assert not ok

    def test_invalid_direction_raises(self):
        coef = pd.Series({0: 0.1})
        with pytest.raises(ValueError):
            check_monotonicity(coef, "upward")
