"""
Tests for utils.py: split-coding helpers, binning, adjacency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_glm_cluster.utils import (
    build_split_coding_matrix,
    apply_split_coding,
    levels_to_onehot,
    delta_to_beta,
    bin_numeric,
    groups_from_delta,
    exposure_adjusted_target,
)


class TestDeltaToBeta:
    def test_cumsum_identity(self):
        delta = np.array([1.0, 0.5, 0.0, -0.3])
        beta = delta_to_beta(delta)
        np.testing.assert_array_almost_equal(beta, np.cumsum(delta))

    def test_single_element(self):
        beta = delta_to_beta(np.array([2.5]))
        np.testing.assert_array_almost_equal(beta, [2.5])

    def test_all_zeros_gives_zeros(self):
        delta = np.zeros(6)
        beta = delta_to_beta(delta)
        np.testing.assert_array_equal(beta, np.zeros(6))


class TestGroupsFromDelta:
    def test_all_nonzero_creates_unique_groups(self):
        delta = np.array([1.0, 0.5, 0.3, 0.1])
        groups = groups_from_delta(delta)
        assert len(np.unique(groups)) == 4

    def test_zero_delta_merges_adjacent(self):
        # Levels 0,1 merged (delta[1]=0), levels 2,3 merged (delta[3]=0)
        delta = np.array([1.0, 0.0, 0.5, 0.0])
        groups = groups_from_delta(delta, tol=1e-8)
        # groups should be [0, 0, 1, 1]
        assert groups[0] == groups[1]
        assert groups[2] == groups[3]
        assert groups[0] != groups[2]

    def test_all_zeros_single_group(self):
        delta = np.array([0.0, 0.0, 0.0])
        groups = groups_from_delta(delta)
        assert len(np.unique(groups)) == 1

    def test_tolerance_respected(self):
        # Small but above-tol delta should NOT merge
        delta = np.array([1.0, 1e-6])
        groups = groups_from_delta(delta, tol=1e-8)
        assert groups[0] != groups[1]

        # Same delta below tol SHOULD merge
        groups_merged = groups_from_delta(delta, tol=1e-5)
        assert groups_merged[0] == groups_merged[1]

    def test_output_dtype(self):
        delta = np.array([1.0, 0.0, 0.5])
        groups = groups_from_delta(delta)
        assert groups.dtype == np.int64


class TestLevelsToOnehot:
    def test_simple_categorical(self):
        series = pd.Series(["A", "B", "A", "C"])
        onehot, cats = levels_to_onehot(series)
        assert cats == ["A", "B", "C"]
        assert onehot.shape == (4, 3)
        assert onehot[0, 0] == 1.0  # A
        assert onehot[1, 1] == 1.0  # B
        assert onehot[3, 2] == 1.0  # C

    def test_row_sums_to_one(self):
        series = pd.Series([0, 1, 2, 1, 0])
        onehot, cats = levels_to_onehot(series)
        np.testing.assert_array_equal(onehot.sum(axis=1), np.ones(5))

    def test_custom_categories(self):
        series = pd.Series(["B", "A"])
        onehot, cats = levels_to_onehot(series, categories=["A", "B", "C"])
        assert cats == ["A", "B", "C"]
        assert onehot.shape == (2, 3)
        assert onehot[0, 1] == 1.0  # B is index 1
        assert onehot[1, 0] == 1.0  # A is index 0

    def test_dtype_float64(self):
        series = pd.Series(["X", "Y"])
        onehot, _ = levels_to_onehot(series)
        assert onehot.dtype == np.float64

    def test_unknown_level_gives_zero_row(self):
        """Level not in categories → zero row (no one-hot bit set)."""
        series = pd.Series(["A", "UNKNOWN", "B"])
        onehot, cats = levels_to_onehot(series, categories=["A", "B"])
        assert cats == ["A", "B"]
        np.testing.assert_array_equal(onehot[1, :], [0.0, 0.0])


class TestBinNumeric:
    def test_fewer_unique_than_bins(self):
        """When fewer unique values than bins, use direct encoding."""
        series = pd.Series([1, 2, 3, 1, 2])
        binned, cats = bin_numeric(series, n_bins=10)
        assert set(binned.unique()) == {0, 1, 2}  # 3 unique values → 3 bins

    def test_quantile_binning(self):
        series = pd.Series(np.arange(100, dtype=float))
        binned, cats = bin_numeric(series, n_bins=10, strategy="quantile")
        assert binned.nunique() <= 10
        assert binned.min() >= 0

    def test_uniform_binning(self):
        series = pd.Series(np.arange(100, dtype=float))
        binned, cats = bin_numeric(series, n_bins=5, strategy="uniform")
        assert binned.nunique() <= 5

    def test_output_length_matches_input(self):
        series = pd.Series(np.random.default_rng(0).normal(size=200))
        binned, cats = bin_numeric(series, n_bins=20)
        assert len(binned) == 200

    def test_cats_are_sorted_integers(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        binned, cats = bin_numeric(series, n_bins=3)
        assert cats == sorted(cats)


class TestExposureAdjustedTarget:
    def test_rate_equals_y_over_exposure(self):
        y = np.array([2.0, 4.0, 1.0])
        exp = np.array([2.0, 4.0, 1.0])
        y_adj, sw = exposure_adjusted_target(y, exp)
        np.testing.assert_array_almost_equal(y_adj, np.ones(3))

    def test_weights_equal_exposure(self):
        y = np.array([1.0, 2.0])
        exp = np.array([0.5, 1.5])
        _, sw = exposure_adjusted_target(y, exp)
        np.testing.assert_array_almost_equal(sw, exp)

    def test_zero_exposure_handled(self):
        """Zero exposure should not cause division by zero."""
        y = np.array([0.0, 1.0])
        exp = np.array([0.0, 1.0])
        y_adj, sw = exposure_adjusted_target(y, exp)
        assert np.isfinite(y_adj).all()

    def test_output_lengths(self):
        y = np.arange(5, dtype=float)
        exp = np.ones(5)
        y_adj, sw = exposure_adjusted_target(y, exp)
        assert len(y_adj) == 5
        assert len(sw) == 5
