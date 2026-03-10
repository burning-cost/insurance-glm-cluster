"""
Tests for penalties.py: R2VF split-coding and fused lasso construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_glm_cluster.penalties import (
    R2VFRanker,
    build_r2vf_design_matrix,
    fit_fused_lasso,
    lambda_grid,
)
from insurance_glm_cluster.utils import (
    build_split_coding_matrix,
    apply_split_coding,
    levels_to_onehot,
)


# ---------------------------------------------------------------------------
# Split-coding matrix
# ---------------------------------------------------------------------------

class TestBuildSplitCodingMatrix:
    def test_shape(self):
        T = build_split_coding_matrix(4)
        assert T.shape == (4, 4)

    def test_lower_triangular_ones(self):
        T = build_split_coding_matrix(3)
        expected = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=float)
        np.testing.assert_array_equal(T, expected)

    def test_single_level(self):
        T = build_split_coding_matrix(1)
        np.testing.assert_array_equal(T, np.array([[1.0]]))

    def test_dtype_float64(self):
        T = build_split_coding_matrix(5)
        assert T.dtype == np.float64


class TestApplySplitCoding:
    def test_identity_for_single_level(self):
        """Single-level factor: split-coding should be identity."""
        X = np.array([[1.0], [0.0], [1.0]])
        X_delta = apply_split_coding(X)
        np.testing.assert_array_almost_equal(X_delta, X)

    def test_three_levels_correctness(self):
        """
        For three one-hot columns [x0, x1, x2]:
          X_delta[:, 0] = x0 + x1 + x2
          X_delta[:, 1] = x1 + x2
          X_delta[:, 2] = x2
        """
        X = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=float)
        X_delta = apply_split_coding(X)
        expected = np.array([
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ], dtype=float)
        np.testing.assert_array_equal(X_delta, expected)

    def test_reconstruction_property(self):
        """
        δ_i = β_i - β_{i-1}. If we set δ = [1, 0.5, 0], then
        β = [1, 1.5, 1.5]. The split-coded design should reproduce this.
        """
        # One-hot for 3 levels, 6 observations (2 per level)
        X = np.zeros((6, 3))
        X[:2, 0] = 1
        X[2:4, 1] = 1
        X[4:, 2] = 1

        delta = np.array([1.0, 0.5, 0.0])
        X_delta = apply_split_coding(X)
        fitted = X_delta @ delta  # should be β for each observation

        expected_beta = [1.0, 1.0, 1.5, 1.5, 1.5, 1.5]
        np.testing.assert_array_almost_equal(fitted, expected_beta)

    def test_output_shape(self):
        X = np.eye(5)
        X_delta = apply_split_coding(X)
        assert X_delta.shape == (5, 5)


# ---------------------------------------------------------------------------
# R2VFRanker
# ---------------------------------------------------------------------------

class TestR2VFRanker:
    def test_ranking_separates_groups(self, nominal_dataset):
        """
        R2VF ranker should assign higher coefficients to high-risk levels
        (H, I, J) and lower coefficients to low-risk levels (A, B, C).
        After ranking (argsort ascending), all low-risk should appear before
        all high-risk.
        """
        ds = nominal_dataset
        X = ds["X"]
        y = ds["y"]
        exposure = ds["exposure"]
        makes = ds["makes"]

        from insurance_glm_cluster.utils import exposure_adjusted_target

        y_adj, sw = exposure_adjusted_target(y, exposure)

        # Build one-hot
        onehot, cats = levels_to_onehot(X["vehicle_make"], makes)
        step1_slices = {"vehicle_make": slice(0, len(makes))}

        ranker = R2VFRanker(alpha=2.0)
        ranker.fit(onehot, y_adj, step1_slices, sample_weight=sw)

        ranking = ranker.get_ranking("vehicle_make")
        coef = ranker.get_coefficients("vehicle_make")

        # Coefficient for low-risk levels should be lower than high-risk
        low_risk_idx = [makes.index(m) for m in "ABC"]
        high_risk_idx = [makes.index(m) for m in "HIJ"]

        avg_low = np.mean(coef[low_risk_idx])
        avg_high = np.mean(coef[high_risk_idx])
        assert avg_low < avg_high, (
            f"Low-risk avg coef {avg_low:.3f} should be < high-risk {avg_high:.3f}"
        )

    def test_ranking_length(self, nominal_dataset):
        ds = nominal_dataset
        makes = ds["makes"]
        onehot, _ = levels_to_onehot(ds["X"]["vehicle_make"], makes)
        ranker = R2VFRanker(alpha=2.0)
        ranker.fit(
            onehot, ds["y"] / ds["exposure"],
            {"vehicle_make": slice(0, len(makes))},
            sample_weight=ds["exposure"],
        )
        assert len(ranker.get_ranking("vehicle_make")) == len(makes)

    def test_ranking_is_permutation(self, nominal_dataset):
        ds = nominal_dataset
        makes = ds["makes"]
        onehot, _ = levels_to_onehot(ds["X"]["vehicle_make"], makes)
        ranker = R2VFRanker(alpha=2.0)
        ranker.fit(
            onehot, ds["y"] / ds["exposure"],
            {"vehicle_make": slice(0, len(makes))},
            sample_weight=ds["exposure"],
        )
        ranking = ranker.get_ranking("vehicle_make")
        assert sorted(ranking) == list(range(len(makes)))

    def test_missing_factor_raises(self, nominal_dataset):
        ds = nominal_dataset
        makes = ds["makes"]
        onehot, _ = levels_to_onehot(ds["X"]["vehicle_make"], makes)
        ranker = R2VFRanker()
        ranker.fit(onehot, ds["y"] / ds["exposure"], {"vehicle_make": slice(0, len(makes))})
        with pytest.raises(ValueError, match="not in fitted rankings"):
            ranker.get_ranking("unknown_factor")


# ---------------------------------------------------------------------------
# build_r2vf_design_matrix
# ---------------------------------------------------------------------------

class TestBuildR2VFDesignMatrix:
    def test_output_shape_ordinal_only(self):
        """Design matrix should have one column per level."""
        onehot = np.eye(4, dtype=float)  # 4 levels, 4 observations
        X_delta, slices, reordered = build_r2vf_design_matrix(
            {"age": onehot},
            {"age": [0, 1, 2, 3]},
            nominal_rankings=None,
        )
        assert X_delta.shape == (4, 4)
        assert "age" in slices
        assert slices["age"] == slice(0, 4)

    def test_nominal_reordering_applied(self):
        """For nominals, columns should be reordered by the ranking."""
        # 3 levels, ranking reverses order
        onehot = np.eye(3, dtype=float)
        X_delta, slices, reordered = build_r2vf_design_matrix(
            {"make": onehot},
            {"make": ["A", "B", "C"]},
            nominal_rankings={"make": [2, 1, 0]},  # reverse
        )
        # Reordered categories should be reversed
        assert reordered["make"] == ["C", "B", "A"]

    def test_multi_factor_slices(self):
        """Slices should be non-overlapping and cover all columns."""
        oh1 = np.eye(3, dtype=float)
        oh2 = np.eye(4, dtype=float)[:3, :]  # 3 obs, 4 levels
        X_delta, slices, _ = build_r2vf_design_matrix(
            {"age": oh1, "ncd": oh2},
            {"age": [0, 1, 2], "ncd": [0, 1, 2, 3]},
        )
        assert X_delta.shape[1] == 7
        assert slices["age"] == slice(0, 3)
        assert slices["ncd"] == slice(3, 7)


# ---------------------------------------------------------------------------
# fit_fused_lasso
# ---------------------------------------------------------------------------

class TestFitFusedLasso:
    def test_high_lambda_zeroes_all(self):
        """Very large lambda should zero all delta coefficients."""
        X = np.random.default_rng(0).normal(size=(100, 5))
        y = np.random.default_rng(0).normal(size=100)
        delta = fit_fused_lasso(X, y, lam=1e6)
        np.testing.assert_array_almost_equal(delta, np.zeros(5), decimal=3)

    def test_zero_lambda_fits_well(self):
        """Very small lambda should give near-OLS fit."""
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 3))
        beta_true = np.array([1.0, -0.5, 0.2])
        y = X @ beta_true + rng.normal(scale=0.01, size=200)
        delta = fit_fused_lasso(X, y, lam=1e-8)
        # Should recover approximately: not exact because Lasso intercept
        np.testing.assert_array_almost_equal(delta, beta_true, decimal=1)

    def test_output_length(self):
        X = np.eye(5)
        y = np.ones(5)
        delta = fit_fused_lasso(X, y, lam=0.1)
        assert len(delta) == 5

    def test_sample_weight_accepted(self):
        """Should not raise when sample_weight is provided."""
        X = np.random.default_rng(2).normal(size=(50, 3))
        y = np.random.default_rng(2).exponential(size=50)
        w = np.ones(50) * 2.0
        delta = fit_fused_lasso(X, y, lam=0.1, sample_weight=w)
        assert delta.shape == (3,)


# ---------------------------------------------------------------------------
# lambda_grid
# ---------------------------------------------------------------------------

class TestLambdaGrid:
    def test_grid_is_decreasing(self):
        X = np.random.default_rng(3).normal(size=(100, 5))
        y = np.random.default_rng(3).normal(size=100)
        grid = lambda_grid(X, y, n_points=20)
        assert np.all(np.diff(grid) < 0), "Lambda grid should be decreasing"

    def test_grid_length(self):
        X = np.random.default_rng(4).normal(size=(50, 3))
        y = np.random.default_rng(4).normal(size=50)
        grid = lambda_grid(X, y, n_points=30)
        assert len(grid) == 30

    def test_all_positive(self):
        X = np.random.default_rng(5).normal(size=(80, 4))
        y = np.random.default_rng(5).normal(size=80)
        grid = lambda_grid(X, y, n_points=10)
        assert np.all(grid > 0)
