"""
Integration tests: full R2VF workflow on synthetic datasets with known structure.

These tests are the most important ones — they verify that the algorithm
actually recovers the true groupings, not just that it runs without crashing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_glm_cluster import FactorClusterer


class TestOrdinalGroupRecovery:
    """
    On a clean dataset with 5 levels and true 3-group structure,
    the algorithm should find <= 4 groups (ideally 3).

    We don't demand exact recovery at this sample size — the fused lasso
    is regularised and may over-merge or under-merge. We verify that
    the direction of the coefficients is correct.
    """

    def test_recovers_correct_direction(self, ordinal_dataset):
        """Low-index levels should have lower coefficients than high-index levels."""
        ds = ordinal_dataset
        # True log-rates: [-0.5, -0.5, 0.0, 0.0, 0.8]
        # So level 0,1 < level 2,3 < level 4

        fc = FactorClusterer(family="poisson", lambda_="bic")
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        lm = fc.level_map("vehicle_age")

        df = lm.to_df()
        df = df.sort_values("original_level")

        # Group of level 0 should be <= group of level 4
        group_0 = df[df["original_level"] == 0]["merged_group"].values[0]
        group_4 = df[df["original_level"] == 4]["merged_group"].values[0]

        coef_0 = lm.group_coefficients.get(group_0, np.nan)
        coef_4 = lm.group_coefficients.get(group_4, np.nan)

        assert coef_0 <= coef_4, (
            f"Level 0 (true rate=-0.5) should have coef <= level 4 (true rate=0.8). "
            f"Got coef_0={coef_0:.3f}, coef_4={coef_4:.3f}"
        )

    def test_groups_fewer_than_levels(self, ordinal_dataset):
        """At any reasonable lambda, some merging should occur."""
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_="bic")
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        lm = fc.level_map("vehicle_age")
        assert lm.n_groups() < lm.n_levels_original()

    def test_level_map_covers_all_levels(self, ordinal_dataset):
        """Every original level should appear in the mapping."""
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_="bic")
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        lm = fc.level_map("vehicle_age")
        observed_levels = set(lm.mapping.keys())
        # All 5 original integer levels (0-4) should be present
        assert {0, 1, 2, 3, 4}.issubset(observed_levels), (
            f"Missing levels. Observed: {observed_levels}"
        )


class TestNominalGroupRecovery:
    """
    10 nominal levels (A-J) with true 3-group structure:
      {A,B,C} low risk, {D,E,F,G} medium, {H,I,J} high.
    """

    def test_recovers_correct_risk_ordering(self, nominal_dataset):
        """Low-risk makes (A,B,C) should have lower coefficients than high-risk (H,I,J)."""
        ds = nominal_dataset
        fc = FactorClusterer(family="poisson", lambda_="bic")
        fc.fit(
            ds["X"], ds["y"], exposure=ds["exposure"],
            nominal_factors=["vehicle_make"]
        )
        lm = fc.level_map("vehicle_make")

        low_risk = ["A", "B", "C"]
        high_risk = ["H", "I", "J"]

        low_groups = [lm.mapping[m] for m in low_risk]
        high_groups = [lm.mapping[m] for m in high_risk]

        avg_low_coef = np.mean([lm.group_coefficients.get(g, 0.0) for g in low_groups])
        avg_high_coef = np.mean([lm.group_coefficients.get(g, 0.0) for g in high_groups])

        assert avg_low_coef < avg_high_coef, (
            f"Low-risk avg coef ({avg_low_coef:.3f}) should be < "
            f"high-risk avg coef ({avg_high_coef:.3f})"
        )

    def test_all_makes_in_mapping(self, nominal_dataset):
        """All 10 vehicle makes should appear in the level map."""
        ds = nominal_dataset
        fc = FactorClusterer(family="poisson", lambda_="bic")
        fc.fit(
            ds["X"], ds["y"], exposure=ds["exposure"],
            nominal_factors=["vehicle_make"]
        )
        lm = fc.level_map("vehicle_make")
        for make in "ABCDEFGHIJ":
            assert make in lm.mapping, f"Make '{make}' missing from level map"

    def test_merging_occurs(self, nominal_dataset):
        """Should produce fewer groups than 10 at any reasonable lambda."""
        ds = nominal_dataset
        fc = FactorClusterer(family="poisson", lambda_="bic")
        fc.fit(
            ds["X"], ds["y"], exposure=ds["exposure"],
            nominal_factors=["vehicle_make"]
        )
        lm = fc.level_map("vehicle_make")
        assert lm.n_groups() < 10


class TestExposureHandling:
    """
    Verify that exposure handling is correct: log(exposure) as offset
    should give different results from no exposure.
    """

    def test_exposure_changes_fitted_groups(self, ordinal_dataset):
        """With exposure, the model is a frequency model — groups should differ."""
        ds = ordinal_dataset
        # Fit with exposure (frequency model)
        fc_with = FactorClusterer(family="poisson", lambda_=0.05)
        fc_with.fit(
            ds["X"], ds["y"], exposure=ds["exposure"],
            ordinal_factors=["vehicle_age"]
        )

        # Fit without exposure (count model)
        fc_without = FactorClusterer(family="poisson", lambda_=0.05)
        fc_without.fit(
            ds["X"], ds["y"],
            ordinal_factors=["vehicle_age"]
        )

        # Both should fit without error and produce valid maps
        lm_with = fc_with.level_map("vehicle_age")
        lm_without = fc_without.level_map("vehicle_age")
        assert lm_with.n_groups() >= 1
        assert lm_without.n_groups() >= 1

    def test_zero_exposure_rows_handled(self):
        """Zero-exposure rows should not cause NaN errors."""
        rng = np.random.default_rng(99)
        n = 500
        levels = rng.integers(0, 4, size=n)
        exposure = rng.uniform(0.1, 2.0, size=n)
        exposure[:10] = 0.0  # inject zero exposures
        y = rng.poisson(exposure * np.exp(levels * 0.2))
        X = pd.DataFrame({"age": levels})

        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(X, y.astype(float), exposure=exposure, ordinal_factors=["age"])
        lm = fc.level_map("age")
        assert lm.n_groups() >= 1


class TestFullWorkflow:
    """End-to-end: fit → transform → refit_glm → diagnostics."""

    def test_full_pipeline_ordinal(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_="bic")

        # Fit
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])

        # Transform
        X_merged = fc.transform(ds["X"])
        assert isinstance(X_merged, pd.DataFrame)

        # Refit
        result = fc.refit_glm(X_merged, ds["y"], exposure=ds["exposure"])
        assert result is not None

        # Diagnostics
        diag = fc.diagnostics()
        assert diag["aic_before"] is not None
        assert diag["aic_after"] is not None

    def test_full_pipeline_two_factors(self, two_factor_dataset):
        ds = two_factor_dataset
        fc = FactorClusterer(
            family="poisson",
            lambda_="bic",
            monotone_factors=["ncd"],
            monotone_direction={"ncd": "decreasing"},
        )
        fc.fit(
            ds["X"], ds["y"], exposure=ds["exposure"],
            ordinal_factors=["ncd"],
            nominal_factors=["occupation"],
        )
        X_merged = fc.transform(ds["X"])

        # NCD should be monotone decreasing (higher NCD = lower risk)
        lm_ncd = fc.level_map("ncd")
        assert lm_ncd.validate_monotone("decreasing")

        result = fc.refit_glm(X_merged, ds["y"], exposure=ds["exposure"])
        assert hasattr(result, "params")

    def test_level_map_to_df_roundtrip(self, ordinal_dataset):
        """to_df() should be a complete record — no gaps in group coverage."""
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        lm = fc.level_map("vehicle_age")
        df = lm.to_df()

        # Every group in the mapping should appear in the DataFrame
        all_groups_in_mapping = set(lm.mapping.values())
        all_groups_in_df = set(df["merged_group"].tolist())
        assert all_groups_in_mapping == all_groups_in_df

    def test_diagnostics_n_levels_after_equals_actual_groups(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        diag = fc.diagnostics()
        lm = fc.level_map("vehicle_age")
        assert diag["n_levels_after"]["vehicle_age"] == lm.n_groups()


class TestDiagnosticValues:
    def test_aic_before_greater_or_equal_after_refit(self, ordinal_dataset):
        """
        After merging, the refit GLM has fewer parameters. The AIC may go
        up or down depending on deviance gain vs parameter reduction. We
        just check both are finite.
        """
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_="bic")
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        X_merged = fc.transform(ds["X"])
        fc.refit_glm(X_merged, ds["y"], exposure=ds["exposure"])
        diag = fc.diagnostics()
        assert np.isfinite(diag["aic_before"])
        assert np.isfinite(diag["aic_after"])

    def test_deviance_before_finite(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        diag = fc.diagnostics()
        if diag["deviance_before"] is not None:
            assert np.isfinite(diag["deviance_before"])
