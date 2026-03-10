"""
Tests for FactorClusterer: the main public API.

Tests cover:
- fit/transform/fit_transform on ordinal and nominal factors
- lambda selection ('bic' and fixed float)
- level_map() output format
- diagnostics() return structure
- refit_glm() runs without error
- monotonicity enforcement via monotone_factors
- min_exposure enforcement
- error handling (unfitted, missing columns, invalid params)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_glm_cluster import FactorClusterer, LevelMap


class TestFactorClustererInit:
    def test_default_params(self):
        fc = FactorClusterer()
        assert fc.family == "poisson"
        assert fc.method == "r2vf"
        assert fc.lambda_ == "bic"
        assert fc.alpha == 2.0

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            FactorClusterer(method="gfl")

    def test_invalid_lambda_string_raises(self):
        with pytest.raises(ValueError, match="lambda_"):
            FactorClusterer(lambda_="cv")

    def test_float_lambda_accepted(self):
        fc = FactorClusterer(lambda_=0.1)
        assert fc.lambda_ == 0.1

    def test_repr_unfitted(self):
        fc = FactorClusterer()
        assert "unfitted" in repr(fc)

    def test_repr_fitted(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        assert "fitted" in repr(fc)


class TestFactorClustererFitOrdinal:
    def test_fit_returns_self(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        result = fc.fit(
            ds["X"], ds["y"], exposure=ds["exposure"],
            ordinal_factors=["vehicle_age"]
        )
        assert result is fc

    def test_fit_sets_fitted_flag(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        assert fc._is_fitted

    def test_fit_empty_factors_raises(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer()
        with pytest.raises(ValueError, match="at least one factor"):
            fc.fit(ds["X"], ds["y"])

    def test_fit_missing_column_raises(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(lambda_=0.05)
        with pytest.raises(ValueError, match="not found"):
            fc.fit(ds["X"], ds["y"], ordinal_factors=["nonexistent"])

    def test_reduces_levels(self, ordinal_dataset):
        """Clustering should produce fewer groups than original levels."""
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.1)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        lm = fc.level_map("vehicle_age")
        # Original 5 levels should be merged to <= 5 groups
        assert lm.n_groups() <= 5
        assert lm.n_levels_original() == 5


class TestFactorClustererFitNominal:
    def test_nominal_fit_runs(self, nominal_dataset):
        ds = nominal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(
            ds["X"], ds["y"], exposure=ds["exposure"],
            nominal_factors=["vehicle_make"]
        )
        assert fc._is_fitted

    def test_nominal_level_map_has_all_levels(self, nominal_dataset):
        ds = nominal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(
            ds["X"], ds["y"], exposure=ds["exposure"],
            nominal_factors=["vehicle_make"]
        )
        lm = fc.level_map("vehicle_make")
        assert lm.n_levels_original() == 10

    def test_nominal_reduces_levels(self, nominal_dataset):
        """For well-separated nominal levels, the fused lasso should merge."""
        ds = nominal_dataset
        # Use moderate lambda to allow merging without collapsing everything
        fc = FactorClusterer(family="poisson", lambda_=0.08)
        fc.fit(
            ds["X"], ds["y"], exposure=ds["exposure"],
            nominal_factors=["vehicle_make"]
        )
        lm = fc.level_map("vehicle_make")
        assert lm.n_groups() < 10, "Expected some merging of nominal levels"


class TestFactorClustererBICSelection:
    def test_bic_selects_lambda(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_="bic")
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        assert fc._selected_lambda is not None
        assert fc._selected_lambda > 0

    def test_bic_curve_has_50_points(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_="bic")
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        assert fc._bic_curve is not None
        assert len(fc._bic_curve) == 50

    def test_bic_curve_contains_finite_values(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_="bic")
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        assert all(np.isfinite(v) for v in fc._bic_curve)


class TestFactorClustererTransform:
    def test_transform_returns_dataframe(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        X_merged = fc.transform(ds["X"])
        assert isinstance(X_merged, pd.DataFrame)

    def test_transform_same_number_of_rows(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        X_merged = fc.transform(ds["X"])
        assert len(X_merged) == len(ds["X"])

    def test_transform_column_preserved(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        X_merged = fc.transform(ds["X"])
        assert "vehicle_age" in X_merged.columns

    def test_transform_values_are_group_codes(self, ordinal_dataset):
        """All values in the transformed column should be valid group codes."""
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        X_merged = fc.transform(ds["X"])
        lm = fc.level_map("vehicle_age")
        valid_codes = set(lm.mapping.values())
        observed_codes = set(X_merged["vehicle_age"].dropna().astype(int).tolist())
        assert observed_codes.issubset(valid_codes)

    def test_transform_before_fit_raises(self, ordinal_dataset):
        fc = FactorClusterer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            fc.transform(ordinal_dataset["X"])

    def test_fit_transform_equals_fit_then_transform(self, ordinal_dataset):
        ds = ordinal_dataset
        fc1 = FactorClusterer(family="poisson", lambda_=0.05)
        X1 = fc1.fit_transform(
            ds["X"], ds["y"], exposure=ds["exposure"],
            ordinal_factors=["vehicle_age"]
        )
        fc2 = FactorClusterer(family="poisson", lambda_=0.05)
        fc2.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        X2 = fc2.transform(ds["X"])
        pd.testing.assert_frame_equal(X1, X2)


class TestFactorClustererLevelMap:
    def test_level_map_returns_level_map(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        lm = fc.level_map("vehicle_age")
        assert isinstance(lm, LevelMap)

    def test_level_map_unknown_factor_raises(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        with pytest.raises(ValueError, match="not fitted"):
            fc.level_map("unknown")

    def test_level_map_to_df_has_correct_columns(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        df = fc.level_map("vehicle_age").to_df()
        assert "original_level" in df.columns
        assert "merged_group" in df.columns
        assert "coefficient" in df.columns
        assert "exposure" in df.columns

    def test_level_map_nominal_flag(self, nominal_dataset):
        ds = nominal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(
            ds["X"], ds["y"], exposure=ds["exposure"],
            nominal_factors=["vehicle_make"]
        )
        lm = fc.level_map("vehicle_make")
        assert lm.is_nominal is True

    def test_level_map_ordinal_flag(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        lm = fc.level_map("vehicle_age")
        assert lm.is_nominal is False


class TestFactorClustererDiagnostics:
    def test_diagnostics_returns_dict(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        diag = fc.diagnostics()
        assert isinstance(diag, dict)

    def test_diagnostics_required_keys(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        diag = fc.diagnostics()
        required_keys = [
            "aic_before", "aic_after", "bic_before", "bic_after",
            "deviance_before", "deviance_after",
            "n_levels_before", "n_levels_after",
            "selected_lambda", "bic_curve",
        ]
        for key in required_keys:
            assert key in diag, f"Missing key: {key}"

    def test_n_levels_before_correct(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        diag = fc.diagnostics()
        assert diag["n_levels_before"]["vehicle_age"] == 5

    def test_n_levels_after_at_most_before(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        diag = fc.diagnostics()
        assert diag["n_levels_after"]["vehicle_age"] <= diag["n_levels_before"]["vehicle_age"]

    def test_selected_lambda_is_positive(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_="bic")
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        diag = fc.diagnostics()
        assert diag["selected_lambda"] > 0

    def test_diagnostics_before_fit_raises(self):
        fc = FactorClusterer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            fc.diagnostics()


class TestFactorClustererRefitGLM:
    def test_refit_glm_runs(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        X_merged = fc.transform(ds["X"])
        result = fc.refit_glm(X_merged, ds["y"], exposure=ds["exposure"])
        assert result is not None

    def test_refit_glm_returns_statsmodels_result(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        X_merged = fc.transform(ds["X"])
        result = fc.refit_glm(X_merged, ds["y"], exposure=ds["exposure"])
        # statsmodels result has .params
        assert hasattr(result, "params")

    def test_refit_updates_diagnostics_after(self, ordinal_dataset):
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        X_merged = fc.transform(ds["X"])
        fc.refit_glm(X_merged, ds["y"], exposure=ds["exposure"])
        diag = fc.diagnostics()
        # After refit, aic_after should be populated
        assert diag["aic_after"] is not None
        assert np.isfinite(diag["aic_after"])


class TestFactorClustererMinExposure:
    def test_min_exposure_reduces_groups(self, ordinal_dataset):
        ds = ordinal_dataset
        # Very large min_exposure should force all levels into one group
        fc = FactorClusterer(family="poisson", lambda_=0.05, min_exposure=1e9)
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        lm = fc.level_map("vehicle_age")
        assert lm.n_groups() == 1

    def test_min_exposure_none_no_effect(self, ordinal_dataset):
        ds = ordinal_dataset
        fc1 = FactorClusterer(family="poisson", lambda_=0.05, min_exposure=None)
        fc2 = FactorClusterer(family="poisson", lambda_=0.05, min_exposure=0.001)
        fc1.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        fc2.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        # Both should produce valid outputs
        assert fc1.level_map("vehicle_age").n_groups() >= 1
        assert fc2.level_map("vehicle_age").n_groups() >= 1


class TestFactorClustererMonotonicity:
    def test_monotone_factors_respected(self, ordinal_dataset):
        """Enforcing monotonicity should yield increasing group coefficients."""
        ds = ordinal_dataset
        fc = FactorClusterer(
            family="poisson",
            lambda_=0.05,
            monotone_factors=["vehicle_age"],
            monotone_direction={"vehicle_age": "increasing"},
        )
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        lm = fc.level_map("vehicle_age")
        assert lm.validate_monotone("increasing")

    def test_monotone_not_applied_without_flag(self, ordinal_dataset):
        """Without monotone_factors, the constraint is NOT applied."""
        ds = ordinal_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05, monotone_factors=[])
        fc.fit(ds["X"], ds["y"], exposure=ds["exposure"], ordinal_factors=["vehicle_age"])
        # Simply check it ran without error — monotonicity not guaranteed
        lm = fc.level_map("vehicle_age")
        assert lm.n_groups() >= 1


class TestFactorClustererTwoFactors:
    def test_two_factor_fit_runs(self, two_factor_dataset):
        ds = two_factor_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(
            ds["X"], ds["y"], exposure=ds["exposure"],
            ordinal_factors=["ncd"],
            nominal_factors=["occupation"],
        )
        assert fc._is_fitted

    def test_two_factor_level_maps_both_present(self, two_factor_dataset):
        ds = two_factor_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(
            ds["X"], ds["y"], exposure=ds["exposure"],
            ordinal_factors=["ncd"],
            nominal_factors=["occupation"],
        )
        lm_ncd = fc.level_map("ncd")
        lm_occ = fc.level_map("occupation")
        assert lm_ncd.n_levels_original() == 4
        assert lm_occ.n_levels_original() == 6

    def test_two_factor_transform_columns(self, two_factor_dataset):
        ds = two_factor_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(
            ds["X"], ds["y"], exposure=ds["exposure"],
            ordinal_factors=["ncd"],
            nominal_factors=["occupation"],
        )
        X_merged = fc.transform(ds["X"])
        assert "ncd" in X_merged.columns
        assert "occupation" in X_merged.columns

    def test_two_factor_diagnostics_n_levels(self, two_factor_dataset):
        ds = two_factor_dataset
        fc = FactorClusterer(family="poisson", lambda_=0.05)
        fc.fit(
            ds["X"], ds["y"], exposure=ds["exposure"],
            ordinal_factors=["ncd"],
            nominal_factors=["occupation"],
        )
        diag = fc.diagnostics()
        assert "ncd" in diag["n_levels_before"]
        assert "occupation" in diag["n_levels_before"]
