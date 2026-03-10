"""
Tests for backends.py: statsmodels GLM adapter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_glm_cluster.backends import StatsmodelsBackend, get_backend


class TestStatsmodelsBackend:
    def _make_poisson_data(self, n: int = 200, seed: int = 0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 2))
        beta = np.array([0.3, -0.2])
        mu = np.exp(X @ beta)
        y = rng.poisson(mu).astype(float)
        return X, y, mu

    def test_poisson_fit_runs(self):
        X, y, _ = self._make_poisson_data()
        import statsmodels.api as sm
        X_const = sm.add_constant(X)
        backend = StatsmodelsBackend(family="poisson", link="log")
        backend.fit(X_const, y)
        assert backend._result is not None

    def test_coef_length(self):
        X, y, _ = self._make_poisson_data()
        import statsmodels.api as sm
        X_const = sm.add_constant(X)
        backend = StatsmodelsBackend(family="poisson")
        backend.fit(X_const, y)
        assert len(backend.coef_) == 3  # intercept + 2 features

    def test_fitted_values_shape(self):
        X, y, _ = self._make_poisson_data()
        import statsmodels.api as sm
        X_const = sm.add_constant(X)
        backend = StatsmodelsBackend(family="poisson")
        backend.fit(X_const, y)
        assert backend.fitted_values_.shape == (200,)

    def test_log_likelihood_is_finite(self):
        X, y, _ = self._make_poisson_data()
        import statsmodels.api as sm
        X_const = sm.add_constant(X)
        backend = StatsmodelsBackend(family="poisson")
        backend.fit(X_const, y)
        assert np.isfinite(backend.log_likelihood_)

    def test_aic_bic_finite(self):
        X, y, _ = self._make_poisson_data()
        import statsmodels.api as sm
        X_const = sm.add_constant(X)
        backend = StatsmodelsBackend(family="poisson")
        backend.fit(X_const, y)
        assert np.isfinite(backend.aic_)
        assert np.isfinite(backend.bic_)

    def test_deviance_positive(self):
        X, y, _ = self._make_poisson_data()
        import statsmodels.api as sm
        X_const = sm.add_constant(X)
        backend = StatsmodelsBackend(family="poisson")
        backend.fit(X_const, y)
        assert backend.deviance_ >= 0

    def test_exposure_as_offset(self):
        """With exposure, log(exposure) should be used as offset."""
        rng = np.random.default_rng(1)
        n = 100
        X = rng.normal(size=(n, 1))
        exposure = rng.uniform(0.5, 2.0, size=n)
        mu = exposure * np.exp(X[:, 0] * 0.5)
        y = rng.poisson(mu).astype(float)

        import statsmodels.api as sm
        X_const = sm.add_constant(X)
        backend = StatsmodelsBackend(family="poisson")
        backend.fit(X_const, y, exposure=exposure)
        # Fitted values should be on the count scale (mu = rate * exposure)
        assert backend.fitted_values_.shape == (n,)
        assert np.isfinite(backend.log_likelihood_)

    def test_gamma_family_supported(self):
        rng = np.random.default_rng(2)
        n = 100
        X = rng.normal(size=(n, 2))
        import statsmodels.api as sm
        X_const = sm.add_constant(X)
        mu = np.exp(X @ np.array([0.2, -0.1]))
        y = rng.gamma(shape=2.0, scale=mu / 2.0)
        backend = StatsmodelsBackend(family="gamma")
        backend.fit(X_const, y)
        assert np.isfinite(backend.log_likelihood_)

    def test_not_fitted_raises(self):
        backend = StatsmodelsBackend()
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = backend.coef_

    def test_both_exposure_and_offset_raises(self):
        backend = StatsmodelsBackend()
        X = np.eye(3)
        y = np.ones(3)
        with pytest.raises(ValueError, match="not both"):
            backend.fit(X, y, exposure=np.ones(3), offset=np.zeros(3))

    def test_invalid_family_raises(self):
        with pytest.raises(ValueError, match="family must be one of"):
            StatsmodelsBackend(family="binomial")

    def test_invalid_link_raises(self):
        backend = StatsmodelsBackend(family="poisson", link="probit")
        import statsmodels.api as sm
        X = sm.add_constant(np.eye(3))
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Unsupported link"):
            backend.fit(X, y)

    def test_result_returns_raw_object(self):
        X, y, _ = self._make_poisson_data()
        import statsmodels.api as sm
        X_const = sm.add_constant(X)
        backend = StatsmodelsBackend(family="poisson")
        backend.fit(X_const, y)
        result = backend.result()
        assert result is not None
        assert hasattr(result, "params")


class TestGetBackend:
    def test_returns_statsmodels(self):
        backend = get_backend("statsmodels", family="poisson")
        assert isinstance(backend, StatsmodelsBackend)

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("xgboost")

    def test_tweedie_family_supported(self):
        backend = get_backend("statsmodels", family="tweedie", tweedie_power=1.6)
        assert backend.family == "tweedie"
        assert backend.tweedie_power == 1.6
