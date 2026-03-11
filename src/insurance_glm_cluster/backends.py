"""
GLM backend adapters for statsmodels and glum.

The library uses statsmodels by default for fitting unpenalised GLMs
(the refit step after clustering). glum is supported as a faster optional
backend, particularly useful for large datasets (> 500k rows).

Design rationale: the clustering step uses sklearn (Lasso on the split-coded
matrix) and does not need a full GLM backend. The backends here are only
used for the unpenalised refit and for computing log-likelihoods/deviance
for diagnostics.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


SUPPORTED_FAMILIES = ("poisson", "gamma", "tweedie")


class StatsmodelsBackend:
    """
    Adapter for fitting unpenalised GLMs using statsmodels.

    Handles Poisson, Gamma, and Tweedie families with log link. Supports
    exposure as an offset (log(exposure)) which is the statistically correct
    treatment for frequency models.

    Parameters
    ----------
    family : str
        GLM family: 'poisson', 'gamma', or 'tweedie'.
    link : str
        Link function: 'log' (default), 'identity'.
    tweedie_power : float
        Power parameter for Tweedie family (1 < p < 2 for compound Poisson).
    """

    def __init__(
        self,
        family: str = "poisson",
        link: str = "log",
        tweedie_power: float = 1.5,
    ) -> None:
        if family not in SUPPORTED_FAMILIES:
            raise ValueError(
                f"family must be one of {SUPPORTED_FAMILIES}, got '{family}'."
            )
        self.family = family
        self.link = link
        self.tweedie_power = tweedie_power
        self._result: Any = None

    def _get_sm_family(self) -> Any:
        """Construct the statsmodels family object."""
        import statsmodels.api as sm

        link_map = {
            "log": sm.families.links.Log(),
            "identity": sm.families.links.Identity(),
        }
        link_obj = link_map.get(self.link)
        if link_obj is None:
            raise ValueError(f"Unsupported link function: '{self.link}'.")

        if self.family == "poisson":
            return sm.families.Poisson(link=link_obj)
        elif self.family == "gamma":
            return sm.families.Gamma(link=link_obj)
        elif self.family == "tweedie":
            return sm.families.Tweedie(
                link=link_obj, var_power=self.tweedie_power
            )
        else:
            raise ValueError(f"Unknown family: '{self.family}'.")

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        exposure: NDArray[np.float64] | None = None,
        offset: NDArray[np.float64] | None = None,
    ) -> "StatsmodelsBackend":
        """
        Fit an unpenalised GLM.

        Parameters
        ----------
        X : NDArray[np.float64]
            Design matrix (n_samples, n_features). Should include an intercept
            column if needed, or use the add_constant parameter.
        y : NDArray[np.float64]
            Response variable.
        exposure : NDArray[np.float64], optional
            Exposure array. Added as log(exposure) offset. Mutually exclusive
            with the offset parameter.
        offset : NDArray[np.float64], optional
            Pre-computed offset (on the linear predictor scale).

        Returns
        -------
        self
        """
        import statsmodels.api as sm

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Build offset: log(exposure) for frequency models
        if exposure is not None and offset is not None:
            raise ValueError(
                "Provide either 'exposure' or 'offset', not both."
            )
        if exposure is not None:
            exp_arr = np.asarray(exposure, dtype=np.float64)
            safe_exp = np.where(exp_arr > 0, exp_arr, 1e-300)
            glm_offset = np.log(safe_exp)
        elif offset is not None:
            glm_offset = np.asarray(offset, dtype=np.float64)
        else:
            glm_offset = None

        family_obj = self._get_sm_family()
        model = sm.GLM(
            y,
            X,
            family=family_obj,
            offset=glm_offset,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._result = model.fit(disp=False)

        return self

    @property
    def coef_(self) -> NDArray[np.float64]:
        """Fitted coefficients (including intercept)."""
        if self._result is None:
            raise RuntimeError("Model has not been fitted.")
        return np.asarray(self._result.params, dtype=np.float64)

    @property
    def fitted_values_(self) -> NDArray[np.float64]:
        """Fitted means on the response scale."""
        if self._result is None:
            raise RuntimeError("Model has not been fitted.")
        return np.asarray(self._result.fittedvalues, dtype=np.float64)

    @property
    def log_likelihood_(self) -> float:
        """Log-likelihood of the fitted model."""
        if self._result is None:
            raise RuntimeError("Model has not been fitted.")
        return float(self._result.llf)

    @property
    def aic_(self) -> float:
        """AIC of the fitted model."""
        if self._result is None:
            raise RuntimeError("Model has not been fitted.")
        return float(self._result.aic)

    @property
    def bic_(self) -> float:
        """BIC of the fitted model."""
        if self._result is None:
            raise RuntimeError("Model has not been fitted.")
        return float(self._result.bic)

    @property
    def deviance_(self) -> float:
        """Model deviance."""
        if self._result is None:
            raise RuntimeError("Model has not been fitted.")
        return float(self._result.deviance)

    @property
    def n_params_(self) -> int:
        """Number of estimated parameters."""
        if self._result is None:
            raise RuntimeError("Model has not been fitted.")
        return int(self._result.df_model) + 1  # +1 for intercept

    def result(self) -> Any:
        """Return the raw statsmodels GLMResultsWrapper."""
        return self._result


class GlumBackend:
    """
    Adapter for fitting GLMs using glum (optional fast backend).

    glum (github.com/QuantActuary/glum) provides IRLS and L-BFGS GLM fitting
    with proper offset support and is substantially faster than statsmodels
    on large datasets. It requires the 'glum' extra.

    Parameters
    ----------
    family : str
        GLM family: 'poisson', 'gamma', or 'tweedie'.
    link : str
        Link function: 'log' (default).
    tweedie_power : float
        Power parameter for Tweedie family.
    """

    def __init__(
        self,
        family: str = "poisson",
        link: str = "log",
        tweedie_power: float = 1.5,
    ) -> None:
        try:
            import glum  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "glum is required for the glum backend. "
                "Install with: pip install insurance-glm-cluster[fast]"
            ) from exc

        if family not in SUPPORTED_FAMILIES:
            raise ValueError(
                f"family must be one of {SUPPORTED_FAMILIES}, got '{family}'."
            )
        self.family = family
        self.link = link
        self.tweedie_power = tweedie_power
        self._result: Any = None
        self._log_likelihood: float | None = None

    def _get_glum_family(self) -> Any:
        """Construct the glum distribution object."""
        from glum import TweedieDistribution

        if self.family == "poisson":
            return TweedieDistribution(power=1)
        elif self.family == "gamma":
            return TweedieDistribution(power=2)
        elif self.family == "tweedie":
            return TweedieDistribution(power=self.tweedie_power)

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        exposure: NDArray[np.float64] | None = None,
        offset: NDArray[np.float64] | None = None,
    ) -> "GlumBackend":
        """
        Fit an unpenalised GLM using glum.

        Parameters
        ----------
        X : NDArray[np.float64]
            Design matrix (n_samples, n_features).
        y : NDArray[np.float64]
            Response variable.
        exposure : NDArray[np.float64], optional
            Exposure (used as sample_weight in glum Poisson frequency models).
        offset : NDArray[np.float64], optional
            Pre-computed offset.

        Returns
        -------
        self
        """
        from glum import GeneralizedLinearRegressor

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        family_obj = self._get_glum_family()

        if exposure is not None and offset is not None:
            raise ValueError("Provide either 'exposure' or 'offset', not both.")

        if exposure is not None:
            exp_arr = np.asarray(exposure, dtype=np.float64)
            safe_exp = np.where(exp_arr > 0, exp_arr, 1e-300)
            glm_offset = np.log(safe_exp)
        elif offset is not None:
            glm_offset = np.asarray(offset, dtype=np.float64)
        else:
            glm_offset = None

        glm = GeneralizedLinearRegressor(
            family=family_obj,
            link=self.link,
            alpha=0,  # unpenalised
            fit_intercept=True,
        )

        fit_kwargs: dict[str, Any] = {}
        if glm_offset is not None:
            fit_kwargs["offset"] = glm_offset

        self._result = glm.fit(X, y, **fit_kwargs)
        return self

    @property
    def coef_(self) -> NDArray[np.float64]:
        """Fitted coefficients (excluding intercept)."""
        if self._result is None:
            raise RuntimeError("Model has not been fitted.")
        coef = self._result.coef_.astype(np.float64)
        intercept = np.array([self._result.intercept_], dtype=np.float64)
        return np.concatenate([intercept, coef])

    @property
    def fitted_values_(self) -> NDArray[np.float64]:
        """Fitted means on the response scale."""
        if self._result is None:
            raise RuntimeError("Model has not been fitted.")
        return self._result.predict().astype(np.float64)

    def result(self) -> Any:
        """Return the raw glum fitted model."""
        return self._result


def get_backend(
    backend: str = "statsmodels",
    family: str = "poisson",
    link: str = "log",
    tweedie_power: float = 1.5,
) -> StatsmodelsBackend | GlumBackend:
    """
    Instantiate a GLM backend by name.

    Parameters
    ----------
    backend : str
        'statsmodels' (default) or 'glum'.
    family : str
        GLM family.
    link : str
        Link function.
    tweedie_power : float
        Tweedie power parameter.

    Returns
    -------
    StatsmodelsBackend or GlumBackend
    """
    if backend == "statsmodels":
        return StatsmodelsBackend(
            family=family, link=link, tweedie_power=tweedie_power
        )
    elif backend == "glum":
        return GlumBackend(
            family=family, link=link, tweedie_power=tweedie_power
        )
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose 'statsmodels' or 'glum'."
        )
