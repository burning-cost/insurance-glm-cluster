"""
FactorClusterer: main class for automated GLM factor level clustering.

Implements the R2VF algorithm (Ben Dror, arXiv:2503.01521, 2025) which
converts the O(K²) generalised fused lasso for nominal factors into an
O(K) problem via a two-step ranking and fusion procedure.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .utils import (
    levels_to_onehot,
    exposure_adjusted_target,
    groups_from_delta,
    delta_to_beta,
    bin_numeric,
)
from .penalties import (
    R2VFRanker,
    build_r2vf_design_matrix,
    fit_fused_lasso,
    lambda_grid,
)
from .diagnostics import bic_lambda_selection
from .level_map import LevelMap
from .backends import get_backend, StatsmodelsBackend
from .constraints import (
    enforce_min_exposure,
    enforce_min_claims,
    enforce_monotonicity,
    check_monotonicity,
)


class FactorClusterer:
    """
    Automated GLM factor level clustering via the R2VF algorithm.

    Collapses high-cardinality categorical factors (e.g. 500 vehicle makes)
    into pricing bands by fitting a fused lasso on the factor dummies. For
    nominal (unordered) factors, uses R2VF's two-step approach to first
    rank levels via Ridge regression before applying the fusion penalty.

    Parameters
    ----------
    family : str
        GLM family: 'poisson', 'gamma', or 'tweedie'.
    link : str
        Link function: 'log' (default for all three families).
    method : str
        Clustering method. Currently only 'r2vf' is supported.
    lambda_ : float or 'bic'
        Regularisation strength for the fused lasso (Step 2).
        If 'bic', selects lambda by minimising BIC over a grid of 50 values.
    n_ordinal_bins : int
        Initial number of bins for numeric/ordinal factors. Reduces the
        design matrix size before fusion. Recommended: 30.
    m_nominal_bins : int
        Initial number of bins for nominal factors in Step 1 (ranking).
        Caps the number of dummies for high-cardinality nominals. Recommended:
        75. This guards against overfitting in Step 1 with very large factors
        (e.g. 500+ vehicle makes).
    alpha : float
        Regularisation type for Step 1: 1.0 = Lasso, 2.0 = Ridge. Ridge is
        preferred because it preserves the full coefficient ranking rather
        than zeroing levels before Step 2 can run.
    min_exposure : float, optional
        Minimum exposure per merged group after fusion. Groups below this
        threshold are absorbed into the nearest neighbour by coefficient value.
    min_claims : int, optional
        Minimum claim count per merged group.
    monotone_factors : list[str]
        Factor names to enforce monotonicity on. Applied after fusion, before
        the unpenalised refit.
    monotone_direction : dict[str, str]
        Per-factor monotone direction: {'vehicle_age': 'increasing'}.
        Factors in monotone_factors but not here default to 'increasing'.
    backend : str
        GLM backend for the unpenalised refit step: 'statsmodels' or 'glum'.
    tweedie_power : float
        Tweedie power parameter (relevant only when family='tweedie').
    random_state : int
        Random seed.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from insurance_glm_cluster import FactorClusterer
    >>> # Fit on a DataFrame with vehicle_age (ordinal) and vehicle_make (nominal)
    >>> clusterer = FactorClusterer(family='poisson', lambda_='bic')
    >>> clusterer.fit(
    ...     X, y, exposure=exposure,
    ...     ordinal_factors=['vehicle_age'],
    ...     nominal_factors=['vehicle_make'],
    ... )
    >>> X_merged = clusterer.transform(X)
    """

    def __init__(
        self,
        family: str = "poisson",
        link: str = "log",
        method: str = "r2vf",
        lambda_: float | str = "bic",
        n_ordinal_bins: int = 30,
        m_nominal_bins: int = 75,
        alpha: float = 2.0,
        min_exposure: float | None = None,
        min_claims: int | None = None,
        monotone_factors: list[str] | None = None,
        monotone_direction: dict[str, str] | None = None,
        backend: str = "statsmodels",
        tweedie_power: float = 1.5,
        random_state: int = 42,
    ) -> None:
        if method != "r2vf":
            raise ValueError(
                f"method must be 'r2vf' (Phase 2 will add chi2_adjacent). "
                f"Got '{method}'."
            )
        if isinstance(lambda_, str) and lambda_ != "bic":
            raise ValueError(
                f"lambda_ must be a float or 'bic', got '{lambda_}'."
            )

        self.family = family
        self.link = link
        self.method = method
        self.lambda_ = lambda_
        self.n_ordinal_bins = n_ordinal_bins
        self.m_nominal_bins = m_nominal_bins
        self.alpha = alpha
        self.min_exposure = min_exposure
        self.min_claims = min_claims
        self.monotone_factors = list(monotone_factors or [])
        self.monotone_direction = dict(monotone_direction or {})
        self.backend = backend
        self.tweedie_power = tweedie_power
        self.random_state = random_state

        # Fitted state
        self._is_fitted: bool = False
        self._ordinal_factors: list[str] = []
        self._nominal_factors: list[str] = []
        self._factor_categories: dict[str, list] = {}
        self._factor_reordered_categories: dict[str, list] = {}
        self._group_assignments: dict[str, NDArray[np.int64]] = {}
        self._group_coefficients: dict[str, pd.Series] = {}
        self._group_exposures: dict[str, pd.Series] = {}
        self._selected_lambda: float | None = None
        self._bic_curve: list[float] | None = None
        self._n_obs: int = 0
        self._diag_before: dict[str, Any] = {}
        self._diag_after: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: NDArray[np.float64] | pd.Series,
        exposure: NDArray[np.float64] | pd.Series | None = None,
        offset: NDArray[np.float64] | pd.Series | None = None,
        ordinal_factors: list[str] | None = None,
        nominal_factors: list[str] | None = None,
    ) -> "FactorClusterer":
        """
        Fit the R2VF clustering model.

        Runs Steps 1 (ranking nominals via Ridge) and 2 (fusing all factors
        via split-coded Lasso), then applies exposure and monotonicity
        constraints.

        Parameters
        ----------
        X : pd.DataFrame
            Input features. Must contain all columns in ordinal_factors and
            nominal_factors.
        y : array-like
            Response variable (e.g. claim counts for Poisson frequency).
        exposure : array-like, optional
            Exposure (e.g. earned years). Required for Poisson frequency.
        offset : array-like, optional
            Pre-computed linear predictor offset. Alternative to exposure.
        ordinal_factors : list[str], optional
            Column names to treat as ordinal (natural ordering preserved).
            Numeric columns are automatically binned.
        nominal_factors : list[str], optional
            Column names to treat as nominal (R2VF Step 1 ranking applied).

        Returns
        -------
        self
        """
        ordinal_factors = list(ordinal_factors or [])
        nominal_factors = list(nominal_factors or [])
        all_factors = ordinal_factors + nominal_factors

        if not all_factors:
            raise ValueError(
                "Provide at least one factor in ordinal_factors or nominal_factors."
            )

        missing = [f for f in all_factors if f not in X.columns]
        if missing:
            raise ValueError(f"Factors not found in X: {missing}")

        y_arr = np.asarray(y, dtype=np.float64)
        self._n_obs = len(y_arr)
        exposure_arr = (
            np.asarray(exposure, dtype=np.float64)
            if exposure is not None
            else None
        )

        self._ordinal_factors = ordinal_factors
        self._nominal_factors = nominal_factors

        # ------------------------------------------------------------------
        # Step 0: encode all factors into one-hot matrices
        # ------------------------------------------------------------------
        factor_onehot: dict[str, NDArray[np.float64]] = {}

        for factor in ordinal_factors:
            col = X[factor]
            if pd.api.types.is_numeric_dtype(col):
                binned, cats = bin_numeric(col, self.n_ordinal_bins)
                self._factor_categories[factor] = cats
                onehot, _ = levels_to_onehot(binned, cats)
            else:
                cats = sorted(col.dropna().unique().tolist())
                self._factor_categories[factor] = cats
                onehot, _ = levels_to_onehot(col, cats)
            factor_onehot[factor] = onehot

        for factor in nominal_factors:
            col = X[factor]
            # Cap nominal levels at m_nominal_bins most frequent
            value_counts = col.value_counts()
            if len(value_counts) > self.m_nominal_bins:
                top_cats = value_counts.head(self.m_nominal_bins).index.tolist()
            else:
                top_cats = value_counts.index.tolist()
            self._factor_categories[factor] = top_cats
            onehot, _ = levels_to_onehot(col, top_cats)
            factor_onehot[factor] = onehot

        # ------------------------------------------------------------------
        # Compute exposure-adjusted target for sklearn
        # ------------------------------------------------------------------
        if exposure_arr is not None:
            y_adj, sample_weight = exposure_adjusted_target(y_arr, exposure_arr)
        else:
            y_adj = y_arr
            sample_weight = None

        # ------------------------------------------------------------------
        # Step 1 (nominal factors only): rank levels via Ridge regression
        # ------------------------------------------------------------------
        nominal_rankings: dict[str, list] = {}

        if nominal_factors:
            # Build full dummy matrix for Step 1: all factors stacked
            step1_blocks = []
            step1_slices: dict[str, slice] = {}
            col_offset = 0
            for factor in all_factors:
                oh = factor_onehot[factor]
                n_lev = oh.shape[1]
                step1_blocks.append(oh)
                step1_slices[factor] = slice(col_offset, col_offset + n_lev)
                col_offset += n_lev

            X_step1 = np.hstack(step1_blocks) if step1_blocks else np.empty((self._n_obs, 0))

            ranker = R2VFRanker(alpha=self.alpha, random_state=self.random_state)
            ranker.fit(
                X_step1,
                y_adj,
                factor_slices={f: step1_slices[f] for f in nominal_factors},
                sample_weight=sample_weight,
            )
            for factor in nominal_factors:
                nominal_rankings[factor] = ranker.get_ranking(factor)

        # ------------------------------------------------------------------
        # Step 2: fused lasso on split-coded matrix
        # ------------------------------------------------------------------
        X_delta, delta_slices, reordered_cats = build_r2vf_design_matrix(
            factor_data=factor_onehot,
            factor_categories=self._factor_categories,
            nominal_rankings=nominal_rankings if nominal_factors else None,
        )
        self._factor_reordered_categories = reordered_cats

        # Select lambda
        if self.lambda_ == "bic":
            lam_grid = lambda_grid(X_delta, y_adj, n_points=50, sample_weight=sample_weight)
            selected_lam, best_delta_coef, bic_curve = bic_lambda_selection(
                X_delta=X_delta,
                y=y_adj,
                lambda_grid=lam_grid,
                factor_slices=delta_slices,
                family=self.family,
                sample_weight=sample_weight,
            )
            self._selected_lambda = selected_lam
            self._bic_curve = bic_curve
            delta_coef = best_delta_coef
        else:
            lam_val = float(self.lambda_)  # type: ignore[arg-type]
            self._selected_lambda = lam_val
            delta_coef = fit_fused_lasso(X_delta, y_adj, lam=lam_val, sample_weight=sample_weight)

        # ------------------------------------------------------------------
        # Step 3: extract group assignments from delta coefficients
        # ------------------------------------------------------------------
        for factor in all_factors:
            sl = delta_slices[factor]
            factor_delta = delta_coef[sl]
            groups = groups_from_delta(factor_delta, tol=1e-8)

            ordered_cats = self._factor_reordered_categories[factor]
            n_levels = len(ordered_cats)

            # Compute per-level exposure (distribute per original level)
            if exposure_arr is not None:
                factor_col = X[factor]
                level_exposures = np.zeros(n_levels, dtype=np.float64)
                for i, cat in enumerate(ordered_cats):
                    mask = factor_col == cat
                    level_exposures[i] = exposure_arr[mask].sum()
            else:
                level_exposures = np.ones(n_levels, dtype=np.float64)

            # Group-level coefficients (cumsum of deltas = level beta)
            level_betas = delta_to_beta(factor_delta)

            # Group coefficient = mean of level betas in group
            unique_groups = np.unique(groups)
            group_coef_dict: dict[int, float] = {}
            for g in unique_groups:
                mask = groups == g
                group_coef_dict[g] = float(np.mean(level_betas[mask]))
            group_coef_series = pd.Series(group_coef_dict)

            # Apply min_exposure constraint
            if self.min_exposure is not None:
                group_level_coef = np.array(
                    [group_coef_dict[g] for g in groups], dtype=np.float64
                )
                groups = enforce_min_exposure(
                    groups, level_exposures, self.min_exposure, group_level_coef
                )
                # Recompute group coefficients after absorption
                unique_groups = np.unique(groups)
                group_coef_dict = {}
                for g in unique_groups:
                    mask = groups == g
                    group_coef_dict[g] = float(np.mean(level_betas[mask]))
                group_coef_series = pd.Series(group_coef_dict)

            # Apply min_claims constraint
            if self.min_claims is not None:
                factor_col = X[factor]
                level_claims = np.zeros(n_levels, dtype=np.float64)
                for i, cat in enumerate(ordered_cats):
                    mask = factor_col == cat
                    level_claims[i] = y_arr[mask].sum()
                group_level_coef = np.array(
                    [group_coef_dict[g] for g in groups], dtype=np.float64
                )
                groups = enforce_min_claims(
                    groups, level_claims, self.min_claims, group_level_coef
                )
                unique_groups = np.unique(groups)
                group_coef_dict = {}
                for g in unique_groups:
                    mask = groups == g
                    group_coef_dict[g] = float(np.mean(level_betas[mask]))
                group_coef_series = pd.Series(group_coef_dict)

            # Apply monotonicity enforcement
            if factor in self.monotone_factors:
                direction = self.monotone_direction.get(factor, "increasing")
                group_coef_series = enforce_monotonicity(group_coef_series, direction)

            # Compute per-group exposures
            unique_groups = np.unique(groups)
            group_exp_dict: dict[int, float] = {}
            for g in unique_groups:
                mask = groups == g
                group_exp_dict[g] = float(level_exposures[mask].sum())
            group_exp_series = pd.Series(group_exp_dict)

            self._group_assignments[factor] = groups
            self._group_coefficients[factor] = group_coef_series
            self._group_exposures[factor] = group_exp_series

        # ------------------------------------------------------------------
        # Diagnostics: before (full dummies) and after (merged groups)
        # ------------------------------------------------------------------
        self._compute_diagnostics_before(X, y_arr, exposure_arr, all_factors)
        self._is_fitted = True

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace original factor columns with merged group codes.

        Parameters
        ----------
        X : pd.DataFrame
            Input features. Must contain all fitted factor columns.

        Returns
        -------
        pd.DataFrame
            Copy of X with factor columns replaced by integer group codes.
            Group codes are 0-indexed within each factor.
        """
        self._check_fitted()
        X_out = X.copy()
        all_factors = self._ordinal_factors + self._nominal_factors

        for factor in all_factors:
            if factor not in X.columns:
                raise ValueError(f"Factor '{factor}' not found in X.")

            lm = self.level_map(factor)
            mapping = lm.mapping
            X_out[factor] = X[factor].map(mapping).astype("Int64")

        return X_out

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: NDArray[np.float64] | pd.Series,
        exposure: NDArray[np.float64] | pd.Series | None = None,
        offset: NDArray[np.float64] | pd.Series | None = None,
        ordinal_factors: list[str] | None = None,
        nominal_factors: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : array-like
            Response variable.
        exposure : array-like, optional
            Exposure.
        offset : array-like, optional
            Offset.
        ordinal_factors : list[str], optional
            Ordinal factor column names.
        nominal_factors : list[str], optional
            Nominal factor column names.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with merged group codes.
        """
        return self.fit(
            X, y, exposure=exposure, offset=offset,
            ordinal_factors=ordinal_factors, nominal_factors=nominal_factors,
        ).transform(X)

    # ------------------------------------------------------------------
    # Refit
    # ------------------------------------------------------------------

    def refit_glm(
        self,
        X_merged: pd.DataFrame,
        y: NDArray[np.float64] | pd.Series,
        exposure: NDArray[np.float64] | pd.Series | None = None,
        offset: NDArray[np.float64] | pd.Series | None = None,
    ) -> Any:
        """
        Fit an unpenalised GLM on the merged factor design matrix.

        The penalised clustering step introduces shrinkage bias in the
        coefficient estimates. This refit on the merged (but not penalised)
        design matrix recovers unbiased estimates.

        Parameters
        ----------
        X_merged : pd.DataFrame
            Design matrix after transform(). One column per factor, values
            are integer group codes.
        y : array-like
            Response variable.
        exposure : array-like, optional
            Exposure.
        offset : array-like, optional
            Pre-computed offset.

        Returns
        -------
        statsmodels GLMResultsWrapper (or glum fitted model if backend='glum').
        """
        self._check_fitted()
        import statsmodels.api as sm

        y_arr = np.asarray(y, dtype=np.float64)
        exposure_arr = (
            np.asarray(exposure, dtype=np.float64)
            if exposure is not None
            else None
        )

        # One-hot encode the merged groups, add intercept
        dummies_list = []
        all_factors = self._ordinal_factors + self._nominal_factors

        for factor in all_factors:
            if factor not in X_merged.columns:
                raise ValueError(f"Factor '{factor}' not in X_merged.")
            col = X_merged[factor].astype(str)
            dummies = pd.get_dummies(col, prefix=factor, drop_first=True, dtype=float)
            dummies_list.append(dummies)

        if dummies_list:
            X_design = pd.concat(dummies_list, axis=1)
        else:
            X_design = pd.DataFrame(index=X_merged.index)

        X_with_const = sm.add_constant(X_design.values, has_constant="add")

        backend_obj = get_backend(
            backend=self.backend,
            family=self.family,
            link=self.link,
            tweedie_power=self.tweedie_power,
        )
        backend_obj.fit(
            X_with_const, y_arr,
            exposure=exposure_arr, offset=offset,
        )

        # Update diagnostics_after with refit stats
        if isinstance(backend_obj, StatsmodelsBackend):
            self._diag_after = {
                "aic": backend_obj.aic_,
                "bic": backend_obj.bic_,
                "deviance": backend_obj.deviance_,
                "n_params": backend_obj.n_params_,
                "log_likelihood": backend_obj.log_likelihood_,
            }

        return backend_obj.result()

    # ------------------------------------------------------------------
    # Level map
    # ------------------------------------------------------------------

    def level_map(self, factor: str) -> LevelMap:
        """
        Return the LevelMap for a specific factor.

        Parameters
        ----------
        factor : str
            Factor name (must be in ordinal_factors or nominal_factors).

        Returns
        -------
        LevelMap
            Container with the original-to-group mapping, group coefficients,
            and group exposures.
        """
        self._check_fitted()
        all_factors = self._ordinal_factors + self._nominal_factors
        if factor not in all_factors:
            raise ValueError(
                f"Factor '{factor}' was not fitted. "
                f"Fitted factors: {all_factors}"
            )

        ordered_cats = self._factor_reordered_categories[factor]
        groups = self._group_assignments[factor]

        mapping = {cat: int(groups[i]) for i, cat in enumerate(ordered_cats)}

        return LevelMap(
            factor_name=factor,
            mapping=mapping,
            group_coefficients=self._group_coefficients[factor],
            group_exposures=self._group_exposures[factor],
            original_levels=ordered_cats,
            is_nominal=factor in self._nominal_factors,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        """
        Return a diagnostics dictionary comparing before and after clustering.

        Before: unpenalised GLM fitted on all original factor dummies.
        After: the refit GLM on merged groups (populated after refit_glm()).

        Returns
        -------
        dict
            Keys: 'aic_before', 'aic_after', 'bic_before', 'bic_after',
            'deviance_before', 'deviance_after', 'n_levels_before',
            'n_levels_after', 'selected_lambda', 'bic_curve'.
        """
        self._check_fitted()
        all_factors = self._ordinal_factors + self._nominal_factors

        n_levels_before = {
            f: len(self._factor_categories[f]) for f in all_factors
        }
        n_levels_after = {
            f: int(np.unique(self._group_assignments[f]).size)
            for f in all_factors
        }

        result: dict[str, Any] = {
            "aic_before": self._diag_before.get("aic"),
            "aic_after": self._diag_after.get("aic"),
            "bic_before": self._diag_before.get("bic"),
            "bic_after": self._diag_after.get("bic"),
            "deviance_before": self._diag_before.get("deviance"),
            "deviance_after": self._diag_after.get("deviance"),
            "n_levels_before": n_levels_before,
            "n_levels_after": n_levels_after,
            "selected_lambda": self._selected_lambda,
            "bic_curve": self._bic_curve,
        }
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "FactorClusterer has not been fitted. Call fit() first."
            )

    def _compute_diagnostics_before(
        self,
        X: pd.DataFrame,
        y_arr: NDArray[np.float64],
        exposure_arr: NDArray[np.float64] | None,
        all_factors: list[str],
    ) -> None:
        """Fit unpenalised GLM on full dummies and record AIC/BIC/deviance."""
        try:
            import statsmodels.api as sm

            dummies_list = []
            for factor in all_factors:
                col = X[factor].astype(str)
                dummies = pd.get_dummies(col, prefix=factor, drop_first=True, dtype=float)
                dummies_list.append(dummies)

            if dummies_list:
                X_full = pd.concat(dummies_list, axis=1)
            else:
                X_full = pd.DataFrame(index=X.index)

            X_with_const = sm.add_constant(X_full.values, has_constant="add")

            backend_obj = get_backend(
                backend=self.backend,
                family=self.family,
                link=self.link,
                tweedie_power=self.tweedie_power,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                backend_obj.fit(X_with_const, y_arr, exposure=exposure_arr)

            if isinstance(backend_obj, StatsmodelsBackend):
                self._diag_before = {
                    "aic": backend_obj.aic_,
                    "bic": backend_obj.bic_,
                    "deviance": backend_obj.deviance_,
                }
        except Exception as exc:
            # Diagnostics before are best-effort; don't fail the whole fit
            warnings.warn(
                f"Could not compute before-clustering diagnostics: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._diag_before = {}

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"FactorClusterer(family='{self.family}', method='{self.method}', "
            f"lambda_={self.lambda_!r}, status={status})"
        )
