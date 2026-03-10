"""
Diagnostics: BIC/AIC/deviance computation for GLM factor clustering.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def poisson_log_likelihood(
    y: NDArray[np.float64],
    mu: NDArray[np.float64],
    exposure: NDArray[np.float64] | None = None,
) -> float:
    """
    Compute the Poisson log-likelihood.

    l(μ; y) = Σ_i [ y_i * log(μ_i) - μ_i - log(y_i!) ]

    The factorial term is constant w.r.t. μ and is omitted here, as it
    cancels in AIC/BIC comparisons.

    Parameters
    ----------
    y : NDArray[np.float64]
        Observed counts.
    mu : NDArray[np.float64]
        Fitted means. For frequency models with exposure, mu = exposure * rate.
    exposure : NDArray[np.float64], optional
        Exposure. If provided, mu is expected to be the rate (mu_full = mu *
        exposure is used in the likelihood). If None, mu is used directly.

    Returns
    -------
    float
        Log-likelihood value.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    if exposure is not None:
        exp_arr = np.asarray(exposure, dtype=np.float64)
        mu_full = mu * exp_arr
    else:
        mu_full = mu

    # Avoid log(0)
    safe_mu = np.where(mu_full > 0, mu_full, 1e-300)
    # Avoid y*log(0): when y=0, contribution is -mu_full (log term is 0)
    log_term = np.where(y > 0, y * np.log(safe_mu), 0.0)
    return float(np.sum(log_term - mu_full))


def gamma_log_likelihood(
    y: NDArray[np.float64],
    mu: NDArray[np.float64],
    dispersion: float = 1.0,
) -> float:
    """
    Compute the Gamma log-likelihood (up to a constant).

    For the Gamma GLM with log link, the log-likelihood is proportional to:
    l(μ; y, φ) ∝ Σ_i [ -y_i / (φ * μ_i) - log(μ_i) ]

    Parameters
    ----------
    y : NDArray[np.float64]
        Observed severities (must be > 0).
    mu : NDArray[np.float64]
        Fitted means.
    dispersion : float
        Dispersion parameter φ.

    Returns
    -------
    float
        Log-likelihood value.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    safe_mu = np.where(mu > 0, mu, 1e-300)
    # Full Gamma log-likelihood: (1/φ) * [log(y/μ) - y/μ] - log(y) - log(φ/φ_0)
    # For comparison purposes we use the kernel only:
    ll = np.sum((1.0 / dispersion) * (np.log(y / safe_mu) - y / safe_mu))
    return float(ll)


def poisson_deviance(
    y: NDArray[np.float64],
    mu: NDArray[np.float64],
) -> float:
    """
    Compute the Poisson deviance D = 2 * Σ_i [y_i * log(y_i/μ_i) - (y_i - μ_i)].

    Parameters
    ----------
    y : NDArray[np.float64]
        Observed counts.
    mu : NDArray[np.float64]
        Fitted means.

    Returns
    -------
    float
        Poisson deviance.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    safe_mu = np.where(mu > 0, mu, 1e-300)
    # Avoid 0 * log(0) = 0
    log_term = np.where(y > 0, y * np.log(y / safe_mu), 0.0)
    return float(2.0 * np.sum(log_term - (y - mu)))


def gamma_deviance(
    y: NDArray[np.float64],
    mu: NDArray[np.float64],
) -> float:
    """
    Compute the Gamma deviance D = 2 * Σ_i [ -log(y_i/μ_i) + (y_i - μ_i)/μ_i ].

    Parameters
    ----------
    y : NDArray[np.float64]
        Observed values.
    mu : NDArray[np.float64]
        Fitted means.

    Returns
    -------
    float
        Gamma deviance.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    safe_mu = np.where(mu > 0, mu, 1e-300)
    safe_y = np.where(y > 0, y, 1e-300)
    return float(2.0 * np.sum(-np.log(safe_y / safe_mu) + (y - mu) / safe_mu))


def compute_bic(
    log_likelihood: float,
    n_params: int,
    n_obs: int,
) -> float:
    """
    Compute the Bayesian Information Criterion.

    BIC = -2 * l + k * ln(n)

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the fitted model.
    n_params : int
        Number of effective parameters (number of distinct merged groups across
        all factors, plus intercept).
    n_obs : int
        Number of observations.

    Returns
    -------
    float
        BIC value. Lower is better.
    """
    return -2.0 * log_likelihood + n_params * np.log(n_obs)


def compute_aic(
    log_likelihood: float,
    n_params: int,
) -> float:
    """
    Compute the Akaike Information Criterion.

    AIC = -2 * l + 2 * k

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the fitted model.
    n_params : int
        Number of effective parameters.

    Returns
    -------
    float
        AIC value. Lower is better.
    """
    return -2.0 * log_likelihood + 2.0 * n_params


def bic_lambda_selection(
    X_delta: NDArray[np.float64],
    y: NDArray[np.float64],
    lambda_grid: NDArray[np.float64],
    factor_slices: dict[str, "slice"],
    family: str = "poisson",
    sample_weight: NDArray[np.float64] | None = None,
    max_iter: int = 10_000,
) -> tuple[float, NDArray[np.float64], list[float]]:
    """
    Select lambda by minimising BIC over a grid of regularisation strengths.

    At each lambda, count effective parameters as the number of distinct
    non-zero delta groups across all factors (i.e. total merged groups).
    Compute BIC from the Poisson log-likelihood of the penalised fit on the
    *training* data (not held-out, so this is an in-sample criterion).

    Parameters
    ----------
    X_delta : NDArray[np.float64]
        Split-coded design matrix.
    y : NDArray[np.float64]
        Response (exposure-adjusted for frequency).
    lambda_grid : NDArray[np.float64]
        Decreasing array of lambda values to evaluate.
    factor_slices : dict[str, slice]
        Maps each factor name to its column slice in X_delta.
    family : str
        GLM family. Currently only 'poisson' supported for BIC selection.
    sample_weight : NDArray[np.float64], optional
        Sample weights (exposure).
    max_iter : int
        Max iterations per Lasso fit.

    Returns
    -------
    float
        Optimal lambda (minimises BIC).
    NDArray[np.float64]
        Delta coefficients at optimal lambda.
    list[float]
        BIC values at each lambda in the grid.
    """
    from sklearn.linear_model import Lasso

    n_obs = X_delta.shape[0]
    bic_values: list[float] = []
    best_bic = np.inf
    best_lam = lambda_grid[-1]
    best_coef = np.zeros(X_delta.shape[1])

    for lam in lambda_grid:
        lasso = Lasso(
            alpha=float(lam),
            fit_intercept=True,
            max_iter=max_iter,
            warm_start=False,
        )
        lasso.fit(X_delta, y, sample_weight=sample_weight)
        delta_coef = lasso.coef_
        intercept = lasso.intercept_

        # Count effective parameters: distinct groups per factor + 1 (intercept)
        n_effective = 1  # intercept
        for factor, sl in factor_slices.items():
            delta_factor = delta_coef[sl]
            # Number of groups = number of non-zero deltas + 1 (first group always exists)
            n_nonzero = int(np.sum(np.abs(delta_factor) > 1e-8))
            n_effective += n_nonzero + 1

        # Fitted values: X_delta @ delta + intercept
        mu_pred = X_delta @ delta_coef + intercept

        if family == "poisson":
            if sample_weight is not None:
                # Reverse the exposure adjustment: mu_pred is the rate
                # y_adj = y / exposure, so y_count = y_adj * exposure
                # mu_count = mu_pred * exposure
                y_counts = y * sample_weight / sample_weight.mean() * y.mean() / y.mean()
                # Simpler: compute Poisson LL on rate scale with weights
                safe_mu = np.where(mu_pred > 0, mu_pred, 1e-300)
                log_term = np.where(y > 0, y * np.log(safe_mu), 0.0)
                ll = float(np.sum(sample_weight * (log_term - mu_pred)))
            else:
                safe_mu = np.where(mu_pred > 0, mu_pred, 1e-300)
                log_term = np.where(y > 0, y * np.log(safe_mu), 0.0)
                ll = float(np.sum(log_term - mu_pred))
        else:
            # Generic: negative RSS as proxy log-likelihood
            residuals = y - mu_pred
            ll = -0.5 * float(np.sum(residuals**2))

        bic = compute_bic(ll, n_effective, n_obs)
        bic_values.append(bic)

        if bic < best_bic:
            best_bic = bic
            best_lam = float(lam)
            best_coef = delta_coef.copy()

    return best_lam, best_coef, bic_values
