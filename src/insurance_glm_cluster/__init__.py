"""
insurance-glm-cluster
=====================

Automated GLM factor level clustering for insurance pricing.

Implements the R2VF algorithm (Ben Dror, arXiv:2503.01521, 2025) which
collapses high-cardinality categorical factors (e.g. 500 vehicle makes)
into pricing bands using a two-step fused lasso approach.

Key classes
-----------
FactorClusterer
    Main class. Fit on (X, y, exposure), transform to get merged codes.
LevelMap
    Result container for a single factor's merged groupings.

Quick start
-----------
>>> from insurance_glm_cluster import FactorClusterer
>>> clusterer = FactorClusterer(family='poisson', lambda_='bic')
>>> clusterer.fit(X, y, exposure=exposure,
...               ordinal_factors=['vehicle_age'],
...               nominal_factors=['vehicle_make'])
>>> X_merged = clusterer.transform(X)
>>> lm = clusterer.level_map('vehicle_make')
>>> lm.to_df()
"""

from .clusterer import FactorClusterer
from .level_map import LevelMap
from .diagnostics import (
    poisson_deviance,
    gamma_deviance,
    compute_bic,
    compute_aic,
)
from .backends import get_backend, StatsmodelsBackend, GlumBackend
from .constraints import enforce_min_exposure, enforce_monotonicity
from .penalties import R2VFRanker, build_r2vf_design_matrix
from .utils import build_split_coding_matrix, apply_split_coding

__version__ = "0.1.0"

__all__ = [
    "FactorClusterer",
    "LevelMap",
    "R2VFRanker",
    "build_r2vf_design_matrix",
    "build_split_coding_matrix",
    "apply_split_coding",
    "enforce_min_exposure",
    "enforce_monotonicity",
    "poisson_deviance",
    "gamma_deviance",
    "compute_bic",
    "compute_aic",
    "get_backend",
    "StatsmodelsBackend",
    "GlumBackend",
]
