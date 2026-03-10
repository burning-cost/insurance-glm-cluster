"""
Shared test fixtures for insurance-glm-cluster.

All synthetic datasets are constructed so that the correct answer is known
in advance. This makes tests falsifiable — they fail if the algorithm
doesn't recover the true grouping structure, not just if it runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Shared random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def ordinal_dataset(rng: np.random.Generator) -> dict:
    """
    Synthetic Poisson frequency dataset with a single ordinal factor.

    True structure: 5 levels where:
      - Levels 0, 1 share the same true log-rate (group A)
      - Levels 2, 3 share the same true log-rate (group B)
      - Level 4 has a distinct log-rate (group C)

    This is the minimal detectable structure: 5 → 3 groups.
    """
    n = 2000
    true_log_rates = [-0.5, -0.5, 0.0, 0.0, 0.8]  # 3 distinct values
    true_groups = [0, 0, 1, 1, 2]

    levels = rng.integers(0, 5, size=n)
    exposure = rng.uniform(0.5, 2.0, size=n)
    log_mu = np.array([true_log_rates[lv] for lv in levels])
    mu = exposure * np.exp(log_mu)
    y = rng.poisson(mu)

    X = pd.DataFrame({"vehicle_age": levels})

    return {
        "X": X,
        "y": y.astype(float),
        "exposure": exposure,
        "true_log_rates": true_log_rates,
        "true_groups": true_groups,
        "n_true_groups": 3,
    }


@pytest.fixture(scope="session")
def nominal_dataset(rng: np.random.Generator) -> dict:
    """
    Synthetic Poisson frequency dataset with a single nominal factor.

    10 nominal levels (vehicle make letters A-J) with true groupings:
      - {A, B, C}: low risk, log-rate = -0.4
      - {D, E, F, G}: medium risk, log-rate = 0.0
      - {H, I, J}: high risk, log-rate = 0.5

    The R2VF ranker should rank levels by coefficient, placing {A,B,C} at
    one end and {H,I,J} at the other.
    """
    n = 3000
    makes = list("ABCDEFGHIJ")
    true_log_rates = {
        "A": -0.4, "B": -0.4, "C": -0.4,
        "D": 0.0, "E": 0.0, "F": 0.0, "G": 0.0,
        "H": 0.5, "I": 0.5, "J": 0.5,
    }
    true_groups = {
        "A": 0, "B": 0, "C": 0,
        "D": 1, "E": 1, "F": 1, "G": 1,
        "H": 2, "I": 2, "J": 2,
    }

    level_idx = rng.integers(0, 10, size=n)
    level_labels = [makes[i] for i in level_idx]
    exposure = rng.uniform(0.5, 2.0, size=n)
    log_mu = np.array([true_log_rates[lv] for lv in level_labels])
    mu = exposure * np.exp(log_mu)
    y = rng.poisson(mu)

    X = pd.DataFrame({"vehicle_make": level_labels})

    return {
        "X": X,
        "y": y.astype(float),
        "exposure": exposure,
        "makes": makes,
        "true_log_rates": true_log_rates,
        "true_groups": true_groups,
        "n_true_groups": 3,
    }


@pytest.fixture(scope="session")
def two_factor_dataset(rng: np.random.Generator) -> dict:
    """
    Dataset with both ordinal (ncd) and nominal (occupation) factors.

    NCD (ordinal): 4 levels → 2 groups (0,1 together; 2,3 together)
    Occupation (nominal): 6 levels → 2 groups ({A,B,C} vs {D,E,F})
    """
    n = 4000
    ncd_log_rates = [0.3, 0.3, -0.2, -0.2]
    occ_log_rates = {"A": 0.2, "B": 0.2, "C": 0.2, "D": -0.1, "E": -0.1, "F": -0.1}

    ncd = rng.integers(0, 4, size=n)
    occ_idx = rng.integers(0, 6, size=n)
    occ_labels = [list("ABCDEF")[i] for i in occ_idx]
    exposure = rng.uniform(0.5, 2.0, size=n)

    log_mu = (
        np.array([ncd_log_rates[v] for v in ncd])
        + np.array([occ_log_rates[v] for v in occ_labels])
    )
    mu = exposure * np.exp(log_mu)
    y = rng.poisson(mu)

    X = pd.DataFrame({"ncd": ncd, "occupation": occ_labels})

    return {
        "X": X,
        "y": y.astype(float),
        "exposure": exposure,
        "ncd_log_rates": ncd_log_rates,
        "occ_log_rates": occ_log_rates,
    }


@pytest.fixture(scope="session")
def small_ordinal_known() -> dict:
    """
    Tiny deterministic dataset for split-coding correctness tests.

    5 observations, 3 factor levels with perfectly separated responses.
    """
    X = pd.DataFrame({"age": [0, 0, 1, 2, 2]})
    y = np.array([1.0, 1.0, 2.0, 4.0, 4.0])
    exposure = np.ones(5)
    return {"X": X, "y": y, "exposure": exposure}
