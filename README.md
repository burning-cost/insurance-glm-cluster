# insurance-glm-cluster

Automated GLM factor-level clustering for UK motor insurance pricing.

## The problem

Every motor pricing actuary knows this: you have a factor with 16 vehicle age bands, and you need to work out which ones can be merged without losing predictive signal. Do you merge band 8 with 7 or with 9? What about the extremes where there are three policies and one claim?

Today this is done manually. You plot the loss ratios, eyeball the pattern, argue about it in a model governance meeting, and end up with something defensible but not optimal. With 20 rating factors and hundreds of levels between them, that process takes weeks.

This library automates it using the R2VF algorithm (Ben Dror 2025, arXiv:2503.01521). The core idea: for ordinal factors, the fused lasso (which merges adjacent levels) reduces to a standard L1 lasso after a change of basis. That means you can use existing, well-tested solvers rather than writing a custom optimiser.

## What it does

R2VF Step 2: fits a penalised GLM on the split-coded design matrix. When the lasso shrinks a "difference" coefficient to zero, it's merging two adjacent levels. BIC picks the regularisation strength automatically.

R2VF Step 3: refits an unpenalised GLM on the merged encoding. This removes the shrinkage bias from Step 2 and gives you proper MLE estimates.

**MVP scope**: ordinal factors, Poisson and Gamma families, BIC lambda selection, min-exposure constraint.

## Installation

```bash
pip install insurance-glm-cluster
```

## Quick start

```python
import pandas as pd
from insurance_glm_cluster import FactorClusterer

fc = FactorClusterer(
    family='poisson',     # claim frequency
    lambda_='bic',        # automatic lambda selection
    min_exposure=500.0,   # merge groups with < 500 exposure years
)

fc.fit(
    X,
    y,                           # claim counts
    exposure=exposure,           # years at risk
    ordinal_factors=['vehicle_age', 'ncd_years'],
)

# Inspect the groupings
lm = fc.level_map('vehicle_age')
print(lm.to_df())
#  original_level  merged_group  coefficient  group_exposure
#               0             0        0.000        2341.2
#               1             0        0.000        2287.8
#               2             0        0.000        2319.4
#               3             1        0.312        2201.3
#               ...

# Recode and refit
X_merged = fc.transform(X)
result = fc.refit_glm(X_merged, y, exposure=exposure)
```

## API

### `FactorClusterer`

| Parameter | Type | Description |
|-----------|------|-------------|
| `family` | `'poisson'` \| `'gamma'` | Response distribution |
| `lambda_` | `float` \| `'bic'` | Regularisation strength, or auto-select |
| `n_lambda` | `int` | Grid size for BIC search (default 50) |
| `min_exposure` | `float` | Minimum group exposure (default 0, disabled) |
| `tol` | `float` | Zero-threshold for delta coefficients (default 1e-8) |

#### `.fit(X, y, exposure, ordinal_factors)`
Fits Step 2 (penalised fusion) and determines merged groups.

#### `.transform(X)`
Returns a copy of `X` with factor columns replaced by integer group labels.

#### `.refit_glm(X, y, exposure)`
Fits Step 3 (unpenalised refit) and returns a `statsmodels.GLMResults` object.

#### `.level_map(factor)`
Returns a `LevelMap` for the named factor.

#### `.diagnostic_path`
`DiagnosticPath` object with BIC, deviance, and n_groups per lambda. `None` if lambda was fixed.

### `LevelMap`

```python
lm = fc.level_map('vehicle_age')
lm.n_levels         # 16
lm.n_groups         # 3
lm.to_df()          # tidy DataFrame: original_level, merged_group, coefficient, group_exposure
lm.group_summary()  # one row per group with list of constituent levels
lm.apply(series)    # recode a series of original values to group labels
```

## Algorithm notes

**Split-coding**: for an ordinal factor with K levels and coefficients β, define δⱼ = βⱼ - βⱼ₋₁. The fused lasso penalty λ·Σ|δⱼ| is a plain L1 penalty on the deltas. Build the design matrix so column j has 1s for all observations with level ≥ j. Now the lasso on this matrix is equivalent to the fused lasso on the original one-hot matrix.

**Exposure handling for Poisson**: fitting on (y/exposure, weight=exposure) is algebraically equivalent to Poisson GLM with log(exposure) offset. sklearn's `PoissonRegressor` uses this trick internally.

**BIC lambda selection**: fits 50 lambdas from lambda_max to lambda_max/1000 on a log scale. lambda_max is the point where all factors collapse to a single group. BIC = -2·ℓ + K_eff·log(n) where K_eff counts distinct groups across all factors.

**Min-exposure**: after fusion, groups below `min_exposure` are absorbed into their nearest-coefficient neighbour (not nearest-level — a tiny group gets absorbed into whichever group it already looks most like).

## References

Ben Dror, R. (2025). *R2VF: Regularized Ranking for Variable Fusion in GLMs*. arXiv:2503.01521.
