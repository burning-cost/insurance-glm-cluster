# insurance-glm-cluster

Automated GLM factor level clustering for insurance pricing.

## The problem

You've got 500 vehicle makes in your motor book. Your pricing GLM needs to handle them. You can't fit 500 dummies — the data is too thin, the model will overfit, and you'll end up with nonsense relativities for rare makes.

The traditional fix is manual grouping: spend a week in Excel, consult a book of makes and models, build a lookup table, argue with underwriters. This works but doesn't scale, introduces analyst bias, and has to be redone every model cycle.

**insurance-glm-cluster automates this.** It collapses high-cardinality categorical factors into pricing bands using regularised regression, with proper statistical underpinning and no arbitrary decisions.

## How it works

The library implements the R2VF algorithm (Ben Dror, arXiv:2503.01521, 2025). The key insight is that the standard fused lasso approach — penalising differences between adjacent factor level coefficients — requires a natural ordering. Ordinal factors (vehicle age, NCD years) have one; nominal factors (vehicle make, occupation) don't.

R2VF solves this in two steps:

**Step 1 — Ranking.** Fit a Ridge GLM on all factor dummies simultaneously. The resulting coefficients give a data-driven ordering for each nominal factor: levels with similar risk profiles end up adjacent, levels with different profiles end up far apart.

**Step 2 — Fusion.** Re-encode each nominal factor as ordinal using the Step 1 ranking. Apply a standard fused lasso (via the split-coding trick) to all factors. Where the fused lasso penalty drives adjacent-level differences to zero, those levels are merged.

**Step 3 — Refit.** Fit an unpenalised GLM on the merged groupings to remove shrinkage bias from Step 2.

The split-coding trick is what makes this practical without cvxpy or specialised solvers: transform the design matrix so that standard L1 (sklearn Lasso) achieves the fused lasso objective. No quadratic programming required.

## Installation

```bash
pip install insurance-glm-cluster
```

With the faster glum backend:
```bash
pip install insurance-glm-cluster[fast]
```

With plotting:
```bash
pip install insurance-glm-cluster[plot]
```

## Quick start

```python
from insurance_glm_cluster import FactorClusterer

clusterer = FactorClusterer(
    family='poisson',
    link='log',
    lambda_='bic',              # select regularisation via BIC
    min_exposure=500,           # merge groups with < 500 earned years
    monotone_factors=['ncd'],   # enforce NCD to be monotone decreasing
    monotone_direction={'ncd': 'decreasing'},
)

clusterer.fit(
    X,
    y,
    exposure=exposure,
    ordinal_factors=['vehicle_age', 'ncd'],
    nominal_factors=['vehicle_make', 'occupation'],
)

# Merged group codes — drop-in replacement for original columns
X_merged = clusterer.transform(X)

# Inspect the groupings
lm = clusterer.level_map('vehicle_make')
print(lm.to_df())
#   original_level  merged_group  coefficient  exposure
# 0          AUDI             0        -0.12    4521.3
# 1           BMW             0        -0.12    3892.1
# 2          FORD             1         0.08   18920.4
# ...

# Unpenalised GLM on merged factors
result = clusterer.refit_glm(X_merged, y, exposure=exposure)
print(result.summary())

# Diagnostics
diag = clusterer.diagnostics()
print(f"Vehicle make: {diag['n_levels_before']['vehicle_make']} → "
      f"{diag['n_levels_after']['vehicle_make']} groups")
print(f"AIC before: {diag['aic_before']:.1f}, after: {diag['aic_after']:.1f}")
```

## API

### `FactorClusterer`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `family` | str | `'poisson'` | GLM family: `'poisson'`, `'gamma'`, `'tweedie'` |
| `link` | str | `'log'` | Link function |
| `method` | str | `'r2vf'` | Clustering method (only `'r2vf'` in Phase 1) |
| `lambda_` | float \| `'bic'` | `'bic'` | Regularisation strength or BIC selection |
| `n_ordinal_bins` | int | 30 | Initial bins for numeric/ordinal factors |
| `m_nominal_bins` | int | 75 | Maximum dummy levels for nominal factors in Step 1 |
| `alpha` | float | 2.0 | 1.0 = Lasso, 2.0 = Ridge for nominal Step 1 |
| `min_exposure` | float | None | Minimum exposure per merged group |
| `min_claims` | int | None | Minimum claims per merged group |
| `monotone_factors` | list | `[]` | Factors to enforce monotonicity on |
| `monotone_direction` | dict | `{}` | Per-factor direction: `'increasing'` or `'decreasing'` |
| `backend` | str | `'statsmodels'` | GLM backend for refit: `'statsmodels'` or `'glum'` |
| `random_state` | int | 42 | Random seed |

### `LevelMap`

Returned by `clusterer.level_map(factor_name)`.

```python
lm.to_df()                      # DataFrame: original_level | merged_group | coefficient | exposure
lm.n_groups()                   # int: number of merged groups
lm.n_levels_original()          # int: original cardinality
lm.compression_ratio()          # float: levels / groups
lm.validate_monotone('increasing')  # bool
lm.plot()                       # matplotlib Figure (requires [plot] extra)
```

## Design decisions

**Why R2VF and not generalised fused lasso directly?**
GFL with all-pairs penalties is O(K²) in the number of levels. For 500 vehicle makes, that's 125,000 penalty terms. R2VF reduces this to O(K) by using the Step 1 ranking to impose an ordering, then running standard (1D) fused lasso.

**Why sklearn Lasso for the fusion step, not statsmodels?**
Statsmodels doesn't do L1 penalised GLMs. The split-coding trick converts the fused lasso into a standard L1 problem on a transformed design matrix, which sklearn Lasso solves efficiently via coordinate descent. The regression target is exposure-adjusted (y/exposure with exposure as sample weights) to approximate the Poisson log-likelihood within sklearn's Gaussian-only Lasso.

**Why BIC for lambda selection?**
Cross-validation on insurance data is methodologically awkward: policies across years are correlated, and CV folds will contain leakage from multi-year policyholders. BIC selects a lambda that balances fit and complexity in-sample, which is appropriate when the goal is factor grouping rather than held-out prediction.

**Why is the refit step separate from fit()?**
Actuaries need to review the groupings before committing to a refit. The `level_map()` output is designed for this: you can inspect, challenge, and manually adjust the groups before running `refit_glm()`. Keeping the steps separate also means the clustering output is backend-agnostic.

## References

- Ben Dror, I. (2025). *Variable Fusion for Insurance Pricing: R2VF Algorithm*. arXiv:2503.01521.
- Tibshirani, R. J., & Taylor, J. (2011). The solution path of the generalized lasso. *Annals of Statistics*, 39(3), 1335–1371.
