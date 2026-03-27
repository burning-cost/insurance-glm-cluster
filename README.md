⚠️ **This package has been merged into [`insurance-glm-tools`](https://github.com/burning-cost/insurance-glm-tools).** This repository is archived. Install `insurance-glm-tools` instead.

# insurance-glm-cluster

> **DEPRECATED.** This repository is archived. All functionality has been reconciled into [insurance-glm-tools](https://github.com/burning-cost/insurance-glm-tools), which is the canonical home for GLM factor clustering going forward.
>
> **Migrate by replacing:**
> ```python
> # Old
> from insurance_glm_cluster import FactorClusterer
> from insurance_glm_cluster.constraints import enforce_min_claims, enforce_monotonicity, check_monotonicity
> from insurance_glm_cluster.utils import build_split_coding_matrix, apply_split_coding
>
> # New
> from insurance_glm_tools.cluster import FactorClusterer
> from insurance_glm_tools.cluster import enforce_min_claims, enforce_monotonicity, check_monotonicity
> from insurance_glm_tools.cluster import build_split_coding_matrix, apply_split_coding
> ```
>
> The `insurance-glm-tools` cluster subpackage is a superset of this library: it has a better `DiagnosticPath`, exposure-weighted coefficient averaging in the constraint enforcement, and the full R2VF `FactorClusterer` API. Everything unique to this repo (`enforce_min_claims`, `enforce_monotonicity`, `check_monotonicity`, `build_split_coding_matrix`, `apply_split_coding`) was ported in full on 2026-03-14.

---

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

Use `insurance-glm-tools` instead. This package is no longer maintained.

```bash
pip install insurance-glm-tools
```

## References

- Ben Dror, I. (2025). *Variable Fusion for Insurance Pricing: R2VF Algorithm*. arXiv:2503.01521.
- Tibshirani, R. J., & Taylor, J. (2011). The solution path of the generalized lasso. *Annals of Statistics*, 39(3), 1335–1371.
