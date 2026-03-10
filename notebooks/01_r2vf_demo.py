# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-glm-cluster: R2VF Demo
# MAGIC
# MAGIC This notebook demonstrates the full R2VF workflow on a synthetic motor insurance dataset:
# MAGIC
# MAGIC 1. Generate a realistic synthetic dataset (Poisson frequency)
# MAGIC 2. Fit FactorClusterer with BIC lambda selection
# MAGIC 3. Inspect the LevelMap for each factor
# MAGIC 4. Refit the unpenalised GLM on merged groups
# MAGIC 5. Compare AIC/BIC before and after
# MAGIC
# MAGIC The synthetic data is constructed so that the true groupings are known,
# MAGIC letting you verify that the algorithm recovers them.

# COMMAND ----------

# MAGIC %pip install insurance-glm-cluster matplotlib

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic data
# MAGIC
# MAGIC Motor frequency model: claim count ~ Poisson(exposure * rate)
# MAGIC
# MAGIC **Factors:**
# MAGIC - `vehicle_age` (ordinal, 0–9): true groupings are {0,1}, {2,3,4}, {5,6,7,8,9}
# MAGIC - `vehicle_make` (nominal, 20 makes): true groupings are {A–E} low, {F–L} medium, {M–T} high

# COMMAND ----------

rng = np.random.default_rng(42)
N = 20_000

# Vehicle age: 10 levels, 3 true groups
vehicle_age = rng.integers(0, 10, size=N)
age_log_rates = {
    0: -0.4, 1: -0.4,                          # group 0: young, high risk
    2: 0.0, 3: 0.0, 4: 0.0,                    # group 1: prime age
    5: 0.2, 6: 0.2, 7: 0.2, 8: 0.2, 9: 0.2,   # group 2: older, lower risk
}

# Vehicle make: 20 nominal levels (A-T), 3 true groups
makes = list("ABCDEFGHIJKLMNOPQRST")
make_labels_map = {
    **{m: -0.3 for m in "ABCDE"},     # low risk
    **{m: 0.0 for m in "FGHIJKLM"},   # medium risk
    **{m: 0.4 for m in "NOPQRST"},    # high risk
}
vehicle_make = rng.choice(makes, size=N)

exposure = rng.uniform(0.3, 2.0, size=N)
log_mu = (
    np.array([age_log_rates[a] for a in vehicle_age])
    + np.array([make_labels_map[m] for m in vehicle_make])
)
mu = exposure * np.exp(log_mu)
claim_count = rng.poisson(mu)

X = pd.DataFrame({
    "vehicle_age": vehicle_age,
    "vehicle_make": vehicle_make,
})
y = claim_count.astype(float)

print(f"Dataset: {N:,} policies, {int(y.sum()):,} claims, "
      f"claim frequency {y.sum() / exposure.sum():.4f}")
print(f"\nVehicle age levels: {sorted(X['vehicle_age'].unique())}")
print(f"Vehicle make levels: {sorted(X['vehicle_make'].unique())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit FactorClusterer

# COMMAND ----------

from insurance_glm_cluster import FactorClusterer

clusterer = FactorClusterer(
    family='poisson',
    link='log',
    method='r2vf',
    lambda_='bic',           # select regularisation via BIC
    n_ordinal_bins=30,       # cap ordinal binning at 30
    m_nominal_bins=75,       # cap nominal dummies for Step 1 at 75
    alpha=2.0,               # Ridge for Step 1 (preserves full ranking)
    min_exposure=None,
    random_state=42,
)

clusterer.fit(
    X, y, exposure=exposure,
    ordinal_factors=["vehicle_age"],
    nominal_factors=["vehicle_make"],
)

print(f"Selected lambda: {clusterer._selected_lambda:.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. BIC curve
# MAGIC
# MAGIC The BIC curve shows how many groups are optimal across the lambda grid.

# COMMAND ----------

diag = clusterer.diagnostics()
bic_curve = diag["bic_curve"]
# Reconstruct the lambda grid from the selected lambda position
lam_min_ratio = 0.001
n_points = 50
# Approximate grid for plotting
lam_max_approx = clusterer._selected_lambda / (lam_min_ratio ** (np.argmin(bic_curve) / (n_points - 1)))
lam_grid_plot = np.exp(np.linspace(np.log(lam_max_approx), np.log(lam_max_approx * lam_min_ratio), n_points))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(np.log(lam_grid_plot), bic_curve, 'b-', linewidth=1.5)
ax.axvline(np.log(clusterer._selected_lambda), color='red', linestyle='--', label=f'Selected λ={clusterer._selected_lambda:.5f}')
ax.set_xlabel("log(λ)")
ax.set_ylabel("BIC")
ax.set_title("BIC vs Regularisation Strength")
ax.legend()
fig.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Inspect vehicle_age level map

# COMMAND ----------

lm_age = clusterer.level_map("vehicle_age")
print(f"Vehicle age: {lm_age.n_levels_original()} levels → {lm_age.n_groups()} groups")
print(f"Compression ratio: {lm_age.compression_ratio():.1f}x")
print()
print(lm_age.to_df().to_string(index=False))

# COMMAND ----------

# Plot vehicle_age groupings
fig = lm_age.plot(figsize=(10, 4), title="Vehicle Age Merged Groups")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify vehicle_age recovery
# MAGIC
# MAGIC True groupings: {0,1}, {2,3,4}, {5,6,7,8,9}

# COMMAND ----------

df_age = lm_age.to_df().sort_values("original_level")
print("Level → Group mapping:")
for _, row in df_age.iterrows():
    true_group = (
        "group_0 (low risk)"  if row["original_level"] in [0, 1] else
        "group_1 (mid risk)"  if row["original_level"] in [2, 3, 4] else
        "group_2 (lower risk)"
    )
    print(f"  Age {row['original_level']:2d} → cluster {row['merged_group']} | "
          f"coef={row['coefficient']:+.3f} | true={true_group}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Inspect vehicle_make level map

# COMMAND ----------

lm_make = clusterer.level_map("vehicle_make")
print(f"Vehicle make: {lm_make.n_levels_original()} levels → {lm_make.n_groups()} groups")
print()
df_make = lm_make.to_df().sort_values(["merged_group", "original_level"])
print(df_make.to_string(index=False))

# COMMAND ----------

# True vs recovered groupings for vehicle make
print("\nTrue vs recovered groups:")
print(f"{'Make':>5} | {'True group':>12} | {'Recovered group':>15} | {'Coef':>8}")
print("-" * 50)
for make in sorted(makes):
    true_grp = "low" if make in "ABCDE" else ("medium" if make in "FGHIJKLM" else "high")
    rec_grp = lm_make.mapping.get(make, -1)
    coef = lm_make.group_coefficients.get(rec_grp, float('nan'))
    print(f"{make:>5} | {true_grp:>12} | {rec_grp:>15} | {coef:>+8.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Transform and refit unpenalised GLM

# COMMAND ----------

X_merged = clusterer.transform(X)
print("Merged DataFrame (first 10 rows):")
print(X_merged.head(10))
print(f"\nUnique vehicle_age groups: {sorted(X_merged['vehicle_age'].dropna().unique())}")
print(f"Unique vehicle_make groups: {sorted(X_merged['vehicle_make'].dropna().unique())}")

# COMMAND ----------

result = clusterer.refit_glm(X_merged, y, exposure=exposure)
print(result.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Diagnostics: before vs after

# COMMAND ----------

diag = clusterer.diagnostics()

print("=" * 60)
print("CLUSTERING DIAGNOSTICS")
print("=" * 60)

print("\nLevel reduction:")
for factor in ["vehicle_age", "vehicle_make"]:
    before = diag["n_levels_before"][factor]
    after = diag["n_levels_after"][factor]
    print(f"  {factor}: {before} → {after} groups ({before/after:.1f}x compression)")

print(f"\nGLM fit quality:")
print(f"  AIC  before: {diag['aic_before']:.1f}")
print(f"  AIC  after:  {diag['aic_after']:.1f}")
print(f"  BIC  before: {diag['bic_before']:.1f}")
print(f"  BIC  after:  {diag['bic_after']:.1f}")
print(f"  Deviance before: {diag['deviance_before']:.1f}")
print(f"  Deviance after:  {diag['deviance_after']:.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Monotonicity example
# MAGIC
# MAGIC For vehicle age, we expect higher-age vehicles to be lower risk (they're
# MAGIC typically driven less, by more experienced drivers). Enforce this.

# COMMAND ----------

clusterer_mono = FactorClusterer(
    family='poisson',
    lambda_='bic',
    monotone_factors=['vehicle_age'],
    monotone_direction={'vehicle_age': 'increasing'},
    random_state=42,
)
clusterer_mono.fit(
    X, y, exposure=exposure,
    ordinal_factors=['vehicle_age'],
    nominal_factors=['vehicle_make'],
)

lm_age_mono = clusterer_mono.level_map('vehicle_age')
print("Monotone-enforced vehicle age coefficients:")
df_mono = lm_age_mono.to_df().sort_values("original_level")
print(df_mono[["original_level", "merged_group", "coefficient"]].to_string(index=False))
print(f"\nMonotone increasing: {lm_age_mono.validate_monotone('increasing')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Min-exposure example
# MAGIC
# MAGIC Force every merged group to have at least 1000 earned years.

# COMMAND ----------

clusterer_me = FactorClusterer(
    family='poisson',
    lambda_='bic',
    min_exposure=1000.0,
    random_state=42,
)
clusterer_me.fit(
    X, y, exposure=exposure,
    ordinal_factors=['vehicle_age'],
    nominal_factors=['vehicle_make'],
)

print("Vehicle make groups with min_exposure=1000:")
lm_make_me = clusterer_me.level_map('vehicle_make')
df_make_me = lm_make_me.to_df().sort_values(["merged_group", "original_level"])
print(df_make_me[["original_level", "merged_group", "exposure"]].to_string(index=False))
print(f"\nGroups: {lm_make_me.n_groups()} (vs {lm_make.n_groups()} without constraint)")
