"""
Tests for level_map.py: LevelMap container, to_df(), and validate_monotone().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_glm_cluster.level_map import LevelMap


def make_level_map(
    factor_name: str = "vehicle_make",
    n_levels: int = 6,
    n_groups: int = 3,
    is_nominal: bool = True,
) -> LevelMap:
    """Helper: build a LevelMap with controlled structure."""
    levels = [f"L{i}" for i in range(n_levels)]
    # Assign first 2 → group 0, next 2 → group 1, last 2 → group 2
    mapping = {}
    for i, lv in enumerate(levels):
        mapping[lv] = i // 2

    n_groups_actual = len(set(mapping.values()))
    group_coef = pd.Series(
        {g: g * 0.3 for g in range(n_groups_actual)}
    )
    group_exp = pd.Series(
        {g: 100.0 * (g + 1) for g in range(n_groups_actual)}
    )

    return LevelMap(
        factor_name=factor_name,
        mapping=mapping,
        group_coefficients=group_coef,
        group_exposures=group_exp,
        original_levels=levels,
        is_nominal=is_nominal,
    )


class TestLevelMapBasics:
    def test_n_groups(self):
        lm = make_level_map(n_levels=6, n_groups=3)
        assert lm.n_groups() == 3

    def test_n_levels_original(self):
        lm = make_level_map(n_levels=6)
        assert lm.n_levels_original() == 6

    def test_compression_ratio(self):
        lm = make_level_map(n_levels=6, n_groups=3)
        assert abs(lm.compression_ratio() - 2.0) < 1e-10

    def test_repr_contains_factor_name(self):
        lm = make_level_map(factor_name="ncd")
        assert "ncd" in repr(lm)

    def test_repr_shows_level_and_group_counts(self):
        lm = make_level_map(n_levels=6, n_groups=3)
        r = repr(lm)
        assert "6" in r
        assert "3" in r


class TestLevelMapToDF:
    def test_to_df_columns(self):
        lm = make_level_map()
        df = lm.to_df()
        assert set(df.columns) == {"original_level", "merged_group", "coefficient", "exposure"}

    def test_to_df_row_count(self):
        lm = make_level_map(n_levels=6)
        df = lm.to_df()
        assert len(df) == 6

    def test_to_df_sorted_by_group(self):
        lm = make_level_map(n_levels=6)
        df = lm.to_df()
        assert list(df["merged_group"]) == sorted(df["merged_group"].tolist())

    def test_to_df_coefficients_match_groups(self):
        lm = make_level_map(n_levels=6, n_groups=3)
        df = lm.to_df()
        # Group 0 should have coefficient 0.0, group 1 → 0.3, group 2 → 0.6
        for _, row in df.iterrows():
            expected_coef = row["merged_group"] * 0.3
            assert abs(row["coefficient"] - expected_coef) < 1e-10

    def test_to_df_exposures_match_groups(self):
        lm = make_level_map(n_levels=6, n_groups=3)
        df = lm.to_df()
        for _, row in df.iterrows():
            expected_exp = (row["merged_group"] + 1) * 100.0
            assert abs(row["exposure"] - expected_exp) < 1e-10

    def test_to_df_empty_mapping(self):
        lm = LevelMap(
            factor_name="empty",
            mapping={},
            group_coefficients=pd.Series(dtype=float),
            group_exposures=pd.Series(dtype=float),
            original_levels=[],
        )
        df = lm.to_df()
        assert len(df) == 0


class TestValidateMonotone:
    def test_increasing_monotone_passes(self):
        lm = make_level_map(n_levels=6, n_groups=3)
        # Coefficients are 0.0, 0.3, 0.6 — monotone increasing
        assert lm.validate_monotone("increasing") is True

    def test_increasing_violated_fails(self):
        lm = LevelMap(
            factor_name="test",
            mapping={"A": 0, "B": 1, "C": 2},
            group_coefficients=pd.Series({0: 0.5, 1: 0.1, 2: 0.9}),
            group_exposures=pd.Series({0: 10.0, 1: 10.0, 2: 10.0}),
            original_levels=["A", "B", "C"],
        )
        assert lm.validate_monotone("increasing") is False

    def test_decreasing_monotone_passes(self):
        lm = LevelMap(
            factor_name="test",
            mapping={"A": 0, "B": 1, "C": 2},
            group_coefficients=pd.Series({0: 0.9, 1: 0.5, 2: 0.1}),
            group_exposures=pd.Series({0: 10.0, 1: 10.0, 2: 10.0}),
            original_levels=["A", "B", "C"],
        )
        assert lm.validate_monotone("decreasing") is True

    def test_invalid_direction_raises(self):
        lm = make_level_map()
        with pytest.raises(ValueError):
            lm.validate_monotone("sideways")


class TestLevelMapSingleGroup:
    def test_all_levels_same_group(self):
        lm = LevelMap(
            factor_name="make",
            mapping={"A": 0, "B": 0, "C": 0},
            group_coefficients=pd.Series({0: 0.2}),
            group_exposures=pd.Series({0: 300.0}),
            original_levels=["A", "B", "C"],
        )
        assert lm.n_groups() == 1
        assert lm.n_levels_original() == 3
        assert lm.compression_ratio() == 3.0

    def test_to_df_with_single_group(self):
        lm = LevelMap(
            factor_name="make",
            mapping={"A": 0, "B": 0},
            group_coefficients=pd.Series({0: 0.5}),
            group_exposures=pd.Series({0: 200.0}),
            original_levels=["A", "B"],
        )
        df = lm.to_df()
        assert len(df) == 2
        assert (df["merged_group"] == 0).all()
