"""
LevelMap: result container for a single factor's merged groupings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


@dataclass
class LevelMap:
    """
    Container for the merged groupings of a single GLM factor.

    Stores the original-to-merged mapping, group relativities, and
    per-group exposure. Produced by ``FactorClusterer.level_map()``.

    Parameters
    ----------
    factor_name : str
        Name of the factor (e.g. 'vehicle_make').
    mapping : dict
        Maps each original level to its merged group integer code.
    group_coefficients : pd.Series
        Unpenalised GLM coefficient for each merged group.
        Index is the group integer code.
    group_exposures : pd.Series
        Total exposure (or observation count) for each merged group.
        Index is the group integer code.
    original_levels : list
        All original levels in their Step 2 ordering (the ordering used for
        fusion). For ordinal factors this is the natural order; for nominals
        it is the R2VF Step 1 ranking.
    is_nominal : bool
        Whether the factor was treated as nominal in Step 1.
    """

    factor_name: str
    mapping: dict
    group_coefficients: pd.Series
    group_exposures: pd.Series
    original_levels: list
    is_nominal: bool = False

    def to_df(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per original level.

        Returns
        -------
        pd.DataFrame
            Columns: original_level, merged_group, coefficient, exposure.
            Sorted by merged_group, then by original level position.
        """
        rows = []
        for level, group in self.mapping.items():
            rows.append(
                {
                    "original_level": level,
                    "merged_group": group,
                    "coefficient": self.group_coefficients.get(group, np.nan),
                    "exposure": self.group_exposures.get(group, np.nan),
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["merged_group", "original_level"]).reset_index(
                drop=True
            )
        return df

    def n_groups(self) -> int:
        """Return the number of merged groups."""
        return int(self.group_coefficients.nunique())

    def n_levels_original(self) -> int:
        """Return the number of original levels before merging."""
        return len(self.mapping)

    def compression_ratio(self) -> float:
        """
        Return the ratio of original levels to merged groups.

        A ratio of 5.0 means 5 original levels per merged group on average.
        """
        n_grp = self.n_groups()
        if n_grp == 0:
            return float("nan")
        return self.n_levels_original() / n_grp

    def validate_monotone(
        self, direction: str = "increasing"
    ) -> bool:
        """
        Check whether merged group coefficients are monotone.

        Parameters
        ----------
        direction : str
            'increasing' or 'decreasing'.

        Returns
        -------
        bool
            True if the group coefficients (sorted by group code) satisfy
            the monotonicity constraint.
        """
        coef = self.group_coefficients.sort_index().values
        if direction == "increasing":
            return bool(np.all(np.diff(coef) >= 0))
        elif direction == "decreasing":
            return bool(np.all(np.diff(coef) <= 0))
        else:
            raise ValueError(f"Unknown direction '{direction}'.")

    def plot(
        self,
        figsize: tuple[float, float] = (10, 4),
        title: str | None = None,
        show_exposure: bool = True,
    ) -> "plt.Figure":  # type: ignore[name-defined]  # noqa: F821
        """
        Plot merged group relativities as a bar chart.

        Requires matplotlib. Bars are coloured by merged group; exposure is
        shown as a secondary axis if available.

        Parameters
        ----------
        figsize : tuple[float, float]
            Figure size in inches.
        title : str, optional
            Plot title. Defaults to the factor name.
        show_exposure : bool
            Whether to overlay a step-plot of per-level exposure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install insurance-glm-cluster[plot]"
            ) from exc

        df = self.to_df()
        if df.empty:
            raise ValueError("No level data to plot.")

        fig, ax = plt.subplots(figsize=figsize)

        groups = df["merged_group"].values
        n_groups = int(groups.max()) + 1
        colours = plt.cm.tab20(np.linspace(0, 1, max(n_groups, 2)))  # type: ignore[attr-defined]
        bar_colours = [colours[g % len(colours)] for g in groups]

        x_pos = np.arange(len(df))
        ax.bar(x_pos, df["coefficient"].values, color=bar_colours, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            df["original_level"].astype(str).values,
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel("GLM Coefficient (log scale)")
        ax.set_xlabel("Original Level")

        if show_exposure and df["exposure"].notna().any():
            ax2 = ax.twinx()
            ax2.step(
                x_pos,
                df["exposure"].fillna(0).values,
                color="grey",
                alpha=0.4,
                linewidth=1.5,
                where="mid",
            )
            ax2.set_ylabel("Exposure", color="grey")
            ax2.tick_params(axis="y", labelcolor="grey")

        plot_title = title or f"Factor: {self.factor_name}"
        plot_title += f" ({self.n_levels_original()} → {self.n_groups()} groups)"
        ax.set_title(plot_title)

        fig.tight_layout()
        return fig

    def __repr__(self) -> str:
        return (
            f"LevelMap(factor='{self.factor_name}', "
            f"levels={self.n_levels_original()}, "
            f"groups={self.n_groups()}, "
            f"nominal={self.is_nominal})"
        )
