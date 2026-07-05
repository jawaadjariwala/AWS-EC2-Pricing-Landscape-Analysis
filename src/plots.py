"""Reusable plotting helpers for the AWS pricing landscape analysis.

The plots in this project are designed against the visualization rules
from class: every chart has a clear title, axis labels with units, legible
ticks, no wasted ink, and natural numeric scales. Where data is dense or
multi-dimensional, color and marker shape encode additional dimensions
rather than relying on extra axes. Interactive plots (Plotly) are used
when filtering and zoom would otherwise be needed to read the chart.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VENDOR_COLOR_MAP = {
    "AWS Graviton": "#FF9900",
    "Intel": "#0071C5",
    "AMD": "#ED1C24",
    "Apple": "#A3AAAE",
    "Other": "#7F7F7F",
}


def configure_default_matplotlib_style() -> None:
    """Apply a consistent, readable Matplotlib style across every plot."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 140,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.autolayout": True,
        }
    )


def plot_price_vs_vcpu_by_vendor(
    enriched_dataframe: pd.DataFrame,
    region_code: str = "us-east-1",
    operating_system: str = "linux",
    axis: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot on-demand hourly price versus vCPU count, colored by vendor.

    Uses log scales on both axes because price and vCPU each span more
    than three orders of magnitude. The log-log view exposes the
    near-power-law relationship between vCPU count and price within each
    vendor's lineup.
    """
    region_mask = enriched_dataframe["region_code"] == region_code
    os_mask = enriched_dataframe["operating_system"] == operating_system
    plot_frame = enriched_dataframe[region_mask & os_mask].dropna(
        subset=["vcpu_count", "ondemand_usd_per_hour"]
    )
    if axis is None:
        _, axis = plt.subplots(figsize=(8.5, 5.5))
    for vendor_name, color_value in VENDOR_COLOR_MAP.items():
        vendor_frame = plot_frame[plot_frame["processor_vendor"] == vendor_name]
        if vendor_frame.empty:
            continue
        axis.scatter(
            vendor_frame["vcpu_count"],
            vendor_frame["ondemand_usd_per_hour"],
            s=14,
            alpha=0.55,
            color=color_value,
            label=vendor_name,
            edgecolors="none",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("vCPU count")
    axis.set_ylabel("On-demand price (USD / hour)")
    axis.set_title(
        f"EC2 on-demand price vs. vCPU count — {region_code}, {operating_system}"
    )
    axis.legend(title="Processor vendor", loc="upper left")
    return axis


def plot_efficiency_frontier(
    frontier_dataframe: pd.DataFrame,
    axis: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot the cost-efficiency frontier in (memory, price) space.

    Each instance is plotted as a dot; frontier instances are highlighted
    with a larger orange marker and connected with a step line that traces
    the lowest achievable price as memory increases. Dominated instances
    are shown in muted gray to give a sense of how much of the lineup
    sits above the frontier.
    """
    plot_frame = frontier_dataframe.dropna(
        subset=["memory_gib", "ondemand_usd_per_hour"]
    )
    dominated_frame = plot_frame[~plot_frame["is_on_frontier"]]
    frontier_frame = plot_frame[plot_frame["is_on_frontier"]].sort_values("memory_gib")
    if axis is None:
        _, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.scatter(
        dominated_frame["memory_gib"],
        dominated_frame["ondemand_usd_per_hour"],
        s=10,
        alpha=0.35,
        color="#9aa0a6",
        label=f"Dominated ({len(dominated_frame)})",
        edgecolors="none",
    )
    axis.scatter(
        frontier_frame["memory_gib"],
        frontier_frame["ondemand_usd_per_hour"],
        s=22,
        color="#FF9900",
        label=f"Pareto-optimal ({len(frontier_frame)})",
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    axis.step(
        frontier_frame["memory_gib"],
        frontier_frame["ondemand_usd_per_hour"],
        where="post",
        color="#FF9900",
        alpha=0.6,
        linewidth=1,
        zorder=2,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Memory (GiB)")
    axis.set_ylabel("On-demand price (USD / hour)")
    axis.set_title("Cost-efficiency Pareto frontier — us-east-1 Linux on-demand")
    axis.legend(loc="upper left")
    return axis


def plot_dominance_tax_distribution(
    frontier_dataframe: pd.DataFrame,
    axis: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a histogram of the dominance tax for dominated instances.

    Adds a vertical line at the median to support the "typical premium"
    statement in the report.
    """
    dominated_frame = frontier_dataframe[~frontier_dataframe["is_on_frontier"]]
    tax_values = dominated_frame["dominance_tax_pct"].to_numpy()
    if axis is None:
        _, axis = plt.subplots(figsize=(8.5, 4.5))
    axis.hist(
        tax_values,
        bins=40,
        color="#0071C5",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
    )
    median_tax = float(np.median(tax_values))
    axis.axvline(
        median_tax,
        color="#FF9900",
        linewidth=2,
        label=f"Median = {median_tax:.1f}%",
    )
    axis.set_xlabel("Dominance tax (% over cheapest equivalent-or-better)")
    axis.set_ylabel("Number of dominated instance types")
    axis.set_title("How much you overpay by choosing a dominated instance")
    axis.legend(loc="upper right")
    return axis


def plot_pca_spec_space(
    pca_components: np.ndarray,
    color_values: np.ndarray,
    color_label: str,
    title_suffix: str,
    axis: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Scatter PCA components colored by a continuous variable.

    Designed for 2-component PCA outputs of the (vCPU, RAM, network,
    storage, GPU) instance spec space, colored by price-per-vCPU or
    similar. Uses a log-normalized colormap when the color values span
    multiple orders of magnitude.
    """
    if axis is None:
        _, axis = plt.subplots(figsize=(8.5, 6.0))
    positive_color_values = color_values[color_values > 0]
    range_ratio = (
        positive_color_values.max() / positive_color_values.min()
        if positive_color_values.size
        else 1.0
    )
    if range_ratio > 50:
        floor_value = positive_color_values.min()
        clipped_values = np.where(color_values > 0, color_values, floor_value)
        color_values_for_plot = np.log10(clipped_values)
        colorbar_label = f"log₁₀ {color_label}"
    else:
        color_values_for_plot = color_values
        colorbar_label = color_label
    scatter_handle = axis.scatter(
        pca_components[:, 0],
        pca_components[:, 1],
        c=color_values_for_plot,
        cmap="viridis",
        s=18,
        alpha=0.8,
        edgecolors="none",
    )
    axis.set_xlabel("Principal component 1")
    axis.set_ylabel("Principal component 2")
    axis.set_title(f"PCA of EC2 instance spec space — {title_suffix}")
    colorbar = plt.colorbar(scatter_handle, ax=axis)
    colorbar.set_label(colorbar_label)
    return axis


def plot_vendor_generation_grid(
    vendor_pivot_dataframe: pd.DataFrame,
    polynomial_fits_by_vendor: dict,
    axis: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot cost-per-vCPU vs. family generation, one line per vendor.

    The pivot table must be indexed by ``family_generation`` with one
    column per vendor. A linear fit (in log-cost space) is overlaid for
    each vendor so the reader can see whether per-vCPU cost is rising,
    flat, or falling within each vendor across generations.
    """
    if axis is None:
        _, axis = plt.subplots(figsize=(9.0, 5.5))
    for vendor_name in vendor_pivot_dataframe.columns:
        if vendor_name not in VENDOR_COLOR_MAP:
            continue
        cost_series = vendor_pivot_dataframe[vendor_name].dropna()
        if cost_series.empty:
            continue
        color_value = VENDOR_COLOR_MAP[vendor_name]
        axis.scatter(
            cost_series.index,
            cost_series.values,
            s=70,
            color=color_value,
            label=vendor_name,
            zorder=3,
            edgecolors="white",
            linewidths=0.6,
        )
        if vendor_name in polynomial_fits_by_vendor:
            polynomial_coefficients = polynomial_fits_by_vendor[vendor_name]
            generation_axis = np.linspace(
                cost_series.index.min(), cost_series.index.max(), 100
            )
            fit_curve = np.exp(np.polyval(polynomial_coefficients, generation_axis))
            axis.plot(
                generation_axis,
                fit_curve,
                color=color_value,
                linewidth=1.8,
                alpha=0.7,
            )
    axis.set_yscale("log")
    integer_generations = sorted(
        int(generation_value) for generation_value in vendor_pivot_dataframe.index
    )
    axis.set_xticks(integer_generations)
    axis.set_xlabel("Family generation number")
    axis.set_ylabel("Cost per vCPU-hour (USD), 2xlarge")
    axis.set_title(
        "Per-vCPU cost across generations and vendors (M/C/R, 2xlarge, us-east-1)"
    )
    axis.legend(title="Processor vendor", loc="upper right")
    return axis
