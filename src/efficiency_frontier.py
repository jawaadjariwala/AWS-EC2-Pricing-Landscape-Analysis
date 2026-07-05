"""Compute the cost-efficiency Pareto frontier across EC2 instance types.

When an organization shops for compute capacity, the relevant question is
rarely "which instance type is cheapest?" but rather "given that I need at
least *V* vCPUs and *M* GiB of memory, which instance is cheapest?". This
turns instance selection into a multi-objective optimization problem in
(vCPU, RAM, price) space.

We solve this with the standard Pareto-dominance test: instance A is
*dominated* by instance B if B has at least as much vCPU and at least as
much memory as A while costing strictly less per hour. The set of
non-dominated instances forms the *efficient frontier* — for every
workload requirement there is at least one frontier instance that is
optimal. Every dominated instance is one that no rational shopper should
ever choose.

This module computes the frontier and quantifies the "tax" paid by
choosing a dominated instance — the percentage of hourly cost that could
be saved by switching to a frontier instance with at least as much
capacity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _select_pricing_slice(
    enriched_dataframe: pd.DataFrame,
    region_code: str,
    operating_system: str,
) -> pd.DataFrame:
    """Filter the enriched DataFrame to a single (region, OS) slice.

    The Pareto frontier is only meaningful within a single region and OS,
    since prices vary on both axes. This helper isolates the slice and
    drops rows lacking the required spec columns.
    """
    region_mask = enriched_dataframe["region_code"] == region_code
    os_mask = enriched_dataframe["operating_system"] == operating_system
    sliced = enriched_dataframe[region_mask & os_mask].copy()
    sliced = sliced.dropna(subset=["vcpu_count", "memory_gib", "ondemand_usd_per_hour"])
    sliced = sliced[sliced["ondemand_usd_per_hour"] > 0]
    return sliced.reset_index(drop=True)


def _compute_dominance_mask(
    vcpu_array: np.ndarray,
    memory_array: np.ndarray,
    price_array: np.ndarray,
) -> np.ndarray:
    """Return a boolean mask of non-dominated rows (the Pareto frontier).

    For each row i, row i is dominated if there exists row j with
    ``vcpu[j] >= vcpu[i]`` and ``memory[j] >= memory[i]`` and
    ``price[j] < price[i]``. The frontier is the complement of the
    dominated set. Implementation is fully vectorized via NumPy
    broadcasting, which is much faster than a Python double loop on
    datasets of ~1000 instances.
    """
    vcpu_geq = vcpu_array[None, :] >= vcpu_array[:, None]
    memory_geq = memory_array[None, :] >= memory_array[:, None]
    price_lt = price_array[None, :] < price_array[:, None]
    is_dominated = (vcpu_geq & memory_geq & price_lt).any(axis=1)
    return ~is_dominated


def compute_efficiency_frontier(
    enriched_dataframe: pd.DataFrame,
    region_code: str = "us-east-1",
    operating_system: str = "linux",
) -> pd.DataFrame:
    """Compute the cost-efficiency Pareto frontier for a region and OS.

    Returns a copy of the (region, OS) slice with two new columns:

    - ``is_on_frontier`` : True when the instance is Pareto-optimal in
      (vCPU, RAM, on-demand price) space.
    - ``cheapest_dominator_price_usd`` : for dominated instances, the
      minimum on-demand price among instances that dominate this one
      (i.e., the price of the cheapest equally-or-better alternative).
      Equal to the row's own price for frontier instances.
    - ``dominance_tax_pct`` : the percentage premium paid relative to the
      cheapest dominator. Zero for frontier instances; a positive value
      for dominated instances quantifying the avoidable premium.
    """
    pricing_slice = _select_pricing_slice(
        enriched_dataframe, region_code, operating_system
    )
    vcpu_array = pricing_slice["vcpu_count"].to_numpy(dtype=float)
    memory_array = pricing_slice["memory_gib"].to_numpy(dtype=float)
    price_array = pricing_slice["ondemand_usd_per_hour"].to_numpy(dtype=float)
    frontier_mask = _compute_dominance_mask(vcpu_array, memory_array, price_array)
    pricing_slice["is_on_frontier"] = frontier_mask
    vcpu_geq = vcpu_array[None, :] >= vcpu_array[:, None]
    memory_geq = memory_array[None, :] >= memory_array[:, None]
    weakly_dominates = vcpu_geq & memory_geq
    candidate_prices = np.where(weakly_dominates, price_array[None, :], np.inf)
    cheapest_dominator_price = candidate_prices.min(axis=1)
    pricing_slice["cheapest_dominator_price_usd"] = cheapest_dominator_price
    pricing_slice["dominance_tax_pct"] = (
        100.0 * (price_array - cheapest_dominator_price) / price_array
    )
    pricing_slice.loc[pricing_slice["is_on_frontier"], "dominance_tax_pct"] = 0.0
    return pricing_slice.reset_index(drop=True)


def summarize_frontier_membership_by_vendor(
    frontier_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize how each processor vendor populates the frontier.

    Returns a DataFrame with per-vendor counts of frontier and
    dominated instances and the average dominance tax for the dominated
    portion. This is one of the actionable insights of the analysis: it
    quantifies which vendor's instance lineup is most cost-efficient.
    """
    grouping = frontier_dataframe.groupby("processor_vendor")
    summary = grouping.agg(
        total_instance_count=("instance_type", "count"),
        frontier_instance_count=("is_on_frontier", "sum"),
        mean_dominance_tax_pct=("dominance_tax_pct", "mean"),
        median_dominance_tax_pct=("dominance_tax_pct", "median"),
    )
    summary["frontier_share_pct"] = (
        100.0 * summary["frontier_instance_count"] / summary["total_instance_count"]
    )
    return summary.sort_values("frontier_share_pct", ascending=False)


if __name__ == "__main__":
    from src.feature_extraction import enrich_pricing_dataframe
    from src.pricing_loader import load_pricing_dataframe

    enriched_dataframe = enrich_pricing_dataframe(load_pricing_dataframe())
    frontier_dataframe = compute_efficiency_frontier(enriched_dataframe)
    print(f"Total instances in us-east-1 linux: {len(frontier_dataframe)}")
    print(f"On-frontier instances: {frontier_dataframe['is_on_frontier'].sum()}")
    dominated_only = frontier_dataframe[~frontier_dataframe["is_on_frontier"]]
    print(
        f"Mean dominance tax (dominated): "
        f"{dominated_only['dominance_tax_pct'].mean():.1f}%"
    )
    print("\nFrontier share by vendor:")
    print(summarize_frontier_membership_by_vendor(frontier_dataframe))
