"""Quantify cost-per-vCPU deflation across EC2 instance generations.

AWS releases a new generation of each instance family roughly every two
to three years, typically with newer processors that improve performance
and reduce price-per-vCPU. The purpose of this module is to quantify
that deflation rate so engineers can put a number on the cost of running
a stale instance generation longer than necessary.

The analysis fits a polynomial in log-cost space against the family
generation number, restricted to the M, C, and R families which together
account for the bulk of general-purpose, compute-optimized, and
memory-optimized workloads. Polynomial fitting on the log of cost
linearizes the multiplicative nature of price decreases and exposes a
near-constant per-generation discount factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CORE_FAMILY_CODES = ("m", "c", "r")

CANONICAL_VENDOR_SUFFIXES = {
    "Intel": ("", "i"),
    "AMD": ("a",),
    "AWS Graviton": ("g",),
}


def summarize_cost_per_vcpu_by_generation(
    enriched_dataframe: pd.DataFrame,
    region_code: str = "us-east-1",
    operating_system: str = "linux",
    family_codes: tuple = CORE_FAMILY_CODES,
) -> pd.DataFrame:
    """Compute median cost-per-vCPU per family generation.

    Filters to a single (region, OS) and a chosen set of family codes,
    then groups by ``family_generation`` and reports the median of
    ``cost_per_vcpu_hour_usd`` along with the count of instance types
    contributing to each generation. The 25th and 75th percentiles are
    also reported so the report can show uncertainty bounds.
    """
    region_mask = enriched_dataframe["region_code"] == region_code
    os_mask = enriched_dataframe["operating_system"] == operating_system
    family_mask = enriched_dataframe["family_code"].isin(family_codes)
    sliced = enriched_dataframe[region_mask & os_mask & family_mask].copy()
    sliced = sliced.dropna(subset=["family_generation", "cost_per_vcpu_hour_usd"])
    grouped = sliced.groupby("family_generation").agg(
        median_cost_per_vcpu_hour_usd=("cost_per_vcpu_hour_usd", "median"),
        p25_cost_per_vcpu_hour_usd=(
            "cost_per_vcpu_hour_usd",
            lambda series: series.quantile(0.25),
        ),
        p75_cost_per_vcpu_hour_usd=(
            "cost_per_vcpu_hour_usd",
            lambda series: series.quantile(0.75),
        ),
        instance_count=("instance_type", "nunique"),
    )
    grouped = grouped.reset_index()
    grouped = grouped.sort_values("family_generation").reset_index(drop=True)
    return grouped


def fit_log_cost_polynomial(
    generation_summary: pd.DataFrame,
    polynomial_degree: int = 1,
) -> tuple[np.ndarray, float]:
    """Fit a polynomial to log(cost) vs. generation number.

    A linear fit (degree 1) corresponds to a constant per-generation
    multiplicative discount and is the most defensible model unless the
    residuals show a clear curvature. Returns the polynomial coefficients
    (highest degree first, in the form expected by ``np.polyval``) and
    the implied per-generation deflation factor (e.g., 0.85 means each
    new generation costs 15% less per vCPU on average).
    """
    generation_values = generation_summary["family_generation"].to_numpy(dtype=float)
    cost_values = generation_summary["median_cost_per_vcpu_hour_usd"].to_numpy(
        dtype=float
    )
    log_cost_values = np.log(cost_values)
    polynomial_coefficients = np.polyfit(
        generation_values, log_cost_values, polynomial_degree
    )
    per_generation_slope = polynomial_coefficients[-2]
    per_generation_deflation_factor = float(np.exp(per_generation_slope))
    return polynomial_coefficients, per_generation_deflation_factor


def compare_vendors_at_matched_size(
    enriched_dataframe: pd.DataFrame,
    region_code: str = "us-east-1",
    operating_system: str = "linux",
    size_class: str = "2xlarge",
    family_codes: tuple = CORE_FAMILY_CODES,
) -> pd.DataFrame:
    """Compare cost per vCPU across vendors and generations at fixed size.

    Restricting to a single size class (e.g., ``"2xlarge"``) holds vCPU
    count and memory roughly constant across the comparison, so any
    remaining variation in ``cost_per_vcpu_hour_usd`` is attributable to
    the processor vendor and the family generation rather than to
    differences in instance size or memory ratio. Only the *canonical*
    suffix variants for each vendor are retained (Intel: no suffix or
    "i", AMD: "a", Graviton: "g") to avoid mixing in storage-optimised
    or network-optimised variants whose prices reflect those extras.
    """
    region_mask = enriched_dataframe["region_code"] == region_code
    os_mask = enriched_dataframe["operating_system"] == operating_system
    size_mask = enriched_dataframe["size_class"] == size_class
    family_mask = enriched_dataframe["family_code"].isin(family_codes)
    sliced = enriched_dataframe[region_mask & os_mask & size_mask & family_mask].copy()
    canonical_rows = []
    for vendor, allowed_suffixes in CANONICAL_VENDOR_SUFFIXES.items():
        vendor_mask = sliced["processor_vendor"] == vendor
        suffix_mask = sliced["attributes"].isin(allowed_suffixes)
        canonical_rows.append(sliced[vendor_mask & suffix_mask])
    canonical_dataframe = pd.concat(canonical_rows, ignore_index=True)
    canonical_dataframe = (
        canonical_dataframe[
            [
                "instance_type",
                "family_code",
                "family_generation",
                "processor_vendor",
                "attributes",
                "vcpu_count",
                "memory_gib",
                "ondemand_usd_per_hour",
                "cost_per_vcpu_hour_usd",
            ]
        ]
        .sort_values(["family_code", "family_generation", "processor_vendor"])
        .reset_index(drop=True)
    )
    return canonical_dataframe


def summarize_vendor_comparison(
    vendor_comparison_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate the matched-size comparison into a (generation, vendor) grid.

    Returns a wide-format DataFrame indexed by family_generation with one
    column per vendor, holding the median cost per vCPU-hour. Useful for
    the "is the deflation real, or is it Graviton substitution?" plot in
    the report.
    """
    pivoted = vendor_comparison_dataframe.pivot_table(
        index="family_generation",
        columns="processor_vendor",
        values="cost_per_vcpu_hour_usd",
        aggfunc="median",
    )
    return pivoted.sort_index()


if __name__ == "__main__":
    from src.feature_extraction import enrich_pricing_dataframe
    from src.pricing_loader import load_pricing_dataframe

    enriched_dataframe = enrich_pricing_dataframe(load_pricing_dataframe())
    generation_summary = summarize_cost_per_vcpu_by_generation(enriched_dataframe)
    print("Cost per vCPU by family generation (M/C/R, us-east-1, linux):")
    print(generation_summary.to_string(index=False))
    polynomial_coefficients, deflation_factor = fit_log_cost_polynomial(
        generation_summary
    )
    print(
        f"\nPer-generation cost factor: {deflation_factor:.3f} "
        f"({(1 - deflation_factor) * 100:.1f}% cheaper per generation)"
    )
