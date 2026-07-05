"""Aggregate EC2 pricing across regions and project regions onto a world map.

AWS publishes a single list price per (instance, region, OS), but those
prices vary substantially across the ~30 commercial regions. The variation
is generally believed to reflect a mix of local electricity costs, real
estate, demand, regulatory burden, and AWS's own pricing strategy, but the
variation itself is rarely visualized in aggregate.

This module reduces the per-instance regional prices into a single
"regional price index" — the median price-per-vCPU within a chosen
operating system — and pairs that index with hand-curated approximate
geographic coordinates for each AWS region. The coordinates are taken from
the AWS public documentation of region locations and are sufficient for
dot-on-map visualizations; precise data-center coordinates are not
disclosed by AWS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REGION_COORDINATES = {
    "us-east-1": {"city": "N. Virginia, USA", "latitude": 38.13, "longitude": -78.45},
    "us-east-2": {"city": "Ohio, USA", "latitude": 40.42, "longitude": -82.91},
    "us-west-1": {
        "city": "N. California, USA",
        "latitude": 37.35,
        "longitude": -121.96,
    },
    "us-west-2": {"city": "Oregon, USA", "latitude": 45.84, "longitude": -119.7},
    "ca-central-1": {"city": "Montreal, Canada", "latitude": 45.5, "longitude": -73.6},
    "ca-west-1": {"city": "Calgary, Canada", "latitude": 51.05, "longitude": -114.07},
    "sa-east-1": {"city": "Sao Paulo, Brazil", "latitude": -23.55, "longitude": -46.63},
    "eu-west-1": {"city": "Ireland", "latitude": 53.35, "longitude": -6.26},
    "eu-west-2": {"city": "London, UK", "latitude": 51.51, "longitude": -0.13},
    "eu-west-3": {"city": "Paris, France", "latitude": 48.86, "longitude": 2.35},
    "eu-central-1": {
        "city": "Frankfurt, Germany",
        "latitude": 50.11,
        "longitude": 8.68,
    },
    "eu-central-2": {
        "city": "Zurich, Switzerland",
        "latitude": 47.38,
        "longitude": 8.54,
    },
    "eu-north-1": {"city": "Stockholm, Sweden", "latitude": 59.33, "longitude": 18.07},
    "eu-south-1": {"city": "Milan, Italy", "latitude": 45.46, "longitude": 9.19},
    "eu-south-2": {"city": "Spain", "latitude": 40.42, "longitude": -3.7},
    "me-south-1": {"city": "Bahrain", "latitude": 26.07, "longitude": 50.55},
    "me-central-1": {"city": "UAE", "latitude": 24.45, "longitude": 54.38},
    "il-central-1": {"city": "Tel Aviv, Israel", "latitude": 32.08, "longitude": 34.78},
    "af-south-1": {
        "city": "Cape Town, South Africa",
        "latitude": -33.92,
        "longitude": 18.42,
    },
    "ap-east-1": {"city": "Hong Kong", "latitude": 22.32, "longitude": 114.17},
    "ap-northeast-1": {"city": "Tokyo, Japan", "latitude": 35.68, "longitude": 139.69},
    "ap-northeast-2": {
        "city": "Seoul, S. Korea",
        "latitude": 37.57,
        "longitude": 126.98,
    },
    "ap-northeast-3": {"city": "Osaka, Japan", "latitude": 34.69, "longitude": 135.5},
    "ap-southeast-1": {"city": "Singapore", "latitude": 1.35, "longitude": 103.82},
    "ap-southeast-2": {
        "city": "Sydney, Australia",
        "latitude": -33.87,
        "longitude": 151.21,
    },
    "ap-southeast-3": {
        "city": "Jakarta, Indonesia",
        "latitude": -6.21,
        "longitude": 106.85,
    },
    "ap-southeast-4": {
        "city": "Melbourne, Australia",
        "latitude": -37.81,
        "longitude": 144.96,
    },
    "ap-southeast-5": {"city": "Malaysia", "latitude": 3.14, "longitude": 101.69},
    "ap-southeast-7": {"city": "Thailand", "latitude": 13.75, "longitude": 100.5},
    "ap-south-1": {"city": "Mumbai, India", "latitude": 19.08, "longitude": 72.88},
    "ap-south-2": {"city": "Hyderabad, India", "latitude": 17.39, "longitude": 78.49},
    "mx-central-1": {"city": "Mexico", "latitude": 19.43, "longitude": -99.13},
}


def restrict_to_geocoded_commercial_regions(
    enriched_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Drop rows whose region is not in the commercial geocoded set.

    This excludes AWS GovCloud, China, Local Zones, and Wavelength Zones,
    which are priced separately and would distort the global comparison.
    """
    geocoded_codes = set(REGION_COORDINATES.keys())
    return enriched_dataframe[
        enriched_dataframe["region_code"].isin(geocoded_codes)
    ].reset_index(drop=True)


def compute_regional_price_index(
    enriched_dataframe: pd.DataFrame,
    operating_system: str = "linux",
    reference_region: str = "us-east-1",
) -> pd.DataFrame:
    """Compute a per-region price index relative to a reference region.

    The price index is the median ``cost_per_vcpu_hour_usd`` over all
    instance types currently offered in a region for the given OS,
    expressed as a ratio to the reference region's median. Restricting
    to a single OS keeps the comparison fair (Windows and Linux have
    very different median costs).

    Returns a DataFrame indexed by region code with columns:

    - ``city`` : human-readable location label
    - ``latitude``, ``longitude`` : approximate region coordinates
    - ``median_cost_per_vcpu_hour_usd`` : raw median cost per vCPU
    - ``instances_offered`` : number of instance types priced in this region
    - ``price_index`` : ratio to the reference region (1.0 = parity)
    - ``price_premium_pct`` : (price_index - 1) * 100, signed percentage
      premium (positive = more expensive than reference)
    """
    geocoded_dataframe = restrict_to_geocoded_commercial_regions(enriched_dataframe)
    os_dataframe = geocoded_dataframe[
        geocoded_dataframe["operating_system"] == operating_system
    ]
    grouped = os_dataframe.groupby("region_code").agg(
        median_cost_per_vcpu_hour_usd=("cost_per_vcpu_hour_usd", "median"),
        instances_offered=("instance_type", "nunique"),
    )
    coordinates_frame = pd.DataFrame.from_dict(REGION_COORDINATES, orient="index")
    regional_table = grouped.join(coordinates_frame, how="inner")
    reference_value = regional_table.loc[
        reference_region, "median_cost_per_vcpu_hour_usd"
    ]
    regional_table["price_index"] = (
        regional_table["median_cost_per_vcpu_hour_usd"] / reference_value
    )
    regional_table["price_premium_pct"] = 100.0 * (regional_table["price_index"] - 1.0)
    regional_table = regional_table.reset_index().rename(
        columns={"index": "region_code"}
    )
    return regional_table


def compare_instance_across_regions(
    enriched_dataframe: pd.DataFrame,
    instance_type: str,
    operating_system: str = "linux",
) -> pd.DataFrame:
    """Return the on-demand prices for a single instance across all regions.

    Useful for "spot the cheapest region for this exact instance" lookups
    and for showing concrete dollar differences alongside the aggregate
    price index.
    """
    geocoded_dataframe = restrict_to_geocoded_commercial_regions(enriched_dataframe)
    instance_mask = geocoded_dataframe["instance_type"] == instance_type
    os_mask = geocoded_dataframe["operating_system"] == operating_system
    instance_frame = geocoded_dataframe[instance_mask & os_mask][
        ["region_code", "ondemand_usd_per_hour"]
    ].copy()
    coordinates_frame = (
        pd.DataFrame.from_dict(REGION_COORDINATES, orient="index")
        .reset_index()
        .rename(columns={"index": "region_code"})
    )
    enriched_frame = instance_frame.merge(
        coordinates_frame, on="region_code", how="inner"
    )
    cheapest_price = enriched_frame["ondemand_usd_per_hour"].min()
    enriched_frame["premium_vs_cheapest_pct"] = (
        100.0
        * (enriched_frame["ondemand_usd_per_hour"] - cheapest_price)
        / cheapest_price
    )
    return enriched_frame.sort_values("ondemand_usd_per_hour").reset_index(drop=True)


if __name__ == "__main__":
    from src.feature_extraction import enrich_pricing_dataframe
    from src.pricing_loader import load_pricing_dataframe

    enriched_dataframe = enrich_pricing_dataframe(load_pricing_dataframe())
    regional_table = compute_regional_price_index(enriched_dataframe)
    print("Regional price index (linux, vs us-east-1):")
    print(
        regional_table[["region_code", "city", "price_index", "price_premium_pct"]]
        .sort_values("price_index")
        .to_string(index=False)
    )
    print("\nSample instance comparison (m5.large) across regions:")
    instance_comparison = compare_instance_across_regions(
        enriched_dataframe, "m5.large"
    )
    print(instance_comparison.head(10).to_string(index=False))
