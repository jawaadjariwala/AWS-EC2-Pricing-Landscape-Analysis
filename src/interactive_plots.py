"""Interactive Plotly visualizations of EC2 pricing.

Two interactive figures are provided:

1. ``build_regional_price_map`` plots every commercial AWS region as a
   point on a world map, with marker color encoding the regional price
   premium versus a reference region. Hovering reveals the exact premium
   and instance count.

2. ``build_instance_explorer`` plots every EC2 instance as a point in
   (memory, on-demand price) space, colored by processor vendor and
   sized by vCPU count. Filtering and zoom are essential here because
   the catalog has more than a thousand instance types in the
   us-east-1/Linux slice alone.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_regional_price_map(
    regional_table: pd.DataFrame,
    title: Optional[str] = None,
) -> go.Figure:
    """Build an interactive world map of regional price premiums.

    The ``regional_table`` is the output of
    ``geo_pricing.compute_regional_price_index`` and must include the
    ``latitude``, ``longitude``, ``city``, ``price_premium_pct``, and
    ``instances_offered`` columns.
    """
    plot_title = title or (
        "AWS regional price premium vs. us-east-1 "
        "(median Linux on-demand cost per vCPU-hour)"
    )
    figure = px.scatter_geo(
        regional_table,
        lat="latitude",
        lon="longitude",
        color="price_premium_pct",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        size="instances_offered",
        size_max=22,
        hover_name="city",
        hover_data={
            "region_code": True,
            "price_premium_pct": ":+.1f",
            "instances_offered": True,
            "latitude": False,
            "longitude": False,
        },
        projection="natural earth",
        title=plot_title,
    )
    figure.update_layout(
        coloraxis_colorbar=dict(title="Premium vs.<br>us-east-1 (%)"),
        margin=dict(l=10, r=10, t=60, b=10),
        height=520,
    )
    figure.update_geos(showcountries=True, countrycolor="#cccccc", showcoastlines=False)
    return figure


def build_instance_explorer(
    enriched_dataframe: pd.DataFrame,
    region_code: str = "us-east-1",
    operating_system: str = "linux",
    title: Optional[str] = None,
) -> go.Figure:
    """Build an interactive scatter of instances in (memory, price) space.

    Marker color encodes processor vendor, marker size encodes vCPU
    count, and the hover tooltip surfaces the instance type and
    price-per-vCPU. Both axes are log-scaled because both span more
    than three orders of magnitude across the lineup.
    """
    region_mask = enriched_dataframe["region_code"] == region_code
    os_mask = enriched_dataframe["operating_system"] == operating_system
    plot_frame = enriched_dataframe[region_mask & os_mask].dropna(
        subset=[
            "memory_gib",
            "ondemand_usd_per_hour",
            "vcpu_count",
            "cost_per_vcpu_hour_usd",
        ]
    )
    figure_title = title or (
        f"Interactive EC2 instance explorer — {region_code}, "
        f"{operating_system} on-demand"
    )
    figure = px.scatter(
        plot_frame,
        x="memory_gib",
        y="ondemand_usd_per_hour",
        color="processor_vendor",
        size="vcpu_count",
        size_max=22,
        hover_name="instance_type",
        hover_data={
            "vcpu_count": True,
            "memory_gib": ":.1f",
            "ondemand_usd_per_hour": ":.4f",
            "cost_per_vcpu_hour_usd": ":.4f",
            "family_label": True,
            "processor_vendor": False,
        },
        log_x=True,
        log_y=True,
        color_discrete_map={
            "AWS Graviton": "#FF9900",
            "Intel": "#0071C5",
            "AMD": "#ED1C24",
            "Apple": "#A3AAAE",
            "Other": "#7F7F7F",
        },
        title=figure_title,
        labels={
            "memory_gib": "Memory (GiB)",
            "ondemand_usd_per_hour": "On-demand price (USD / hour)",
            "processor_vendor": "Processor vendor",
            "vcpu_count": "vCPU count",
            "cost_per_vcpu_hour_usd": "Cost per vCPU-hour (USD)",
            "family_label": "AWS family label",
        },
    )
    figure.update_layout(
        margin=dict(l=10, r=10, t=60, b=10),
        height=560,
        legend=dict(title="Processor vendor"),
    )
    return figure
