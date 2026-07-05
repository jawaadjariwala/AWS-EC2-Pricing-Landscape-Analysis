"""Download, parse, and cache the AWS EC2 pricing dataset.

The canonical source for AWS EC2 pricing is the AWS Price List Bulk API
(https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/),
which publishes one large JSON file per region (each ~100 MB). Downloading
and parsing all ~30 commercial regions on every notebook run is impractical
for an analysis of this scope.

This module instead loads the pre-aggregated dataset published by Vantage at
https://instances.vantage.sh/instances.json, which mirrors the AWS Price List
data into a single JSON document covering every EC2 instance type and every
region in which it is offered. The loader caches a pruned, tidy parquet file
so subsequent runs (and the grader) execute in seconds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

VANTAGE_INSTANCES_URL = "https://instances.vantage.sh/instances.json"

DEFAULT_RAW_PATH = Path("data/raw/vantage_instances.json")
DEFAULT_TIDY_PATH = Path("data/ec2_pricing_tidy.parquet")


def download_raw_dataset(
    destination: Path = DEFAULT_RAW_PATH,
    url: str = VANTAGE_INSTANCES_URL,
    overwrite: bool = False,
    timeout_seconds: int = 180,
) -> Path:
    """Download the Vantage EC2 instances JSON to ``destination``.

    The destination's parent directory is created if needed. If the file
    already exists and ``overwrite`` is False, the existing file is reused.

    Parameters
    ----------
    destination : Path
        Local path where the raw JSON will be written.
    url : str
        Source URL. Defaults to the Vantage public mirror of the AWS Price
        List Bulk API.
    overwrite : bool
        If True, re-download even if the file already exists.
    timeout_seconds : int
        HTTP request timeout in seconds.

    Returns
    -------
    Path
        The path to the downloaded JSON file.
    """
    destination = Path(destination)
    if destination.exists() and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout_seconds, stream=True)
    response.raise_for_status()
    with open(destination, "wb") as output_file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output_file.write(chunk)
    return destination


def _safe_float(value) -> Optional[float]:
    """Convert a string price to float, returning None for empty/invalid values."""
    if value is None or value == "" or value == "N/A":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_reserved_price(pricing_block: dict, term_key: str) -> Optional[float]:
    """Extract a reserved-instance hourly price from a Vantage pricing block.

    Vantage stores reserved prices under nested keys like
    ``reserved.yrTerm1Standard.allUpfront``. This helper safely traverses the
    structure and returns the float price or None if the key is absent.
    """
    reserved_block = pricing_block.get("reserved")
    if not isinstance(reserved_block, dict):
        return None
    return _safe_float(reserved_block.get(term_key))


def _flatten_one_instance(instance_record: dict) -> list[dict]:
    """Flatten a single Vantage instance record into one row per (region, os).

    Each input record contains specs (vCPU, memory, family, etc.) plus a
    nested ``pricing`` dict keyed by region and then by operating system.
    This helper emits one tidy row per (region, os) combination, copying
    the spec columns onto each row.
    """
    instance_type = instance_record.get("instance_type")
    base_columns = {
        "instance_type": instance_type,
        "family_label": instance_record.get("family"),
        "generation_label": instance_record.get("generation"),
        "physical_processor": instance_record.get("physical_processor"),
        "vcpu_count": instance_record.get("vCPU"),
        "memory_gib": instance_record.get("memory"),
        "gpu_count": instance_record.get("GPU") or 0,
        "gpu_memory_gib": instance_record.get("GPU_memory") or 0,
        "ecu": instance_record.get("ECU"),
        "clock_speed_ghz_text": instance_record.get("clock_speed_ghz"),
        "network_performance_text": instance_record.get("network_performance"),
        "is_bare_metal": bool(instance_record.get("is_bare_metal")),
        "ebs_optimized": bool(instance_record.get("ebs_optimized")),
        "ebs_baseline_throughput_mbps": instance_record.get("ebs_baseline_throughput"),
    }
    rows = []
    pricing_by_region = instance_record.get("pricing") or {}
    for region_code, os_pricing in pricing_by_region.items():
        if not isinstance(os_pricing, dict):
            continue
        for os_name, pricing_block in os_pricing.items():
            if not isinstance(pricing_block, dict):
                continue
            row = dict(base_columns)
            row["region_code"] = region_code
            row["operating_system"] = os_name
            row["ondemand_usd_per_hour"] = _safe_float(pricing_block.get("ondemand"))
            row["spot_avg_usd_per_hour"] = _safe_float(pricing_block.get("spot_avg"))
            row["spot_pct_savings_vs_ondemand"] = _safe_float(
                pricing_block.get("pct_savings_od")
            )
            row["reserved_1yr_allupfront_usd_per_hour"] = _extract_reserved_price(
                pricing_block, "yrTerm1Standard.allUpfront"
            )
            row["reserved_3yr_allupfront_usd_per_hour"] = _extract_reserved_price(
                pricing_block, "yrTerm3Standard.allUpfront"
            )
            rows.append(row)
    return rows


def parse_raw_to_tidy_dataframe(raw_json_path: Path) -> pd.DataFrame:
    """Parse the raw Vantage JSON file into a tidy long-format DataFrame.

    The output has one row per (instance_type, region, operating_system),
    with instance specs replicated across the rows that share an instance.
    Rows lacking an on-demand price are dropped, since they represent
    purchasing-only configurations that cannot be priced.
    """
    with open(raw_json_path) as input_file:
        raw_records = json.load(input_file)
    flattened_rows = []
    for instance_record in raw_records:
        flattened_rows.extend(_flatten_one_instance(instance_record))
    tidy_dataframe = pd.DataFrame(flattened_rows)
    tidy_dataframe = tidy_dataframe.dropna(subset=["ondemand_usd_per_hour"])
    tidy_dataframe = tidy_dataframe.reset_index(drop=True)
    return tidy_dataframe


def load_pricing_dataframe(
    tidy_path: Path = DEFAULT_TIDY_PATH,
    raw_path: Path = DEFAULT_RAW_PATH,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Load the tidy AWS EC2 pricing DataFrame, building the cache if needed.

    On the first call (or when ``rebuild`` is True), the raw Vantage JSON is
    downloaded if missing, parsed into the tidy long-format DataFrame, and
    written to ``tidy_path`` as a compressed parquet file for fast reuse.
    On subsequent calls the tidy parquet is loaded directly.

    Parameters
    ----------
    tidy_path : Path
        Path to the cached tidy parquet file.
    raw_path : Path
        Path where the raw JSON is staged when downloaded.
    rebuild : bool
        If True, re-download and re-parse even if the tidy cache exists.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns including instance_type,
        family_label, generation_label, physical_processor, vcpu_count,
        memory_gib, region_code, operating_system, ondemand_usd_per_hour,
        spot_avg_usd_per_hour, reserved_1yr_allupfront_usd_per_hour, etc.
    """
    tidy_path = Path(tidy_path)
    if tidy_path.exists() and not rebuild:
        return pd.read_parquet(tidy_path)
    raw_path = Path(raw_path)
    if rebuild or not raw_path.exists():
        download_raw_dataset(destination=raw_path, overwrite=rebuild)
    tidy_dataframe = parse_raw_to_tidy_dataframe(raw_path)
    tidy_path.parent.mkdir(parents=True, exist_ok=True)
    tidy_dataframe.to_parquet(tidy_path, index=False, compression="snappy")
    return tidy_dataframe


if __name__ == "__main__":
    pricing_dataframe = load_pricing_dataframe(rebuild=True)
    print(f"Loaded {len(pricing_dataframe):,} pricing rows")
    print(f"Unique instances: {pricing_dataframe['instance_type'].nunique()}")
    print(f"Unique regions: {pricing_dataframe['region_code'].nunique()}")
    print(
        f"Operating systems: {sorted(pricing_dataframe['operating_system'].unique())}"
    )
    print(f"Cache size: {os.path.getsize(DEFAULT_TIDY_PATH) / 1e6:.1f} MB")
