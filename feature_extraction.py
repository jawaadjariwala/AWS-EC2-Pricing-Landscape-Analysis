"""Derive structured features from raw EC2 instance metadata.

The Vantage dataset gives us free-text fields (``physical_processor`` like
"Intel Xeon 8375C (Ice Lake)", ``network_performance_text`` like
"Up to 25 Gigabit") and the AWS instance type string (e.g., "m6i.2xlarge")
that compactly encodes family, generation, and processor vendor in its name.
This module parses those fields into clean numeric and categorical columns
suitable for plotting, clustering, and frontier analysis.

The naming convention parsed here is documented in the AWS EC2 documentation
(see https://docs.aws.amazon.com/ec2/latest/instancetypes/instance-type-names.html).
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

INSTANCE_TYPE_PATTERN = re.compile(
    r"^(?P<family>[a-z]+?)(?P<generation>\d+)?(?P<attributes>[a-z\-]*)\.(?P<size>[a-z0-9\-]+)$"
)

PROCESSOR_VENDOR_PATTERNS = [
    ("AWS Graviton", "AWS Graviton"),
    ("AMD EPYC", "AMD"),
    ("Intel Xeon", "Intel"),
    ("Apple M", "Apple"),
]

GRAVITON_GENERATION_PATTERN = re.compile(r"Graviton(\d+)?", re.IGNORECASE)


def parse_instance_type_name(instance_type: str) -> dict:
    """Parse an EC2 instance type string into structured components.

    Examples
    --------
    >>> parse_instance_type_name("m6i.2xlarge")
    {'family_code': 'm', 'family_generation': 6, 'attributes': 'i', 'size_class': '2xlarge'}
    >>> parse_instance_type_name("c7gn.16xlarge")
    {'family_code': 'c', 'family_generation': 7, 'attributes': 'gn', 'size_class': '16xlarge'}
    >>> parse_instance_type_name("hpc7a.96xlarge")
    {'family_code': 'hpc', 'family_generation': 7, 'attributes': 'a', 'size_class': '96xlarge'}
    """
    match = INSTANCE_TYPE_PATTERN.match(instance_type)
    if not match:
        return {
            "family_code": None,
            "family_generation": None,
            "attributes": None,
            "size_class": None,
        }
    parts = match.groupdict()
    generation_text = parts.get("generation")
    return {
        "family_code": parts["family"],
        "family_generation": (
            int(generation_text) if generation_text is not None else None
        ),
        "attributes": parts.get("attributes") or "",
        "size_class": parts["size"],
    }


def classify_processor_vendor(physical_processor_text: Optional[str]) -> str:
    """Classify a free-text processor description into a vendor label.

    Returns one of "AWS Graviton", "Intel", "AMD", "Apple", or "Other".
    """
    if not isinstance(physical_processor_text, str):
        return "Other"
    for needle, vendor in PROCESSOR_VENDOR_PATTERNS:
        if needle.lower() in physical_processor_text.lower():
            return vendor
    return "Other"


def extract_graviton_generation(
    physical_processor_text: Optional[str],
) -> Optional[int]:
    """Extract the Graviton generation number from a processor description.

    AWS Graviton processors are labeled "AWS Graviton2 Processor",
    "AWS Graviton3 Processor", etc. The original Graviton has no number.
    Returns None for non-Graviton processors and 1 for the original Graviton.
    """
    if classify_processor_vendor(physical_processor_text) != "AWS Graviton":
        return None
    match = GRAVITON_GENERATION_PATTERN.search(physical_processor_text)
    if not match:
        return None
    generation_text = match.group(1)
    return int(generation_text) if generation_text else 1


def parse_network_bandwidth_gbps(
    network_performance_text: Optional[str],
) -> Optional[float]:
    """Parse a free-text network performance description into Gbps.

    AWS uses strings like "Up to 25 Gigabit", "10 Gigabit",
    "100 Gigabit", and qualitative labels like "Low" or "Moderate" for
    older instance types. This parser returns the numeric Gbps value when
    present and None otherwise.
    """
    if not isinstance(network_performance_text, str):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*Gigabit", network_performance_text)
    if match:
        return float(match.group(1))
    if "Mbps" in network_performance_text or "Megabit" in network_performance_text:
        mbps_match = re.search(r"(\d+(?:\.\d+)?)", network_performance_text)
        if mbps_match:
            return float(mbps_match.group(1)) / 1000.0
    return None


def parse_clock_speed_ghz(clock_speed_text: Optional[str]) -> Optional[float]:
    """Parse a clock speed string like '2.5 GHz' into a float."""
    if not isinstance(clock_speed_text, str):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", clock_speed_text)
    return float(match.group(1)) if match else None


def enrich_pricing_dataframe(pricing_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add derived feature columns to the tidy pricing DataFrame.

    Adds the following columns by parsing the existing fields:

    - ``family_code`` and ``family_generation`` from the instance type name
    - ``attributes`` (the AWS suffix string like "g", "i", "n", "d")
    - ``size_class`` (e.g., "2xlarge", "metal")
    - ``processor_vendor`` ("AWS Graviton", "Intel", "AMD", "Apple", "Other")
    - ``graviton_generation`` (1, 2, 3, 4 for Graviton instances; None otherwise)
    - ``network_bandwidth_gbps`` (numeric Gbps where parseable)
    - ``clock_speed_ghz`` (numeric GHz where parseable)
    - ``cost_per_vcpu_hour_usd`` (on-demand price divided by vCPU count)
    - ``cost_per_gib_hour_usd`` (on-demand price divided by memory in GiB)

    The input DataFrame is not modified; a new DataFrame is returned.
    """
    enriched = pricing_dataframe.copy()
    parsed_names = enriched["instance_type"].apply(parse_instance_type_name)
    parsed_frame = pd.DataFrame(parsed_names.to_list())
    for column_name in parsed_frame.columns:
        enriched[column_name] = parsed_frame[column_name].to_numpy()
    enriched["processor_vendor"] = enriched["physical_processor"].apply(
        classify_processor_vendor
    )
    enriched["graviton_generation"] = enriched["physical_processor"].apply(
        extract_graviton_generation
    )
    enriched["network_bandwidth_gbps"] = enriched["network_performance_text"].apply(
        parse_network_bandwidth_gbps
    )
    enriched["clock_speed_ghz"] = enriched["clock_speed_ghz_text"].apply(
        parse_clock_speed_ghz
    )
    safe_vcpu = enriched["vcpu_count"].replace(0, np.nan)
    safe_memory = enriched["memory_gib"].replace(0, np.nan)
    enriched["cost_per_vcpu_hour_usd"] = enriched["ondemand_usd_per_hour"] / safe_vcpu
    enriched["cost_per_gib_hour_usd"] = enriched["ondemand_usd_per_hour"] / safe_memory
    return enriched


if __name__ == "__main__":
    from src.pricing_loader import load_pricing_dataframe

    pricing_dataframe = load_pricing_dataframe()
    enriched_dataframe = enrich_pricing_dataframe(pricing_dataframe)
    print("Enriched columns:", list(enriched_dataframe.columns))
    print(
        "\nProcessor vendor counts:\n",
        enriched_dataframe["processor_vendor"].value_counts(),
    )
    print(
        "\nFamily code counts:\n",
        enriched_dataframe["family_code"].value_counts().head(15),
    )
    print(
        "\nNetwork bandwidth distribution (Gbps):",
        enriched_dataframe["network_bandwidth_gbps"].describe(),
    )
