"""Reduce the EC2 instance specification space to two dimensions via PCA.

The EC2 catalog gives every instance a five-to-seven dimensional spec
vector (vCPU count, memory, network bandwidth, EBS throughput, GPU
count, GPU memory, etc.). Plotting this directly is impossible without
hiding most of the structure. Principal component analysis projects the
catalog into the two directions of greatest variance in the standardized
spec space so the entire instance lineup can be shown on a single
scatter plot, with price-per-vCPU encoded as the color dimension.

This module is careful to apply PCA only after standardization (z-score
scaling), since the raw spec columns span very different units (cores,
gibibytes, gigabits) and an unscaled PCA would collapse to whichever
column has the largest absolute magnitudes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DEFAULT_SPEC_COLUMNS = (
    "vcpu_count",
    "memory_gib",
    "network_bandwidth_gbps",
    "ebs_baseline_throughput_mbps",
    "gpu_count",
)


def project_spec_space_to_two_components(
    enriched_dataframe: pd.DataFrame,
    region_code: str = "us-east-1",
    operating_system: str = "linux",
    spec_columns: tuple = DEFAULT_SPEC_COLUMNS,
) -> tuple[np.ndarray, pd.DataFrame, PCA]:
    """Run two-component PCA on the standardized instance spec matrix.

    Returns a tuple of:

    1. ``components`` : ``(n_instances, 2)`` array of PC1, PC2 coordinates.
    2. ``aligned_dataframe`` : the rows of the input that survived the
       dropna step, in the same order as ``components``. This frame still
       carries the spec, price, vendor, and family columns and can be
       used directly for color/marker encoding.
    3. ``pca_model`` : the fitted scikit-learn PCA object, useful for
       reporting explained variance and component loadings.
    """
    region_mask = enriched_dataframe["region_code"] == region_code
    os_mask = enriched_dataframe["operating_system"] == operating_system
    sliced = enriched_dataframe[region_mask & os_mask].copy()
    aligned_dataframe = sliced.dropna(
        subset=list(spec_columns) + ["cost_per_vcpu_hour_usd"]
    ).reset_index(drop=True)
    spec_matrix = aligned_dataframe[list(spec_columns)].to_numpy(dtype=float)
    scaler = StandardScaler()
    standardized_matrix = scaler.fit_transform(spec_matrix)
    pca_model = PCA(n_components=2)
    components = pca_model.fit_transform(standardized_matrix)
    return components, aligned_dataframe, pca_model


def describe_pca_loadings(
    pca_model: PCA,
    spec_columns: tuple = DEFAULT_SPEC_COLUMNS,
) -> pd.DataFrame:
    """Return a tidy DataFrame of PCA component loadings per spec column.

    The loadings reveal what each principal component "means" in terms
    of the original spec columns. Component 1 typically captures overall
    instance size; component 2 typically distinguishes compute-heavy
    from memory-heavy instances along an orthogonal axis.
    """
    loadings_matrix = pca_model.components_
    loadings_dataframe = pd.DataFrame(
        loadings_matrix.T,
        index=list(spec_columns),
        columns=["principal_component_1", "principal_component_2"],
    )
    loadings_dataframe.index.name = "spec_column"
    return loadings_dataframe.reset_index()
