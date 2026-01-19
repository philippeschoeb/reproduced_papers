"""Nearest centroid QML reproduced paper library package."""

from .classifier import MLQuantumNearestCentroid, QuantumNearestCentroid
from .synthetic_data import generate_paper_datasets, generate_synthetic_data
from .defaults import default_config

__all__ = [
    "default_config",
    "QuantumNearestCentroid",
    "MLQuantumNearestCentroid",
    "generate_synthetic_data",
    "generate_paper_datasets",
]