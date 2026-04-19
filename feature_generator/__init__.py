"""Feature generator for time-series stock prediction with headline signals."""

from feature_generator.generator import generate_features
from feature_generator.training_builder import (
    build_training_rows,
    build_training_dataset,
    compute_sample_weights,
)

__all__ = [
    "generate_features",
    "build_training_rows",
    "build_training_dataset",
    "compute_sample_weights",
]
