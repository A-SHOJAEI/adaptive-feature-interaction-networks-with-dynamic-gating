"""Adaptive Feature Interaction Networks with Dynamic Gating.

A novel architecture for tabular data that learns to dynamically gate feature
interactions based on sample-specific importance scores.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from adaptive_feature_interaction_networks_with_dynamic_gating.models.model import (
    AdaptiveFeatureInteractionNetwork,
)

__all__ = ["AdaptiveFeatureInteractionNetwork"]
