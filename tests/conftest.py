"""Pytest configuration and fixtures."""

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def sample_data():
    """Generate sample tabular data for testing.

    Returns:
        Tuple of (X_df, y_series) where X_df is features and y_series is labels
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    return X_df, y_series


@pytest.fixture
def sample_config():
    """Generate sample configuration for testing.

    Returns:
        Configuration dictionary
    """
    return {
        "seed": 42,
        "data": {
            "dataset_name": "synthetic",
            "n_samples": 1000,
            "n_features": 20,
            "n_classes": 2,
        },
        "model": {
            "hidden_dims": [64, 32],
            "num_gates": 16,
            "interaction_orders": [2],
            "gate_hidden_dim": 32,
            "gate_temperature": 0.5,
            "dropout": 0.2,
        },
        "training": {
            "num_epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
    }


@pytest.fixture
def device():
    """Get device for testing.

    Returns:
        torch.device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_model(device):
    """Create a sample model for testing.

    Args:
        device: Device to create model on

    Returns:
        AdaptiveFeatureInteractionNetwork instance
    """
    from adaptive_feature_interaction_networks_with_dynamic_gating.models.model import (
        AdaptiveFeatureInteractionNetwork,
    )

    model = AdaptiveFeatureInteractionNetwork(
        input_dim=10,
        hidden_dims=[32, 16],
        num_gates=8,
        interaction_orders=[2],
        num_classes=1,
        dropout=0.2,
    )
    return model.to(device)
