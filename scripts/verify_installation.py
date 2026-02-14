#!/usr/bin/env python
"""Verification script to test project installation and basic functionality."""

import sys
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 60)
print("Adaptive Feature Interaction Networks - Installation Check")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    import torch
    import numpy as np
    import pandas as pd
    import sklearn
    import yaml
    print("   ✓ Core dependencies imported")
except ImportError as e:
    print(f"   ✗ Missing dependency: {e}")
    sys.exit(1)

try:
    from adaptive_feature_interaction_networks_with_dynamic_gating.models.model import (
        AdaptiveFeatureInteractionNetwork,
    )
    from adaptive_feature_interaction_networks_with_dynamic_gating.models.components import (
        DynamicGatingLayer,
        SparsityRegularizedLoss,
    )
    from adaptive_feature_interaction_networks_with_dynamic_gating.data.loader import (
        load_dataset,
    )
    from adaptive_feature_interaction_networks_with_dynamic_gating.training.trainer import (
        Trainer,
    )
    print("   ✓ Project modules imported")
except ImportError as e:
    print(f"   ✗ Failed to import project modules: {e}")
    sys.exit(1)

# Test model creation
print("\n2. Testing model creation...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptiveFeatureInteractionNetwork(
        input_dim=10,
        hidden_dims=[32, 16],
        num_gates=8,
        interaction_orders=[2],
        num_classes=1,
    )
    model = model.to(device)
    print(f"   ✓ Model created with {model.count_parameters():,} parameters")
    print(f"   ✓ Using device: {device}")
except Exception as e:
    print(f"   ✗ Failed to create model: {e}")
    sys.exit(1)

# Test forward pass
print("\n3. Testing forward pass...")
try:
    x = torch.randn(4, 10).to(device)
    logits, aux = model(x)
    assert logits.shape == (4, 1), f"Expected shape (4, 1), got {logits.shape}"
    assert "gate_scores" in aux
    assert "order_weights" in aux
    print(f"   ✓ Forward pass successful")
    print(f"   ✓ Output shape: {logits.shape}")
    print(f"   ✓ Auxiliary outputs: {list(aux.keys())}")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    sys.exit(1)

# Test data loading
print("\n4. Testing data loading...")
try:
    X_df, y_series, metadata = load_dataset(
        dataset_name="synthetic",
        n_samples=100,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )
    assert X_df.shape == (100, 20)
    assert len(y_series) == 100
    print(f"   ✓ Dataset loaded: {X_df.shape[0]} samples, {X_df.shape[1]} features")
except Exception as e:
    print(f"   ✗ Data loading failed: {e}")
    sys.exit(1)

# Test config loading
print("\n5. Testing configuration...")
try:
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    assert "model" in config
    assert "training" in config
    assert "data" in config
    print(f"   ✓ Configuration loaded from {config_path.name}")
except Exception as e:
    print(f"   ✗ Config loading failed: {e}")
    sys.exit(1)

# Test loss function
print("\n6. Testing loss function...")
try:
    criterion = SparsityRegularizedLoss(task="binary", sparsity_weight=0.01)
    logits = torch.randn(4, 1).to(device)
    targets = torch.randint(0, 2, (4,)).to(device)
    gate_scores = torch.randn(4, 8).to(device)
    loss, loss_dict = criterion(logits, targets, gate_scores)
    assert "base_loss" in loss_dict
    assert "sparsity_loss" in loss_dict
    print(f"   ✓ Loss computation successful")
    print(f"   ✓ Loss components: {list(loss_dict.keys())}")
except Exception as e:
    print(f"   ✗ Loss computation failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All checks passed! Installation verified successfully.")
print("=" * 60)
print("\nNext steps:")
print("  - Run training: python scripts/train.py")
print("  - Run tests: pytest tests/ -v")
print("  - See README.md for more information")
print("=" * 60)
