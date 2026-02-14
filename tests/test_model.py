"""Tests for model architecture and components."""

import numpy as np
import pytest
import torch

from adaptive_feature_interaction_networks_with_dynamic_gating.models.model import (
    AdaptiveFeatureInteractionNetwork,
)
from adaptive_feature_interaction_networks_with_dynamic_gating.models.components import (
    DynamicGatingLayer,
    FeatureInteractionLayer,
    InteractionOrderSelector,
    SparsityRegularizedLoss,
    GradientBoostingEnsemble,
)


class TestDynamicGatingLayer:
    """Tests for dynamic gating layer."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = DynamicGatingLayer(
            input_dim=10, num_gates=8, hidden_dim=32
        )
        assert layer.input_dim == 10
        assert layer.num_gates == 8

    def test_forward_pass(self, device):
        """Test forward pass."""
        layer = DynamicGatingLayer(
            input_dim=10, num_gates=8, hidden_dim=32
        ).to(device)

        x = torch.randn(16, 10).to(device)
        gated_scores, raw_gates = layer(x)

        assert gated_scores.shape == (16, 8)
        assert raw_gates.shape == (16, 8)
        # Check softmax property
        assert torch.allclose(gated_scores.sum(dim=1), torch.ones(16).to(device))

    def test_temperature_effect(self, device):
        """Test that temperature affects sparsity."""
        layer_low_temp = DynamicGatingLayer(
            input_dim=10, num_gates=8, temperature=0.1
        ).to(device)
        layer_high_temp = DynamicGatingLayer(
            input_dim=10, num_gates=8, temperature=2.0
        ).to(device)

        x = torch.randn(16, 10).to(device)
        scores_low, _ = layer_low_temp(x)
        scores_high, _ = layer_high_temp(x)

        # Lower temperature should produce more sparse distribution
        entropy_low = -(scores_low * torch.log(scores_low + 1e-8)).sum(dim=1).mean()
        entropy_high = -(scores_high * torch.log(scores_high + 1e-8)).sum(dim=1).mean()
        assert entropy_low < entropy_high


class TestFeatureInteractionLayer:
    """Tests for feature interaction layer."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = FeatureInteractionLayer(
            input_dim=5, interaction_order=2, output_dim=32
        )
        assert layer.input_dim == 5
        assert layer.interaction_order == 2

    def test_pairwise_interactions(self, device):
        """Test pairwise interaction computation."""
        layer = FeatureInteractionLayer(
            input_dim=5, interaction_order=2, output_dim=32
        ).to(device)

        x = torch.randn(8, 5).to(device)
        interactions = layer(x)

        assert interactions.shape == (8, 32)

    def test_triplet_interactions(self, device):
        """Test triplet interaction computation."""
        layer = FeatureInteractionLayer(
            input_dim=5, interaction_order=3, output_dim=64
        ).to(device)

        x = torch.randn(8, 5).to(device)
        interactions = layer(x)

        assert interactions.shape == (8, 64)

    def test_interaction_count(self):
        """Test correct number of interactions."""
        layer = FeatureInteractionLayer(
            input_dim=5, interaction_order=2, output_dim=32, use_projection=False
        )

        # 5 features: 5 + C(5,2) = 5 + 10 = 15 features
        expected_pairwise = (5 * 4) // 2
        assert layer.num_pairwise == expected_pairwise


class TestInteractionOrderSelector:
    """Tests for interaction order selector."""

    def test_initialization(self):
        """Test selector initialization."""
        selector = InteractionOrderSelector(input_dim=10, num_orders=3)
        assert selector.input_dim == 10
        assert selector.num_orders == 3

    def test_forward_pass(self, device):
        """Test forward pass."""
        selector = InteractionOrderSelector(input_dim=10, num_orders=3).to(device)

        x = torch.randn(16, 10).to(device)
        weights = selector(x)

        assert weights.shape == (16, 3)
        # Check softmax property
        assert torch.allclose(weights.sum(dim=1), torch.ones(16).to(device))


class TestSparsityRegularizedLoss:
    """Tests for custom loss function."""

    def test_binary_classification(self, device):
        """Test loss for binary classification."""
        criterion = SparsityRegularizedLoss(task="binary", sparsity_weight=0.01)

        logits = torch.randn(16, 1).to(device)
        targets = torch.randint(0, 2, (16,)).to(device)
        gate_activations = torch.randn(16, 8).to(device)

        loss, loss_dict = criterion(logits, targets, gate_activations)

        assert isinstance(loss, torch.Tensor)
        assert "base_loss" in loss_dict
        assert "sparsity_loss" in loss_dict
        assert "total_loss" in loss_dict

    def test_multiclass_classification(self, device):
        """Test loss for multiclass classification."""
        criterion = SparsityRegularizedLoss(
            task="multiclass", num_classes=5, sparsity_weight=0.01
        )

        logits = torch.randn(16, 5).to(device)
        targets = torch.randint(0, 5, (16,)).to(device)
        gate_activations = torch.randn(16, 8).to(device)

        loss, loss_dict = criterion(logits, targets, gate_activations)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

    def test_sparsity_regularization(self, device):
        """Test that sparsity weight affects loss."""
        criterion_no_sparsity = SparsityRegularizedLoss(
            task="binary", sparsity_weight=0.0
        )
        criterion_with_sparsity = SparsityRegularizedLoss(
            task="binary", sparsity_weight=0.1
        )

        logits = torch.randn(16, 1).to(device)
        targets = torch.randint(0, 2, (16,)).to(device)
        gate_activations = torch.ones(16, 8).to(device)

        loss_no_sparsity, _ = criterion_no_sparsity(
            logits, targets, gate_activations
        )
        loss_with_sparsity, _ = criterion_with_sparsity(
            logits, targets, gate_activations
        )

        # Loss with sparsity should be higher due to regularization
        assert loss_with_sparsity > loss_no_sparsity


class TestAdaptiveFeatureInteractionNetwork:
    """Tests for main model."""

    def test_initialization(self):
        """Test model initialization."""
        model = AdaptiveFeatureInteractionNetwork(
            input_dim=10,
            hidden_dims=[32, 16],
            num_gates=8,
            interaction_orders=[2],
            num_classes=1,
        )

        assert model.input_dim == 10
        assert model.num_gates == 8

    def test_forward_pass(self, sample_model, device):
        """Test forward pass."""
        model = sample_model
        x = torch.randn(16, 10).to(device)

        logits, auxiliary_outputs = model(x)

        assert logits.shape == (16, 1)
        assert "gate_scores" in auxiliary_outputs
        assert "order_weights" in auxiliary_outputs
        assert auxiliary_outputs["gate_scores"].shape == (16, 8)

    def test_multiclass_output(self, device):
        """Test multiclass classification."""
        model = AdaptiveFeatureInteractionNetwork(
            input_dim=10,
            hidden_dims=[32, 16],
            num_gates=8,
            interaction_orders=[2],
            num_classes=5,
        ).to(device)

        x = torch.randn(16, 10).to(device)
        logits, _ = model(x)

        assert logits.shape == (16, 5)

    def test_parameter_count(self, sample_model):
        """Test parameter counting."""
        count = sample_model.count_parameters()
        assert count > 0
        assert isinstance(count, int)

    def test_feature_importance(self, sample_model, device):
        """Test feature importance computation."""
        model = sample_model
        x = torch.randn(16, 10).to(device)

        importance = model.get_feature_importance(x)

        assert "gate_importance" in importance
        assert "order_importance" in importance

    def test_sparsity_computation(self, sample_model, device):
        """Test sparsity metric computation."""
        model = sample_model
        gate_scores = torch.randn(16, 8).to(device)
        gate_scores = torch.softmax(gate_scores, dim=-1)

        sparsity = model.compute_sparsity(gate_scores)

        assert isinstance(sparsity, float)
        assert 0.0 <= sparsity <= 1.0

    def test_gradient_flow(self, sample_model, device):
        """Test that gradients flow through the model."""
        model = sample_model
        model.train()

        x = torch.randn(16, 10).to(device)
        targets = torch.randint(0, 2, (16,)).float().to(device)

        logits, auxiliary_outputs = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(), targets
        )
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_batch_norm_eval_mode(self, sample_model, device):
        """Test that batch norm behaves differently in eval mode."""
        model = sample_model
        x = torch.randn(4, 10).to(device)

        # Train mode
        model.train()
        out1, _ = model(x)

        # Eval mode
        model.eval()
        out2, _ = model(x)

        # Outputs should be different due to batch norm
        assert not torch.allclose(out1, out2)
