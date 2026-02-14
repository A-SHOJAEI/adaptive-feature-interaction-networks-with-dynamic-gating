"""Core model implementation for Adaptive Feature Interaction Networks.

This module implements the main AFIN model with dynamic gating and adaptive
interaction selection.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from adaptive_feature_interaction_networks_with_dynamic_gating.models.components import (
    DynamicGatingLayer,
    FeatureInteractionLayer,
    InteractionOrderSelector,
)

logger = logging.getLogger(__name__)


class AdaptiveFeatureInteractionNetwork(nn.Module):
    """Adaptive Feature Interaction Network with Dynamic Gating.

    This is the core novel architecture that combines:
    1. Dynamic gating for sample-specific feature selection
    2. Multi-order feature interactions (pairwise, triplet, higher-order)
    3. Meta-learning for adaptive interaction order weighting
    4. Differentiable architecture search principles

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        num_gates: Number of gating units
        interaction_orders: List of interaction orders to use (e.g., [2, 3])
        gate_hidden_dim: Hidden dimension for gate network
        gate_temperature: Temperature for gate softmax (lower = more sparse)
        num_classes: Number of output classes (1 for binary, >2 for multiclass)
        dropout: Dropout probability
        use_batch_norm: Whether to use batch normalization
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [256, 128, 64],
        num_gates: int = 32,
        interaction_orders: list[int] = [2, 3],
        gate_hidden_dim: int = 64,
        gate_temperature: float = 0.5,
        num_classes: int = 1,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_gates = num_gates
        self.interaction_orders = interaction_orders
        self.num_classes = num_classes

        logger.info(
            f"Initializing AdaptiveFeatureInteractionNetwork with input_dim={input_dim}, "
            f"hidden_dims={hidden_dims}, num_gates={num_gates}, "
            f"interaction_orders={interaction_orders}"
        )

        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)

        # Dynamic gating layer for sample-specific feature selection
        self.gating_layer = DynamicGatingLayer(
            input_dim=input_dim,
            num_gates=num_gates,
            hidden_dim=gate_hidden_dim,
            temperature=gate_temperature,
            dropout=dropout,
        )

        # Feature interaction layers for each order
        self.interaction_layers = nn.ModuleList()
        self.interaction_output_dims = []

        for order in interaction_orders:
            interaction_dim = hidden_dims[0] // len(interaction_orders)
            layer = FeatureInteractionLayer(
                input_dim=input_dim,
                interaction_order=order,
                output_dim=interaction_dim,
                use_projection=True,
            )
            self.interaction_layers.append(layer)
            self.interaction_output_dims.append(interaction_dim)

        # Interaction order selector (meta-learning component)
        self.order_selector = InteractionOrderSelector(
            input_dim=input_dim,
            num_orders=len(interaction_orders),
            hidden_dim=32,
        )

        # Gate projection to match interaction dimension
        gate_output_dim = hidden_dims[0] // 4
        self.gate_projection = nn.Sequential(
            nn.Linear(num_gates, gate_output_dim),
            nn.ReLU(),
        )

        # Combine all features
        total_feature_dim = sum(self.interaction_output_dims) + gate_output_dim

        # Deep network layers
        layers = []
        prev_dim = total_feature_dim

        for i, hidden_dim in enumerate(hidden_dims[1:], 1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.deep_network = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)

        # Initialize weights
        self._initialize_weights()

        logger.info(
            f"Model initialized with {self.count_parameters():,} parameters"
        )

    def _initialize_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tuple of (logits, auxiliary_outputs) where:
                - logits: Model predictions (batch_size, num_classes)
                - auxiliary_outputs: Dict containing intermediate outputs:
                    - gate_scores: Gate activation values
                    - order_weights: Interaction order weights
                    - interaction_features: Features from each interaction order
        """
        # Input normalization
        x_norm = self.input_norm(x)

        # Dynamic gating
        gate_scores, gate_logits = self.gating_layer(x_norm)

        # Project gate scores
        gate_features = self.gate_projection(gate_scores)

        # Compute feature interactions for each order
        interaction_features_list = []
        for layer in self.interaction_layers:
            interaction_features = layer(x_norm)
            interaction_features_list.append(interaction_features)

        # Get adaptive order weights (meta-learning component)
        order_weights = self.order_selector(x_norm)

        # Weight interaction features by order importance
        weighted_interactions = []
        for i, interaction_features in enumerate(interaction_features_list):
            # Broadcast order weights to match feature dimensions
            weight = order_weights[:, i:i+1]
            weighted = interaction_features * weight
            weighted_interactions.append(weighted)

        # Concatenate all features
        combined_features = torch.cat(
            weighted_interactions + [gate_features], dim=1
        )

        # Deep network
        hidden_features = self.deep_network(combined_features)

        # Output layer
        logits = self.output_layer(hidden_features)

        # Prepare auxiliary outputs for loss computation and analysis
        auxiliary_outputs = {
            "gate_scores": gate_scores,
            "gate_logits": gate_logits,
            "order_weights": order_weights,
            "interaction_features": torch.cat(interaction_features_list, dim=1),
        }

        return logits, auxiliary_outputs

    def count_parameters(self) -> int:
        """Count total trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_feature_importance(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute feature importance scores for input samples.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Dictionary containing various importance scores
        """
        with torch.no_grad():
            x_norm = self.input_norm(x)
            gate_scores, _ = self.gating_layer(x_norm)
            order_weights = self.order_selector(x_norm)

            return {
                "gate_importance": gate_scores,
                "order_importance": order_weights,
            }

    def compute_sparsity(self, gate_scores: torch.Tensor) -> float:
        """Compute sparsity metric for gate activations.

        Args:
            gate_scores: Gate activation values (batch_size, num_gates)

        Returns:
            Sparsity score (0 = dense, 1 = maximally sparse)
        """
        # Compute Gini coefficient as sparsity measure
        sorted_scores, _ = torch.sort(gate_scores, dim=-1)
        n = gate_scores.size(-1)
        indices = torch.arange(1, n + 1, device=gate_scores.device).float()
        gini = (2 * torch.sum(sorted_scores * indices, dim=-1) /
                (n * torch.sum(sorted_scores, dim=-1))) - (n + 1) / n
        return gini.mean().item()
