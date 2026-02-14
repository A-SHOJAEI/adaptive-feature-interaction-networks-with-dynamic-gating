"""Custom components for adaptive feature interaction networks.

This module contains custom layers, loss functions, and modules that implement
the core novelty of the architecture: dynamic gating and adaptive interaction
selection.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DynamicGatingLayer(nn.Module):
    """Dynamic gating layer that learns sample-specific importance scores.

    This layer computes attention-like weights for each feature based on the
    input sample, enabling instance-dependent feature selection.

    Args:
        input_dim: Dimension of input features
        num_gates: Number of gating units
        hidden_dim: Dimension of hidden layer for gate computation
        temperature: Temperature parameter for softmax (lower = more sparse)
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        num_gates: int,
        hidden_dim: int = 64,
        temperature: float = 0.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_gates = num_gates
        self.temperature = temperature

        # Gate network
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_gates),
        )

        # Initialize with small weights for stability
        for module in self.gate_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gating scores.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tuple of (gated_scores, raw_gates) where:
                - gated_scores: Normalized gate values (batch_size, num_gates)
                - raw_gates: Raw gate logits (batch_size, num_gates)
        """
        # Compute raw gate logits
        raw_gates = self.gate_network(x)

        # Apply temperature-scaled softmax for sparsity
        gated_scores = F.softmax(raw_gates / self.temperature, dim=-1)

        return gated_scores, raw_gates


class FeatureInteractionLayer(nn.Module):
    """Feature interaction layer that computes interactions of different orders.

    This layer computes pairwise, triplet, and higher-order feature interactions
    with configurable interaction orders.

    Args:
        input_dim: Dimension of input features
        interaction_order: Maximum order of interactions (2=pairwise, 3=triplet, etc.)
        output_dim: Dimension of output
        use_projection: Whether to project interaction features
    """

    def __init__(
        self,
        input_dim: int,
        interaction_order: int = 2,
        output_dim: int = 128,
        use_projection: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.interaction_order = interaction_order
        self.output_dim = output_dim

        # Compute number of interaction features
        self.num_pairwise = (input_dim * (input_dim - 1)) // 2

        if interaction_order >= 3:
            self.num_triplet = (input_dim * (input_dim - 1) * (input_dim - 2)) // 6
        else:
            self.num_triplet = 0

        total_interactions = input_dim + self.num_pairwise + self.num_triplet

        # Projection layer
        if use_projection:
            self.projection = nn.Sequential(
                nn.Linear(total_interactions, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )
        else:
            self.projection = nn.Identity()
            self.output_dim = total_interactions

        logger.info(
            f"FeatureInteractionLayer initialized with {self.num_pairwise} pairwise "
            f"and {self.num_triplet} triplet interactions"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute feature interactions.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Interaction features of shape (batch_size, output_dim)
        """
        batch_size = x.size(0)
        interactions = [x]  # Start with original features

        # Pairwise interactions (element-wise products)
        if self.interaction_order >= 2:
            pairwise = []
            for i in range(self.input_dim):
                for j in range(i + 1, self.input_dim):
                    pairwise.append(x[:, i] * x[:, j])
            if pairwise:
                pairwise_tensor = torch.stack(pairwise, dim=1)
                interactions.append(pairwise_tensor)

        # Triplet interactions (three-way products)
        if self.interaction_order >= 3:
            triplet = []
            for i in range(self.input_dim):
                for j in range(i + 1, self.input_dim):
                    for k in range(j + 1, self.input_dim):
                        triplet.append(x[:, i] * x[:, j] * x[:, k])
            if triplet:
                triplet_tensor = torch.stack(triplet, dim=1)
                interactions.append(triplet_tensor)

        # Concatenate all interactions
        interaction_features = torch.cat(interactions, dim=1)

        # Project to output dimension
        output = self.projection(interaction_features)

        return output


class InteractionOrderSelector(nn.Module):
    """Meta-learning module that adaptively weights different interaction orders.

    This is a key novelty: instead of using fixed interaction orders, this module
    learns to weight pairwise vs triplet vs higher-order interactions differently
    for each sample.

    Args:
        input_dim: Dimension of input features
        num_orders: Number of interaction orders to consider
        hidden_dim: Hidden dimension for selector network
    """

    def __init__(
        self,
        input_dim: int,
        num_orders: int = 3,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_orders = num_orders

        # Selector network that produces per-order weights
        self.selector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_orders),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample interaction order weights.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Order weights of shape (batch_size, num_orders)
        """
        weights = self.selector(x)
        return weights


class SparsityRegularizedLoss(nn.Module):
    """Custom loss function with sparsity regularization on gate activations.

    This loss encourages the model to use sparse feature interactions, which
    improves interpretability and inference speed. Combines standard BCE/CE loss
    with L1 regularization on gate activations.

    Args:
        task: Task type ('binary' or 'multiclass')
        sparsity_weight: Weight for sparsity regularization term
        num_classes: Number of classes for multiclass classification
    """

    def __init__(
        self,
        task: str = "binary",
        sparsity_weight: float = 0.001,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.task = task
        self.sparsity_weight = sparsity_weight
        self.num_classes = num_classes

        if task == "binary":
            self.base_loss = nn.BCEWithLogitsLoss()
        elif task == "multiclass":
            self.base_loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        gate_activations: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss with sparsity regularization.

        Args:
            logits: Model predictions (batch_size, num_classes or 1)
            targets: Ground truth labels (batch_size,)
            gate_activations: Gate activation values (batch_size, num_gates)

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains:
                - base_loss: Classification loss
                - sparsity_loss: Sparsity regularization
                - total_loss: Combined loss
        """
        # Base classification loss
        if self.task == "binary":
            if logits.dim() > 1 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            base_loss = self.base_loss(logits, targets.float())
        else:
            base_loss = self.base_loss(logits, targets.long())

        # Sparsity regularization on gate activations
        sparsity_loss = torch.tensor(0.0, device=logits.device)
        if gate_activations is not None:
            # L1 regularization to encourage sparsity
            sparsity_loss = torch.mean(torch.abs(gate_activations))

        # Total loss
        total_loss = base_loss + self.sparsity_weight * sparsity_loss

        loss_dict = {
            "base_loss": base_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict


class GradientBoostingEnsemble(nn.Module):
    """Ensemble wrapper that combines neural network with gradient boosting.

    This module integrates XGBoost/LightGBM with the neural architecture through
    a learnable weighted ensemble, bridging deep learning and gradient boosting.

    Args:
        neural_output_dim: Output dimension of neural network
        num_gb_models: Number of gradient boosting models in ensemble
        num_classes: Number of output classes
    """

    def __init__(
        self,
        neural_output_dim: int,
        num_gb_models: int = 3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.neural_output_dim = neural_output_dim
        self.num_gb_models = num_gb_models
        self.num_classes = num_classes

        # Ensemble weight network
        total_models = 1 + num_gb_models  # Neural + GB models
        self.ensemble_weights = nn.Parameter(
            torch.ones(total_models) / total_models
        )

        # Final prediction layer
        self.output_layer = nn.Linear(neural_output_dim, num_classes)

    def forward(
        self,
        neural_features: torch.Tensor,
        gb_predictions: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute ensemble prediction.

        Args:
            neural_features: Features from neural network
            gb_predictions: List of predictions from GB models

        Returns:
            Final ensemble prediction
        """
        # Neural network prediction
        neural_pred = self.output_layer(neural_features)

        if gb_predictions is None or len(gb_predictions) == 0:
            return neural_pred

        # Normalize ensemble weights
        weights = F.softmax(self.ensemble_weights, dim=0)

        # Weighted combination
        ensemble_pred = weights[0] * neural_pred
        for i, gb_pred in enumerate(gb_predictions):
            ensemble_pred += weights[i + 1] * gb_pred

        return ensemble_pred
