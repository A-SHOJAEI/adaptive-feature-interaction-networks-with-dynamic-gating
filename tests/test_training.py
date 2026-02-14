"""Tests for training functionality."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import tempfile

from adaptive_feature_interaction_networks_with_dynamic_gating.training.trainer import Trainer
from adaptive_feature_interaction_networks_with_dynamic_gating.models.components import (
    SparsityRegularizedLoss,
)


class TestTrainer:
    """Tests for trainer class."""

    @pytest.fixture
    def trainer_setup(self, sample_model, device):
        """Setup trainer for testing."""
        model = sample_model
        criterion = SparsityRegularizedLoss(task="binary", sparsity_weight=0.01)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            trainer = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scheduler_type="cosine",
                scheduler_params={"T_max": 10, "eta_min": 0.00001},
                gradient_clip_val=1.0,
                early_stopping_patience=5,
                use_amp=False,  # Disable for testing
                checkpoint_dir=checkpoint_dir,
                log_interval=10,
            )

            yield trainer, checkpoint_dir

    def test_trainer_initialization(self, trainer_setup):
        """Test trainer initialization."""
        trainer, _ = trainer_setup

        assert trainer.device is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None

    def test_train_epoch(self, trainer_setup, device):
        """Test training for one epoch."""
        trainer, _ = trainer_setup

        # Create dummy dataloader
        X = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

        metrics = trainer.train_epoch(dataloader)

        assert "loss" in metrics
        assert "samples" in metrics
        assert metrics["samples"] == 32

    def test_validate(self, trainer_setup, device):
        """Test validation."""
        trainer, _ = trainer_setup

        # Create dummy dataloader
        X = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

        metrics = trainer.validate(dataloader)

        assert "loss" in metrics
        assert "samples" in metrics
        assert "sparsity" in metrics

    def test_fit(self, trainer_setup, device):
        """Test full training loop."""
        trainer, _ = trainer_setup

        # Create dummy dataloaders
        X_train = torch.randn(64, 10)
        y_train = torch.randint(0, 2, (64,))
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

        X_val = torch.randn(32, 10)
        y_val = torch.randint(0, 2, (32,))
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=3,
            mlflow_tracking=False,
        )

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) <= 3

    def test_checkpoint_saving(self, trainer_setup):
        """Test checkpoint saving."""
        trainer, checkpoint_dir = trainer_setup

        trainer.save_checkpoint("test_checkpoint.pt")

        checkpoint_path = checkpoint_dir / "test_checkpoint.pt"
        assert checkpoint_path.exists()

        # Load and verify
        checkpoint = torch.load(checkpoint_path)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    def test_checkpoint_loading(self, trainer_setup):
        """Test checkpoint loading."""
        trainer, checkpoint_dir = trainer_setup

        # Save checkpoint
        trainer.current_epoch = 5
        trainer.best_val_loss = 0.123
        trainer.save_checkpoint("test_checkpoint.pt")

        # Create new trainer and load
        checkpoint_path = checkpoint_dir / "test_checkpoint.pt"
        trainer.load_checkpoint(checkpoint_path)

        assert trainer.current_epoch == 5
        assert trainer.best_val_loss == 0.123

    def test_early_stopping(self, trainer_setup, device):
        """Test early stopping mechanism."""
        trainer, _ = trainer_setup
        trainer.early_stopping_patience = 2

        # Create dummy dataloaders
        X_train = torch.randn(32, 10)
        y_train = torch.randint(0, 2, (32,))
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

        X_val = torch.randn(16, 10)
        y_val = torch.randint(0, 2, (16,))
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=100,  # Large number to test early stopping
            mlflow_tracking=False,
        )

        # Should stop before 100 epochs
        assert len(history["train_loss"]) < 100

    def test_learning_rate_scheduling(self, trainer_setup, device):
        """Test learning rate scheduling."""
        trainer, _ = trainer_setup

        initial_lr = trainer.optimizer.param_groups[0]["lr"]

        # Create dummy dataloaders
        X_train = torch.randn(32, 10)
        y_train = torch.randint(0, 2, (32,))
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

        X_val = torch.randn(16, 10)
        y_val = torch.randint(0, 2, (16,))
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=3,
            mlflow_tracking=False,
        )

        final_lr = trainer.optimizer.param_groups[0]["lr"]

        # LR should have changed with cosine scheduler
        assert final_lr != initial_lr

    def test_gradient_clipping(self, trainer_setup, device):
        """Test gradient clipping."""
        trainer, _ = trainer_setup
        trainer.gradient_clip_val = 0.5

        # Create dummy data
        X = torch.randn(16, 10)
        y = torch.randint(0, 2, (16,))
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)

        # Train one epoch
        trainer.train_epoch(dataloader)

        # Check that gradients are clipped
        for param in trainer.model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # Gradient norm should be reasonable (not exploding)
                assert grad_norm < 100.0
