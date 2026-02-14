"""Training loop with advanced features.

This module implements a comprehensive trainer with:
- Learning rate scheduling
- Early stopping
- Gradient clipping
- Mixed precision training
- Checkpoint saving
- MLflow integration
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for Adaptive Feature Interaction Networks.

    Args:
        model: Model to train
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scheduler_type: Type of LR scheduler ('cosine', 'plateau', or None)
        scheduler_params: Parameters for scheduler
        gradient_clip_val: Maximum gradient norm (None to disable)
        early_stopping_patience: Patience for early stopping (None to disable)
        use_amp: Whether to use automatic mixed precision
        checkpoint_dir: Directory to save checkpoints
        log_interval: Logging interval in batches
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        scheduler_type: Optional[str] = "cosine",
        scheduler_params: Optional[Dict] = None,
        gradient_clip_val: Optional[float] = 1.0,
        early_stopping_patience: Optional[int] = 10,
        use_amp: bool = True,
        checkpoint_dir: Optional[Path] = None,
        log_interval: int = 50,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.gradient_clip_val = gradient_clip_val
        self.early_stopping_patience = early_stopping_patience
        self.use_amp = use_amp and torch.cuda.is_available()
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval

        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler() if self.use_amp else None

        # Initialize learning rate scheduler
        self.scheduler = None
        if scheduler_type == "cosine":
            params = scheduler_params or {"T_max": 100, "eta_min": 0.00001}
            self.scheduler = CosineAnnealingLR(optimizer, **params)
        elif scheduler_type == "plateau":
            params = scheduler_params or {"mode": "min", "factor": 0.5, "patience": 5}
            self.scheduler = ReduceLROnPlateau(optimizer, **params)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_metric = 0.0
        self.epochs_without_improvement = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_metric": [],
            "val_metric": [],
            "lr": [],
        }

        # Create checkpoint directory
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Trainer initialized with device={device}, use_amp={self.use_amp}, "
            f"scheduler={scheduler_type}"
        )

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                logits, auxiliary_outputs = self.model(inputs)
                loss, loss_dict = self.criterion(
                    logits, targets, auxiliary_outputs.get("gate_scores")
                )

            # Backward pass
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.gradient_clip_val is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )
                self.optimizer.step()

            # Update metrics
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Store predictions for metric calculation
            with torch.no_grad():
                if logits.size(1) == 1:  # Binary classification
                    probs = torch.sigmoid(logits).squeeze()
                else:  # Multiclass
                    probs = torch.softmax(logits, dim=1)
                all_predictions.append(probs.cpu())
                all_targets.append(targets.cpu())

            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / total_samples
                logger.info(
                    f"Epoch {self.current_epoch} [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.4f}"
                )

        # Compute epoch metrics
        avg_loss = total_loss / total_samples
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        metrics = {
            "loss": avg_loss,
            "samples": total_samples,
        }

        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        all_gate_scores = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                logits, auxiliary_outputs = self.model(inputs)
                loss, loss_dict = self.criterion(
                    logits, targets, auxiliary_outputs.get("gate_scores")
                )

                # Update metrics
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Store predictions
                if logits.size(1) == 1:  # Binary classification
                    probs = torch.sigmoid(logits).squeeze()
                else:  # Multiclass
                    probs = torch.softmax(logits, dim=1)
                all_predictions.append(probs.cpu())
                all_targets.append(targets.cpu())
                all_gate_scores.append(
                    auxiliary_outputs["gate_scores"].cpu()
                )

        # Compute metrics
        avg_loss = total_loss / total_samples
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        all_gate_scores = torch.cat(all_gate_scores)

        # Compute sparsity
        sparsity = self.model.compute_sparsity(all_gate_scores)

        metrics = {
            "loss": avg_loss,
            "samples": total_samples,
            "sparsity": sparsity,
        }

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        mlflow_tracking: bool = False,
    ) -> Dict[str, list]:
        """Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            mlflow_tracking: Whether to log to MLflow

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        # MLflow logging (optional)
        mlflow_available = False
        if mlflow_tracking:
            try:
                import mlflow
                mlflow_available = True
                logger.info("MLflow tracking enabled")
            except ImportError:
                logger.warning("MLflow not available, tracking disabled")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Training phase
            train_metrics = self.train_epoch(train_loader)

            # Validation phase
            val_metrics = self.validate(val_loader)

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log metrics
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Sparsity: {val_metrics['sparsity']:.4f}, "
                f"LR: {current_lr:.6f}, "
                f"Time: {epoch_time:.2f}s"
            )

            # Update history
            self.training_history["train_loss"].append(train_metrics["loss"])
            self.training_history["val_loss"].append(val_metrics["loss"])
            self.training_history["lr"].append(current_lr)

            # MLflow logging
            if mlflow_available:
                try:
                    mlflow.log_metrics(
                        {
                            "train_loss": train_metrics["loss"],
                            "val_loss": val_metrics["loss"],
                            "val_sparsity": val_metrics["sparsity"],
                            "learning_rate": current_lr,
                        },
                        step=epoch,
                    )
                except Exception as e:
                    logger.warning(f"MLflow logging failed: {e}")

            # Save best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.epochs_without_improvement = 0
                if self.checkpoint_dir is not None:
                    self.save_checkpoint("best_model.pt")
                    logger.info("Saved best model checkpoint")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if (
                self.early_stopping_patience is not None
                and self.epochs_without_improvement >= self.early_stopping_patience
            ):
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(patience={self.early_stopping_patience})"
                )
                break

            # Save periodic checkpoint
            if self.checkpoint_dir is not None and (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        logger.info(f"Training completed. Best val loss: {self.best_val_loss:.4f}")
        return self.training_history

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        if self.checkpoint_dir is None:
            return

        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint.get("training_history", {})

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
