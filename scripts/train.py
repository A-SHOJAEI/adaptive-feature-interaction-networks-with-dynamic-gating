#!/usr/bin/env python
"""Training script for Adaptive Feature Interaction Networks.

This script trains the AFIN model with the following features:
- Mixed precision training
- Learning rate scheduling
- Early stopping
- Checkpoint saving
- MLflow tracking (optional)
- Gradient clipping
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_feature_interaction_networks_with_dynamic_gating.data.loader import (
    load_dataset,
    create_dataloaders,
)
from adaptive_feature_interaction_networks_with_dynamic_gating.data.preprocessing import (
    TabularPreprocessor,
)
from adaptive_feature_interaction_networks_with_dynamic_gating.models.model import (
    AdaptiveFeatureInteractionNetwork,
)
from adaptive_feature_interaction_networks_with_dynamic_gating.models.components import (
    SparsityRegularizedLoss,
)
from adaptive_feature_interaction_networks_with_dynamic_gating.training.trainer import Trainer
from adaptive_feature_interaction_networks_with_dynamic_gating.evaluation.analysis import (
    plot_training_curves,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train Adaptive Feature Interaction Network"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow tracking",
    )
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")

    # Set random seed
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed)

    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create output directories
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    checkpoints_dir = project_root / "checkpoints"
    results_dir = project_root / "results"
    models_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    try:
        # Load dataset
        logger.info("Loading dataset...")
        data_config = config.get("data", {})
        X_df, y_series, metadata = load_dataset(
            dataset_name=data_config.get("dataset_name", "synthetic"),
            n_samples=data_config.get("n_samples", 10000),
            n_features=data_config.get("n_features", 50),
            n_informative=data_config.get("n_informative", 30),
            n_redundant=data_config.get("n_redundant", 10),
            n_classes=data_config.get("n_classes", 2),
            random_state=seed,
        )

        logger.info(
            f"Dataset loaded: {X_df.shape[0]} samples, {X_df.shape[1]} features, "
            f"{metadata['n_classes']} classes"
        )

        # Split data into train/val/test
        from sklearn.model_selection import train_test_split

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_df, y_series, test_size=0.2, random_state=seed, stratify=y_series
        )

        # Second split: separate train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=seed, stratify=y_temp
        )

        logger.info(
            f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        # Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = TabularPreprocessor(
            scaling_method=data_config.get("scaling_method", "standard"),
            categorical_encoding="label",
            handle_missing="mean",
        )

        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)

        y_train_np = y_train.values
        y_val_np = y_val.values
        y_test_np = y_test.values

        # Create dataloaders
        batch_size = config.get("training", {}).get("batch_size", 256)
        dataloaders = create_dataloaders(
            X_train_processed,
            y_train_np,
            X_val_processed,
            y_val_np,
            X_test_processed,
            y_test_np,
            batch_size=batch_size,
            num_workers=0,
            shuffle_train=True,
        )

        # Initialize model
        logger.info("Initializing model...")
        model_config = config.get("model", {})
        input_dim = X_train_processed.shape[1]
        num_classes = 1 if metadata["n_classes"] == 2 else metadata["n_classes"]

        model = AdaptiveFeatureInteractionNetwork(
            input_dim=input_dim,
            hidden_dims=model_config.get("hidden_dims", [256, 128, 64]),
            num_gates=model_config.get("num_gates", 32),
            interaction_orders=model_config.get("interaction_orders", [2, 3]),
            gate_hidden_dim=model_config.get("gate_hidden_dim", 64),
            gate_temperature=model_config.get("gate_temperature", 0.5),
            num_classes=num_classes,
            dropout=model_config.get("dropout", 0.3),
            use_batch_norm=model_config.get("use_batch_norm", True),
        )

        model = model.to(device)
        logger.info(f"Model has {model.count_parameters():,} parameters")

        # Initialize loss function
        task = "binary" if metadata["n_classes"] == 2 else "multiclass"
        criterion = SparsityRegularizedLoss(
            task=task,
            sparsity_weight=model_config.get("sparsity_weight", 0.001),
            num_classes=num_classes if task == "multiclass" else 2,
        )

        # Initialize optimizer
        training_config = config.get("training", {})
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config.get("learning_rate", 0.001),
            weight_decay=training_config.get("weight_decay", 0.0001),
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scheduler_type=training_config.get("scheduler_type", "cosine"),
            scheduler_params=training_config.get("scheduler_params", {
                "T_max": training_config.get("num_epochs", 100),
                "eta_min": 0.00001,
            }),
            gradient_clip_val=training_config.get("gradient_clip_val", 1.0),
            early_stopping_patience=training_config.get("early_stopping_patience", 10),
            use_amp=training_config.get("use_amp", True),
            checkpoint_dir=checkpoints_dir,
            log_interval=training_config.get("log_interval", 50),
        )

        # Train model
        logger.info("Starting training...")
        num_epochs = training_config.get("num_epochs", 100)
        history = trainer.fit(
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            num_epochs=num_epochs,
            mlflow_tracking=args.mlflow,
        )

        # Save final model
        final_model_path = models_dir / "final_model.pt"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Saved final model to {final_model_path}")

        # Plot training curves
        logger.info("Plotting training curves...")
        plot_training_curves(history, results_dir, "training_curves.png")

        # Save training history
        import json
        history_path = results_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")

        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
