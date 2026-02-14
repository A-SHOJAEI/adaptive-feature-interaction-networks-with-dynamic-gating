#!/usr/bin/env python
"""Evaluation script for Adaptive Feature Interaction Networks.

This script loads a trained model and evaluates it on the test set,
computing comprehensive metrics and generating analysis plots.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split

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
from adaptive_feature_interaction_networks_with_dynamic_gating.evaluation.metrics import (
    compute_metrics,
    compute_per_class_metrics,
    compute_confusion_matrix,
    compute_feature_interaction_sparsity,
)
from adaptive_feature_interaction_networks_with_dynamic_gating.evaluation.analysis import (
    analyze_results,
    plot_confusion_matrix,
    save_predictions,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Evaluate Adaptive Feature Interaction Network"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    return parser.parse_args()


def main() -> None:
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")

    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create output directory
    project_root = Path(__file__).parent.parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(exist_ok=True)

    try:
        # Load dataset
        logger.info("Loading dataset...")
        data_config = config.get("data", {})
        seed = config.get("seed", 42)
        X_df, y_series, metadata = load_dataset(
            dataset_name=data_config.get("dataset_name", "synthetic"),
            n_samples=data_config.get("n_samples", 10000),
            n_features=data_config.get("n_features", 50),
            n_informative=data_config.get("n_informative", 30),
            n_redundant=data_config.get("n_redundant", 10),
            n_classes=data_config.get("n_classes", 2),
            random_state=seed,
        )

        # Split data (same as training)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_df, y_series, test_size=0.2, random_state=seed, stratify=y_series
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=seed, stratify=y_temp
        )

        # Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = TabularPreprocessor(
            scaling_method=data_config.get("scaling_method", "standard"),
            categorical_encoding="label",
            handle_missing="mean",
        )
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        y_test_np = y_test.values

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

        # Load checkpoint
        checkpoint_path = project_root / args.checkpoint
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            logger.warning("Using randomly initialized model")
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")

        model = model.to(device)
        model.eval()

        # Create test dataloader
        batch_size = config.get("training", {}).get("batch_size", 256)
        X_test_tensor = torch.FloatTensor(X_test_processed)
        y_test_tensor = torch.LongTensor(y_test_np)

        # Evaluate model
        logger.info("Evaluating model on test set...")
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_gate_scores = []

        inference_times = []

        with torch.no_grad():
            for i in range(0, len(X_test_tensor), batch_size):
                batch_x = X_test_tensor[i:i+batch_size].to(device)
                batch_y = y_test_tensor[i:i+batch_size]

                # Measure inference time
                start_time = time.time()
                logits, auxiliary_outputs = model(batch_x)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                # Get predictions
                if num_classes == 1:  # Binary
                    probs = torch.sigmoid(logits).squeeze()
                    preds = (probs > 0.5).long()
                    probs = probs.cpu().numpy()
                else:  # Multiclass
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    probs = probs.cpu().numpy()

                all_predictions.append(preds.cpu().numpy())
                all_probabilities.append(probs)
                all_targets.append(batch_y.numpy())
                all_gate_scores.append(
                    auxiliary_outputs["gate_scores"].cpu().numpy()
                )

        # Concatenate results
        all_predictions = np.concatenate(all_predictions)
        all_probabilities = np.concatenate(all_probabilities)
        all_targets = np.concatenate(all_targets)
        all_gate_scores = np.concatenate(all_gate_scores)

        # Compute metrics
        logger.info("Computing metrics...")
        task = "binary" if metadata["n_classes"] == 2 else "multiclass"

        metrics = compute_metrics(
            y_true=all_targets,
            y_pred=all_predictions,
            y_pred_proba=all_probabilities,
            task=task,
            average="macro",
        )

        # Compute additional metrics
        sparsity = compute_feature_interaction_sparsity(all_gate_scores)
        metrics["feature_interaction_sparsity"] = sparsity

        avg_inference_time = np.mean(inference_times)
        metrics["avg_inference_time_per_batch"] = avg_inference_time
        metrics["inference_throughput_samples_per_sec"] = batch_size / avg_inference_time

        # Compute per-class metrics
        per_class_metrics = compute_per_class_metrics(
            all_targets, all_predictions
        )

        # Compute confusion matrix
        cm = compute_confusion_matrix(all_targets, all_predictions)

        # Save results
        logger.info("Saving results...")
        analyze_results(metrics, output_dir, experiment_name="test_evaluation")

        # Save per-class metrics
        per_class_path = output_dir / "per_class_metrics.json"
        with open(per_class_path, "w") as f:
            json.dump(per_class_metrics, f, indent=2)
        logger.info(f"Saved per-class metrics to {per_class_path}")

        # Plot confusion matrix
        class_names = [f"Class {i}" for i in range(metadata["n_classes"])]
        plot_confusion_matrix(
            cm, class_names, output_dir, "confusion_matrix.png", normalize=True
        )

        # Save predictions
        save_predictions(
            all_probabilities, all_targets, output_dir, "test_predictions.csv"
        )

        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Test Set Size: {len(all_targets)}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        if 'auc_roc' in metrics:
            print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        if 'log_loss' in metrics:
            print(f"Log Loss: {metrics['log_loss']:.4f}")
        print(f"Feature Interaction Sparsity: {metrics['feature_interaction_sparsity']:.4f}")
        print(f"Inference Throughput: {metrics['inference_throughput_samples_per_sec']:.2f} samples/sec")
        print("="*60)

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
