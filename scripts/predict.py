#!/usr/bin/env python
"""Prediction script for Adaptive Feature Interaction Networks.

This script loads a trained model and performs inference on new data.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_feature_interaction_networks_with_dynamic_gating.data.preprocessing import (
    TabularPreprocessor,
)
from adaptive_feature_interaction_networks_with_dynamic_gating.models.model import (
    AdaptiveFeatureInteractionNetwork,
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
        description="Make predictions with Adaptive Feature Interaction Network"
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
        default="models/final_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file or JSON string",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions (default: stdout)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference",
    )
    return parser.parse_args()


def load_input_data(input_path: str) -> pd.DataFrame:
    """Load input data from file or JSON string.

    Args:
        input_path: Path to CSV file or JSON string

    Returns:
        Input data as DataFrame
    """
    # Try to load as file
    if Path(input_path).exists():
        if input_path.endswith(".csv"):
            data = pd.read_csv(input_path)
        elif input_path.endswith(".json"):
            data = pd.read_json(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
        logger.info(f"Loaded data from {input_path}: {data.shape}")
    else:
        # Try to parse as JSON string
        try:
            data_dict = json.loads(input_path)
            data = pd.DataFrame([data_dict])
            logger.info(f"Parsed JSON input: {data.shape}")
        except json.JSONDecodeError:
            raise ValueError(
                f"Input must be a valid file path or JSON string: {input_path}"
            )

    return data


def main() -> None:
    """Main prediction function."""
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

    project_root = Path(__file__).parent.parent

    try:
        # Load input data
        logger.info("Loading input data...")
        input_data = load_input_data(args.input)

        # For demonstration, we need to fit the preprocessor on training data
        # In production, the preprocessor should be saved during training
        logger.info("Note: Using fresh preprocessor. In production, load the saved preprocessor from training.")

        from adaptive_feature_interaction_networks_with_dynamic_gating.data.loader import load_dataset
        from sklearn.model_selection import train_test_split

        # Load training data to fit preprocessor (this should be cached in production)
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

        X_train, _, _, _ = train_test_split(
            X_df, y_series, test_size=0.2, random_state=seed, stratify=y_series
        )

        # Fit preprocessor
        preprocessor = TabularPreprocessor(
            scaling_method=data_config.get("scaling_method", "standard"),
            categorical_encoding="label",
            handle_missing="mean",
        )
        preprocessor.fit(X_train)

        # Ensure input has same features
        missing_features = set(X_train.columns) - set(input_data.columns)
        if missing_features:
            logger.warning(f"Input missing features: {missing_features}. Filling with zeros.")
            for feat in missing_features:
                input_data[feat] = 0

        # Reorder columns to match training
        input_data = input_data[X_train.columns]

        # Preprocess input
        logger.info("Preprocessing input data...")
        input_processed = preprocessor.transform(input_data)

        # Initialize model
        logger.info("Loading model...")
        model_config = config.get("model", {})
        input_dim = input_processed.shape[1]
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
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

        model = model.to(device)
        model.eval()

        # Make predictions
        logger.info("Making predictions...")
        input_tensor = torch.FloatTensor(input_processed).to(device)

        with torch.no_grad():
            logits, auxiliary_outputs = model(input_tensor)

            # Get predictions and probabilities
            if num_classes == 1:  # Binary
                probs = torch.sigmoid(logits).squeeze()
                preds = (probs > 0.5).long()
                probs = probs.cpu().numpy()
                preds = preds.cpu().numpy()
            else:  # Multiclass
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                probs = probs.cpu().numpy()
                preds = preds.cpu().numpy()

            # Get gate scores for interpretability
            gate_scores = auxiliary_outputs["gate_scores"].cpu().numpy()

        # Prepare output
        results = []
        for i in range(len(input_data)):
            result = {
                "sample_id": i,
                "predicted_class": int(preds[i]) if preds.ndim > 0 else int(preds),
            }

            # Add probabilities
            if num_classes == 1:
                result["probability_class_1"] = float(probs[i]) if probs.ndim > 0 else float(probs)
                result["probability_class_0"] = 1.0 - result["probability_class_1"]
            else:
                for cls in range(num_classes):
                    result[f"probability_class_{cls}"] = float(probs[i, cls])

            # Add confidence (max probability)
            if num_classes == 1:
                result["confidence"] = max(result["probability_class_0"], result["probability_class_1"])
            else:
                result["confidence"] = float(np.max(probs[i]))

            # Add top active gates (for interpretability)
            top_gate_indices = np.argsort(gate_scores[i])[-5:][::-1]
            result["top_active_gates"] = top_gate_indices.tolist()

            results.append(result)

        # Output results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix == ".json":
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
            elif output_path.suffix == ".csv":
                results_df = pd.DataFrame(results)
                results_df.to_csv(output_path, index=False)
            else:
                raise ValueError(f"Unsupported output format: {output_path.suffix}")

            logger.info(f"Saved predictions to {output_path}")
        else:
            # Print to stdout
            print("\nPredictions:")
            print(json.dumps(results, indent=2))

        logger.info("Prediction completed successfully!")

    except Exception as e:
        logger.error(f"Prediction failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
