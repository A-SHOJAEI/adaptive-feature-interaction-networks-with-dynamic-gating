"""Results analysis and visualization utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def analyze_results(
    metrics: Dict[str, float],
    output_dir: Path,
    experiment_name: str = "experiment",
) -> None:
    """Analyze and save experiment results.

    Args:
        metrics: Dictionary of computed metrics
        output_dir: Directory to save results
        experiment_name: Name of experiment
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics as JSON
    results_path = output_dir / f"{experiment_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"Results for {experiment_name}")
    print("=" * 60)
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name:30s}: {value:.4f}")
        else:
            print(f"{metric_name:30s}: {value}")
    print("=" * 60 + "\n")


def plot_training_curves(
    history: Dict[str, List[float]],
    output_dir: Path,
    filename: str = "training_curves.png",
) -> None:
    """Plot training and validation curves.

    Args:
        history: Training history dictionary
        output_dir: Directory to save plot
        filename: Output filename
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curves
    if "train_loss" in history and "val_loss" in history:
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="s")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True)

    # Learning rate
    if "lr" in history:
        epochs = range(1, len(history["lr"]) + 1)
        axes[1].plot(epochs, history["lr"], label="Learning Rate", marker="o", color="green")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Learning Rate")
        axes[1].set_title("Learning Rate Schedule")
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_yscale("log")

    plt.tight_layout()
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved training curves to {output_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]],
    output_dir: Path,
    filename: str = "confusion_matrix.png",
    normalize: bool = True,
) -> None:
    """Plot confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: Names of classes
        output_dir: Directory to save plot
        filename: Output filename
        normalize: Whether to normalize the confusion matrix
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")


def plot_feature_importance(
    importance_scores: np.ndarray,
    feature_names: Optional[List[str]],
    output_dir: Path,
    filename: str = "feature_importance.png",
    top_k: int = 20,
) -> None:
    """Plot feature importance scores.

    Args:
        importance_scores: Feature importance scores
        feature_names: Names of features
        output_dir: Directory to save plot
        filename: Output filename
        top_k: Number of top features to display
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get top k features
    top_indices = np.argsort(importance_scores)[-top_k:]
    top_scores = importance_scores[top_indices]

    if feature_names is not None:
        top_names = [feature_names[i] for i in top_indices]
    else:
        top_names = [f"Feature {i}" for i in top_indices]

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_k), top_scores, color="steelblue")
    plt.yticks(range(top_k), top_names)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title(f"Top {top_k} Most Important Features")
    plt.tight_layout()

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved feature importance plot to {output_path}")


def compare_models(
    results_list: List[Dict[str, float]],
    model_names: List[str],
    output_dir: Path,
    filename: str = "model_comparison.png",
) -> None:
    """Compare metrics across multiple models.

    Args:
        results_list: List of result dictionaries
        model_names: Names of models
        output_dir: Directory to save plot
        filename: Output filename
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get common metrics
    all_metrics = set()
    for results in results_list:
        all_metrics.update(results.keys())
    common_metrics = sorted(all_metrics)

    # Filter to numeric metrics only
    numeric_metrics = []
    for metric in common_metrics:
        if all(
            isinstance(results.get(metric), (int, float))
            for results in results_list
        ):
            numeric_metrics.append(metric)

    if not numeric_metrics:
        logger.warning("No common numeric metrics found for comparison")
        return

    # Create comparison plot
    n_metrics = len(numeric_metrics)
    fig, axes = plt.subplots(
        (n_metrics + 1) // 2, 2, figsize=(15, 5 * ((n_metrics + 1) // 2))
    )
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, metric in enumerate(numeric_metrics):
        values = [results.get(metric, 0) for results in results_list]
        axes[idx].bar(model_names, values, color="steelblue")
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(f"{metric} Comparison")
        axes[idx].tick_params(axis="x", rotation=45)

    # Remove extra subplots
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved model comparison to {output_path}")


def save_predictions(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    output_dir: Path,
    filename: str = "predictions.csv",
) -> None:
    """Save predictions to CSV file.

    Args:
        predictions: Predicted labels or probabilities
        true_labels: True labels
        output_dir: Directory to save file
        filename: Output filename
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    if predictions.ndim == 1:
        df = pd.DataFrame({
            "true_label": true_labels,
            "prediction": predictions,
        })
    else:
        df = pd.DataFrame({
            "true_label": true_labels,
        })
        for i in range(predictions.shape[1]):
            df[f"prob_class_{i}"] = predictions[:, i]

    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")
