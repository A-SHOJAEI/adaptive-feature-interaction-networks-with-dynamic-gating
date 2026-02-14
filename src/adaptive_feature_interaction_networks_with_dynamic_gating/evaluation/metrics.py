"""Evaluation metrics for classification tasks.

This module provides comprehensive metrics for binary and multiclass classification,
including AUC-ROC, log loss, accuracy, precision, recall, and F1 score.
"""

import logging
from typing import Dict, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss as sklearn_log_loss,
    classification_report,
    confusion_matrix,
)
import torch

logger = logging.getLogger(__name__)


def compute_auc_roc(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    average: str = "macro",
    multi_class: str = "ovr",
) -> float:
    """Compute AUC-ROC score.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        average: Averaging strategy for multiclass
        multi_class: Multiclass strategy ('ovr' or 'ovo')

    Returns:
        AUC-ROC score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    try:
        if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
            # Binary classification
            auc_roc = roc_auc_score(y_true, y_pred)
        else:
            # Multiclass classification
            auc_roc = roc_auc_score(
                y_true, y_pred, average=average, multi_class=multi_class
            )
        return float(auc_roc)
    except Exception as e:
        logger.warning(f"Failed to compute AUC-ROC: {e}")
        return 0.0


def compute_log_loss(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-15,
) -> float:
    """Compute log loss (cross-entropy loss).

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        eps: Epsilon for numerical stability

    Returns:
        Log loss value
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    try:
        # Clip predictions for numerical stability
        y_pred = np.clip(y_pred, eps, 1 - eps)
        logloss = sklearn_log_loss(y_true, y_pred)
        return float(logloss)
    except Exception as e:
        logger.warning(f"Failed to compute log loss: {e}")
        return float("inf")


def compute_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    y_pred_proba: Optional[Union[np.ndarray, torch.Tensor]] = None,
    task: str = "binary",
    average: str = "macro",
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted class labels
        y_pred_proba: Predicted probabilities (for AUC-ROC and log loss)
        task: Task type ('binary' or 'multiclass')
        average: Averaging strategy for multiclass metrics

    Returns:
        Dictionary of metrics
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_pred_proba, torch.Tensor):
        y_pred_proba = y_pred_proba.cpu().numpy()

    metrics = {}

    # Basic metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(
        precision_score(y_true, y_pred, average=average, zero_division=0)
    )
    metrics["recall"] = float(
        recall_score(y_true, y_pred, average=average, zero_division=0)
    )
    metrics["f1"] = float(
        f1_score(y_true, y_pred, average=average, zero_division=0)
    )

    # Probability-based metrics
    if y_pred_proba is not None:
        metrics["auc_roc"] = compute_auc_roc(y_true, y_pred_proba)
        metrics["log_loss"] = compute_log_loss(y_true, y_pred_proba)

    logger.info(f"Computed metrics: {metrics}")
    return metrics


def compute_per_class_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    class_names: Optional[list] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-class metrics.

    Args:
        y_true: True labels
        y_pred: Predicted class labels
        class_names: Names of classes (optional)

    Returns:
        Dictionary mapping class names to their metrics
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Get classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Get unique classes
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

    per_class_metrics = {}
    for cls in unique_classes:
        cls_str = str(int(cls))
        if cls_str in report:
            class_name = class_names[cls] if class_names else f"Class_{cls}"
            per_class_metrics[class_name] = report[cls_str]

    return per_class_metrics


def compute_confusion_matrix(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    normalize: Optional[str] = None,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted class labels
        normalize: Normalization mode ('true', 'pred', 'all', or None)

    Returns:
        Confusion matrix
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    return cm


def compute_feature_interaction_sparsity(
    gate_activations: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.01,
) -> float:
    """Compute feature interaction sparsity metric.

    This measures the proportion of gate activations below a threshold,
    indicating sparse feature usage.

    Args:
        gate_activations: Gate activation values (batch_size, num_gates)
        threshold: Threshold for considering a gate inactive

    Returns:
        Sparsity score (0 to 1, higher = more sparse)
    """
    if isinstance(gate_activations, torch.Tensor):
        gate_activations = gate_activations.cpu().numpy()

    # Compute proportion of activations below threshold
    below_threshold = (gate_activations < threshold).mean()
    return float(below_threshold)


def compute_inference_speedup(
    inference_time_proposed: float,
    inference_time_baseline: float,
) -> float:
    """Compute inference speedup vs baseline.

    Args:
        inference_time_proposed: Inference time of proposed model (seconds)
        inference_time_baseline: Inference time of baseline model (seconds)

    Returns:
        Speedup factor (>1 means faster)
    """
    if inference_time_proposed <= 0:
        return 0.0
    speedup = inference_time_baseline / inference_time_proposed
    return float(speedup)
