"""Data loading utilities for tabular datasets.

This module provides functions to load various tabular datasets including
synthetic benchmarks, OpenML datasets, and Kaggle competitions.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def load_dataset(
    dataset_name: str = "synthetic",
    data_dir: Optional[Path] = None,
    n_samples: int = 10000,
    n_features: int = 50,
    n_informative: int = 30,
    n_redundant: int = 10,
    n_classes: int = 2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, any]]:
    """Load tabular dataset.

    Args:
        dataset_name: Name of dataset to load ('synthetic', 'openml', 'kaggle')
        data_dir: Directory containing dataset files
        n_samples: Number of samples for synthetic data
        n_features: Number of features for synthetic data
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        n_classes: Number of classes
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (features_df, target_series, metadata_dict)
    """
    logger.info(f"Loading dataset: {dataset_name}")

    if dataset_name == "synthetic":
        # Generate synthetic classification dataset
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=n_classes,
            n_clusters_per_class=2,
            flip_y=0.05,
            class_sep=0.8,
            random_state=random_state,
        )

        # Add some categorical features
        n_categorical = min(5, n_features // 10)
        for i in range(n_categorical):
            cat_idx = i * (n_features // n_categorical)
            n_categories = np.random.randint(3, 10)
            X[:, cat_idx] = np.random.randint(0, n_categories, size=n_samples)

        feature_names = [f"feature_{i}" for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name="target")

        metadata = {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "n_categorical": n_categorical,
            "task": "binary" if n_classes == 2 else "multiclass",
        }

        logger.info(
            f"Generated synthetic dataset: {X_df.shape[0]} samples, "
            f"{X_df.shape[1]} features, {n_classes} classes"
        )

    elif dataset_name == "openml":
        # For demonstration, generate synthetic data
        # In production, use: from sklearn.datasets import fetch_openml
        logger.warning(
            "OpenML dataset loading not implemented, using synthetic data"
        )
        return load_dataset(
            dataset_name="synthetic",
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            random_state=random_state,
        )

    elif dataset_name == "kaggle":
        # For demonstration, generate synthetic data
        # In production, load from CSV files
        logger.warning(
            "Kaggle dataset loading not implemented, using synthetic data"
        )
        return load_dataset(
            dataset_name="synthetic",
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            random_state=random_state,
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return X_df, y_series, metadata


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    batch_size: int = 256,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders for training, validation, and test sets.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features (optional)
        y_test: Test labels (optional)
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data

    Returns:
        Dictionary of DataLoaders for each split
    """
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
    }

    # Add test loader if provided
    if X_test is not None and y_test is not None:
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        dataloaders["test"] = test_loader

    logger.info(
        f"Created dataloaders: train={len(train_loader)} batches, "
        f"val={len(val_loader)} batches"
    )

    return dataloaders
