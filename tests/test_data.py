"""Tests for data loading and preprocessing."""

import numpy as np
import pandas as pd
import pytest
import torch

from adaptive_feature_interaction_networks_with_dynamic_gating.data.loader import (
    load_dataset,
    create_dataloaders,
)
from adaptive_feature_interaction_networks_with_dynamic_gating.data.preprocessing import (
    TabularPreprocessor,
)


class TestDataLoader:
    """Tests for data loading functionality."""

    def test_load_synthetic_dataset(self):
        """Test loading synthetic dataset."""
        X_df, y_series, metadata = load_dataset(
            dataset_name="synthetic",
            n_samples=100,
            n_features=10,
            n_classes=2,
            random_state=42,
        )

        assert isinstance(X_df, pd.DataFrame)
        assert isinstance(y_series, pd.Series)
        assert X_df.shape == (100, 10)
        assert len(y_series) == 100
        assert metadata["n_classes"] == 2

    def test_load_dataset_multiclass(self):
        """Test loading multiclass dataset."""
        X_df, y_series, metadata = load_dataset(
            dataset_name="synthetic",
            n_samples=100,
            n_features=10,
            n_classes=5,
            random_state=42,
        )

        assert metadata["n_classes"] == 5
        assert y_series.nunique() == 5

    def test_create_dataloaders(self):
        """Test creating PyTorch dataloaders."""
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.randn(20, 10)
        y_val = np.random.randint(0, 2, 20)

        dataloaders = create_dataloaders(
            X_train, y_train, X_val, y_val, batch_size=32
        )

        assert "train" in dataloaders
        assert "val" in dataloaders
        assert len(dataloaders["train"]) > 0
        assert len(dataloaders["val"]) > 0

        # Check batch shapes
        for batch_x, batch_y in dataloaders["train"]:
            assert batch_x.shape[1] == 10
            assert batch_y.shape[0] == batch_x.shape[0]
            break


class TestTabularPreprocessor:
    """Tests for tabular data preprocessing."""

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = TabularPreprocessor(
            scaling_method="standard",
            categorical_encoding="label",
        )
        assert preprocessor.scaling_method == "standard"
        assert preprocessor.categorical_encoding == "label"

    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        X_df, y_series = sample_data
        preprocessor = TabularPreprocessor()
        X_processed = preprocessor.fit_transform(X_df)

        assert isinstance(X_processed, np.ndarray)
        assert X_processed.shape[0] == X_df.shape[0]
        assert preprocessor.is_fitted

    def test_transform_without_fit(self, sample_data):
        """Test that transform raises error before fit."""
        X_df, _ = sample_data
        preprocessor = TabularPreprocessor()

        with pytest.raises(ValueError):
            preprocessor.transform(X_df)

    def test_scaling(self, sample_data):
        """Test feature scaling."""
        X_df, _ = sample_data
        preprocessor = TabularPreprocessor(scaling_method="standard")
        X_processed = preprocessor.fit_transform(X_df)

        # Check that features are approximately standardized
        assert np.abs(X_processed.mean()) < 0.1
        assert np.abs(X_processed.std() - 1.0) < 0.2

    def test_categorical_encoding(self):
        """Test categorical feature encoding."""
        # Create data with categorical features
        X_df = pd.DataFrame({
            "cat1": ["A", "B", "A", "C", "B"],
            "cat2": ["X", "Y", "X", "Y", "X"],
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        preprocessor = TabularPreprocessor(categorical_encoding="label")
        X_processed = preprocessor.fit_transform(X_df)

        assert isinstance(X_processed, np.ndarray)
        assert X_processed.shape[0] == 5

    def test_missing_value_handling(self):
        """Test missing value imputation."""
        X_df = pd.DataFrame({
            "num1": [1.0, 2.0, np.nan, 4.0, 5.0],
            "num2": [np.nan, 2.0, 3.0, 4.0, 5.0],
        })

        preprocessor = TabularPreprocessor(handle_missing="mean")
        X_processed = preprocessor.fit_transform(X_df)

        # Check no NaN values remain
        assert not np.isnan(X_processed).any()

    def test_feature_type_detection(self):
        """Test automatic feature type detection."""
        X_df = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "num2": [1.5, 2.5, 3.5, 4.5, 5.5],
            "cat1": ["A", "B", "A", "C", "B"],
        })

        preprocessor = TabularPreprocessor()
        preprocessor.fit(X_df)

        assert len(preprocessor.numerical_features) > 0
        assert "num1" in preprocessor.numerical_features
        assert "num2" in preprocessor.numerical_features
