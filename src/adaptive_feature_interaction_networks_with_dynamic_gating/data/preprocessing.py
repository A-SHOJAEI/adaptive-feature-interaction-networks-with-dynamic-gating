"""Data preprocessing utilities for tabular data.

This module provides preprocessing pipelines for handling categorical features,
missing values, scaling, and feature engineering.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class TabularPreprocessor:
    """Preprocessor for tabular data with categorical and numerical features.

    This class handles:
    - Missing value imputation
    - Categorical encoding
    - Numerical feature scaling
    - Feature type detection

    Args:
        numerical_features: List of numerical feature names (auto-detected if None)
        categorical_features: List of categorical feature names (auto-detected if None)
        scaling_method: Scaling method ('standard', 'minmax', or None)
        categorical_encoding: Encoding method ('label', 'onehot')
        handle_missing: How to handle missing values ('mean', 'median', 'most_frequent')
    """

    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        scaling_method: str = "standard",
        categorical_encoding: str = "label",
        handle_missing: str = "mean",
    ) -> None:
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.scaling_method = scaling_method
        self.categorical_encoding = categorical_encoding
        self.handle_missing = handle_missing

        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.num_imputer: Optional[SimpleImputer] = None
        self.cat_imputer: Optional[SimpleImputer] = None
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TabularPreprocessor":
        """Fit preprocessor on training data.

        Args:
            X: Input features
            y: Target variable (unused, for sklearn compatibility)

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting preprocessor on {X.shape[0]} samples, {X.shape[1]} features")

        # Auto-detect feature types if not specified
        if self.numerical_features is None or self.categorical_features is None:
            self._detect_feature_types(X)

        self.feature_names = list(X.columns)

        # Fit numerical imputer
        if self.numerical_features:
            strategy = self.handle_missing if self.handle_missing in ["mean", "median"] else "mean"
            self.num_imputer = SimpleImputer(strategy=strategy)
            X_num = X[self.numerical_features]
            self.num_imputer.fit(X_num)

        # Fit categorical imputer and encoders
        if self.categorical_features:
            self.cat_imputer = SimpleImputer(strategy="most_frequent")
            X_cat = X[self.categorical_features]
            self.cat_imputer.fit(X_cat)

            # Fit label encoders
            X_cat_imputed = pd.DataFrame(
                self.cat_imputer.transform(X_cat),
                columns=self.categorical_features,
            )
            for col in self.categorical_features:
                le = LabelEncoder()
                le.fit(X_cat_imputed[col].astype(str))
                self.label_encoders[col] = le

        # Fit scaler on numerical features
        if self.scaling_method == "standard" and self.numerical_features:
            self.scaler = StandardScaler()
            X_num = X[self.numerical_features]
            if self.num_imputer is not None:
                X_num = pd.DataFrame(
                    self.num_imputer.transform(X_num),
                    columns=self.numerical_features,
                )
            self.scaler.fit(X_num)

        self.is_fitted = True
        logger.info("Preprocessor fitted successfully")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor.

        Args:
            X: Input features

        Returns:
            Transformed features as numpy array

        Raises:
            ValueError: If preprocessor has not been fitted
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        X_transformed = []

        # Transform numerical features
        if self.numerical_features:
            X_num = X[self.numerical_features]
            if self.num_imputer is not None:
                X_num = self.num_imputer.transform(X_num)
            if self.scaler is not None:
                X_num = self.scaler.transform(X_num)
            X_transformed.append(X_num)

        # Transform categorical features
        if self.categorical_features:
            X_cat = X[self.categorical_features]
            if self.cat_imputer is not None:
                X_cat = self.cat_imputer.transform(X_cat)

            X_cat_encoded = []
            for i, col in enumerate(self.categorical_features):
                col_data = X_cat[:, i].astype(str)
                # Handle unseen categories
                le = self.label_encoders[col]
                encoded = np.array([
                    le.transform([val])[0] if val in le.classes_ else -1
                    for val in col_data
                ])
                X_cat_encoded.append(encoded.reshape(-1, 1))

            if X_cat_encoded:
                X_cat_encoded = np.hstack(X_cat_encoded)
                X_transformed.append(X_cat_encoded)

        # Concatenate all features
        if X_transformed:
            X_final = np.hstack(X_transformed)
        else:
            X_final = X.values

        return X_final

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit preprocessor and transform data in one step.

        Args:
            X: Input features
            y: Target variable (unused)

        Returns:
            Transformed features as numpy array
        """
        return self.fit(X, y).transform(X)

    def _detect_feature_types(self, X: pd.DataFrame) -> None:
        """Auto-detect numerical and categorical features.

        Args:
            X: Input features
        """
        numerical = []
        categorical = []

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                # Check if it's actually categorical (few unique values)
                n_unique = X[col].nunique()
                if n_unique <= 20 and n_unique < len(X) * 0.05:
                    categorical.append(col)
                else:
                    numerical.append(col)
            else:
                categorical.append(col)

        self.numerical_features = numerical
        self.categorical_features = categorical

        logger.info(
            f"Auto-detected {len(numerical)} numerical and "
            f"{len(categorical)} categorical features"
        )

    def get_feature_names(self) -> List[str]:
        """Get feature names after transformation.

        Returns:
            List of feature names
        """
        if self.feature_names is None:
            return []
        return self.feature_names
