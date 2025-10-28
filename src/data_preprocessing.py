from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = ["location"]
NUMERIC_FEATURES = ["size_sqft", "rooms", "bathrooms", "age"]
TARGET_COLUMN = "price"


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_data(csv_path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    df = pd.read_csv(csv_path)
    expected_cols = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET_COLUMN])
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing expected columns: {sorted(missing)}")
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def build_preprocessor(
    categorical_features: List[str] | None = None,
    numeric_features: List[str] | None = None,
) -> ColumnTransformer:
    """Create a preprocessing transformer: impute + encode + scale."""
    cat_feats = categorical_features or CATEGORICAL_FEATURES
    num_feats = numeric_features or NUMERIC_FEATURES

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, cat_feats),
            ("num", numeric_pipeline, num_feats),
        ]
    )
    return preprocessor


def make_pipeline(model) -> Pipeline:
    """Wrap a model with preprocessing pipeline."""
    pre = build_preprocessor()
    return Pipeline(steps=[("preprocess", pre), ("model", model)])
