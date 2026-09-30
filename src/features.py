"""Feature engineering and preprocessing pipelines for Abalone Age Estimation."""

from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

# pyrefly: ignore [missing-import]
from src.data import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor(with_scaling: bool = True) -> ColumnTransformer:
    """Build ColumnTransformer for categorical and numerical features.

    - Categorical features (`Sex`): OneHotEncoder (handle_unknown="ignore")
    - Numerical features: StandardScaler if with_scaling=True, else passthrough.

    Args:
        with_scaling: Whether to standardize numerical features.

    Returns:
        ColumnTransformer: Configured transformer.
    """
    numeric_transformer = StandardScaler() if with_scaling else "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def build_knn_pipeline(
    n_neighbors: int = 5,
    weights: str = "uniform",
    with_scaling: bool = True,
) -> Pipeline:
    """Construct full scikit-learn Pipeline encapsulating preprocessing and KNeighborsRegressor.

    Args:
        n_neighbors: Number of nearest neighbors (K).
        weights: Weight function used in prediction ('uniform' or 'distance').
        with_scaling: Whether to apply StandardScaler to numerical features.

    Returns:
        Pipeline: Complete trainable scikit-learn pipeline.
    """
    preprocessor = build_preprocessor(with_scaling=with_scaling)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "knn",
                KNeighborsRegressor(
                    n_neighbors=n_neighbors,
                    weights=weights,
                ),
            ),
        ]
    )
    return pipeline
