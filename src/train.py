"""Training and Cross-Validation module for K-NN and baseline models."""

import os
from typing import Dict, Any, Sequence, Tuple
# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.features import build_knn_pipeline


def train_baseline_median(y_train: pd.Series) -> float:
    """Compute baseline prediction value (median Rings on training data).

    Args:
        y_train: Training target values.

    Returns:
        float: Median value of training target.
    """
    return float(y_train.median())


def cross_validate_knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k_values: Sequence[int] = range(1, 31),
    weights_list: Sequence[str] = ("uniform", "distance"),
    cv_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Execute K-Fold Cross-Validation strictly on training data to search for optimal K and weights.

    Args:
        X_train: Training features.
        y_train: Training targets.
        k_values: Range or list of K values to explore.
        weights_list: List of weights schemes ('uniform', 'distance').
        cv_folds: Number of CV splits.
        random_state: Random seed for KFold shuffling.

    Returns:
        pd.DataFrame: Table of K, Weights, Mean MAE, and MAE standard deviation.
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = []

    for weights in weights_list:
        for k in k_values:
            pipeline = build_knn_pipeline(n_neighbors=k, weights=weights, with_scaling=True)
            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring="neg_mean_absolute_error",
            )
            mean_mae = float(-scores.mean())
            std_mae = float(scores.std())
            results.append(
                {
                    "K": k,
                    "Weights": weights,
                    "MAE": mean_mae,
                    "MAE_STD": std_mae,
                }
            )

    return pd.DataFrame(results)


def select_best_hyperparameters(cv_results_df: pd.DataFrame) -> Dict[str, Any]:
    """Identify best hyperparameter configuration minimizing MAE.

    Args:
        cv_results_df: DataFrame containing CV evaluation grid.

    Returns:
        Dict[str, Any]: Best hyperparameter settings and corresponding validation score.
    """
    best_idx = cv_results_df["MAE"].idxmin()
    best_row = cv_results_df.loc[best_idx]
    return {
        "best_k": int(best_row["K"]),
        "best_weights": str(best_row["Weights"]),
        "best_cv_mae": float(best_row["MAE"]),
        "best_cv_std": float(best_row["MAE_STD"]),
    }


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_neighbors: int,
    weights: str = "uniform",
    with_scaling: bool = True,
) -> Pipeline:
    """Fit final model pipeline on entire training partition using selected hyperparameters.

    Args:
        X_train: Training features.
        y_train: Training targets.
        n_neighbors: Number of neighbors.
        weights: Neighbor weighting scheme.
        with_scaling: Whether to scale features.

    Returns:
        Pipeline: Fitted scikit-learn Pipeline.
    """
    pipeline = build_knn_pipeline(
        n_neighbors=n_neighbors,
        weights=weights,
        with_scaling=with_scaling,
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model_pipeline(
    pipeline: Pipeline,
    filepath: str = "models/abalone_knn_pipeline.pkl",
) -> None:
    """Persist pipeline to disk using joblib.

    Args:
        pipeline: Fitted pipeline.
        filepath: Destination file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)


def save_cv_results(
    cv_results_df: pd.DataFrame,
    filepath: str = "reports/cv_results.csv",
) -> None:
    """Save Cross-Validation table to CSV.

    Args:
        cv_results_df: DataFrame with CV results.
        filepath: Output CSV file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv_results_df.to_csv(filepath, index=False)
