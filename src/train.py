"""Training and Cross-Validation module for K-NN and baseline models."""

import json
import os
from typing import Dict, Any, Sequence, Tuple
# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import numpy as np
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
    scaling_list: Sequence[bool] = (True, False),
    cv_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Execute K-Fold Cross-Validation strictly on training data across K, weights, and scaling.

    Compares preprocessing configurations (StandardScaler vs No Scale) strictly
    on the training partition to ensure zero test-data leakage.

    Args:
        X_train: Training features.
        y_train: Training targets.
        k_values: Range or list of K values to explore.
        weights_list: List of weights schemes ('uniform', 'distance').
        scaling_list: List of scaling options (True for StandardScaler, False for unscaled).
        cv_folds: Number of CV splits.
        random_state: Random seed for KFold shuffling.

    Returns:
        pd.DataFrame: Table of Scaling, Weights, K, Mean MAE, and MAE standard deviation.
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = []

    for scaling in scaling_list:
        scaling_label = "StandardScaler" if scaling else "No Scale"
        for weights in weights_list:
            for k in k_values:
                pipeline = build_knn_pipeline(n_neighbors=k, weights=weights, with_scaling=scaling)
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
                        "Scaling": scaling_label,
                        "with_scaling": bool(scaling),
                        "Weights": weights,
                        "K": int(k),
                        "MAE": mean_mae,
                        "MAE_STD": std_mae,
                    }
                )

    return pd.DataFrame(results)


def get_ablation_cv_summary(cv_results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the best CV performance for each scaling configuration on training set.

    Allows fair comparison of preprocessing options strictly using training validation.

    Args:
        cv_results_df: DataFrame containing all CV runs.

    Returns:
        pd.DataFrame: Summary table comparing best CV MAE between Scaled and Unscaled.
    """
    records = []
    for scaling_label, group in cv_results_df.groupby("Scaling"):
        best_idx = group["MAE"].idxmin()
        best_row = group.loc[best_idx]
        records.append(
            {
                "Preprocessing": scaling_label,
                "Best_K": int(best_row["K"]),
                "Best_Weights": str(best_row["Weights"]),
                "CV_MAE": float(best_row["MAE"]),
                "CV_STD": float(best_row["MAE_STD"]),
            }
        )
    return pd.DataFrame(records)


def select_best_hyperparameters(
    cv_results_df: pd.DataFrame,
    force_scaling: bool = True,
) -> Dict[str, Any]:
    """Identify best hyperparameter configuration minimizing MAE on CV.

    Args:
        cv_results_df: DataFrame containing CV evaluation grid.
        force_scaling: If True, selects the best configuration among StandardScaler models
                       (as recommended by assignment specification). If False, selects overall best.

    Returns:
        Dict[str, Any]: Best hyperparameter settings and corresponding validation score.
    """
    subset = cv_results_df[cv_results_df["with_scaling"] == force_scaling] if force_scaling is not None else cv_results_df
    best_idx = subset["MAE"].idxmin()
    best_row = subset.loc[best_idx]
    return {
        "best_k": int(best_row["K"]),
        "best_weights": str(best_row["Weights"]),
        "best_scaling": str(best_row["Scaling"]),
        "with_scaling": bool(best_row["with_scaling"]),
        "best_cv_mae": float(best_row["MAE"]),
        "best_cv_std": float(best_row["MAE_STD"]),
    }


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_neighbors: int,
    weights: str = "distance",
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


def refit_final_model(
    X_train_val: pd.DataFrame,
    y_train_val: pd.Series,
    n_neighbors: int,
    weights: str = "distance",
    with_scaling: bool = True,
) -> Pipeline:
    """Refit the locked pipeline on combined Train + Validation partition (85% of data).

    Re-fits the entire pipeline from scratch (including StandardScaler, OneHotEncoder,
    and KNeighborsRegressor) using the exact configuration chosen during Validation selection.
    Ensures zero leakage to the held-out Test set while maximizing training sample density.

    Args:
        X_train_val: Combined Train and Validation features (e.g. 85% data).
        y_train_val: Combined Train and Validation targets.
        n_neighbors: Number of neighbors locked from validation.
        weights: Neighbor weighting scheme locked from validation.
        with_scaling: Whether to scale features locked from validation.

    Returns:
        Pipeline: Re-fitted scikit-learn Pipeline on 85% data for production serving & test evaluation.
    """
    pipeline = build_knn_pipeline(
        n_neighbors=n_neighbors,
        weights=weights,
        with_scaling=with_scaling,
    )
    pipeline.fit(X_train_val, y_train_val)
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


def save_model_metadata(
    metadata: Dict[str, Any],
    filepath: str = "models/model_metadata.json",
) -> None:
    """Save model metadata, dataset bounds, and experimental results to JSON.

    Args:
        metadata: Metadata dictionary.
        filepath: Output JSON path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
