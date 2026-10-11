"""Evaluation, error analysis, and visualization module for Abalone Age Estimation."""

import os
from typing import Dict, Any, Sequence
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# pyrefly: ignore [missing-import]
from src.data import NUMERIC_FEATURES


def calculate_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """Calculate regression metrics: MAE, RMSE, R².

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted values.

    Returns:
        Dict[str, float]: Dictionary containing MAE, RMSE, and R2.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def analyze_errors_by_age_group(
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> pd.Series:
    """Analyze MAE across biological age categories:

    - Young: Rings <= 8
    - Middle: 8 < Rings <= 12
    - Old: Rings > 12

    Args:
        y_test: Actual test rings.
        y_pred: Predicted test rings.

    Returns:
        pd.Series: MAE aggregated by AgeGroup.
    """
    error_df = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Predicted": y_pred,
        }
    )
    error_df["AgeGroup"] = pd.cut(
        error_df["Actual"],
        bins=[0, 8, 12, np.inf],
        labels=["Young", "Middle", "Old"],
    )

    group_mae = (
        error_df.groupby("AgeGroup", observed=True)
        .apply(lambda group: mean_absolute_error(group["Actual"], group["Predicted"]))
    )
    return group_mae


def plot_eda_figures(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: str = "reports/figures",
) -> None:
    """Generate exploratory data analysis charts strictly on training set."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Rings target distribution
    plt.figure(figsize=(8, 5))
    plt.hist(y_train, bins=20, color="#3b82f6", edgecolor="black", alpha=0.8)
    plt.title("Distribution of Rings - Training Set", fontsize=14, fontweight="bold")
    plt.xlabel("Rings", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rings_distribution.png"), dpi=300)
    plt.close()

    # 2. Sex distribution
    plt.figure(figsize=(6, 4))
    counts = X_train["Sex"].value_counts()
    counts.plot(kind="bar", color=["#60a5fa", "#34d399", "#f87171"], edgecolor="black")
    plt.title("Sex Distribution - Training Set", fontsize=14, fontweight="bold")
    plt.xlabel("Sex", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sex_distribution.png"), dpi=300)
    plt.close()

    # 3. Numeric features vs Rings
    for feat in NUMERIC_FEATURES:
        plt.figure(figsize=(7, 5))
        plt.scatter(X_train[feat], y_train, color="#2563eb", alpha=0.4, edgecolors="none")
        plt.xlabel(feat, fontsize=12)
        plt.ylabel("Rings", fontsize=12)
        plt.title(f"{feat} vs Rings (Training Set)", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{feat}_vs_rings.png"), dpi=300)
        plt.close()


def plot_mae_vs_k(
    cv_results_df: pd.DataFrame,
    output_path: str = "reports/figures/mae_vs_k.png",
) -> None:
    """Plot Cross-Validation MAE curve across various K values for both weighting schemes and scalings."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if "Scaling" in cv_results_df.columns and len(cv_results_df["Scaling"].unique()) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
        scalings = ["StandardScaler", "No Scale"]
        for idx, sc in enumerate(scalings):
            ax = axes[idx]
            sub_sc = cv_results_df[cv_results_df["Scaling"] == sc]
            for weights, color, marker in [("uniform", "#2563eb", "o"), ("distance", "#10b981", "s")]:
                subset = sub_sc[sub_sc["Weights"] == weights]
                if not subset.empty:
                    ax.plot(
                        subset["K"],
                        subset["MAE"],
                        marker=marker,
                        label=f"Weights = {weights}",
                        color=color,
                        linewidth=2,
                        markersize=5,
                    )
            ax.set_title(f"Ablation CV: {sc}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Number of Neighbors (K)", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=10)
        axes[0].set_ylabel("Cross-Validation MAE", fontsize=11)
        fig.suptitle("Cross-Validation MAE vs. K (Ablation: Scaled vs. Unscaled on Train)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.figure(figsize=(9, 6))
        for weights, color, marker in [("uniform", "#2563eb", "o"), ("distance", "#10b981", "s")]:
            subset = cv_results_df[cv_results_df["Weights"] == weights]
            if not subset.empty:
                plt.plot(
                    subset["K"],
                    subset["MAE"],
                    marker=marker,
                    label=f"Weights = {weights}",
                    color=color,
                    linewidth=2,
                    markersize=6,
                )
        plt.xlabel("Number of Neighbors (K)", fontsize=12)
        plt.ylabel("Cross-Validation MAE", fontsize=12)
        plt.title("Cross-Validation MAE vs. K", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()


def plot_model_comparison(
    results_df: pd.DataFrame,
    output_path: str = "reports/figures/model_comparison.png",
) -> None:
    """Plot bar chart comparing model metrics on test set."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(9, 6))

    x = np.arange(len(results_df))
    plt.bar(
        x,
        results_df["R2"],
        color=["#94a3b8", "#f59e0b", "#10b981"],
        edgecolor="black",
        width=0.55,
    )
    plt.xticks(x, results_df["Model"], rotation=15, fontsize=11)
    plt.ylabel("R² Score", fontsize=12)
    plt.title("Model Performance Comparison (R² Score on Test Set)", fontsize=14, fontweight="bold")
    plt.axhline(0, color="gray", linewidth=1)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_actual_vs_predicted(
    y_test: pd.Series,
    y_pred: np.ndarray,
    output_path: str = "reports/figures/actual_vs_predicted.png",
) -> None:
    """Plot actual vs predicted scatter plot with ideal diagonal reference line."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(7, 7))

    plt.scatter(y_test, y_pred, alpha=0.45, color="#3b82f6", edgecolors="none")
    min_val = min(float(y_test.min()), float(y_pred.min()))
    max_val = max(float(y_test.max()), float(y_pred.max()))
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color="#ef4444",
        linewidth=2,
        label="Ideal (y = x)",
    )

    plt.xlabel("Actual Rings", fontsize=12)
    plt.ylabel("Predicted Rings", fontsize=12)
    plt.title("Actual vs. Predicted Rings (Test Set)", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_residuals(
    y_test: pd.Series,
    y_pred: np.ndarray,
    output_path: str = "reports/figures/residual_plot.png",
) -> None:
    """Plot residual error vs actual rings."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, residuals, alpha=0.45, color="#8b5cf6", edgecolors="none")
    plt.axhline(0, color="#ef4444", linestyle="--", linewidth=1.5)
    plt.xlabel("Actual Rings", fontsize=12)
    plt.ylabel("Residual (Actual - Predicted)", fontsize=12)
    plt.title("Residual Analysis (Test Set)", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_evaluation_tables(
    results_df: pd.DataFrame,
    group_mae_series: pd.Series,
    reports_dir: str = "reports",
    cv_ablation_df: Any = None,
) -> None:
    """Save model performance summary tables to CSV."""
    os.makedirs(reports_dir, exist_ok=True)
    results_df.to_csv(os.path.join(reports_dir, "model_results.csv"), index=False)
    group_mae_series.to_csv(os.path.join(reports_dir, "age_group_mae.csv"))
    if cv_ablation_df is not None and isinstance(cv_ablation_df, pd.DataFrame):
        cv_ablation_df.to_csv(os.path.join(reports_dir, "cv_ablation.csv"), index=False)

