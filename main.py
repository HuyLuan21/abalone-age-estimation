"""Master pipeline orchestrator for Abalone Age Estimation Project.

Orchestrates data processing, EDA, baseline modeling, hyperparameter search,
model ablation (scaled vs. unscaled), test evaluation, and artifact generation.
"""

import argparse
import sys
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

from src.data import load_data, check_data_quality, split_data
from src.features import build_knn_pipeline
from src.train import (
    train_baseline_median,
    cross_validate_knn,
    select_best_hyperparameters,
    train_final_model,
    save_model_pipeline,
    save_cv_results,
)
from src.evaluate import (
    calculate_metrics,
    analyze_errors_by_age_group,
    plot_eda_figures,
    plot_mae_vs_k,
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_residuals,
    save_evaluation_tables,
)


def run_pipeline(
    data_path: str = "data/raw/abalone.data",
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    skip_eda: bool = False,
) -> None:
    """Execute the full end-to-end Machine Learning pipeline."""
    print("=" * 75)
    print(" PROJECT 06 - ABALONE AGE ESTIMATION (K-NN REGRESSION)")
    print("=" * 75)

    # 1. Load Data
    print("\n[1/7] Loading dataset and verifying schema...")
    df = load_data(data_path)
    quality = check_data_quality(df)
    print(f"  ✓ Total observations: {quality['num_rows']}, Columns: {quality['num_columns']}")
    print(f"  ✓ Missing values    : {sum(quality['missing_values'].values())}")
    print(f"  ✓ Duplicate rows    : {quality['duplicate_rows']}")

    # 2. Split Data (Strict Leakage Prevention)
    print("\n[2/7] Splitting data into Train (80%) and Test (20%)...")
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    print(f"  ✓ Train size: {X_train.shape[0]} samples")
    print(f"  ✓ Test size : {X_test.shape[0]} samples")

    # 3. Exploratory Data Analysis (Only on Train)
    if not skip_eda:
        print("\n[3/7] Generating EDA figures on Training set...")
        plot_eda_figures(X_train, y_train, output_dir="reports/figures")
        print("  ✓ Saved EDA figures to reports/figures/")
    else:
        print("\n[3/7] Skipping EDA generation (--skip-eda flag enabled).")

    # 4. Baseline Modeling (Median of Train Rings)
    print("\n[4/7] Evaluating Baseline (Median of Train Rings)...")
    baseline_val = train_baseline_median(y_train)
    baseline_pred = np.full(shape=len(y_test), fill_value=baseline_val)
    baseline_metrics = calculate_metrics(y_test, baseline_pred)
    print(f"  ✓ Median Baseline value : {baseline_val:.2f}")
    print(f"  ✓ Baseline Test MAE     : {baseline_metrics['MAE']:.3f}")
    print(f"  ✓ Baseline Test RMSE    : {baseline_metrics['RMSE']:.3f}")
    print(f"  ✓ Baseline Test R²      : {baseline_metrics['R2']:.3f}")

    # 5. Cross-Validation on Training set (Hyperparameter Tuning)
    print(f"\n[5/7] Running {cv_folds}-Fold Cross-Validation on Train (K=1..30, weights=uniform/distance)...")
    cv_results_df = cross_validate_knn(
        X_train,
        y_train,
        k_values=range(1, 31),
        weights_list=["uniform", "distance"],
        cv_folds=cv_folds,
        random_state=random_state,
    )
    save_cv_results(cv_results_df, "reports/cv_results.csv")
    plot_mae_vs_k(cv_results_df, "reports/figures/mae_vs_k.png")

    best_config = select_best_hyperparameters(cv_results_df)
    best_k = best_config["best_k"]
    best_weights = best_config["best_weights"]
    print(f"  ✓ Best Hyperparameters  : K = {best_k}, Weights = '{best_weights}'")
    print(f"  ✓ Best CV MAE           : {best_config['best_cv_mae']:.4f} ± {best_config['best_cv_std']:.4f}")
    print("  ✓ Saved reports/cv_results.csv and reports/figures/mae_vs_k.png")

    # 6. Ablation Experiment: k-NN No Scale vs k-NN StandardScaler
    print("\n[6/7] Running Ablation Experiment: Unscaled vs. StandardScaler...")
    # Unscaled pipeline
    unscaled_pipeline = build_knn_pipeline(n_neighbors=best_k, weights=best_weights, with_scaling=False)
    unscaled_pipeline.fit(X_train, y_train)
    unscaled_pred = unscaled_pipeline.predict(X_test)
    unscaled_metrics = calculate_metrics(y_test, unscaled_pred)

    # Scaled pipeline (Final Model)
    final_pipeline = train_final_model(
        X_train,
        y_train,
        n_neighbors=best_k,
        weights=best_weights,
        with_scaling=True,
    )
    final_pred = final_pipeline.predict(X_test)
    final_metrics = calculate_metrics(y_test, final_pred)

    # Save model artifact
    save_model_pipeline(final_pipeline, "models/abalone_knn_pipeline.pkl")
    print("  ✓ Saved model artifact to models/abalone_knn_pipeline.pkl")

    # 7. Final Evaluation & Error Analysis
    print("\n[7/7] Generating final evaluations and error analysis...")
    results_df = pd.DataFrame(
        [
            {"Model": "Median Baseline", **baseline_metrics},
            {"Model": "KNN - No Scale", **unscaled_metrics},
            {"Model": "KNN - StandardScaler", **final_metrics},
        ]
    )

    plot_model_comparison(results_df, "reports/figures/model_comparison.png")
    plot_actual_vs_predicted(y_test, final_pred, "reports/figures/actual_vs_predicted.png")
    plot_residuals(y_test, final_pred, "reports/figures/residual_plot.png")

    group_mae = analyze_errors_by_age_group(y_test, final_pred)
    save_evaluation_tables(results_df, group_mae, "reports")

    # Summary Output
    print("\n" + "=" * 75)
    print(" SUMMARY OF EXPERIMENTAL RESULTS (TEST SET EVALUATION)")
    print("=" * 75)
    print(results_df.round(4).to_string(index=False))

    print("\n" + "-" * 75)
    print(" ERROR ANALYSIS ACROSS AGE GROUPS (MAE)")
    print("-" * 75)
    for group, mae in group_mae.items():
        print(f"  • {group:8s} : MAE = {mae:.3f} rings")

    print("\n" + "=" * 75)
    print(" Pipeline completed successfully. All artifacts saved.")
    print("=" * 75)


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Abalone Age Estimation using K-NN Regression Pipeline"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/abalone.data",
        help="Path to abalone raw data file (default: data/raw/abalone.data)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of dataset for test split (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for Cross Validation (default: 5)",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip generation of exploratory data figures",
    )

    args = parser.parse_args()
    run_pipeline(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        skip_eda=args.skip_eda,
    )


if __name__ == "__main__":
    main()