"""Master pipeline orchestrator for Abalone Age Estimation Project.

Orchestrates data processing, EDA, baseline modeling, hyperparameter search,
preprocessing ablation (scaled vs. unscaled on train CV), final test evaluation,
and artifact/metadata generation.
"""

import argparse
import sys
from typing import Dict, Any

# Ensure terminal encoding compatibility across Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

from src.data import load_data, check_data_quality, split_data, NUMERIC_FEATURES
from src.features import build_knn_pipeline
from src.train import (
    train_baseline_median,
    cross_validate_knn,
    get_ablation_cv_summary,
    select_best_hyperparameters,
    train_final_model,
    save_model_pipeline,
    save_cv_results,
    save_model_metadata,
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


def compute_feature_statistics(X_train: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute statistical boundaries and metadata dynamically from the training set."""
    descriptions = {
        "Length": "Chiều dài lớn nhất của vỏ (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 mm)",
        "Diameter": "Đường kính vuông góc với chiều dài (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 mm)",
        "Height": "Chiều cao kèm thịt trong vỏ (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 mm)",
        "Whole_weight": "Tổng khối lượng toàn bộ cá thể (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 gam)",
        "Shucked_weight": "Khối lượng phần thịt (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 gam)",
        "Viscera_weight": "Khối lượng nội tạng sau khi mổ (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 gam)",
        "Shell_weight": "Khối lượng vỏ sấy khô (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 gam)",
    }
    units = {
        "Length": "mm",
        "Diameter": "mm",
        "Height": "mm",
        "Whole_weight": "g",
        "Shucked_weight": "g",
        "Viscera_weight": "g",
        "Shell_weight": "g",
    }

    stats = {}
    for col in NUMERIC_FEATURES:
        series = X_train[col]
        stats[col] = {
            "unit": units.get(col, "-"),
            "desc": descriptions.get(col, col),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
        }
    return stats


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
    print(f"  [OK] Total observations: {quality['num_rows']}, Columns: {quality['num_columns']}")
    print(f"  [OK] Missing values    : {sum(quality['missing_values'].values())}")
    print(f"  [OK] Duplicate rows    : {quality['duplicate_rows']}")

    # 2. Split Data (Strict Leakage Prevention: Test set isolated before any modeling)
    print("\n[2/7] Splitting data into Train (80%) and Test (20%)...")
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    print(f"  [OK] Train size: {X_train.shape[0]} samples")
    print(f"  [OK] Test size : {X_test.shape[0]} samples")

    # Dynamic training statistics
    feature_stats = compute_feature_statistics(X_train)

    # 3. Exploratory Data Analysis (Strictly on Train)
    if not skip_eda:
        print("\n[3/7] Generating EDA figures on Training set...")
        plot_eda_figures(X_train, y_train, output_dir="reports/figures")
        print("  [OK] Saved EDA figures to reports/figures/")
    else:
        print("\n[3/7] Skipping EDA generation (--skip-eda flag enabled).")

    # 4. Baseline Modeling (Median of Train Rings)
    print("\n[4/7] Evaluating Baseline (Median of Train Rings)...")
    baseline_val = train_baseline_median(y_train)
    baseline_pred = np.full(shape=len(y_test), fill_value=baseline_val)
    baseline_metrics = calculate_metrics(y_test, baseline_pred)
    print(f"  [OK] Median Baseline value : {baseline_val:.2f}")
    print(f"  [OK] Baseline Test MAE     : {baseline_metrics['MAE']:.3f}")
    print(f"  [OK] Baseline Test RMSE    : {baseline_metrics['RMSE']:.3f}")
    print(f"  [OK] Baseline Test R2      : {baseline_metrics['R2']:.3f}")

    # 5. Cross-Validation on Training set (Hyperparameter Tuning & Ablation Study)
    print(f"\n[5/7] Running {cv_folds}-Fold Cross-Validation on Train (K=1..30, weights, scaling)...")
    cv_results_df = cross_validate_knn(
        X_train,
        y_train,
        k_values=range(1, 31),
        weights_list=["uniform", "distance"],
        scaling_list=[True, False],
        cv_folds=cv_folds,
        random_state=random_state,
    )
    save_cv_results(cv_results_df, "reports/cv_results.csv")
    plot_mae_vs_k(cv_results_df, "reports/figures/mae_vs_k.png")

    cv_ablation_df = get_ablation_cv_summary(cv_results_df)
    cv_ablation_df.to_csv("reports/cv_ablation.csv", index=False)
    print("  [OK] Cross-Validation Ablation Summary on Training Set (5-Fold CV):")
    for _, row in cv_ablation_df.iterrows():
        print(f"    - {row['Preprocessing']:14s}: Best K={row['Best_K']:2d}, Weights='{row['Best_Weights']}', CV MAE={row['CV_MAE']:.4f} +/- {row['CV_STD']:.4f}")

    # Select best configurations based STRICTLY on Train CV
    best_scaled_config = select_best_hyperparameters(cv_results_df, force_scaling=True)
    best_unscaled_config = select_best_hyperparameters(cv_results_df, force_scaling=False)

    best_k = best_scaled_config["best_k"]
    best_weights = best_scaled_config["best_weights"]
    print(f"\n  [OK] Selected Primary Model (StandardScaler): K = {best_k}, Weights = '{best_weights}'")
    print(f"  [OK] Best Scaled CV MAE     : {best_scaled_config['best_cv_mae']:.4f} +/- {best_scaled_config['best_cv_std']:.4f}")
    print(f"  [OK] Best Unscaled CV MAE   : {best_unscaled_config['best_cv_mae']:.4f} (K={best_unscaled_config['best_k']}, Weights='{best_unscaled_config['best_weights']}')")
    print("  [OK] Saved reports/cv_results.csv, reports/cv_ablation.csv and reports/figures/mae_vs_k.png")

    # 6. Fit Final Model Pipelines on Full Training Set
    print("\n[6/7] Fitting final model pipeline on entire training set...")
    final_pipeline = train_final_model(
        X_train,
        y_train,
        n_neighbors=best_k,
        weights=best_weights,
        with_scaling=True,
    )
    save_model_pipeline(final_pipeline, "models/abalone_knn_pipeline.pkl")
    print("  [OK] Saved primary model artifact to models/abalone_knn_pipeline.pkl")

    # Fit unscaled pipeline (with its CV-optimal parameters) for final test reporting
    unscaled_pipeline = build_knn_pipeline(
        n_neighbors=best_unscaled_config["best_k"],
        weights=best_unscaled_config["best_weights"],
        with_scaling=False,
    )
    unscaled_pipeline.fit(X_train, y_train)

    # 7. Final Independent Test Set Evaluation & Error Analysis
    print("\n[7/7] Evaluating independently on Test set (held-out evaluation)...")
    final_pred = final_pipeline.predict(X_test)
    final_metrics = calculate_metrics(y_test, final_pred)

    unscaled_pred = unscaled_pipeline.predict(X_test)
    unscaled_metrics = calculate_metrics(y_test, unscaled_pred)

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
    save_evaluation_tables(results_df, group_mae, "reports", cv_ablation_df=cv_ablation_df)

    # Save comprehensive model and dataset metadata for dynamic API synchronization
    model_metadata = {
        "model_name": "Abalone Age Estimator (K-NN Regression)",
        "algorithm": "K-Nearest Neighbors Regressor with Euclidean Distance",
        "dataset_source": "UCI Machine Learning Repository (DOI: 10.24432/C55C7W)",
        "intended_use": "Hỗ trợ nghiên cứu nuôi trồng và sàng lọc mẫu bào ngư cần kiểm tra vi thể.",
        "total_training_samples": int(len(X_train)),
        "total_test_samples": int(len(X_test)),
        "valid_sex_values": sorted(list(X_train["Sex"].unique())),
        "pipeline_steps": [
            "StandardScaler (7 numerical features)",
            "OneHotEncoder (Sex categorical feature)",
            f"KNeighborsRegressor(n_neighbors={best_k}, weights='{best_weights}')",
        ],
        "hyperparameters": {
            "n_neighbors": best_k,
            "weights": best_weights,
            "metric": "minkowski (p=2 Euclidean)",
            "scaling": "StandardScaler",
            "cv_folds": cv_folds,
            "random_state": random_state,
        },
        "cv_performance": {
            "best_scaled_cv_mae": round(float(best_scaled_config["best_cv_mae"]), 4),
            "best_scaled_cv_std": round(float(best_scaled_config["best_cv_std"]), 4),
            "best_unscaled_cv_mae": round(float(best_unscaled_config["best_cv_mae"]), 4),
            "best_unscaled_cv_std": round(float(best_unscaled_config["best_cv_std"]), 4),
        },
        "performance_metrics": {
            "Test_MAE": round(float(final_metrics["MAE"]), 4),
            "Test_RMSE": round(float(final_metrics["RMSE"]), 4),
            "Test_R2": round(float(final_metrics["R2"]), 4),
            "Baseline_Median_MAE": round(float(baseline_metrics["MAE"]), 4),
        },
        "age_group_mae": {
            "Young (<=8 rings)": round(float(group_mae["Young"]), 4),
            "Middle (9-12 rings)": round(float(group_mae["Middle"]), 4),
            "Old (>12 rings)": round(float(group_mae["Old"]), 4),
        },
        "limitations": [
            f"Mô hình đạt độ chính xác cao nhất ở nhóm non (MAE = {group_mae['Young']:.3f}) và trung bình (MAE = {group_mae['Middle']:.3f}).",
            f"Sai số tăng cao rõ rệt ở nhóm già (>12 vòng, MAE = {group_mae['Old']:.3f}) do mật độ cá thể già ít và sinh trưởng phi tuyến tính.",
            "Quy trình so sánh mô hình (Scaling vs No-Scale) được chọn dựa trên 5-Fold Cross-Validation trên Train để bảo đảm không rò rỉ tập Test.",
            "Ước lượng hỗ trợ sàng lọc sơ bộ, không thay thế quy trình cắt vỏ và đếm vòng kính hiển vi tiêu chuẩn.",
        ],
        "feature_statistics": feature_stats,
    }
    save_model_metadata(model_metadata, "models/model_metadata.json")
    print("  [OK] Saved models/model_metadata.json (Dynamic API synchronization)")

    # Summary Output
    print("\n" + "=" * 75)
    print(" SUMMARY OF EXPERIMENTAL RESULTS (TEST SET EVALUATION)")
    print("=" * 75)
    print(results_df.round(4).to_string(index=False))

    print("\n" + "-" * 75)
    print(" ERROR ANALYSIS ACROSS AGE GROUPS (MAE ON TEST SET)")
    print("-" * 75)
    for group, mae in group_mae.items():
        print(f"  - {group:8s} : MAE = {mae:.3f} rings")

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