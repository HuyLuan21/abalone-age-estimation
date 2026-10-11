"""Master pipeline orchestrator for Abalone Age Estimation Project.

Orchestrates data processing, anomaly cleaning, strict 3-way train/validation/test
partitioning, baseline modeling, 5-Fold cross-validation on train, validation set
evaluation and model locking, single held-out test evaluation, and dynamic
artifact/metadata synchronization.
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
import sklearn

from src.data import (
    load_data,
    check_data_quality,
    clean_anomalies,
    split_train_val_test,
    NUMERIC_FEATURES,
)
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
    """Compute statistical boundaries and dual-unit metadata strictly from training partition."""
    descriptions = {
        "Length": "Chiều dài lớn nhất của vỏ (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 mm)",
        "Diameter": "Đường kính vuông góc với chiều dài (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 mm)",
        "Height": "Chiều cao kèm thịt trong vỏ (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 mm)",
        "Whole_weight": "Tổng khối lượng toàn bộ cá thể (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 gam)",
        "Shucked_weight": "Khối lượng phần thịt (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 gam)",
        "Viscera_weight": "Khối lượng nội tạng sau khi mổ (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 gam)",
        "Shell_weight": "Khối lượng vỏ sấy khô (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 gam)",
    }
    physical_units = {
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
        col_min = round(float(series.min()), 4)
        col_max = round(float(series.max()), 4)
        col_mean = round(float(series.mean()), 4)
        col_std = round(float(series.std()), 4)
        phys_u = physical_units.get(col, "-")

        stats[col] = {
            "desc": descriptions.get(col, col),
            "unit": "UCI scale (đã chia 200)",
            "min": col_min,
            "max": col_max,
            "mean": col_mean,
            "std": col_std,
            "physical_unit": phys_u,
            "physical_min": round(col_min * 200.0, 2),
            "physical_max": round(col_max * 200.0, 2),
            "physical_mean": round(col_mean * 200.0, 2),
        }
    return stats


def run_pipeline(
    data_path: str = "data/raw/abalone.data",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    cv_folds: int = 5,
    skip_eda: bool = False,
) -> None:
    """Execute the full end-to-end Machine Learning pipeline with strict Train/Validation/Test workflow."""
    print("=" * 75)
    print(" PROJECT 06 - ABALONE AGE ESTIMATION (K-NN REGRESSION)")
    print(" STRICT TRAIN / VALIDATION / TEST WORKFLOW")
    print("=" * 75)

    # 1. Load Data & Quality Assurance
    print("\n[1/8] Loading dataset and verifying schema & anomalies...")
    raw_df = load_data(data_path)
    quality = check_data_quality(raw_df)
    print(f"  [OK] Total observations in raw data: {quality['num_rows']}, Columns: {quality['num_columns']}")
    print(f"  [OK] Missing values                 : {sum(quality['missing_values'].values())}")
    print(f"  [OK] Duplicate rows                 : {quality['duplicate_rows']}")

    clean_df, anomalies_df = clean_anomalies(
        raw_df, audit_report_path="reports/removed_anomalies.csv"
    )
    print(f"  [OK] Filtered physical anomalies    : {len(anomalies_df)} samples removed")
    print(f"  [OK] Remaining cleaned observations : {len(clean_df)} samples")

    # 2. Strict 3-Way Splitting (Train 70%, Validation 15%, Test 15%)
    print(f"\n[2/8] Splitting data into Train ({train_ratio*100:.0f}%), Validation ({val_ratio*100:.0f}%), Test ({test_ratio*100:.0f}%)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        clean_df,
        train_size=train_ratio,
        val_size=val_ratio,
        test_size=test_ratio,
        random_state=random_state,
        auto_clean=False,
    )
    print(f"  [OK] Train size     : {X_train.shape[0]} samples ({X_train.shape[0]/len(clean_df)*100:.2f}%)")
    print(f"  [OK] Validation size: {X_val.shape[0]} samples ({X_val.shape[0]/len(clean_df)*100:.2f}%)")
    print(f"  [OK] Test size      : {X_test.shape[0]} samples ({X_test.shape[0]/len(clean_df)*100:.2f}%)")

    # Dynamic training statistics (strictly calculated on Train partition)
    feature_stats = compute_feature_statistics(X_train)

    # 3. Exploratory Data Analysis (Strictly on Training Partition)
    if not skip_eda:
        print("\n[3/8] Generating EDA figures on Training set...")
        plot_eda_figures(X_train, y_train, output_dir="reports/figures")
        print("  [OK] Saved EDA figures to reports/figures/")
    else:
        print("\n[3/8] Skipping EDA generation (--skip-eda flag enabled).")

    # 4. Baseline Modeling (Median of Train Rings Only)
    print("\n[4/8] Evaluating Baseline (Median of Train Rings)...")
    baseline_val = train_baseline_median(y_train)

    baseline_val_pred = np.full(shape=len(y_val), fill_value=baseline_val)
    baseline_val_metrics = calculate_metrics(y_val, baseline_val_pred)

    baseline_test_pred = np.full(shape=len(y_test), fill_value=baseline_val)
    baseline_test_metrics = calculate_metrics(y_test, baseline_test_pred)

    print(f"  [OK] Median Baseline value : {baseline_val:.2f} rings")
    print(f"  [OK] Baseline Validation MAE: {baseline_val_metrics['MAE']:.4f}")
    print(f"  [OK] Baseline Held-out Test MAE: {baseline_test_metrics['MAE']:.4f}")

    # 5. 5-Fold Cross-Validation on Training set (Hyperparameter Tuning & Ablation Study)
    print(f"\n[5/8] Running {cv_folds}-Fold Cross-Validation on Train (K=1..30, weights, scaling)...")
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
    best_overall_config = select_best_hyperparameters(cv_results_df, force_scaling=None)
    best_scaled_config = select_best_hyperparameters(cv_results_df, force_scaling=True)
    best_unscaled_config = select_best_hyperparameters(cv_results_df, force_scaling=False)

    best_k = best_overall_config["best_k"]
    best_weights = best_overall_config["best_weights"]
    best_scaling = best_overall_config["best_scaling"]
    with_scaling = best_overall_config["with_scaling"]

    print(f"\n  [OK] Best Configuration from Train CV ({best_scaling}): K = {best_k}, Weights = '{best_weights}'")
    print(f"  [OK] Best Overall CV MAE    : {best_overall_config['best_cv_mae']:.4f} +/- {best_overall_config['best_cv_std']:.4f}")
    print(f"  [OK] Best Scaled CV MAE     : {best_scaled_config['best_cv_mae']:.4f} (K={best_scaled_config['best_k']}, Weights='{best_scaled_config['best_weights']}')")
    print(f"  [OK] Best Unscaled CV MAE   : {best_unscaled_config['best_cv_mae']:.4f} (K={best_unscaled_config['best_k']}, Weights='{best_unscaled_config['best_weights']}')")

    # 6. Validation Set Evaluation & Model Locking (Hold-out Validation Confirmation)
    print("\n[6/8] Evaluating candidate models on Validation Set (Hold-out Validation)...")
    primary_pipeline = train_final_model(
        X_train,
        y_train,
        n_neighbors=best_k,
        weights=best_weights,
        with_scaling=with_scaling,
    )
    primary_val_pred = primary_pipeline.predict(X_val)
    primary_val_metrics = calculate_metrics(y_val, primary_val_pred)

    # Fit alternative ablation model for comparison on Validation set
    alt_scaling = not with_scaling
    alt_config = best_scaled_config if alt_scaling else best_unscaled_config
    alt_pipeline = build_knn_pipeline(
        n_neighbors=alt_config["best_k"],
        weights=alt_config["best_weights"],
        with_scaling=alt_scaling,
    )
    alt_pipeline.fit(X_train, y_train)
    alt_val_pred = alt_pipeline.predict(X_val)
    alt_val_metrics = calculate_metrics(y_val, alt_val_pred)

    val_records = [
        {"Model": "Median Baseline", **baseline_val_metrics},
    ]
    if with_scaling:
        val_records.append({"Model": "KNN - No Scale", **alt_val_metrics})
        val_records.append({"Model": "KNN - StandardScaler", **primary_val_metrics})
    else:
        val_records.append({"Model": "KNN - No Scale", **primary_val_metrics})
        val_records.append({"Model": "KNN - StandardScaler", **alt_val_metrics})

    val_results_df = pd.DataFrame(val_records)
    val_results_df.to_csv("reports/validation_results.csv", index=False)
    print("  [OK] Saved reports/validation_results.csv")
    print("  [OK] Validation Set Results:")
    print(val_results_df.round(4).to_string(index=False))

    # Lock primary model artifact BEFORE opening test set
    save_model_pipeline(primary_pipeline, "models/abalone_knn_pipeline.pkl")
    print(f"\n  [LOCKED] Primary model pipeline artifact saved to models/abalone_knn_pipeline.pkl ({best_scaling})")

    # 7. Final Independent Test Set Evaluation (Held-out Evaluation - Opened ONLY ONCE)
    print("\n[7/8] Evaluating locked model independently on Held-out Test set (Final Evaluation)...")
    final_test_pred = primary_pipeline.predict(X_test)
    final_test_metrics = calculate_metrics(y_test, final_test_pred)

    # TUÂN THỦ NGUYÊN TẮC NIÊM PHONG TẬP TEST (ZERO DATA LEAKAGE PROTOCOL):
    # alt_pipeline TUYỆT ĐỐI KHÔNG được gọi trên tập Test.
    # Việc đối sánh giữa No Scale và StandardScaler đã được hoàn thành khách quan tại bước 5-Fold CV và Validation Set.
    # Tập Test chỉ mở đúng 1 lần duy nhất để báo cáo hiệu năng mô hình đã chốt (primary_pipeline).

    # Assemble comprehensive results table with both Validation and Test metrics
    test_records = [
        {
            "Model": "Median Baseline",
            "Selection_Stage": "Baseline Benchmark",
            "Val_MAE": baseline_val_metrics["MAE"],
            "Val_RMSE": baseline_val_metrics["RMSE"],
            "Val_R2": baseline_val_metrics["R2"],
            "Test_MAE": baseline_test_metrics["MAE"],
            "Test_RMSE": baseline_test_metrics["RMSE"],
            "Test_R2": baseline_test_metrics["R2"],
            "MAE": baseline_test_metrics["MAE"],
            "RMSE": baseline_test_metrics["RMSE"],
            "R2": baseline_test_metrics["R2"],
        },
        {
            "Model": f"KNN - {best_scaling} (Selected & Locked)",
            "Selection_Stage": "Selected by CV & Locked on Val",
            "Val_MAE": primary_val_metrics["MAE"],
            "Val_RMSE": primary_val_metrics["RMSE"],
            "Val_R2": primary_val_metrics["R2"],
            "Test_MAE": final_test_metrics["MAE"],
            "Test_RMSE": final_test_metrics["RMSE"],
            "Test_R2": final_test_metrics["R2"],
            "MAE": final_test_metrics["MAE"],
            "RMSE": final_test_metrics["RMSE"],
            "R2": final_test_metrics["R2"],
        },
        {
            "Model": f"KNN - {'StandardScaler' if not with_scaling else 'No Scale'} (Ablation Candidate)",
            "Selection_Stage": "Eliminated on Validation",
            "Val_MAE": alt_val_metrics["MAE"],
            "Val_RMSE": alt_val_metrics["RMSE"],
            "Val_R2": alt_val_metrics["R2"],
            "Test_MAE": np.nan,
            "Test_RMSE": np.nan,
            "Test_R2": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
        },
    ]

    results_df = pd.DataFrame(test_records)
    plot_model_comparison(results_df, "reports/figures/model_comparison.png")
    plot_actual_vs_predicted(y_test, final_test_pred, "reports/figures/actual_vs_predicted.png")
    plot_residuals(y_test, final_test_pred, "reports/figures/residual_plot.png")

    group_mae = analyze_errors_by_age_group(y_test, final_test_pred)
    save_evaluation_tables(results_df, group_mae, "reports", cv_ablation_df=cv_ablation_df)

    # 8. Save Comprehensive Model and Dataset Metadata
    print("\n[8/8] Synchronizing dynamic metadata (models/model_metadata.json)...")
    pipeline_steps_list = (
        [
            "StandardScaler (7 numerical features)",
            "OneHotEncoder (Sex categorical feature)",
            f"KNeighborsRegressor(n_neighbors={best_k}, weights='{best_weights}')",
        ]
        if with_scaling
        else [
            "OneHotEncoder (Sex categorical feature)",
            f"KNeighborsRegressor(n_neighbors={best_k}, weights='{best_weights}')",
        ]
    )

    model_metadata = {
        "model_name": "Abalone Age Estimator (K-NN Regression)",
        "algorithm": "K-Nearest Neighbors Regressor with Euclidean Distance",
        "dataset_source": "UCI Machine Learning Repository (DOI: 10.24432/C55C7W)",
        "intended_use": "Hỗ trợ nghiên cứu nuôi trồng và sàng lọc mẫu bào ngư cần kiểm tra vi thể.",
        "data_split": {
            "train_samples": int(len(X_train)),
            "validation_samples": int(len(X_val)),
            "test_samples": int(len(X_test)),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio),
            "random_state": int(random_state),
            "total_clean_samples": int(len(X_train) + len(X_val) + len(X_test)),
            "removed_anomalies": int(len(anomalies_df)),
        },
        "total_training_samples": int(len(X_train)),
        "total_validation_samples": int(len(X_val)),
        "total_test_samples": int(len(X_test)),
        "valid_sex_values": sorted(list(X_train["Sex"].unique())),
        "system_environment": {
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "pipeline_steps": pipeline_steps_list,
        "hyperparameters": {
            "n_neighbors": best_k,
            "weights": best_weights,
            "metric": "minkowski (p=2 Euclidean)",
            "scaling": best_scaling,
            "cv_folds": cv_folds,
            "random_state": random_state,
        },
        "cv_performance": {
            "best_scaled_cv_mae": round(float(best_scaled_config["best_cv_mae"]), 4),
            "best_scaled_cv_std": round(float(best_scaled_config["best_cv_std"]), 4),
            "best_unscaled_cv_mae": round(float(best_unscaled_config["best_cv_mae"]), 4),
            "best_unscaled_cv_std": round(float(best_unscaled_config["best_cv_std"]), 4),
            "selected_cv_mae": round(float(best_overall_config["best_cv_mae"]), 4),
            "selected_cv_std": round(float(best_overall_config["best_cv_std"]), 4),
        },
        "validation_performance": {
            "Validation_MAE": round(float(primary_val_metrics["MAE"]), 4),
            "Validation_RMSE": round(float(primary_val_metrics["RMSE"]), 4),
            "Validation_R2": round(float(primary_val_metrics["R2"]), 4),
            "Baseline_Validation_MAE": round(float(baseline_val_metrics["MAE"]), 4),
        },
        "performance_metrics": {
            "Test_MAE": round(float(final_test_metrics["MAE"]), 4),
            "Test_RMSE": round(float(final_test_metrics["RMSE"]), 4),
            "Test_R2": round(float(final_test_metrics["R2"]), 4),
            "Baseline_Median_MAE": round(float(baseline_test_metrics["MAE"]), 4),
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
            "Tập Validation độc lập xác nhận hiệu năng mô hình trước khi đánh giá một lần duy nhất trên tập Test.",
            "Ước lượng hỗ trợ sàng lọc sơ bộ, không thay thế quy trình cắt vỏ và đếm vòng kính hiển vi tiêu chuẩn.",
        ],
        "feature_statistics": feature_stats,
    }
    save_model_metadata(model_metadata, "models/model_metadata.json")
    print("  [OK] Saved models/model_metadata.json (Dynamic API synchronization)")

    # Summary Output
    print("\n" + "=" * 75)
    print(" SUMMARY OF EXPERIMENTAL RESULTS (VALIDATION & TEST SETS)")
    print("=" * 75)
    display_cols = ["Model", "Val_MAE", "Val_RMSE", "Val_R2", "Test_MAE", "Test_RMSE", "Test_R2"]
    print(results_df[display_cols].round(4).to_string(index=False))

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
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion of dataset for train split (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of dataset for validation split (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of dataset for test split (default: 0.15)",
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
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        skip_eda=args.skip_eda,
    )


if __name__ == "__main__":
    main()