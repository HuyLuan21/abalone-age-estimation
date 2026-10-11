"""Unit tests for pipeline integrity, data splitting, zero leakage, and model serialization.

Validates compliance with Project 06 requirements:
- Clean 3-way split (Train 70%, Validation 15%, Test 15%)
- Disjoint indices and exact sample conservation
- Reproducibility across runs
- Leakage prevention: Preprocessor fit strictly on Train
- Baseline computed solely from Train
- Artifact serialization and deserialization
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ast
import json
# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_data, clean_anomalies, split_train_val_test
from src.features import build_knn_pipeline
from src.train import train_baseline_median, cross_validate_knn, select_best_hyperparameters


def test_anomaly_removal_and_sample_count():
    """Verify raw dataset has 4177 samples and exactly 4 biological/physical anomalies are removed."""
    raw_df = load_data("data/raw/abalone.data")
    assert len(raw_df) == 4177
    clean_df, anomalies = clean_anomalies(raw_df)
    assert len(anomalies) == 4
    assert len(clean_df) == 4173


def test_split_train_val_test_integrity_and_disjointness():
    """Test 70/15/15 partition ratios, index disjointness, and sample conservation."""
    raw_df = load_data("data/raw/abalone.data")
    clean_df, _ = clean_anomalies(raw_df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        clean_df, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42
    )

    # Expected exact sizes for 4,173 samples:
    # Test 15% = 626; Remainder 3547 -> Val 15% (626) -> Train 2921
    assert len(X_train) == 2921
    assert len(X_val) == 626
    assert len(X_test) == 626
    assert len(X_train) + len(X_val) + len(X_test) == 4173

    # Absolute disjointness of indices
    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)

    assert len(train_idx & val_idx) == 0, "Train and Validation share indices!"
    assert len(train_idx & test_idx) == 0, "Train and Test share indices!"
    assert len(val_idx & test_idx) == 0, "Validation and Test share indices!"


def test_split_reproducibility():
    """Test identical splits given the same random_state."""
    raw_df = load_data("data/raw/abalone.data")
    clean_df, _ = clean_anomalies(raw_df)

    X_tr1, X_v1, X_te1, _, _, _ = split_train_val_test(clean_df, random_state=42)
    X_tr2, X_v2, X_te2, _, _, _ = split_train_val_test(clean_df, random_state=42)

    pd.testing.assert_frame_equal(X_tr1, X_tr2)
    pd.testing.assert_frame_equal(X_v1, X_v2)
    pd.testing.assert_frame_equal(X_te1, X_te2)


def test_baseline_computed_only_from_train():
    """Verify baseline ring estimate is strictly calculated on train target."""
    raw_df = load_data("data/raw/abalone.data")
    clean_df, _ = clean_anomalies(raw_df)
    X_train, _, _, y_train, _, _ = split_train_val_test(clean_df, random_state=42)

    baseline_val = train_baseline_median(y_train)
    expected_val = float(y_train.median())
    assert baseline_val == expected_val
    assert baseline_val == 9.0


def test_pipeline_leakage_free_encapsulation():
    """Verify preprocessor is strictly fit within pipeline and does not leak test statistics."""
    raw_df = load_data("data/raw/abalone.data")
    clean_df, _ = clean_anomalies(raw_df)
    X_train, X_val, X_test, y_train, _, _ = split_train_val_test(clean_df, random_state=42)

    pipe = build_knn_pipeline(n_neighbors=5, weights="distance", with_scaling=True)
    pipe.fit(X_train, y_train)

    # Verify that the scaler inside the pipeline only learned from X_train
    scaler = pipe.named_steps["preprocessor"].named_transformers_["numeric"]
    train_mean = X_train[["Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight"]].mean().values
    np.testing.assert_allclose(scaler.mean_, train_mean, rtol=1e-5)


def test_model_artifact_load_and_predict():
    """Verify saved pipeline artifact can be loaded and perform valid inference."""
    artifact_path = "models/abalone_knn_pipeline.pkl"
    assert os.path.exists(artifact_path), f"Artifact missing at {artifact_path}"

    loaded_pipe = joblib.load(artifact_path)
    assert isinstance(loaded_pipe, Pipeline)

    # Dummy sample
    sample_df = pd.DataFrame(
        [
            {
                "Sex": "M",
                "Length": 0.455,
                "Diameter": 0.365,
                "Height": 0.095,
                "Whole_weight": 0.514,
                "Shucked_weight": 0.2245,
                "Viscera_weight": 0.101,
                "Shell_weight": 0.150,
            }
        ]
    )
    pred = loaded_pipe.predict(sample_df)
    assert len(pred) == 1
    assert 1.0 <= pred[0] <= 30.0


def test_metadata_consistency():
    """Verify model_metadata.json matches actual training outputs."""
    metadata_path = "models/model_metadata.json"
    assert os.path.exists(metadata_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["data_split"]["train_samples"] == 2921
    assert meta["data_split"]["validation_samples"] == 626
    assert meta["data_split"]["test_samples"] == 626
    assert "cv_performance" in meta
    assert "validation_performance" in meta
    assert "performance_metrics" in meta
    assert meta["system_environment"]["sklearn_version"] is not None


def test_held_out_test_evaluation_protocol():
    """Verify that Test metrics are strictly reserved for the locked primary model and baseline.

    Ablation candidate models must remain un-evaluated on Test (NaN/null) to prevent multiple
    hypothesis testing and data leakage on the held-out partition.
    """
    results_path = "reports/model_results.csv"
    if os.path.exists(results_path):
        df_res = pd.read_csv(results_path)
        # Identify rows for candidate models
        ablation_rows = df_res[df_res["Model"].str.contains("Ablation Candidate|StandardScaler", case=False, na=False)]
        for _, row in ablation_rows.iterrows():
            if "Ablation Candidate" in str(row.get("Selection_Stage", "")):
                assert pd.isna(row["Test_MAE"]), "Ablation candidate model must NOT have Test_MAE evaluated!"
                assert pd.isna(row["Test_R2"]), "Ablation candidate model must NOT have Test_R2 evaluated!"

        # Identify locked model
        locked_rows = df_res[df_res["Model"].str.contains("Locked Final Model|Selected & Locked", case=False, na=False)]
        assert len(locked_rows) >= 1, "Must contain locked model row in results"
        for _, row in locked_rows.iterrows():
            assert not pd.isna(row["Test_MAE"]), "Locked model must have valid Test_MAE!"
            assert row["Test_MAE"] > 0, "Test_MAE must be positive"


def test_model_locking_precedence_over_test_evaluation():
    """Verify strictly by AST and source analysis that artifact locking precedes any Test evaluation.

    Asserts:
    1. save_model_pipeline occurs before any prediction on X_test in main.py.
    2. No baseline test evaluation touches y_test before save_model_pipeline.
    3. No candidate model (alt_pipeline) calls .predict(X_test).
    """
    main_py_path = os.path.join(os.path.dirname(__file__), "..", "main.py")
    with open(main_py_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    tree = ast.parse(source_code)

    save_pipeline_line = None
    predict_x_test_lines = []
    calculate_metrics_test_lines = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check for save_model_pipeline
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name == "save_model_pipeline":
                save_pipeline_line = node.lineno

            # Check for .predict(X_test)
            if func_name == "predict":
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id == "X_test":
                        caller = ""
                        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                            caller = node.func.value.id
                        predict_x_test_lines.append((node.lineno, caller))

            # Check for calculate_metrics with y_test
            if func_name == "calculate_metrics":
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id == "y_test":
                        calculate_metrics_test_lines.append(node.lineno)

    assert save_pipeline_line is not None, "save_model_pipeline call not found in main.py!"

    # 1. Assert that ALL .predict(X_test) calls occur strictly AFTER model locking
    assert len(predict_x_test_lines) > 0, "No predict(X_test) found in main.py!"
    for line_no, caller in predict_x_test_lines:
        assert line_no > save_pipeline_line, (
            f"predict(X_test) called at line {line_no} BEFORE save_model_pipeline at line {save_pipeline_line}!"
        )
        assert caller != "alt_pipeline", (
            f"alt_pipeline.predict(X_test) called at line {line_no}! Candidate models must not evaluate on Test."
        )

    # 2. Assert that ALL calculate_metrics(y_test, ...) occur strictly AFTER model locking
    assert len(calculate_metrics_test_lines) > 0, "No calculate_metrics(y_test) found in main.py!"
    for line_no in calculate_metrics_test_lines:
        assert line_no > save_pipeline_line, (
            f"calculate_metrics(y_test) called at line {line_no} BEFORE save_model_pipeline at line {save_pipeline_line}!"
        )


if __name__ == "__main__":
    print("Running pipeline verification test suite...")
    test_anomaly_removal_and_sample_count()
    print("[PASS] test_anomaly_removal_and_sample_count")
    test_split_train_val_test_integrity_and_disjointness()
    print("[PASS] test_split_train_val_test_integrity_and_disjointness")
    test_split_reproducibility()
    print("[PASS] test_split_reproducibility")
    test_baseline_computed_only_from_train()
    print("[PASS] test_baseline_computed_only_from_train")
    test_pipeline_leakage_free_encapsulation()
    print("[PASS] test_pipeline_leakage_free_encapsulation")
    test_model_artifact_load_and_predict()
    print("[PASS] test_model_artifact_load_and_predict")
    test_metadata_consistency()
    print("[PASS] test_metadata_consistency")
    test_held_out_test_evaluation_protocol()
    print("[PASS] test_held_out_test_evaluation_protocol")
    test_model_locking_precedence_over_test_evaluation()
    print("[PASS] test_model_locking_precedence_over_test_evaluation")
    print("\n[SUCCESS] ALL PIPELINE UNIT TESTS PASSED SUCCESSFULLY!")


