"""Automated tests for FastAPI endpoints."""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient

try:
    from app.api import app
except ModuleNotFoundError:
    from api import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_card_endpoint():
    """Test model card metadata endpoint."""
    response = client.get("/api/model-card")
    assert response.status_code == 200
    data = response.json()
    assert "algorithm" in data
    assert "performance_metrics" in data
    assert "hyperparameters" in data
    assert "age_group_mae" in data
    assert "validation_performance" in data
    assert "data_split" in data
    assert data["data_split"]["train_samples"] == 2921
    assert data["data_split"]["validation_samples"] == 626
    assert data["data_split"]["test_samples"] == 626
    # Verify pipeline preprocessing and scaling strictly match No Scale
    assert data["hyperparameters"]["scaling"] == "No Scale", (
        f"Expected 'No Scale', but got: {data['hyperparameters']['scaling']}"
    )
    steps_combined = " ".join(data["pipeline_steps"])
    assert "StandardScaler" not in steps_combined, (
        f"Unexpected StandardScaler found in Model Card pipeline_steps: {data['pipeline_steps']}"
    )
    assert "OneHotEncoder" in steps_combined
    assert "KNeighborsRegressor" in steps_combined

    # Verify exact synchronization with reports/age_group_mae.csv
    assert "Young (<=8 rings)" in data["age_group_mae"]
    assert "Middle (9-12 rings)" in data["age_group_mae"]
    assert "Old (>12 rings)" in data["age_group_mae"]
    assert data["age_group_mae"]["Young (<=8 rings)"] < 1.3
    assert data["age_group_mae"]["Old (>12 rings)"] > 3.0


def test_feature_ranges_endpoint():
    """Test feature ranges and statistics endpoint."""
    response = client.get("/api/feature-ranges")
    assert response.status_code == 200
    data = response.json()
    assert "numeric_ranges" in data
    assert "Length" in data["numeric_ranges"]
    assert "valid_sex_values" in data
    # 4,173 cleaned samples -> 85% Train+Val refitted serving = 3,547 samples (initial 70% train = 2,921)
    assert data["total_training_samples"] in (2921, 3547)
    assert "physical_unit" in data["numeric_ranges"]["Length"]
    assert data["numeric_ranges"]["Length"]["physical_unit"] == "mm"


def test_predict_valid_sample():
    """Test prediction endpoint with a standard valid sample."""
    sample = {
        "Sex": "M",
        "Length": 0.455,
        "Diameter": 0.365,
        "Height": 0.095,
        "Whole_weight": 0.5140,
        "Shucked_weight": 0.2245,
        "Viscera_weight": 0.1010,
        "Shell_weight": 0.1500,
    }
    response = client.post("/api/predict", json=sample)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_rings" in data
    assert "estimated_age_years" in data
    assert "age_group" in data
    assert "reference_interval" in data
    assert len(data["nearest_neighbors"]) > 0
    assert data["estimated_age_years"] == round(data["predicted_rings"] + 1.5, 2)


def test_predict_physical_units_sample():
    """Test prediction endpoint with physical mm/g units (auto converted / 200)."""
    # Physical: Length = 91.0 mm (=0.455 UCI), Diameter = 73.0 mm (=0.365 UCI),
    # Height = 19.0 mm (=0.095 UCI), Whole_weight = 102.8 g (=0.514 UCI)
    sample = {
        "Sex": "M",
        "Length": 91.0,
        "Diameter": 73.0,
        "Height": 19.0,
        "Whole_weight": 102.8,
        "Shucked_weight": 44.9,
        "Viscera_weight": 20.2,
        "Shell_weight": 30.0,
    }
    response = client.post("/api/predict", json=sample)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_rings" in data
    assert "estimated_age_years" in data
    assert data["estimated_age_years"] == round(data["predicted_rings"] + 1.5, 2)


def test_predict_out_of_distribution_warning():
    """Test prediction with an out-of-distribution physical sample."""
    sample = {
        "Sex": "F",
        "Length": 0.950,  # Far above normal historical max (~0.815)
        "Diameter": 0.750,
        "Height": 0.350,
        "Whole_weight": 3.500,
        "Shucked_weight": 1.600,
        "Viscera_weight": 0.850,
        "Shell_weight": 1.100,
    }
    response = client.post("/api/predict", json=sample)
    assert response.status_code == 200
    data = response.json()
    assert data["is_out_of_distribution"] is True
    assert len(data["warnings"]) > 0


def test_predict_invalid_diameter_greater_than_length():
    """Test validation rejection when Diameter > Length."""
    sample = {
        "Sex": "I",
        "Length": 0.300,
        "Diameter": 0.400,  # Invalid: Diameter > Length
        "Height": 0.100,
        "Whole_weight": 0.200,
        "Shucked_weight": 0.080,
        "Viscera_weight": 0.040,
        "Shell_weight": 0.060,
    }
    response = client.post("/api/predict", json=sample)
    assert response.status_code == 422


def test_predict_invalid_negative_measurement():
    """Test validation rejection for non-positive values."""
    sample = {
        "Sex": "M",
        "Length": -0.1,  # Invalid negative length
        "Diameter": 0.365,
        "Height": 0.095,
        "Whole_weight": 0.514,
        "Shucked_weight": 0.224,
        "Viscera_weight": 0.101,
        "Shell_weight": 0.150,
    }
    response = client.post("/api/predict", json=sample)
    assert response.status_code == 422


def test_predict_mixed_units_rejection():
    """Test validation rejection when request mixes physical mm with UCI normalized weights."""
    sample = {
        "Sex": "M",
        "Length": 91.0,  # Physical mm
        "Diameter": 73.0,  # Physical mm
        "Height": 19.0,  # Physical mm
        "Whole_weight": 0.514,  # UCI scale (< 4.0 g, mixed units!)
        "Shucked_weight": 0.224,
        "Viscera_weight": 0.101,
        "Shell_weight": 0.150,
    }
    response = client.post("/api/predict", json=sample)
    assert response.status_code == 422


def test_predict_explicit_unit_system():
    """Test explicit unit_system declaration for physical and uci."""
    physical_sample = {
        "Sex": "F",
        "Length": 105.0,
        "Diameter": 82.0,
        "Height": 25.0,
        "Whole_weight": 140.0,
        "Shucked_weight": 60.0,
        "Viscera_weight": 30.0,
        "Shell_weight": 42.0,
        "unit_system": "physical",
    }
    resp1 = client.post("/api/predict", json=physical_sample)
    assert resp1.status_code == 200

    uci_sample = {
        "Sex": "F",
        "Length": 0.525,
        "Diameter": 0.410,
        "Height": 0.125,
        "Whole_weight": 0.700,
        "Shucked_weight": 0.300,
        "Viscera_weight": 0.150,
        "Shell_weight": 0.210,
        "unit_system": "uci",
    }
    resp2 = client.post("/api/predict", json=uci_sample)
    assert resp2.status_code == 200


def test_predict_physical_mode_with_uci_weights_rejected():
    """Verify that unit_system='physical' strictly rejects UCI-scale weights (Reviewer Item 1)."""
    sample = {
        "Sex": "M",
        "Length": 91.0,  # Physical mm
        "Diameter": 73.0,  # Physical mm
        "Height": 19.0,  # Physical mm
        "Whole_weight": 0.514,  # UCI scale weight (< 4.0 g)
        "Shucked_weight": 0.2245,
        "Viscera_weight": 0.101,
        "Shell_weight": 0.15,
        "unit_system": "physical",
    }
    response = client.post("/api/predict", json=sample)
    assert response.status_code == 422
    data = response.json()
    assert "thang đo UCI" in data["detail"] or "khối lượng" in data["detail"]


def test_predict_physical_mode_with_uci_sizes_rejected():
    """Verify that unit_system='physical' strictly rejects UCI-scale dimensions."""
    sample = {
        "Sex": "M",
        "Length": 0.455,  # UCI scale size (<= 1.5)
        "Diameter": 0.365,
        "Height": 0.095,
        "Whole_weight": 102.8,  # Physical grams
        "Shucked_weight": 44.9,
        "Viscera_weight": 20.2,
        "Shell_weight": 30.0,
        "unit_system": "physical",
    }
    response = client.post("/api/predict", json=sample)
    assert response.status_code == 422
    data = response.json()
    assert "kích thước" in data["detail"] or "thang đo UCI" in data["detail"]


def test_predict_uci_mode_with_physical_values_rejected():
    """Verify that unit_system='uci' strictly rejects physical millimeter or gram values."""
    sample = {
        "Sex": "M",
        "Length": 91.0,  # Physical mm (> 1.5)
        "Diameter": 73.0,
        "Height": 19.0,
        "Whole_weight": 102.8,  # Physical g (> 4.0)
        "Shucked_weight": 44.9,
        "Viscera_weight": 20.2,
        "Shell_weight": 30.0,
        "unit_system": "uci",
    }
    response = client.post("/api/predict", json=sample)
    assert response.status_code == 422
    data = response.json()
    assert "vượt ngưỡng thang UCI" in data["detail"]


def test_predict_boundary_cases_auto_unit_detection():
    """Verify auto-detection rejects borderline mixed-unit requests."""
    # Length in mm, but weight in UCI scale without unit_system declared
    sample = {
        "Sex": "I",
        "Length": 45.0,  # Physical mm
        "Diameter": 35.0,
        "Height": 10.0,
        "Whole_weight": 0.250,  # UCI scale
        "Shucked_weight": 0.110,
        "Viscera_weight": 0.050,
        "Shell_weight": 0.075,
    }
    response = client.post("/api/predict", json=sample)
    assert response.status_code == 422


def test_spa_frontend_serving():
    """Verify that FastAPI serves frontend index.html when dist exists, or fallback onboarding HTML when missing."""
    response = client.get("/")
    assert response.status_code == 200
    if os.path.exists("frontend/dist/index.html"):
        assert "<div id=\"root\">" in response.text or "vite" in response.text.lower()
    else:
        assert "Giao diện Web Frontend chưa được biên dịch" in response.text or "Abalone Age Estimator" in response.text


def test_predict_valid_small_juvenile_physical_sample():
    """Verify that a valid small juvenile abalone with small physical weight (<4g) is accepted (P1 issue)."""
    # Equivalent to UCI min sample: Length 15mm (=0.075 UCI), Whole_weight 0.40g (=0.002 UCI)
    juvenile_sample = {
        "Sex": "I",
        "Length": 15.0,
        "Diameter": 11.0,
        "Height": 2.0,
        "Whole_weight": 0.40,
        "Shucked_weight": 0.20,
        "Viscera_weight": 0.10,
        "Shell_weight": 0.10,
        "unit_system": "physical",
    }
    response = client.post("/api/predict", json=juvenile_sample)
    assert response.status_code == 200, f"Juvenile physical sample rejected: {response.text}"
    data = response.json()
    assert "predicted_rings" in data
    assert data["predicted_rings"] > 0


def test_model_card_matches_loaded_pipeline_artifact():
    """Verify that Model Card returned by API matches the serialized pipeline artifact identically."""
    import joblib
    model_path = os.path.join(PROJECT_ROOT, "models", "abalone_knn_pipeline.pkl")
    pipeline = joblib.load(model_path)
    knn = pipeline.named_steps["knn"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Check that fitted pipeline has passthrough for numeric (No Scale)
    for name, trans, _ in preprocessor.transformers_:
        if name == "numeric":
            assert trans == "passthrough" or not hasattr(trans, "mean_"), (
                f"Pipeline should not have fitted scaler, got {trans}"
            )

    response = client.get("/api/model-card")
    assert response.status_code == 200
    card = response.json()
    assert card["hyperparameters"]["n_neighbors"] == knn.n_neighbors
    assert card["hyperparameters"]["weights"] == knn.weights
    assert card["hyperparameters"]["scaling"] == "No Scale"


def test_predict_physical_boundary_samples():
    """Verify edge-case biological physical samples are correctly accepted."""
    # 1. Boundary juvenile: Length = 35.0 mm, Whole_weight = 3.20 g (<4g)
    juvenile_boundary = {
        "Sex": "I",
        "Length": 35.0,
        "Diameter": 27.0,
        "Height": 8.0,
        "Whole_weight": 3.20,
        "Shucked_weight": 1.40,
        "Viscera_weight": 0.70,
        "Shell_weight": 1.00,
        "unit_system": "physical",
    }
    r1 = client.post("/api/predict", json=juvenile_boundary)
    assert r1.status_code == 200, f"Boundary juvenile rejected: {r1.text}"

    # 2. Boundary adult: Length = 40.0 mm, Whole_weight = 7.70 g
    adult_boundary = {
        "Sex": "M",
        "Length": 40.0,
        "Diameter": 31.0,
        "Height": 10.0,
        "Whole_weight": 7.70,
        "Shucked_weight": 3.50,
        "Viscera_weight": 1.60,
        "Shell_weight": 2.40,
        "unit_system": "physical",
    }
    r2 = client.post("/api/predict", json=adult_boundary)
    assert r2.status_code == 200, f"Boundary adult rejected: {r2.text}"

    # 3. Maximum size adult: Length = 160.0 mm, Whole_weight = 520.0 g
    large_adult = {
        "Sex": "F",
        "Length": 160.0,
        "Diameter": 125.0,
        "Height": 45.0,
        "Whole_weight": 520.0,
        "Shucked_weight": 250.0,
        "Viscera_weight": 120.0,
        "Shell_weight": 150.0,
        "unit_system": "physical",
    }
    r3 = client.post("/api/predict", json=large_adult)
    assert r3.status_code == 200, f"Large adult rejected: {r3.text}"


def test_predict_physical_boundary_rejections():
    """Verify that boundary mixed-unit or sub-biological inputs are strictly rejected."""
    # Boundary mixed: Length = 40.0 mm (adult length), but Whole_weight = 0.514 g (UCI scale)
    mixed_boundary = {
        "Sex": "M",
        "Length": 40.0,
        "Diameter": 31.0,
        "Height": 10.0,
        "Whole_weight": 0.514,
        "Shucked_weight": 0.224,
        "Viscera_weight": 0.101,
        "Shell_weight": 0.150,
        "unit_system": "physical",
    }
    r1 = client.post("/api/predict", json=mixed_boundary)
    assert r1.status_code == 422, "Adult length with UCI weight should be rejected"

    # Sub-biological: Whole_weight < 0.1 g
    sub_bio = {
        "Sex": "I",
        "Length": 15.0,
        "Diameter": 11.0,
        "Height": 2.0,
        "Whole_weight": 0.05,
        "Shucked_weight": 0.02,
        "Viscera_weight": 0.01,
        "Shell_weight": 0.01,
        "unit_system": "physical",
    }
    r2 = client.post("/api/predict", json=sub_bio)
    assert r2.status_code == 422, "Weight < 0.1g should be rejected"


if __name__ == "__main__":
    print("Running automated API tests...")
    test_health_endpoint()
    print("[PASS] test_health_endpoint")
    test_model_card_endpoint()
    print("[PASS] test_model_card_endpoint")
    test_model_card_matches_loaded_pipeline_artifact()
    print("[PASS] test_model_card_matches_loaded_pipeline_artifact")
    test_feature_ranges_endpoint()
    print("[PASS] test_feature_ranges_endpoint")
    test_predict_valid_sample()
    print("[PASS] test_predict_valid_sample")
    test_predict_physical_units_sample()
    print("[PASS] test_predict_physical_units_sample")
    test_predict_valid_small_juvenile_physical_sample()
    print("[PASS] test_predict_valid_small_juvenile_physical_sample")
    test_predict_physical_boundary_samples()
    print("[PASS] test_predict_physical_boundary_samples")
    test_predict_physical_boundary_rejections()
    print("[PASS] test_predict_physical_boundary_rejections")
    test_predict_out_of_distribution_warning()
    print("[PASS] test_predict_out_of_distribution_warning")
    test_predict_invalid_diameter_greater_than_length()
    print("[PASS] test_predict_invalid_diameter_greater_than_length")
    test_predict_invalid_negative_measurement()
    print("[PASS] test_predict_invalid_negative_measurement")
    test_predict_mixed_units_rejection()
    print("[PASS] test_predict_mixed_units_rejection")
    test_predict_explicit_unit_system()
    print("[PASS] test_predict_explicit_unit_system")
    test_predict_physical_mode_with_uci_weights_rejected()
    print("[PASS] test_predict_physical_mode_with_uci_weights_rejected")
    test_predict_physical_mode_with_uci_sizes_rejected()
    print("[PASS] test_predict_physical_mode_with_uci_sizes_rejected")
    test_predict_uci_mode_with_physical_values_rejected()
    print("[PASS] test_predict_uci_mode_with_physical_values_rejected")
    test_predict_boundary_cases_auto_unit_detection()
    print("[PASS] test_predict_boundary_cases_auto_unit_detection")
    test_spa_frontend_serving()
    print("[PASS] test_spa_frontend_serving")
    print("\n[SUCCESS] ALL 19 API & FULLSTACK UNIT TESTS PASSED SUCCESSFULLY!")
