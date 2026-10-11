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
    assert len(data["pipeline_steps"]) > 0
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
    # 4,173 cleaned samples (4 anomalies filtered) -> 70% train = 2,921 samples
    assert data["total_training_samples"] == 2921
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


if __name__ == "__main__":
    print("Running automated API tests...")
    test_health_endpoint()
    print("[PASS] test_health_endpoint")
    test_model_card_endpoint()
    print("[PASS] test_model_card_endpoint")
    test_feature_ranges_endpoint()
    print("[PASS] test_feature_ranges_endpoint")
    test_predict_valid_sample()
    print("[PASS] test_predict_valid_sample")
    test_predict_physical_units_sample()
    print("[PASS] test_predict_physical_units_sample")
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
    print("\n[SUCCESS] ALL 10 API UNIT TESTS PASSED SUCCESSFULLY!")
