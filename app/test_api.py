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
    assert len(data["pipeline_steps"]) > 0


def test_feature_ranges_endpoint():
    """Test feature ranges and statistics endpoint."""
    response = client.get("/api/feature-ranges")
    assert response.status_code == 200
    data = response.json()
    assert "numeric_ranges" in data
    assert "Length" in data["numeric_ranges"]
    assert "valid_sex_values" in data


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


def test_predict_out_of_distribution_warning():
    """Test prediction with an out-of-distribution physical sample."""
    sample = {
        "Sex": "F",
        "Length": 0.950,  # Far above normal historical max (0.815)
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
    """Test validation rejection for negative measurements."""
    sample = {
        "Sex": "M",
        "Length": -0.455,  # Invalid: <= 0
        "Diameter": 0.365,
        "Height": 0.095,
        "Whole_weight": 0.514,
        "Shucked_weight": 0.224,
        "Viscera_weight": 0.101,
        "Shell_weight": 0.150,
    }
    response = client.post("/api/predict", json=sample)
    assert response.status_code == 422


if __name__ == "__main__":
    print("Running automated API tests...")
    test_health_endpoint()
    print("✓ test_health_endpoint PASSED")
    test_model_card_endpoint()
    print("✓ test_model_card_endpoint PASSED")
    test_feature_ranges_endpoint()
    print("✓ test_feature_ranges_endpoint PASSED")
    test_predict_valid_sample()
    print("✓ test_predict_valid_sample PASSED")
    test_predict_out_of_distribution_warning()
    print("✓ test_predict_out_of_distribution_warning PASSED")
    test_predict_invalid_diameter_greater_than_length()
    print("✓ test_predict_invalid_diameter_greater_than_length PASSED")
    test_predict_invalid_negative_measurement()
    print("✓ test_predict_invalid_negative_measurement PASSED")
    print("\n🎉 ALL 7 API UNIT TESTS PASSED SUCCESSFULLY!")
