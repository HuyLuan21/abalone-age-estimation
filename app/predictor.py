"""Prediction service and OOD validation logic for Abalone Age Estimation."""

import os
import sys
from typing import Dict, List, Tuple, Any
# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

# Ensure project root is always in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from app.schemas import (
        AbaloneInput,
        NeighborInfo,
        ReferenceInterval,
        PredictionResponse,
        ModelCardResponse,
        DatasetStatsResponse,
        FeatureRangeItem,
    )
except ModuleNotFoundError:
    from schemas import (
        AbaloneInput,
        NeighborInfo,
        ReferenceInterval,
        PredictionResponse,
        ModelCardResponse,
        DatasetStatsResponse,
        FeatureRangeItem,
    )


FEATURE_METADATA = {
    "Length": {
        "unit": "mm",
        "desc": "Chiều dài lớn nhất của vỏ bào ngư",
        "min": 0.075,
        "max": 0.815,
        "mean": 0.524,
        "std": 0.120,
    },
    "Diameter": {
        "unit": "mm",
        "desc": "Đường kính vỏ vuông góc với chiều dài",
        "min": 0.055,
        "max": 0.650,
        "mean": 0.408,
        "std": 0.099,
    },
    "Height": {
        "unit": "mm",
        "desc": "Chiều cao kèm thịt trong vỏ",
        "min": 0.010,
        "max": 0.300,
        "mean": 0.140,
        "std": 0.042,
    },
    "Whole_weight": {
        "unit": "g",
        "desc": "Tổng khối lượng nguyên con",
        "min": 0.002,
        "max": 2.826,
        "mean": 0.829,
        "std": 0.490,
    },
    "Shucked_weight": {
        "unit": "g",
        "desc": "Khối lượng phần thịt bào ngư",
        "min": 0.001,
        "max": 1.488,
        "mean": 0.360,
        "std": 0.222,
    },
    "Viscera_weight": {
        "unit": "g",
        "desc": "Khối lượng nội tạng sau khi mổ",
        "min": 0.001,
        "max": 0.760,
        "mean": 0.181,
        "std": 0.110,
    },
    "Shell_weight": {
        "unit": "g",
        "desc": "Khối lượng vỏ sấy khô",
        "min": 0.002,
        "max": 1.005,
        "mean": 0.239,
        "std": 0.139,
    },
}


class AbalonePredictor:
    """Predictor service wrapping scikit-learn pipeline and domain validations."""

    def __init__(self, model_path: str = "models/abalone_knn_pipeline.pkl"):
        """Initialize predictor and load saved pipeline."""
        self.model_path = model_path
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Load fitted Pipeline from disk."""
        candidate_paths = [
            self.model_path,
            os.path.join(PROJECT_ROOT, self.model_path),
            os.path.join(os.path.dirname(__file__), "..", self.model_path),
            os.path.abspath(self.model_path),
        ]
        found_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                found_path = p
                break

        if not found_path:
            raise FileNotFoundError(
                f"Model pipeline not found at {self.model_path}. "
                "Please run 'python main.py' to train and save the pipeline first."
            )

        self.pipeline = joblib.load(found_path)

    def check_out_of_distribution(self, data: AbaloneInput) -> Tuple[bool, List[str]]:
        """Identify Out-of-Distribution (OOD) or biologically unusual samples.

        Args:
            data: Input abalone measurements.

        Returns:
            Tuple[bool, List[str]]: (is_ood flag, list of descriptive warning messages).
        """
        warnings = []
        is_ood = False

        # 1. Check feature domain bounds
        for feat, meta in FEATURE_METADATA.items():
            val = getattr(data, feat)
            if val < meta["min"] or val > meta["max"]:
                is_ood = True
                warnings.append(
                    f"Giá trị {feat}={val}{meta['unit']} nằm ngoài phạm vi quan sát trong tập huấn luyện "
                    f"[{meta['min']} - {meta['max']}{meta['unit']}]."
                )

        # 2. Biological check: sum of meat, viscera, shell vs whole weight
        sum_parts = data.Shucked_weight + data.Viscera_weight + data.Shell_weight
        if sum_parts > data.Whole_weight * 1.15:
            is_ood = True
            warnings.append(
                f"Tổng khối lượng các thành phần ({sum_parts:.3f}g) vượt quá 115% tổng khối lượng Whole_weight ({data.Whole_weight:.3f}g). "
                "Mẫu có thể có sai số đo lường hoặc mất nước."
            )

        # 3. Height anomaly check
        if data.Height < 0.02 and data.Whole_weight > 0.3:
            is_ood = True
            warnings.append(
                "Chiều cao quá nhỏ so với khối lượng toàn bộ cá thể (khả năng vỏ bị biến dạng hoặc lỗi đo)."
            )

        return is_ood, warnings

    def predict(self, input_data: AbaloneInput) -> PredictionResponse:
        """Run k-NN inference, extract nearest neighbors, and return rich response payload.

        Args:
            input_data: Validated user input measurements.

        Returns:
            PredictionResponse: Complete prediction result with neighbors and intervals.
        """
        # 1. Convert input to DataFrame matching training schema
        df_input = pd.DataFrame(
            [
                {
                    "Sex": input_data.Sex,
                    "Length": input_data.Length,
                    "Diameter": input_data.Diameter,
                    "Height": input_data.Height,
                    "Whole_weight": input_data.Whole_weight,
                    "Shucked_weight": input_data.Shucked_weight,
                    "Viscera_weight": input_data.Viscera_weight,
                    "Shell_weight": input_data.Shell_weight,
                }
            ]
        )

        # 2. Predict Rings
        pred_rings = float(self.pipeline.predict(df_input)[0])
        pred_rings_rounded = round(max(1.0, pred_rings), 2)
        estimated_age = round(pred_rings_rounded + 1.5, 2)

        # 3. Classify Age Group
        if pred_rings_rounded <= 8.0:
            age_group = "Young"
        elif pred_rings_rounded <= 12.0:
            age_group = "Middle"
        else:
            age_group = "Old"

        # 4. Extract Nearest Neighbors from pipeline
        preprocessor = self.pipeline.named_steps["preprocessor"]
        knn = self.pipeline.named_steps["knn"]

        X_trans = preprocessor.transform(df_input)
        distances, indices = knn.kneighbors(X_trans, return_distance=True)

        neighbor_distances = distances[0]
        neighbor_indices = indices[0]

        # Training target rings stored in knn estimator
        y_train_fit = knn._y if hasattr(knn, "_y") else None

        nearest_neighbors = []
        neighbor_rings_list = []

        for dist, idx in zip(neighbor_distances, neighbor_indices):
            n_rings = float(y_train_fit[idx]) if y_train_fit is not None else pred_rings_rounded
            neighbor_rings_list.append(n_rings)
            nearest_neighbors.append(
                NeighborInfo(
                    distance=round(float(dist), 4),
                    rings=round(n_rings, 1),
                    estimated_age=round(n_rings + 1.5, 1),
                )
            )

        ref_min = float(np.min(neighbor_rings_list))
        ref_max = float(np.max(neighbor_rings_list))
        ref_median = float(np.median(neighbor_rings_list))

        reference_interval = ReferenceInterval(
            min_rings=round(ref_min, 1),
            max_rings=round(ref_max, 1),
            median_rings=round(ref_median, 1),
            min_age_years=round(ref_min + 1.5, 1),
            max_age_years=round(ref_max + 1.5, 1),
        )

        # 5. Out-of-Distribution Check
        is_ood, warnings = self.check_out_of_distribution(input_data)

        return PredictionResponse(
            predicted_rings=pred_rings_rounded,
            estimated_age_years=estimated_age,
            age_group=age_group,
            reference_interval=reference_interval,
            nearest_neighbors=nearest_neighbors,
            is_out_of_distribution=is_ood,
            warnings=warnings,
        )

    def get_dataset_stats(self) -> DatasetStatsResponse:
        """Return dataset boundaries and descriptions for client form validation."""
        numeric_ranges = {
            k: FeatureRangeItem(
                unit=v["unit"],
                description=v["desc"],
                min=v["min"],
                max=v["max"],
                mean=v["mean"],
                std=v["std"],
            )
            for k, v in FEATURE_METADATA.items()
        }
        return DatasetStatsResponse(
            numeric_ranges=numeric_ranges,
            valid_sex_values=["M", "F", "I"],
            total_training_samples=3341,
        )

    def get_model_card(self) -> ModelCardResponse:
        """Return Model Card specification."""
        knn = self.pipeline.named_steps["knn"]
        return ModelCardResponse(
            pipeline_steps=[
                "StandardScaler (7 numerical features)",
                "OneHotEncoder (Sex categorical feature)",
                f"KNeighborsRegressor(n_neighbors={knn.n_neighbors}, weights='{knn.weights}')",
            ],
            hyperparameters={
                "n_neighbors": knn.n_neighbors,
                "weights": knn.weights,
                "metric": knn.metric,
                "scaling": "StandardScaler",
                "cv_folds": 5,
            },
            performance_metrics={
                "Test_MAE": 1.551,
                "Test_RMSE": 2.259,
                "Test_R2": 0.529,
                "Baseline_Median_MAE": 2.378,
            },
            age_group_mae={
                "Young (<=8 rings)": 1.341,
                "Middle (9-12 rings)": 1.396,
                "Old (>12 rings)": 2.651,
            },
            limitations=[
                "Mô hình có độ chính xác cao nhất ở nhóm bào ngư non và trung bình (MAE ~ 1.3-1.4).",
                "Độ phân tán và sai số tăng rõ rệt ở nhóm bào ngư già (>12 vòng) do mật độ mẫu huấn luyện ít.",
                "Không thay thế hoàn toàn phương pháp cắt vỏ và soi kính hiển vi sinh học tiêu chuẩn.",
            ],
        )


# Global singleton instance
predictor_instance = None


def get_predictor() -> AbalonePredictor:
    """Dependency getter for singleton predictor instance."""
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = AbalonePredictor()
    return predictor_instance
