"""Prediction service and OOD validation logic for Abalone Age Estimation.

Handles dynamic metadata synchronization, k-NN inference, neighbor explanations,
and domain consistency validation.
"""

import json
import os
import sys
import warnings
from typing import Dict, List, Tuple, Any
# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

# Suppress minor cross-environment version warnings when loading pickle
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

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


# Baseline fallback metadata (used if json/csv artifacts are not yet generated)
DEFAULT_FEATURE_METADATA = {
    "Length": {
        "unit": "mm",
        "desc": "Chiều dài lớn nhất của vỏ (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 mm)",
        "min": 0.075,
        "max": 0.815,
        "mean": 0.524,
        "std": 0.120,
    },
    "Diameter": {
        "unit": "mm",
        "desc": "Đường kính vuông góc với chiều dài (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 mm)",
        "min": 0.055,
        "max": 0.650,
        "mean": 0.408,
        "std": 0.099,
    },
    "Height": {
        "unit": "mm",
        "desc": "Chiều cao kèm thịt trong vỏ (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 mm)",
        "min": 0.010,
        "max": 0.300,
        "mean": 0.140,
        "std": 0.042,
    },
    "Whole_weight": {
        "unit": "g",
        "desc": "Tổng khối lượng toàn bộ cá thể (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 gam)",
        "min": 0.002,
        "max": 2.826,
        "mean": 0.829,
        "std": 0.490,
    },
    "Shucked_weight": {
        "unit": "g",
        "desc": "Khối lượng phần thịt bào ngư (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 gam)",
        "min": 0.001,
        "max": 1.488,
        "mean": 0.360,
        "std": 0.222,
    },
    "Viscera_weight": {
        "unit": "g",
        "desc": "Khối lượng nội tạng sau khi mổ (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 gam)",
        "min": 0.001,
        "max": 0.760,
        "mean": 0.181,
        "std": 0.110,
    },
    "Shell_weight": {
        "unit": "g",
        "desc": "Khối lượng vỏ sấy khô (Dữ liệu gốc UCI chuẩn hóa chia 200, 1 đơn vị = 200 gam)",
        "min": 0.002,
        "max": 1.005,
        "mean": 0.239,
        "std": 0.139,
    },
}


class AbalonePredictor:
    """Predictor service wrapping scikit-learn pipeline, dynamic metadata, and validations."""

    def __init__(
        self,
        model_path: str = "models/abalone_knn_pipeline.pkl",
        metadata_path: str = "models/model_metadata.json",
    ):
        """Initialize predictor, load saved pipeline, and sync dynamic metadata."""
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.pipeline = None
        self.metadata: Dict[str, Any] = {}
        self.feature_stats: Dict[str, Dict[str, Any]] = DEFAULT_FEATURE_METADATA.copy()
        self.total_training_samples: int = 3341
        self.valid_sex_values: List[str] = ["M", "F", "I"]

        self._load_pipeline()
        self._load_metadata()

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

    def _load_metadata(self) -> None:
        """Dynamically load training configuration, statistics, and evaluation metrics."""
        candidate_json_paths = [
            self.metadata_path,
            os.path.join(PROJECT_ROOT, self.metadata_path),
            os.path.join(os.path.dirname(__file__), "..", self.metadata_path),
        ]

        loaded = False
        for p in candidate_json_paths:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                    loaded = True
                    break
                except Exception:
                    pass

        if loaded:
            if "feature_statistics" in self.metadata:
                self.feature_stats = self.metadata["feature_statistics"]
            if "total_training_samples" in self.metadata:
                self.total_training_samples = int(self.metadata["total_training_samples"])
            if "valid_sex_values" in self.metadata:
                self.valid_sex_values = list(self.metadata["valid_sex_values"])
        else:
            # Fallback: check CSV reports
            self._load_metadata_from_csv_reports()

    def _load_metadata_from_csv_reports(self) -> None:
        """Fallback sync from CSV reports directory."""
        reports_dir = os.path.join(PROJECT_ROOT, "reports")
        age_csv = os.path.join(reports_dir, "age_group_mae.csv")
        results_csv = os.path.join(reports_dir, "model_results.csv")

        # 1. Sync age group MAE
        if os.path.exists(age_csv):
            try:
                df_age = pd.read_csv(age_csv)
                # First column is AgeGroup, second column is MAE
                group_col = df_age.columns[0]
                val_col = df_age.columns[1]
                age_mae = {}
                for _, row in df_age.iterrows():
                    grp = str(row[group_col])
                    val = round(float(row[val_col]), 4)
                    if grp == "Young":
                        age_mae["Young (<=8 rings)"] = val
                    elif grp == "Middle":
                        age_mae["Middle (9-12 rings)"] = val
                    elif grp == "Old":
                        age_mae["Old (>12 rings)"] = val
                self.metadata["age_group_mae"] = age_mae
            except Exception:
                pass

        # 2. Sync test performance metrics
        if os.path.exists(results_csv):
            try:
                df_res = pd.read_csv(results_csv)
                perf = {}
                for _, row in df_res.iterrows():
                    m_name = str(row["Model"])
                    if "StandardScaler" in m_name:
                        perf["Test_MAE"] = round(float(row["MAE"]), 4)
                        perf["Test_RMSE"] = round(float(row["RMSE"]), 4)
                        perf["Test_R2"] = round(float(row["R2"]), 4)
                    elif "Baseline" in m_name:
                        perf["Baseline_Median_MAE"] = round(float(row["MAE"]), 4)
                self.metadata["performance_metrics"] = perf
            except Exception:
                pass

    def check_out_of_distribution(self, data: AbaloneInput) -> Tuple[bool, List[str]]:
        """Identify Out-of-Distribution (OOD) or biologically unusual samples.

        Args:
            data: Input abalone measurements.

        Returns:
            Tuple[bool, List[str]]: (is_ood flag, list of descriptive warning messages).
        """
        warnings_list = []
        is_ood = False

        # 1. Check feature domain bounds using dynamic statistics
        for feat, meta in self.feature_stats.items():
            if hasattr(data, feat):
                val = getattr(data, feat)
                min_val = meta.get("min", 0.0)
                max_val = meta.get("max", 5.0)
                unit_str = meta.get("unit", "")
                if val < min_val or val > max_val:
                    is_ood = True
                    warnings_list.append(
                        f"Giá trị {feat}={val}{unit_str} nằm ngoài phạm vi quan sát trong tập huấn luyện "
                        f"[{min_val} - {max_val}{unit_str}]."
                    )

        # 2. Biological check: sum of meat, viscera, shell vs whole weight
        sum_parts = data.Shucked_weight + data.Viscera_weight + data.Shell_weight
        if sum_parts > data.Whole_weight * 1.15:
            is_ood = True
            warnings_list.append(
                f"Tổng khối lượng các thành phần ({sum_parts:.3f}g) vượt quá 115% tổng khối lượng Whole_weight ({data.Whole_weight:.3f}g). "
                "Mẫu có thể có sai số đo lường hoặc mất nước."
            )

        # 3. Height anomaly check
        if data.Height < 0.02 and data.Whole_weight > 0.3:
            is_ood = True
            warnings_list.append(
                "Chiều cao quá nhỏ so với khối lượng toàn bộ cá thể (khả năng vỏ bị biến dạng hoặc lỗi đo)."
            )

        return is_ood, warnings_list

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
        y_train_fit = getattr(knn, "_y", None)

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
        is_ood, warnings_list = self.check_out_of_distribution(input_data)

        return PredictionResponse(
            predicted_rings=pred_rings_rounded,
            estimated_age_years=estimated_age,
            age_group=age_group,
            reference_interval=reference_interval,
            nearest_neighbors=nearest_neighbors,
            is_out_of_distribution=is_ood,
            warnings=warnings_list,
        )

    def get_dataset_stats(self) -> DatasetStatsResponse:
        """Return dataset boundaries and descriptions for client form validation."""
        numeric_ranges = {}
        for k, v in self.feature_stats.items():
            u = v.get("unit", "-")
            min_v = float(v.get("min", 0.0))
            max_v = float(v.get("max", 5.0))
            mean_v = float(v.get("mean", 0.0))
            std_v = float(v.get("std", 1.0))
            phys_u = "g" if "weight" in k.lower() else "mm"
            numeric_ranges[k] = FeatureRangeItem(
                unit=u,
                description=v.get("desc", k),
                min=min_v,
                max=max_v,
                mean=mean_v,
                std=std_v,
                physical_unit=phys_u,
                physical_min=round(min_v * 200, 2),
                physical_max=round(max_v * 200, 2),
                physical_mean=round(mean_v * 200, 2),
            )
        return DatasetStatsResponse(
            numeric_ranges=numeric_ranges,
            valid_sex_values=self.valid_sex_values,
            total_training_samples=self.total_training_samples,
        )

    def get_model_card(self) -> ModelCardResponse:
        """Return Model Card specification synchronized dynamically with experimental results."""
        knn = self.pipeline.named_steps["knn"]

        # Default fallback metrics if metadata is empty
        perf_metrics = self.metadata.get(
            "performance_metrics",
            {
                "Test_MAE": 1.551,
                "Test_RMSE": 2.259,
                "Test_R2": 0.529,
                "Baseline_Median_MAE": 2.378,
            },
        )

        # Exactly synchronized with reports/age_group_mae.csv
        age_mae = self.metadata.get(
            "age_group_mae",
            {
                "Young (<=8 rings)": 1.114,
                "Middle (9-12 rings)": 1.125,
                "Old (>12 rings)": 3.845,
            },
        )

        limitations = self.metadata.get(
            "limitations",
            [
                "Mô hình đạt độ chính xác cao nhất ở nhóm bào ngư non (MAE = 1.114) và trung bình (MAE = 1.125).",
                "Độ phân tán và sai số tăng rõ rệt ở nhóm già (>12 vòng, MAE = 3.845) do mẫu già khan hiếm và sinh trưởng chậm.",
                "Thí nghiệm so sánh Scaling và No-Scale được xác minh qua 5-Fold CV trên Train để chống rò rỉ tập Test.",
                "Không thay thế hoàn toàn phương pháp cắt vỏ và đếm vòng kính hiển vi sinh học tiêu chuẩn.",
            ],
        )

        pipeline_steps = self.metadata.get(
            "pipeline_steps",
            [
                "StandardScaler (7 numerical features)",
                "OneHotEncoder (Sex categorical feature)",
                f"KNeighborsRegressor(n_neighbors={knn.n_neighbors}, weights='{knn.weights}')",
            ],
        )

        hyperparams = self.metadata.get(
            "hyperparameters",
            {
                "n_neighbors": knn.n_neighbors,
                "weights": knn.weights,
                "metric": "minkowski (p=2 Euclidean)",
                "scaling": "StandardScaler",
                "cv_folds": 5,
            },
        )

        return ModelCardResponse(
            pipeline_steps=pipeline_steps,
            hyperparameters=hyperparams,
            performance_metrics=perf_metrics,
            validation_performance=self.metadata.get("validation_performance"),
            cv_performance=self.metadata.get("cv_performance"),
            data_split=self.metadata.get("data_split"),
            age_group_mae=age_mae,
            limitations=limitations,
        )


# Global singleton instance
predictor_instance = None


def get_predictor() -> AbalonePredictor:
    """Dependency getter for singleton predictor instance."""
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = AbalonePredictor()
    return predictor_instance
