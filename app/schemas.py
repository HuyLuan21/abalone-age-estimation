"""Pydantic data schemas for Abalone Age Estimation API.

Includes request/response models, input validation, and metadata schemas.
"""

from typing import List, Dict, Optional, Literal
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field, model_validator


class AbaloneInput(BaseModel):
    """Input features for predicting abalone age/rings."""

    Sex: Literal["M", "F", "I"] = Field(
        ...,
        description="Giới tính cá thể bào ngư: 'M' (Male), 'F' (Female), 'I' (Infant)",
        examples=["M"],
    )
    Length: float = Field(
        ...,
        gt=0.0,
        le=2.0,
        description="Chiều dài lớn nhất của vỏ (mm)",
        examples=[0.455],
    )
    Diameter: float = Field(
        ...,
        gt=0.0,
        le=2.0,
        description="Đường kính vuông góc với chiều dài (mm)",
        examples=[0.365],
    )
    Height: float = Field(
        ...,
        gt=0.0,
        le=2.0,
        description="Chiều cao kèm thịt trong vỏ (mm)",
        examples=[0.095],
    )
    Whole_weight: float = Field(
        ...,
        gt=0.0,
        le=5.0,
        description="Tổng khối lượng toàn bộ cá thể (gam)",
        examples=[0.514],
    )
    Shucked_weight: float = Field(
        ...,
        gt=0.0,
        le=5.0,
        description="Khối lượng phần thịt (gam)",
        examples=[0.2245],
    )
    Viscera_weight: float = Field(
        ...,
        gt=0.0,
        le=5.0,
        description="Khối lượng nội tạng sau khi chảy máu (gam)",
        examples=[0.101],
    )
    Shell_weight: float = Field(
        ...,
        gt=0.0,
        le=5.0,
        description="Khối lượng vỏ sau khi sấy khô (gam)",
        examples=[0.150],
    )

    @model_validator(mode="after")
    def validate_physical_consistency(self) -> "AbaloneInput":
        """Validate logical physical relationships between abalone measurements."""
        # 1. Diameter must not exceed Length
        if self.Diameter > self.Length:
            raise ValueError(
                f"Đường kính (Diameter={self.Diameter}) không thể lớn hơn Chiều dài (Length={self.Length})."
            )

        # 2. Individual weight components cannot exceed whole weight
        for part_name, part_val in [
            ("Khối lượng thịt (Shucked_weight)", self.Shucked_weight),
            ("Khối lượng nội tạng (Viscera_weight)", self.Viscera_weight),
            ("Khối lượng vỏ (Shell_weight)", self.Shell_weight),
        ]:
            if part_val > self.Whole_weight:
                raise ValueError(
                    f"{part_name} ({part_val}) không thể lớn hơn Tổng khối lượng (Whole_weight={self.Whole_weight})."
                )

        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "Sex": "M",
                "Length": 0.455,
                "Diameter": 0.365,
                "Height": 0.095,
                "Whole_weight": 0.5140,
                "Shucked_weight": 0.2245,
                "Viscera_weight": 0.1010,
                "Shell_weight": 0.1500,
            }
        }
    }


class NeighborInfo(BaseModel):
    """Information about a nearest neighbor used in k-NN regression."""

    distance: float = Field(..., description="Khoảng cách Euclidean đã chuẩn hóa đến mẫu mới")
    rings: float = Field(..., description="Số vòng vỏ thực tế của láng giềng này")
    estimated_age: float = Field(..., description="Tuổi tương ứng (Rings + 1.5 năm)")


class ReferenceInterval(BaseModel):
    """Reference interval computed from k nearest neighbors."""

    min_rings: float = Field(..., description="Số vòng nhỏ nhất trong nhóm láng giềng")
    max_rings: float = Field(..., description="Số vòng lớn nhất trong nhóm láng giềng")
    median_rings: float = Field(..., description="Trung vị số vòng trong nhóm láng giềng")
    min_age_years: float = Field(..., description="Tuổi nhỏ nhất trong nhóm láng giềng (năm)")
    max_age_years: float = Field(..., description="Tuổi lớn nhất trong nhóm láng giềng (năm)")


class PredictionResponse(BaseModel):
    """Response payload for age estimation endpoint."""

    predicted_rings: float = Field(..., description="Số vòng vỏ ước lượng (Rings)")
    estimated_age_years: float = Field(
        ..., description="Tuổi ước lượng xấp xỉ tính bằng năm (Rings + 1.5)"
    )
    age_group: Literal["Young", "Middle", "Old"] = Field(
        ...,
        description="Phân nhóm sinh học: Young (<=8 vòng), Middle (9-12 vòng), Old (>12 vòng)",
    )
    reference_interval: ReferenceInterval = Field(
        ..., description="Khoảng tham chiếu từ k láng giềng gần nhất"
    )
    nearest_neighbors: List[NeighborInfo] = Field(
        ..., description="Danh sách các láng giềng gần nhất được tham khảo"
    )
    is_out_of_distribution: bool = Field(
        ..., description="Cảnh báo mẫu đo có nằm ngoài miền phân bố dữ liệu huấn luyện hay không"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Danh sách các cảnh báo (nếu có)"
    )


class FeatureRangeItem(BaseModel):
    """Summary statistics and domain bounds for a feature."""

    unit: str
    description: str
    min: float
    max: float
    mean: float
    std: float


class DatasetStatsResponse(BaseModel):
    """Dataset statistics and feature bounds to assist frontend forms."""

    numeric_ranges: Dict[str, FeatureRangeItem]
    valid_sex_values: List[str]
    total_training_samples: int


class ModelCardResponse(BaseModel):
    """Model card specification containing architecture, metrics, and limitations."""

    model_name: str = "Abalone Age Estimator (K-NN Regression)"
    algorithm: str = "K-Nearest Neighbors Regressor with Euclidean Distance"
    pipeline_steps: List[str]
    hyperparameters: Dict[str, object]
    performance_metrics: Dict[str, float]
    age_group_mae: Dict[str, float]
    dataset_source: str = "UCI Machine Learning Repository (DOI: 10.24432/C55C7W)"
    intended_use: str = "Hỗ trợ nghiên cứu nuôi trồng và sàng lọc mẫu bào ngư cần kiểm tra vi thể."
    limitations: List[str]
