"""Pydantic data schemas for Abalone Age Estimation API.

Includes request/response models, input validation, physical consistency checks,
and metadata schemas aligned with UCI 1/200 scaling.
"""

from typing import List, Dict, Optional, Literal
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field, model_validator


class AbaloneInput(BaseModel):
    """Input features for predicting abalone age/rings.

    Supports both:
    1. UCI Standard Normalized Scale (divided by 200, e.g., Length = 0.455, Weight = 0.514).
    2. Physical Millimeter/Gram scale (e.g., Length = 91.0 mm, Weight = 102.8 g).
       If physical values (> 2.0) are supplied, the validator automatically converts them by dividing by 200.
    """

    Sex: Literal["M", "F", "I"] = Field(
        ...,
        description="Giới tính cá thể bào ngư: 'M' (Đực), 'F' (Cái), 'I' (Non/Chưa trưởng thành)",
        examples=["M"],
    )
    Length: float = Field(
        ...,
        gt=0.0,
        description="Chiều dài lớn nhất của vỏ theo thang UCI (đã chia 200, ví dụ 0.455) hoặc mm thực tế (ví dụ 91.0 mm)",
        examples=[0.455],
    )
    Diameter: float = Field(
        ...,
        gt=0.0,
        description="Đường kính vỏ vuông góc với chiều dài theo thang UCI (đã chia 200, ví dụ 0.365) hoặc mm thực tế (ví dụ 73.0 mm)",
        examples=[0.365],
    )
    Height: float = Field(
        ...,
        gt=0.0,
        description="Chiều cao kèm thịt trong vỏ theo thang UCI (đã chia 200, ví dụ 0.095) hoặc mm thực tế (ví dụ 19.0 mm)",
        examples=[0.095],
    )
    Whole_weight: float = Field(
        ...,
        gt=0.0,
        description="Tổng khối lượng toàn bộ cá thể theo thang UCI (đã chia 200, ví dụ 0.514) hoặc gam thực tế (ví dụ 102.8 g)",
        examples=[0.514],
    )
    Shucked_weight: float = Field(
        ...,
        gt=0.0,
        description="Khối lượng phần thịt theo thang UCI (đã chia 200, ví dụ 0.2245) hoặc gam thực tế (ví dụ 44.9 g)",
        examples=[0.2245],
    )
    Viscera_weight: float = Field(
        ...,
        gt=0.0,
        description="Khối lượng nội tạng sau khi mổ theo thang UCI (đã chia 200, ví dụ 0.101) hoặc gam thực tế (ví dụ 20.2 g)",
        examples=[0.101],
    )
    Shell_weight: float = Field(
        ...,
        gt=0.0,
        description="Khối lượng vỏ sau khi sấy khô theo thang UCI (đã chia 200, ví dụ 0.150) hoặc gam thực tế (ví dụ 30.0 g)",
        examples=[0.150],
    )

    @model_validator(mode="after")
    def normalize_units_and_validate(self) -> "AbaloneInput":
        """Auto-convert physical mm/g units to UCI scale if needed, and validate physical integrity."""
        # 1. Smart unit conversion: if physical measurements (e.g. Length > 2.0 mm or Weight > 5.0 g)
        if self.Length > 2.0:
            self.Length = round(self.Length / 200.0, 4)
        if self.Diameter > 2.0:
            self.Diameter = round(self.Diameter / 200.0, 4)
        if self.Height > 1.5:
            self.Height = round(self.Height / 200.0, 4)
        if self.Whole_weight > 5.0:
            self.Whole_weight = round(self.Whole_weight / 200.0, 4)
        if self.Shucked_weight > 5.0:
            self.Shucked_weight = round(self.Shucked_weight / 200.0, 4)
        if self.Viscera_weight > 5.0:
            self.Viscera_weight = round(self.Viscera_weight / 200.0, 4)
        if self.Shell_weight > 5.0:
            self.Shell_weight = round(self.Shell_weight / 200.0, 4)

        # 2. Strict domain bounds check (in UCI normalized scale)
        if self.Height <= 0.0 or self.Height >= 0.6:
            raise ValueError(
                f"Chiều cao (Height={self.Height}) bất thường hoặc phi sinh học. Phải nằm trong khoảng (0.0, 0.6) theo thang UCI."
            )

        if self.Length > 1.5 or self.Diameter > 1.5:
            raise ValueError(
                f"Kích thước chiều dài hoặc đường kính ({self.Length}, {self.Diameter}) vượt quá giới hạn sinh học hợp lý."
            )

        # 3. Geometric consistency: Diameter must not exceed Length
        if self.Diameter > self.Length:
            raise ValueError(
                f"Đường kính (Diameter={self.Diameter}) không thể lớn hơn Chiều dài (Length={self.Length})."
            )

        # 4. Weight component consistency: components cannot exceed whole weight
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

    distance: float = Field(..., description="Khoảng cách Euclidean đến mẫu mới theo thang đo huấn luyện")
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
    physical_unit: str = "mm"
    physical_min: float = 0.0
    physical_max: float = 0.0
    physical_mean: float = 0.0


class DatasetStatsResponse(BaseModel):
    """Dataset statistics and feature bounds to assist frontend forms in both UCI and physical scales."""

    numeric_ranges: Dict[str, FeatureRangeItem]
    valid_sex_values: List[str]
    total_training_samples: int
    scale_factor: float = 200.0
    scaling_explanation: str = (
        "Dữ liệu gốc UCI đã chia 200 cho toàn bộ 7 biến liên tục. "
        "Hệ thống hỗ trợ nhập theo thang chuẩn UCI (ví dụ 0.455) hoặc mm/gam thực tế (ví dụ 91.0 mm) với tự động quy đổi."
    )


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
