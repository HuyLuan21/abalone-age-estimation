"""FastAPI application entrypoint for Abalone Age Estimation.

Provides RESTful endpoints for real-time model inference, model card metadata,
and data validation bounds.
"""

import os
import sys
from typing import Dict, Any
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, Depends, Request, status
# pyrefly: ignore [missing-import]
from fastapi.exceptions import RequestValidationError
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse, FileResponse
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from app.schemas import (
        AbaloneInput,
        PredictionResponse,
        ModelCardResponse,
        DatasetStatsResponse,
    )
    from app.predictor import get_predictor, AbalonePredictor
except ModuleNotFoundError:
    from schemas import (
        AbaloneInput,
        PredictionResponse,
        ModelCardResponse,
        DatasetStatsResponse,
    )
    from predictor import get_predictor, AbalonePredictor

# Initialize FastAPI App
app = FastAPI(
    title="Abalone Age Estimation API",
    description=(
        "REST API phục vụ dự đoán số vòng vỏ và ước lượng tuổi bào ngư bằng thuật toán K-NN Regression. "
        "Dự án kết thúc học phần Machine Learning cơ bản (Project 06)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Format Pydantic validation errors into clean, readable Vietnamese messages."""
    error_messages = []
    for err in exc.errors():
        msg = err.get("msg", "")
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        loc = [str(x) for x in err.get("loc", []) if x != "body"]
        field_str = ".".join(loc)
        if field_str:
            error_messages.append(f"{field_str}: {msg}")
        else:
            error_messages.append(msg)

    clean_message = "; ".join(error_messages) if error_messages else "Dữ liệu đầu vào không hợp lệ."
    sanitized_errors = []
    for err in exc.errors():
        item = dict(err)
        if "ctx" in item and isinstance(item["ctx"], dict):
            item["ctx"] = {k: str(v) for k, v in item["ctx"].items()}
        sanitized_errors.append(item)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": clean_message, "errors": sanitized_errors},
    )


@app.get(
    "/api/health",
    tags=["System"],
    summary="Kiểm tra trạng thái server và model",
    status_code=status.HTTP_200_OK,
)
def health_check(predictor: AbalonePredictor = Depends(get_predictor)) -> Dict[str, Any]:
    """Health check endpoint confirming FastAPI server and ML pipeline are operational."""
    return {
        "status": "healthy",
        "service": "Abalone Age Estimation API",
        "model_loaded": predictor.pipeline is not None,
        "version": "1.0.0",
    }


@app.get(
    "/api/model-card",
    response_model=ModelCardResponse,
    tags=["Metadata"],
    summary="Lấy thông tin Model Card và chỉ số đánh giá",
)
def get_model_card_info(
    predictor: AbalonePredictor = Depends(get_predictor),
) -> ModelCardResponse:
    """Retrieve structured Model Card containing architecture, hyperparameters,

    evaluation metrics, and ethical/operational limitations.
    """
    try:
        return predictor.get_model_card()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi đọc thông tin Model Card: {str(exc)}",
        )


@app.get(
    "/api/feature-ranges",
    response_model=DatasetStatsResponse,
    tags=["Metadata"],
    summary="Lấy miền giá trị và thống kê của các thuộc tính đầu vào",
)
def get_feature_ranges(
    predictor: AbalonePredictor = Depends(get_predictor),
) -> DatasetStatsResponse:
    """Retrieve statistical bounds and unit metadata for all input features."""
    try:
        return predictor.get_dataset_stats()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lấy thông tin miền giá trị: {str(exc)}",
        )


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Dự đoán tuổi bào ngư từ các phép đo vật lý",
    status_code=status.HTTP_200_OK,
)
def predict_age(
    input_data: AbaloneInput,
    predictor: AbalonePredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Predict abalone rings and age from 8 physical measurements.

    - Computes predicted rings via fitted KNeighborsRegressor pipeline.
    - Estimates age in years (Rings + 1.5).
    - Classifies age into biological groups: Young (<=8), Middle (9-12), Old (>12).
    - Extracts k nearest neighbors and calculates reference confidence interval.
    - Inspects for Out-of-Distribution (OOD) or biologically unusual samples.
    """
    try:
        response = predictor.predict(input_data)
        return response
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dữ liệu đầu vào không hợp lệ: {str(val_err)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi trong quá trình suy luận mô hình: {str(exc)}",
        )


# Serve built frontend SPA if available
frontend_dist = os.path.join(PROJECT_ROOT, "frontend", "dist")
if os.path.exists(frontend_dist):
    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa_frontend(full_path: str):
        if (
            full_path.startswith("api/")
            or full_path == "api"
            or full_path.startswith("docs")
            or full_path.startswith("redoc")
            or full_path == "openapi.json"
        ):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API endpoint not found")

        target_file = os.path.join(frontend_dist, full_path)
        if full_path and os.path.exists(target_file) and os.path.isfile(target_file):
            return FileResponse(target_file)

        index_file = os.path.join(frontend_dist, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Frontend index.html not found")


if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn

    print("=" * 70)
    print("🚀 Khởi động Abalone Age Estimation Backend API Server...")
    print("👉 Swagger UI : http://127.0.0.1:8000/docs")
    print("👉 ReDoc      : http://127.0.0.1:8000/redoc")
    print("=" * 70)
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)
