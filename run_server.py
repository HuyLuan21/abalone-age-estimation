"""Convenient 1-click runner for Abalone Age Estimation Backend API & Web Server."""

# pyrefly: ignore [missing-import]
import uvicorn
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Khởi động Abalone Age Estimation Backend API Server...")
    print("👉 Swagger UI (Tài liệu API tương tác): http://127.0.0.1:8000/docs")
    print("👉 ReDoc                              : http://127.0.0.1:8000/redoc")
    print("👉 Health Check Endpoint             : http://127.0.0.1:8000/api/health")
    print("=" * 70)
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)
