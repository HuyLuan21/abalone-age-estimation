"""Convenient 1-click runner for Abalone Age Estimation Backend API & Web Server."""

# pyrefly: ignore [missing-import]
import uvicorn
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if __name__ == "__main__":
    frontend_dist_index = os.path.join(PROJECT_ROOT, "frontend", "dist", "index.html")
    has_frontend_build = os.path.exists(frontend_dist_index)

    print("=" * 70)
    print("🚀 Khởi động Abalone Age Estimation Server...")
    if has_frontend_build:
        print("✅ Giao diện Web (Production Build)    : http://127.0.0.1:8000/")
    else:
        print("⚠️  Giao diện Web chưa build (Repo mới): http://127.0.0.1:8000/ (Trang hướng dẫn)")
        print("💡 Để build giao diện chính thức       : cd frontend && npm install && npm run build")
        print("💡 Hoặc chạy Vite dev server độc lập   : cd frontend && npm run dev (http://localhost:5173)")
    print("👉 Swagger UI (Tài liệu API tương tác) : http://127.0.0.1:8000/docs")
    print("👉 ReDoc                               : http://127.0.0.1:8000/redoc")
    print("👉 Health Check Endpoint              : http://127.0.0.1:8000/api/health")
    print("=" * 70)
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)
