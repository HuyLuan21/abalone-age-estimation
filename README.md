# Project 06: Ước lượng tuổi bào ngư bằng K-NN Regression

Dự án học máy ước lượng tuổi của bào ngư (*Abalone*) thông qua các phép đo vật lý sử dụng thuật toán **K-Nearest Neighbors (k-NN) Regression**, tuân thủ nghiêm ngặt quy trình chống rò rỉ dữ liệu (data leakage) và quy chuẩn bài tập lớn.

---

## 📁 Cấu trúc thư mục mã nguồn

Dự án được tái cấu trúc dạng **Modular** theo đúng khuyến nghị của đề bài:

```text
abalone-age-estimation/
├── data/
│   ├── README.md               # Mô tả nguồn gốc dữ liệu, bản quyền CC BY 4.0 & SHA-256
│   ├── data_dictionary.md      # Giải thích chi tiết các thuộc tính & thang đo
│   └── raw/
│       ├── abalone.data        # Dữ liệu gốc UCI (4,177 mẫu)
│       └── abalone.names       # Tài liệu gốc mô tả thuộc tính
├── src/                        # Mã nguồn module hóa
│   ├── __init__.py
│   ├── data.py                 # Tải, làm sạch 4 mẫu ngoại lệ & chia Train/Val/Test
│   ├── features.py             # Xây dựng tiền xử lý & scikit-learn Pipeline
│   ├── train.py                # K-Fold CV trên Train, chọn tham số & lưu Model
│   └── evaluate.py             # Đánh giá metric, phân tích sai số & vẽ biểu đồ
├── tests/
│   └── test_pipeline.py        # Kiểm thử toàn vẹn data split, leakage & reproducibility
├── models/
│   ├── abalone_knn_pipeline.pkl# Pipeline k-NN đã huấn luyện hoàn chỉnh (No Scale, K=17)
│   └── model_metadata.json     # Siêu dữ liệu động đồng bộ với API, Model Card & hệ đo
├── reports/
│   ├── removed_anomalies.csv   # Kiểm kê 04 dòng bất thường bị loại trước khi chia tập
│   ├── validation_results.csv  # Bảng đánh giá so sánh trên tập Validation (15%)
│   ├── cv_results.csv          # Bảng kết quả 5-Fold Cross Validation trên Train (70%)
│   ├── cv_ablation.csv         # Bảng so sánh CV Scaled vs No-Scale trên Train
│   ├── model_results.csv       # Bảng tổng hợp hiệu năng Validation & Test
│   ├── age_group_mae.csv       # Phân tích sai số theo nhóm tuổi trên Test
│   └── figures/                # Biểu đồ phân tích EDA và đánh giá mô hình
│       ├── rings_distribution.png
│       ├── sex_distribution.png
│       ├── mae_vs_k.png
│       ├── model_comparison.png
│       ├── actual_vs_predicted.png
│       └── residual_plot.png
├── app/                        # Ứng dụng Web & REST API
├── main.py                     # Script điều phối toàn bộ quy trình Train/Val/Test
├── requirements.txt            # Danh sách thư viện phụ thuộc
└── README.md                   # Tài liệu hướng dẫn sử dụng
```

---

## ⚙️ Cài đặt & Thiết lập môi trường

### 1. Tạo và kích hoạt môi trường ảo

```powershell
# Di chuyển vào thư mục dự án
cd abalone-age-estimation

# Tạo môi trường ảo (Python 3.10+)
python -m venv venv

# Kích hoạt môi trường (Windows PowerShell)
venv\Scripts\activate
```

### 2. Cài đặt các thư viện phụ thuộc

```powershell
pip install -r requirements.txt
```

---

## 🚀 Hướng dẫn chạy

### Chạy toàn bộ Pipeline (1-Click)

Thực hiện lệnh sau để chạy toàn bộ quá trình (Load data → EDA → Baseline → CV chọn K → Thí nghiệm Scale → Test cuối → Xuất biểu đồ & bảng kết quả):

```powershell
python main.py
```

### Chạy với các tùy chọn bổ sung:

```powershell
# Bỏ qua tạo lại hình ảnh EDA (tăng tốc độ chạy)
python main.py --skip-eda

# Tùy chỉnh seed ngẫu nhiên hoặc số fold CV
python main.py --random-state 42 --cv-folds 5
```

### Chạy kiểm tra từng module độc lập:

```powershell
# Kiểm tra tải dữ liệu và phân chia tập
python -m src.data
```

---

## 🌐 Khởi chạy Hệ thống Web & Backend API (Fullstack)

Hệ thống đã được **ghép nối hoàn chỉnh giữa Backend (FastAPI) và Frontend (React + Vite)**:

### 1. Khởi động 1-Click (Khuyên dùng - Chạy cả Giao diện Web + API trên cổng 8000):

Chỉ cần một lệnh duy nhất:
```powershell
python run_server.py
```
* **Ứng dụng Web (Frontend UI):** `http://127.0.0.1:8000/` (Giao diện 3 màn hình Overview, Dự đoán tuổi, Model Card).
* **Swagger UI (Tài liệu API tương tác):** `http://127.0.0.1:8000/docs`
* **ReDoc:** `http://127.0.0.1:8000/redoc`
* **Health Check API:** `http://127.0.0.1:8000/api/health`

### 2. Chế độ Phát triển Frontend (Hot-reload độc lập):

Nếu muốn chỉnh sửa code giao diện frontend theo thời gian thực:
* **Terminal 1 (Backend API):** `python run_server.py` (Lắng nghe tại port `8000`)
* **Terminal 2 (Frontend Dev):** 
  ```powershell
  cd frontend
  yarn dev
  ```
  Truy cập `http://localhost:5173/` (Vite dev server tự động proxy toàn bộ request `/api` sang backend port `8000`).

### 3. Các Endpoint API chính:

| Phương thức | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Kiểm tra trạng thái hoạt động của server và model |
| `GET` | `/api/model-card` | Lấy siêu dữ liệu Model Card, chỉ số đánh giá, giới hạn |
| `GET` | `/api/feature-ranges` | Lấy miền giá trị và thống kê các biến đầu vào |
| `POST` | `/api/predict` | Dự đoán số vòng, tuổi, khoảng tham chiếu láng giềng & cảnh báo OOD |

### 3. Chạy Unit Test kiểm thử hệ thống tự động:

```powershell
# 1. Kiểm thử toàn vẹn Pipeline, phân chia dữ liệu, không rò rỉ và tái lập
python tests/test_pipeline.py

# 2. Kiểm thử toàn diện REST API (10 ca kiểm thử: Endpoint, Schema, Mixed-unit rejection)
python -m app.test_api
```

---

## 🔬 Các thí nghiệm bắt buộc & Kết quả thực tế (Quy trình Train 70% / Val 15% / Test 15%)

1. **Thí nghiệm 1 (Baseline vs. k-NN):** Mô hình k-NN tối ưu vượt trội rõ rệt so với Median Baseline:
   - **Tập Validation (626 mẫu):** Baseline MAE = $2.3674$ $\rightarrow$ KNN tối ưu MAE = **$1.4564$** (giảm 38.5% sai số).
   - **Tập Test độc lập (626 mẫu):** Baseline MAE = $2.5655$, $R^2 = -0.0992$ $\rightarrow$ KNN tối ưu MAE = **$1.5063$**, $R^2 = \mathbf{0.5741}$.
2. **Thí nghiệm 2 (Ablation Study: No Scale vs. StandardScaler):**
   - So sánh bằng 5-Fold Cross-Validation trên tập Train (2,921 mẫu): 
     * `KNN - No Scale` đạt CV MAE = **$1.5213 \pm 0.0570$** ($K=17$, `weights='distance'`).
     * `KNN - StandardScaler` đạt CV MAE = $1.5360 \pm 0.0395$ ($K=17$, `weights='distance'`).
   - Kiểm định độc lập trên tập Validation (626 mẫu):
     * `KNN - No Scale` đạt **Val MAE = 1.4564** ($R^2 = 0.5394$).
     * `KNN - StandardScaler` đạt Val MAE = $1.5190$ ($R^2 = 0.5039$).
   - **Quyết định khóa mô hình (Freeze):** Do `No Scale` chiến thắng ở cả CV lẫn Validation, cấu hình `No Scale (K=17)` được lưu thành artifact chính thức. Theo nguyên tắc niêm phong tập Test (Zero Data Leakage), mô hình StandardScaler bị loại ở bước Validation và **không được đánh giá trên tập Test**.
   - **Đánh giá duy nhất trên tập Test (626 mẫu):** Mô hình đã khóa đạt **Test MAE = 1.5063**, Test RMSE = **$2.2778$**, $R^2 = \mathbf{0.5741}$.
3. **Thí nghiệm 3 (Đồ thị MAE theo K):** Tìm $K$ tối ưu qua 5-Fold CV trên Train ($K=17$, `weights='distance'` cho cả No Scale và StandardScaler), lưu đồ thị tại [`reports/figures/mae_vs_k.png`](reports/figures/mae_vs_k.png).
4. **Thí nghiệm 4 (Phân tích sai số theo nhóm tuổi trên tập Test):**
   - Nhóm non (`Young`, $\le 8$ vòng): $\text{MAE} = \mathbf{1.034}$ vòng.
   - Nhóm trưởng thành (`Middle`, $9-12$ vòng): $\text{MAE} = \mathbf{0.932}$ vòng.
   - Nhóm già (`Old`, $> 12$ vòng): $\text{MAE} = \mathbf{3.732}$ vòng (do hiện tượng bão hòa sinh trưởng kích thước vỏ ở bào ngư già).
