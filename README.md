# Project 06: Ước lượng tuổi bào ngư bằng K-NN Regression

Dự án Machine Learning ước lượng tuổi của bào ngư (*Abalone*) thông qua 8 phép đo vật lý sử dụng thuật toán **K-Nearest Neighbors (k-NN) Regression**, tuân thủ nghiêm ngặt quy trình chống rò rỉ dữ liệu (*Data Leakage Prevention*), giao thức phân chia 3 tầng kèm tái huấn luyện (*3-Stage Train/Val/Test + Refit Protocol*), cùng ứng dụng Web Fullstack (FastAPI + React Vite) phục vụ dự đoán thời gian thực.

---

## 📁 Cấu trúc thư mục mã nguồn

Dự án được cấu trúc dạng **Modular Clean Architecture** theo đúng yêu cầu đề bài:

```text
abalone-age-estimation/
├── data/
│   ├── README.md               # Nguồn gốc dữ liệu UCI (DOI: 10.24432/C55C7W), bản quyền CC BY 4.0
│   ├── data_dictionary.md      # Giải thích chi tiết 8 thuộc tính, thang đo UCI và đơn vị mm/gam
│   └── raw/
│       ├── abalone.data        # Dữ liệu gốc 4,177 quan sát
│       └── abalone.names       # Mô tả gốc từ các tác giả (Warwick J Nash et al.)
├── src/                        # Mã nguồn module hóa
│   ├── __init__.py
│   ├── data.py                 # Tải dữ liệu, lọc 4 mẫu dị biệt vật lý & chia Train/Val/Test
│   ├── features.py             # ColumnTransformer (StandardScaler + OneHotEncoder) & Pipeline
│   ├── train.py                # 5-Fold CV trên Train, chọn mô hình & Refit trên 85% dữ liệu
│   └── evaluate.py             # Đánh giá metric (MAE, RMSE, R²), phân tích lỗi & vẽ đồ thị
├── app/                        # Ứng dụng Backend REST API (FastAPI)
│   ├── api.py                  # Các endpoint RESTful (/api/predict, /api/model-card, ...)
│   ├── predictor.py            # Service suy luận k-NN, trích xuất láng giềng & kiểm định OOD
│   ├── schemas.py              # Pydantic Schemas, chuyển đổi đơn vị linh hoạt UCI ↔ mm/g
│   └── test_api.py             # Bộ kiểm thử API tự động (Schema, Validation, Unit conversion)
├── frontend/                   # Ứng dụng Frontend SPA (React 19 + TypeScript + Vite + Recharts)
│   ├── src/                    # Giao diện 3 màn hình: Giới thiệu, Dự đoán & Dashboard Model Card
│   ├── public/model-reports/   # Artifact tĩnh đồng bộ từ quá trình huấn luyện
│   └── scripts/                # Script kiểm toán và trích xuất số liệu thực nghiệm
├── tests/
│   └── test_pipeline.py        # Kiểm thử toàn vẹn phân chia tập, không rò rỉ & tái lập AST
├── models/
│   ├── abalone_knn_pipeline.pkl# Pipeline scikit-learn hoàn chỉnh (Refit trên 85% Train+Val)
│   └── model_metadata.json     # Siêu dữ liệu động đồng bộ toàn hệ thống
├── reports/
│   ├── removed_anomalies.csv   # Kiểm kê 04 quan sát bất thường bị loại bỏ
│   ├── cv_results.csv          # Bảng chi tiết 120 cấu hình 5-Fold CV trên Train (70%)
│   ├── cv_ablation.csv         # Bảng so sánh tốt nhất CV: No Scale vs. StandardScaler
│   ├── validation_results.csv  # Bảng đấu loại ứng viên trên Validation (15%)
│   ├── model_results.csv       # Bảng tổng hợp kết quả Baseline & k-NN trên Test (15%)
│   ├── age_group_mae.csv       # Sai số MAE trên 3 nhóm tuổi sinh học (Young, Middle, Old)
│   └── figures/                # Toàn bộ biểu đồ trực quan hóa kết quả
├── requirements.txt            # Danh sách thư viện phụ thuộc Python
├── run_server.py               # Script khởi chạy 1-Click Fullstack Web & API
└── README.md                   # Tài liệu hướng dẫn chi tiết
```

---

## ⚙️ Hướng dẫn Cài đặt & Khởi tạo Môi trường (Step-by-Step)

### Bước 1: Clone hoặc mở thư mục dự án

Mở PowerShell hoặc Command Prompt tại thư mục dự án:
```powershell
cd c:\Users\Admin\source\repos\ML\abalone-age-estimation
```

### Bước 2: Tạo và kích hoạt môi trường ảo Python

Khuyến nghị sử dụng **Python 3.10 trở lên** (dự án được kiểm thử tương thích hoàn hảo trên Python 3.10, 3.11, 3.12, 3.14):

```powershell
# 1. Tạo môi trường ảo
python -m venv venv

# 2. Kích hoạt môi trường ảo trên Windows PowerShell
venv\Scripts\Activate.ps1

# (Nếu dùng Command Prompt / CMD thông thường: venv\Scripts\activate.bat)
```

> *Lưu ý nếu PowerShell báo lỗi Execution Policy:* Chạy lệnh `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` rồi kích hoạt lại.

### Bước 3: Cài đặt các thư viện phụ thuộc Python

```powershell
pip install -r requirements.txt
```

*Các thư viện lõi:* `scikit-learn==1.9.1`, `pandas>=2.2.0`, `numpy>=1.26.0`, `joblib>=1.4.0`, `matplotlib>=3.8.0`, `fastapi>=0.110.0`, `uvicorn>=0.28.0`, `pydantic>=2.6.0`, `httpx>=0.27.0`.

### Bước 4: Biên dịch Giao diện Web Frontend (Bắt buộc cho repo mới clone)

Do thư mục bản dựng `frontend/dist` không đưa lên Git để giữ repository tinh gọn, hãy thực hiện biên dịch frontend trước khi khởi động fullstack server:

```powershell
cd frontend
npm install
npm run build
cd ..
```

> **Lưu ý:** Lệnh `npm run build` sẽ tự động kích hoạt hook `prebuild` (`node scripts/prepare-model-reports.mjs`) để đồng bộ toàn bộ bảng kết quả và artifact mới nhất vào giao diện. Nếu khởi động server khi chưa build, backend vẫn chạy và hiển thị trang hướng dẫn cài đặt trực quan tại `http://127.0.0.1:8000/`.

---

## 🚀 Hướng dẫn Chạy dự án

### Cách 1: Khởi động Fullstack 1-Click (Khuyên dùng khi Báo cáo / Chấm thi)

Sau khi đã build frontend ở Bước 4, chỉ cần chạy một lệnh duy nhất:
```powershell
python run_server.py
```
Hệ thống sẽ tự động khởi tạo backend FastAPI và phục vụ giao diện Web React:
* **Giao diện Web ứng dụng (Frontend):** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
  * **Tab 1: Giới thiệu & Phạm vi** (Mục tiêu sinh học, quy chuẩn bài toán, bản quyền dữ liệu).
  * **Tab 2: Dự đoán & Láng giềng** (Form nhập liệu 2 hệ đơn vị, bảng láng giềng gần nhất, khoảng tham chiếu từ K láng giềng & cảnh báo OOD).
  * **Tab 3: Dashboard & Model Card** (3 thí nghiệm bắt buộc, đường cong MAE vs K, phân tích sai số 3 nhóm tuổi, Model Card).
* **Tài liệu API tương tác (Swagger UI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Tài liệu API (ReDoc):** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
* **Kiểm tra trạng thái hệ thống:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

### Cách 2: Chế độ Phát triển Frontend độc lập (Hot-reload UI)

Nếu cần chỉnh sửa mã nguồn giao diện frontend:
* **Terminal 1 (Backend API):**
  ```powershell
  python run_server.py
  ```
* **Terminal 2 (Frontend Dev Server):**
  ```powershell
  cd frontend
  npm install   # hoặc yarn
  npm run dev   # hoặc yarn dev
  ```
  Mở trình duyệt tại [http://localhost:5173/](http://localhost:5173/). Vite dev server đã cấu hình proxy tự động chuyển tiếp request `/api` sang cổng `8000`.

---

### Cách 3: Tái huấn luyện Pipeline (Train / Val / Test / Refit)

Để chạy lại toàn bộ quy trình tiền xử lý, cross-validation, chọn mô hình và xuất toàn bộ artifact/biểu đồ từ đầu:
```powershell
python main.py
```
*Các tùy chọn nâng cao:*
```powershell
# Bỏ qua vẽ lại biểu đồ EDA để chạy nhanh:
python main.py --skip-eda

# Tùy chỉnh seed ngẫu nhiên hoặc số fold:
python main.py --random-state 42 --cv-folds 5
```

---

### Cách 4: Chạy Bộ kiểm thử tự động (Unit Test Suite)

Dự án trang bị 2 bộ kiểm thử chuyên sâu nhằm đảm bảo tính toàn vẹn và chuẩn mực khoa học:

```powershell
# 1. Kiểm thử Pipeline, phân chia dữ liệu, không rò rỉ (AST test) và serialization:
python tests/test_pipeline.py

# 2. Kiểm thử toàn diện API FastAPI (Health, Model Card, Feature Ranges, OOD, Unit conversion):
python -m app.test_api
```

---

## 🔬 Quy trình Khoa học & Kết quả Thực nghiệm

Dự án áp dụng **Giao thức 3 tầng kèm Tái huấn luyện (3-Stage Protocol with Refit)** chuẩn mực trong nghiên cứu Machine Learning:

```
Tập Dữ liệu Đã Làm Sạch (4,173 mẫu sau khi loại 4 mẫu bất thường vật lý)
  ├── Train (70% ~ 2,921 mẫu): Chạy 5-Fold CV tìm siêu tham số tối ưu (K, weights, scaling)
  ├── Validation (15% ~ 626 mẫu): Tuyển chọn ứng viên xuất sắc nhất -> KHÓA CẤU HÌNH
  ├── Refit (85% ~ 3,547 mẫu): Huấn luyện lại cấu hình đã khóa trên Train + Val
  └── Test (15% ~ 626 mẫu): Mở duy nhất 1 lần để đánh giá khách quan cuối cùng
```

### 1. Bảng tuyển chọn trên tập Validation (15% ~ 626 mẫu)

Sau khi chạy 5-Fold CV trên tập Train (70%), 2 ứng viên đại diện được đưa vào vòng so găng trên Validation:
* **Ứng viên A (StandardScaler):** $K=17$, `weights='distance'`
* **Ứng viên B (No Scale):** $K=17$, `weights='distance'`

| Mô hình | Trạng thái | Validation MAE (vòng) | Validation RMSE | Validation $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Median Baseline** | Mốc so sánh | 2.3674 | 3.3157 | -0.0987 |
| **KNN - StandardScaler** | Bị loại | 1.5190 | 2.2281 | 0.5039 |
| **KNN - No Scale** | **Chiến thắng & Khóa cấu hình** | **1.4564** | **2.1468** | **0.5394** |

> **Quyết định:** Cấu hình **`No Scale` ($K=17$, `weights='distance'`)** có sai số Validation thấp hơn rõ rệt nên được khóa làm cấu hình chính thức.

---

### 2. Kết quả Đánh giá Cuối cùng trên tập Test độc lập (15% ~ 626 mẫu)

Sau khi khóa cấu hình tại tập Validation, cả mô hình k-NN lẫn mô hình Baseline đều được **Refit trên toàn bộ 85% dữ liệu (Train + Validation = 3,547 mẫu)** để đảm bảo nguyên tắc so sánh hoàn toàn công bằng (*Fair Comparison Protocol*) về dung lượng dữ liệu học trước khi đánh giá trên tập Test độc lập:

| Mô hình | Huấn luyện trên | Test MAE (vòng) | Test RMSE (vòng) | Test $R^2$ | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Median Baseline** | 85% Train + Val | 2.5655 | 3.6592 | -0.0992 | Mốc ngây thơ (Naive Benchmark) |
| **KNN - No Scale** | **85% Train + Val** | **1.5018** | **2.2567** | **0.5820** | **Vượt trội (giảm 41.5% MAE so với Baseline)** |

*Nhận xét kết quả:* Mô hình k-NN chiến thắng với cấu hình khóa ($K=17$, `weights='distance'`, `No Scale`) đạt hiệu năng vượt trội, giảm hơn $1.06$ vòng sai số tuyệt đối trung bình so với Baseline và giải thích được hơn $58.2\%$ phương sai biến thiên của số vòng trên tập kiểm thử độc lập.

---

### 3. Phân tích Sai số theo Nhóm tuổi Sinh học (Test Set)

| Nhóm tuổi | Tiêu chí vòng vỏ | Số mẫu test | MAE (vòng) | Nhận xét sinh học |
| :--- | :---: | :---: | :---: | :---: |
| **Non (`Young`)** | $\le 8$ vòng | 185 | **1.033** | Độ chính xác cao, kích thước vỏ tỷ lệ thuận mạnh với số vòng |
| **Trung bình (`Middle`)** | $9 - 12$ vòng | 316 | **0.956** | Sai số thấp nhất, mật độ dữ liệu quan sát phong phú nhất |
| **Già (`Old`)** | $> 12$ vòng | 125 | **3.649** | Sai số tăng cao do hiện tượng bão hòa sinh trưởng kích thước ở cá thể già |

---

## 📡 Danh sách REST API Endpoints

| Method | Endpoint | Tham số / Body | Chức năng |
| :---: | :--- | :--- | :--- |
| `GET` | `/api/health` | Không | Kiểm tra trạng thái server và xác nhận model đã nạp vào bộ nhớ |
| `GET` | `/api/model-card` | Không | Trả về cấu trúc Model Card chuẩn, siêu dữ liệu, metrics và giới hạn |
| `GET` | `/api/feature-ranges` | Không | Trả về miền giá trị hợp lệ theo thang UCI và mm/gam thực tế |
| `POST` | `/api/predict` | `AbaloneInput` (JSON) | Dự đoán số vòng, tuổi (năm), nhóm tuổi, láng giềng k-NN và cảnh báo OOD |

---

## 🛡️ Nguyên tắc Đạo đức & Sử dụng có Trách nhiệm

1. **Phạm vi hỗ trợ:** Mô hình được thiết kế nhằm mục đích **sàng lọc sơ bộ** trong nghiên cứu nuôi trồng thủy sản, **không thay thế** quy trình mô học chuẩn (cắt vỏ, mài mỏng và đếm vòng dưới kính hiển vi).
2. **Cảnh báo ngoài miền (OOD):** Hệ thống tích hợp bộ lọc kiểm tra nghiêm ngặt mâu thuẫn sinh học (ví dụ: tổng khối lượng thịt + vỏ > toàn bộ khối lượng cá thể; đường kính lớn hơn chiều dài). Khi phát hiện mẫu bất thường, hệ thống sẽ gắn nhãn cảnh báo rõ ràng thay vì âm thầm đưa ra kết quả sai lệch.
3. **Bản quyền dữ liệu:** Bộ dữ liệu Abalone được cung cấp theo giấy phép mở **Creative Commons Attribution 4.0 International (CC BY 4.0)**, bảo đảm tính minh bạch và trung thực học thuật.
