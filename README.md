# Project 06: Ước lượng tuổi bào ngư bằng K-NN Regression

Dự án học máy ước lượng tuổi của bào ngư (*Abalone*) thông qua các phép đo vật lý sử dụng thuật toán **K-Nearest Neighbors (k-NN) Regression**, tuân thủ nghiêm ngặt quy trình chống rò rỉ dữ liệu (data leakage) và quy chuẩn bài tập lớn.

---

## 📁 Cấu trúc thư mục mã nguồn

Dự án được tái cấu trúc dạng **Modular** theo đúng khuyến nghị của đề bài:

```text
abalone-age-estimation/
├── data/
│   ├── README.md               # Mô tả nguồn gốc dữ liệu & DOI
│   ├── data_dictionary.md      # Giải thích chi tiết các thuộc tính
│   └── raw/
│       ├── abalone.data        # Dữ liệu gốc UCI
│       └── abalone.names       # Tài liệu gốc mô tả thuộc tính
├── src/                        # Mã nguồn module hóa
│   ├── __init__.py
│   ├── data.py                 # Tải, kiểm tra chất lượng & chia Train/Test
│   ├── features.py             # Xây dựng tiền xử lý & scikit-learn Pipeline
│   ├── train.py                # K-Fold CV, chọn tham số & lưu Model
│   └── evaluate.py             # Đánh giá metric, phân tích sai số & vẽ biểu đồ
├── models/
│   └── abalone_knn_pipeline.pkl# Pipeline k-NN đã huấn luyện hoàn chỉnh
├── reports/
│   ├── cv_results.csv          # Bảng kết quả 5-Fold Cross Validation
│   ├── model_results.csv       # Bảng so sánh hiệu năng các mô hình
│   ├── age_group_mae.csv       # Phân tích sai số theo nhóm tuổi
│   └── figures/                # Biểu đồ phân tích EDA và đánh giá mô hình
│       ├── rings_distribution.png
│       ├── sex_distribution.png
│       ├── mae_vs_k.png
│       ├── model_comparison.png
│       ├── actual_vs_predicted.png
│       └── residual_plot.png
├── app/                        # Ứng dụng Web & REST API
├── main.py                     # Script điều phối toàn bộ quy trình
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

## 🔬 Các thí nghiệm bắt buộc & Kết quả

1. **Thí nghiệm 1 (Baseline vs. k-NN):** Mô hình k-NN vượt trội hoàn toàn so với Median Baseline ($R^2 \approx 0.53$ vs $R^2 \approx -0.07$).
2. **Thí nghiệm 2 (Có scale vs. Không scale):** Đánh giá tác động của `StandardScaler` lên khoảng cách Euclidean của k-NN.
3. **Thí nghiệm 3 (Đồ thị MAE theo K):** Tìm được $K$ tối ưu thông qua 5-Fold CV trên tập huấn luyện (`reports/figures/mae_vs_k.png`).
4. **Thí nghiệm 4 (Sai số theo nhóm tuổi):** Phân tích sự khác biệt MAE giữa các nhóm `Young` ($\le 8$), `Middle` ($9-12$), và `Old` ($> 12$).
