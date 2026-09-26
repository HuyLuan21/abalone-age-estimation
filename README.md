# Data Processing Guide

Hướng dẫn cài đặt và chạy phần **xử lý dữ liệu** của dự án **Abalone Age Estimation using K-NN Regression**.

## Yêu cầu môi trường

- Python 3.14 hoặc mới hơn
- Windows (PowerShell) _(có thể chạy trên Linux/macOS với lệnh tương đương)_

## Cài đặt dự án

### 1. Di chuyển vào thư mục dự án

```powershell
cd ../abalone-age-estimation
```

### 2. Tạo môi trường ảo

```powershell
python -m venv venv
```

### 3. Kích hoạt môi trường ảo

```powershell
venv\Scripts\activate
```

Sau khi kích hoạt thành công, đầu dòng lệnh sẽ xuất hiện `(venv)`.

### 4. Cài đặt thư viện

Dự án sử dụng các thư viện được khai báo trong `requirements.txt`.

```powershell
python -m pip install -r requirements.txt
```

Nội dung `requirements.txt`:

```text
pandas
matplotlib
scikit-learn
joblib
streamlit
```

## Cấu trúc dữ liệu

```text
data/
├── README.md
├── data_dictionary.md
└── raw/
    ├── abalone.data
    └── abalone.names
```

- `abalone.data`: dữ liệu gốc từ UCI Machine Learning Repository.
- `abalone.names`: tài liệu mô tả dataset.
- `README.md`: mô tả nguồn dữ liệu.
- `data_dictionary.md`: giải thích từng thuộc tính.

## Cách chạy

Thực hiện lệnh:

```powershell
python main.py
```

## Quy trình xử lý dữ liệu

Chương trình sẽ thực hiện toàn bộ pipeline Machine Learning theo các bước sau:

1. Đọc dữ liệu từ `data/raw/abalone.data`.
2. Gán tên cho các cột.
3. Kiểm tra kích thước và kiểu dữ liệu.
4. Kiểm tra giá trị thiếu.
5. Phân tích phân bố dữ liệu (`Sex`, `Rings`).
6. Chuyển biến `Sex` thành dữ liệu số bằng One-Hot Encoding.
7. Chia dữ liệu theo tỷ lệ 80% Train và 20% Test.
8. Huấn luyện mô hình Baseline (Linear Regression).
9. Huấn luyện k-NN Regression.
10. So sánh k-NN trước và sau khi chuẩn hóa bằng `StandardScaler`.
11. Tìm giá trị K tối ưu thông qua đồ thị MAE theo K.
12. Đánh giá mô hình bằng MAE, RMSE và R².
13. Phân tích sai số theo nhóm tuổi.
14. Lưu Pipeline và các biểu đồ.

## Kết quả đầu ra

Sau khi chạy thành công, chương trình tạo các tệp sau.

### Mô hình

```text
models/
└── abalone_knn_pipeline.pkl
```

Pipeline đã huấn luyện được lưu bằng `joblib` để sử dụng cho ứng dụng web ở bước tiếp theo.

### Biểu đồ

```text
reports/
└── figures/
    ├── rings_distribution.png
    ├── mae_vs_k.png
    ├── model_comparison.png
    └── actual_vs_predicted.png
```

Ý nghĩa các biểu đồ:

- **rings_distribution.png**: Phân bố số vòng của bào ngư.
- **mae_vs_k.png**: Sai số MAE theo từng giá trị K.
- **model_comparison.png**: So sánh hiệu năng giữa các mô hình.
- **actual_vs_predicted.png**: So sánh giá trị dự đoán với giá trị thực tế.

## Các chỉ số đánh giá

Chương trình sử dụng ba chỉ số chính:

| Chỉ số | Ý nghĩa                       |
| ------ | ----------------------------- |
| MAE    | Sai số tuyệt đối trung bình   |
| RMSE   | Sai số căn bậc hai trung bình |
| R²     | Hệ số xác định của mô hình    |

Các chỉ số này được hiển thị trên terminal sau khi hoàn thành quá trình huấn luyện.

## Khắc phục lỗi thường gặp

### Không tìm thấy dataset

Lỗi:

```text
FileNotFoundError: data/raw/abalone.data
```

Kiểm tra lại cấu trúc thư mục:

```text
data/
└── raw/
    └── abalone.data
```

### Thiếu thư viện

Nếu xuất hiện lỗi `ModuleNotFoundError`, cài lại thư viện:

```powershell
python -m pip install -r requirements.txt
```

### Sai Python Interpreter

Kiểm tra interpreter đang sử dụng:

```powershell
python -c "import sys; print(sys.executable)"
```

Kết quả mong muốn:

```text
...\abalone-age-estimation\venv\Scripts\python.exe
```
