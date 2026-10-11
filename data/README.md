# Hồ sơ Dữ liệu Abalone (Data Dossier)

Tài liệu hồ sơ dữ liệu bắt buộc theo quy chuẩn của đề tài nghiên cứu (Mục 65–67 trong đề tài giao nhiệm vụ).

---

## 1. Nguồn gốc & Bản quyền dữ liệu

- **Tên bộ dữ liệu:** Abalone Dataset
- **Nguồn chính thức:** [UCI Machine Learning Repository - Abalone](https://archive.ics.uci.edu/dataset/1/abalone)
- **DOI định danh:** [10.24432/C55C7W](https://doi.org/10.24432/C55C7W)
- **Ngày phát hành gốc:** 01/12/1995 (Tác giả: Warwick J. Nash et al., Marine Resources Division, Tasmania)
- **Giấy phép sử dụng:** Creative Commons Attribution 4.0 International (**CC BY 4.0**)
- **Trích dẫn nghiên cứu gốc:**
  > Nash, W. J., Sellers, T. L., Talbot, S. R., Cawthorn, A. J., & Ford, W. B. (1994). *The Population Biology of Abalone (_Haliotis_ species) in Tasmania. I. Blacklip Abalone (_H. rubra_) from the North Coast and Islands of Bass Strait*. Sea Fisheries Division, Technical Report No. 48.

---

## 2. Tính toàn vẹn & Mã băm kiểm tra (Checksum Verification)

Tệp dữ liệu sử dụng trực tiếp trong dự án được lưu tại `data/raw/abalone.data`:

- **Định dạng:** CSV thuần (không tiêu đề cột, phân tách bằng dấu phẩy)
- **Kích thước tệp:** 191,873 bytes (187.38 KB)
- **Số dòng quan sát:** 4,177 dòng
- **Số thuộc tính:** 9 cột (8 thuộc tính đầu vào + 1 thuộc tính mục tiêu `Rings`)
- **Mã băm MD5:** `a769fd0119787cac09158fe08971e480`
- **Mã băm SHA-256:** `de37cdcdcaaa50c309d514f248f7c2302a5f1f88c168905eba23fe2fbc78449f`

---

## 3. Báo cáo chất lượng dữ liệu & Quy tắc xử lý dữ liệu bất thường (Mục 67)

### 3.1. Thống kê tập dữ liệu thô (Raw Data - 4,177 mẫu)
- **Giá trị khuyết thiếu (Missing values):** **0** (không có giá trị NaN / Null).
- **Dòng trùng lặp hoàn toàn (Duplicate rows):** **0** dòng.
- **Phân bố giới tính (`Sex`):**
  - `M` (Male): 1,528 mẫu (36.58%)
  - `I` (Infant): 1,342 mẫu (32.13%)
  - `F` (Female): 1,307 mẫu (31.29%)
- **Thống kê biến mục tiêu (`Rings`):**
  - Giá trị nhỏ nhất (Min): 1 vòng (~2.5 tuổi)
  - Giá trị lớn nhất (Max): 29 vòng (~30.5 tuổi)
  - Giá trị trung vị (Median): 9.0 vòng (~10.5 tuổi)
  - Giá trị trung bình (Mean): 9.93 vòng ($\pm 3.22$)

### 3.2. Quy tắc loại bỏ 04 quan sát bất thường trước khi chia tập
Theo đúng yêu cầu tại Mục 67 của đề tài *(báo cáo chất lượng và danh sách mọi dòng bị loại kèm lý do)*, đúng **04 dòng dữ liệu vi phạm vật lý và lỗi ghi chép** đã được xác định và loại bỏ trước khi chia Train/Validation/Test, lưu bảng theo dõi tại [`reports/removed_anomalies.csv`](../reports/removed_anomalies.csv):

| Chỉ số dòng gốc | Thuộc tính vi phạm | Giá trị ghi nhận | Lý do khoa học loại bỏ |
| :---: | :--- | :--- | :--- |
| **1210** | `Diameter > Length` | `Diameter = 0.375`, `Length = 0.185` | Vi phạm hình học vỏ bào ngư: Chiều dài theo định nghĩa là trục đo lớn nhất. |
| **1257** | `Height == 0` | `Height = 0.0` mm | Phi sinh học: Cá thể bào ngư sống không thể có chiều cao vỏ bằng 0. |
| **2051** | `Height` ngoại lệ cực đoan | `Height = 1.130` mm | Lỗi nhập liệu (outlier typo): Chiều cao lớn gấp gần 10 lần trung bình (0.14) và lớn hơn cả chiều dài (0.455). |
| **3996** | `Height == 0` | `Height = 0.0` mm | Phi sinh học: Cá thể bào ngư sống không thể có chiều cao vỏ bằng 0. |

### 3.3. Tập dữ liệu sạch đưa vào mô hình (Cleaned Data - 4,173 mẫu)
- **Tổng số quan sát:** **4,173 mẫu** (loại đúng 4 dòng nêu trên).
- **Tính toàn vẹn vật lý:** Toàn bộ các phép đo đều strictly positive ($> 0$), `Height` dao động trong khoảng hợp lý $[0.010, 0.515]$, `Diameter` $\le$ `Length`.
- **Chiến lược phân chia 3 tập độc lập (70% Train / 15% Validation / 15% Test):**
  Tuân thủ nghiêm ngặt quy định chống rò rỉ dữ liệu (Zero Data Leakage) của đề tài:
  - **Tập Huấn luyện (Train set - 70%):** **2,921 mẫu**. Dùng để huấn luyện pipeline và thực hiện 5-Fold Cross-Validation dò tìm siêu tham số ($K$, hàm trọng số `weights`, bộ chuẩn hóa scaler).
  - **Tập Kiểm định (Validation set - 15%):** **626 mẫu**. Dùng làm tập đối chứng độc lập nhằm đánh giá và khóa (lock) cấu hình mô hình tối ưu trước khi mở tập Test.
  - **Tập Kiểm thử niêm phong (Test set - 15%):** **626 mẫu**. Dữ liệu hoàn toàn độc lập (held-out), chỉ được đưa vào đánh giá đúng 1 lần duy nhất để báo cáo hiệu năng thực tế cuối cùng (MAE, RMSE, $R^2$ và phân tích sai số theo nhóm tuổi). Tuyệt đối không can thiệp vào quá trình chọn mô hình hay tinh chỉnh siêu tham số.
- **Kiểm định tính toàn vẹn phân vùng dữ liệu:**
  - Tập chỉ số (indices) hoàn toàn rời nhau: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
  - Bảo toàn chính xác tổng số mẫu: $2,921 + 626 + 626 = 4,173\text{ mẫu}$ ($100\%$).

---

## 4. Hướng dẫn tải & Tái lập dữ liệu từ đầu

Dự án cung cấp script tự động tải dữ liệu gốc từ kho UCI và kiểm tra mã băm SHA-256:

```powershell
python src/download_data.py
```
Script sẽ tự động tải `abalone.data`, đối chiếu mã băm MD5/SHA-256 và thông báo nếu tệp bị sai lệch.
