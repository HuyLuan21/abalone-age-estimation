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

### 3.2. Tiêu chí kiểm soát chất lượng dữ liệu tiên nghiệm (Pre-defined Quality Rules)
Trước khi tiến hành phân chia dữ liệu hoặc huấn luyện mô hình, dữ liệu thô được kiểm định dựa trên **03 Quy tắc Kiểm soát Chất lượng Tiên nghiệm** xuất phát từ hình thái học sinh học của bào ngư và được hiện thực hóa trực tiếp trong mã nguồn [`src/data.py`](../src/data.py):

Biểu thức lọc trong code:
```python
cond_height_zero = df["Height"] <= 0
cond_height_extreme = df["Height"] >= 1.0
cond_diameter_gt_length = df["Diameter"] > df["Length"]
anomaly_mask = cond_height_zero | cond_height_extreme | cond_diameter_gt_length
```

Chi tiết căn cứ khoa học cho từng điều kiện:
1. **Quy tắc 1 - Trục kích thước hình thái sinh học (`df["Diameter"] > df["Length"]`):**
   - *Cơ sở sinh học:* Theo mô tả giải phẫu vỏ của Nash et al. (1994), Chiều dài ($Length$) được định nghĩa là trục đo lớn nhất của vỏ cá thể. Đường kính ($Diameter$) là trục đo vuông góc với chiều dài. Do đó, điều kiện hình học bắt buộc là $\text{Diameter} \le \text{Length}$.
   - *Phát hiện:* Dòng **1210** có `Diameter = 0.375` ($75.0\text{ mm}$) lớn hơn `Length = 0.185` ($37.0\text{ mm}$) $\rightarrow$ Vi phạm định nghĩa trục đo vỏ.
2. **Quy tắc 2 - Cá thể sống / Chiều cao dương nghiêm ngặt (`df["Height"] <= 0`):**
   - *Cơ sở sinh học:* Vỏ của cá thể bào ngư sống luôn có độ dày và thể tích khoang chứa nội tạng, chiều cao thẳng đứng không thể bằng 0.
   - *Phát hiện:* Dòng **1257** và dòng **3996** đều có `Height = 0.0` ($0.0\text{ mm}$) $\rightarrow$ Dữ liệu bị khuyết thiếu hoặc lỗi đo lường khi thu hoạch.
3. **Quy tắc 3 - Giới hạn miền giá trị hình thái tự nhiên (`df["Height"] >= 1.0`):**
   - *Cơ sở sinh học & Ngưỡng chặn:* Chiều cao trung bình của quần thể bào ngư trong tập dữ liệu là $0.140$ (thang UCI, tương ứng $28.0\text{ mm}$). Trong tự nhiên, cá thể già nhất cũng hiếm khi vượt quá $60\text{ mm}$ ($0.30$ thang UCI). Ngưỡng chặn an toàn $\text{Height} \ge 1.0$ (tương đương $\ge 200.0\text{ mm}$ vật lý) được thiết lập để loại trừ các sai số nhập liệu bất thường vượt quá mọi giới hạn sinh học.
   - *Phát hiện:* Dòng **2051** ghi nhận `Height = 1.130` (thang UCI, tương đương $\mathbf{226.0\text{ mm}}$), trong khi chiều dài cá thể chỉ là `Length = 0.455` ($91.0\text{ mm}$). Một cá thể dài $91\text{ mm}$ nhưng cao tới $226\text{ mm}$ (cao gấp 2.5 lần chiều dài và gấp 8 lần trung bình loài) là lỗi ghi chép/gõ phím (typo outlier).

| Dòng gốc | Giá trị UCI (gốc) | Giá trị vật lý quy đổi ($\times 200$) | Điều kiện code vi phạm | Quy tắc sinh học |
| :---: | :--- | :--- | :--- | :--- |
| **1210** | `Diameter=0.375`, `Length=0.185` | `Diameter=75.0 mm`, `Length=37.0 mm` | `Diameter > Length` | Quy tắc 1 (Trục hình thái) |
| **1257** | `Height=0.0` | `Height=0.0 mm` | `Height <= 0` | Quy tắc 2 (Cá thể sống) |
| **2051** | `Height=1.130`, `Length=0.455` | `Height=226.0 mm`, `Length=91.0 mm` | `Height >= 1.0` | Quy tắc 3 (Giới hạn miền vật lý) |
| **3996** | `Height=0.0` | `Height=0.0 mm` | `Height <= 0` | Quy tắc 2 (Cá thể sống) |

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
