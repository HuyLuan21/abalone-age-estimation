# Hồ sơ Dữ liệu Abalone (Data Dossier)

Tài liệu hồ sơ dữ liệu bắt buộc theo quy chuẩn của đề tài nghiên cứu (Mục 65 trong đề tài giao nhiệm vụ).

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

## 3. Báo cáo chất lượng dữ liệu (Data Quality Summary)

- **Giá trị khuyết thiếu (Missing values):** **0** (không có bất kỳ giá trị NaN / Null nào).
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
- **Ngoại lệ vật lý cần chú ý:** Có 2 mẫu trong dữ liệu gốc có `Height = 0` (mã 1258 và 3996); các mẫu này được giữ nguyên để bảo đảm tính khách quan của dữ liệu gốc khi thực hiện k-NN (hoặc được cảnh báo qua bộ lọc OOD).

---

## 4. Hướng dẫn tái tạo / Tải dữ liệu

Để tải lại tệp từ kho lưu trữ UCI chính thức bằng script tự động:

```python
import urllib.request
url = "https://archive.ics.uci.edu/static/public/1/abalone.zip"
# hoặc tải trực tiếp data thô:
# https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data
```
