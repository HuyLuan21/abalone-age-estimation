# Data Dictionary - Bộ dữ liệu Abalone

Tài liệu từ điển dữ liệu đặc tả chi tiết 9 trường thông tin của tập dữ liệu Abalone theo quy chuẩn môn học (Mục 66 trong đề tài giao nhiệm vụ).

## Bảng đặc tả thuộc tính (Data Dictionary)

| Tên biến | Kiểu dữ liệu | Đơn vị gốc | Đơn vị tệp UCI (Scale factor) | Vai trò | Thời điểm có sẵn | Mô tả chi tiết & Ý nghĩa sinh học |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sex** | Nominal | - | Phân loại (`M`, `F`, `I`) | **Feature** | Thu hoạch / Đánh bắt | Giới tính cá thể: `M` (Đực), `F` (Cái), `I` (Bào ngư non / Chưa phân hóa sinh dục). |
| **Length** | Continuous | mm | Giá trị $\div 200$ (tương đương $0.075 - 0.815$) | **Feature** | Đo sống khi thu hoạch | Chiều dài lớn nhất của vỏ (song song với trục dài). |
| **Diameter** | Continuous | mm | Giá trị $\div 200$ (tương đương $0.055 - 0.650$) | **Feature** | Đo sống khi thu hoạch | Đường kính lớn nhất vuông góc với chiều dài. |
| **Height** | Continuous | mm | Giá trị $\div 200$ (tương đương $0.010 - 0.300$) | **Feature** | Đo sống khi thu hoạch | Chiều cao thẳng đứng bao gồm cả phần thịt bên trong vỏ. |
| **Whole_weight** | Continuous | gam (g) | Giá trị $\div 200$ (tương đương $0.002 - 2.826$) | **Feature** | Cân sống khi thu hoạch | Tổng khối lượng toàn bộ cá thể bào ngư nguyên con (vỏ + thịt + nước nội tạng). |
| **Shucked_weight** | Continuous | gam (g) | Giá trị $\div 200$ (tương đương $0.001 - 1.488$) | **Feature** | Sau khi tách vỏ / mổ | Khối lượng phần thịt bào ngư có thể sử dụng được. |
| **Viscera_weight** | Continuous | gam (g) | Giá trị $\div 200$ (tương đương $0.001 - 0.760$) | **Feature** | Sau khi tách vỏ & mổ | Khối lượng hệ cơ quan nội tạng sau khi đã chảy hết máu. |
| **Shell_weight** | Continuous | gam (g) | Giá trị $\div 200$ (tương đương $0.002 - 1.005$) | **Feature** | Sau khi phơi / sấy khô | Khối lượng vỏ sau khi đã rửa sạch và sấy khô hoàn toàn. |
| **Rings** | Integer | vòng | Số nguyên dương ($1 - 29$) | **Target** | Sau khi cắt vỏ & soi kính hiển vi | Số vòng sinh trưởng trên vỏ; Tuổi thực nghiệm xấp xỉ $= \text{Rings} + 1.5$ năm. |

---

## 📌 Ghi chú quan trọng về Đơn vị đo & Hệ số tỉ lệ (Scaling Factor)

1. **Đơn vị kích thước và khối lượng:**
   - Trong nghiên cứu sinh học gốc của Nash et al. (1994), kích thước được đo bằng **milimét (mm)** và khối lượng được cân bằng **gam (g)**.
   - **Tuy nhiên**, trong bộ dữ liệu gốc lưu trữ tại UCI Machine Learning Repository, tác giả đã **chia tất cả 7 giá trị đo liên tục cho 200** (nhằm chuẩn bị dữ liệu cho mạng nơ-ron nhân tạo thời kỳ 1995).
   - *Ví dụ:* Mẫu có `Length = 0.455` thực tế tương ứng với chiều dài sinh học là $0.455 \times 200 = 91.0\text{ mm}$; `Whole_weight = 0.514` tương ứng với khối lượng sinh học $0.514 \times 200 = 102.8\text{ g}$.
2. **Quy tắc tính tuổi:**
   - Số tuổi sinh học của bào ngư được ước lượng bằng công thức chuẩn:
     $$\text{Age (năm)} = \text{Rings} + 1.5$$
3. **Tính toàn vẹn vật lý (Physical Consistency):**
   - Không có giá trị âm hoặc bằng 0 ở các phép đo.
   - Đường kính $\le$ Chiều dài (`Diameter <= Length`).
   - Từng thành phần khối lượng nhỏ không được vượt quá tổng khối lượng (`Shucked_weight`, `Viscera_weight`, `Shell_weight` $\le$ `Whole_weight`).
