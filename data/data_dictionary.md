# Data Dictionary - Bộ dữ liệu Abalone

Tài liệu từ điển dữ liệu đặc tả chi tiết 9 trường thông tin của tập dữ liệu Abalone theo quy chuẩn môn học (Mục 66 trong đề tài giao nhiệm vụ).

## Bảng đặc tả thuộc tính (Data Dictionary)

| Tên biến | Kiểu dữ liệu | Đơn vị gốc | Đơn vị tệp UCI (Scale factor) | Vai trò | Thời điểm có sẵn | Mô tả chi tiết & Ý nghĩa sinh học |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sex** | Nominal | - | Phân loại (`M`, `F`, `I`) | **Feature** | Thu hoạch / Đánh bắt | Giới tính cá thể: `M` (Đực), `F` (Cái), `I` (Bào ngư non / Chưa phân hóa sinh dục). |
| **Length** | Continuous | mm | Giá trị $\div 200$ (tương đương $0.075 - 0.815$) | **Feature** | Đo sống khi thu hoạch | Chiều dài lớn nhất của vỏ (song song với trục dài). |
| **Diameter** | Continuous | mm | Giá trị $\div 200$ (tương đương $0.055 - 0.650$) | **Feature** | Đo sống khi thu hoạch | Đường kính lớn nhất vuông góc với chiều dài. |
| **Height** | Continuous | mm | Giá trị $\div 200$ (tương đương $0.010 - 0.515$) | **Feature** | Đo sống khi thu hoạch | Chiều cao thẳng đứng bao gồm cả phần thịt bên trong vỏ (đã loại bỏ mẫu lỗi $0.0$ và $1.13$). |
| **Whole_weight** | Continuous | gam (g) | Giá trị $\div 200$ (tương đương $0.002 - 2.826$) | **Feature** | Cân sống khi thu hoạch | Tổng khối lượng toàn bộ cá thể bào ngư nguyên con (vỏ + thịt + nước nội tạng). |
| **Shucked_weight** | Continuous | gam (g) | Giá trị $\div 200$ (tương đương $0.001 - 1.488$) | **Feature** | Sau khi tách vỏ / mổ | Khối lượng phần thịt bào ngư có thể sử dụng được. |
| **Viscera_weight** | Continuous | gam (g) | Giá trị $\div 200$ (tương đương $0.001 - 0.760$) | **Feature** | Sau khi tách vỏ & mổ | Khối lượng hệ cơ quan nội tạng sau khi đã chảy hết máu. |
| **Shell_weight** | Continuous | gam (g) | Giá trị $\div 200$ (tương đương $0.002 - 1.005$) | **Feature** | Sau khi phơi / sấy khô | Khối lượng vỏ sau khi đã rửa sạch và sấy khô hoàn toàn. |
| **Rings** | Integer | vòng | Số nguyên dương ($1 - 29$) | **Target** | Sau khi cắt vỏ & soi kính hiển vi | Số vòng sinh trưởng trên vỏ; Tuổi thực nghiệm xấp xỉ $= \text{Rings} + 1.5$ năm. |

---

## 📌 Ghi chú quan trọng về Đơn vị đo, Hệ số tỉ lệ & Tính toàn vẹn

1. **Đơn vị kích thước và khối lượng:**
   - Trong nghiên cứu sinh học gốc của Nash et al. (1994), kích thước được đo bằng **milimét (mm)** và khối lượng được cân bằng **gam (g)**.
   - **Tuy nhiên**, trong bộ dữ liệu gốc lưu trữ tại UCI Machine Learning Repository, tác giả đã **chia tất cả 7 giá trị đo liên tục cho 200** (nhằm chuẩn bị dữ liệu cho mạng nơ-ron nhân tạo thời kỳ 1995).
   - *Ví dụ:* Mẫu có `Length = 0.455` thực tế tương ứng với chiều dài sinh học là $0.455 \times 200 = 91.0\text{ mm}$; `Whole_weight = 0.514` tương ứng với khối lượng sinh học $0.514 \times 200 = 102.8\text{ g}$.
2. **Quy tắc tính tuổi sinh học:**
   - Số tuổi sinh học của bào ngư được ước lượng bằng công thức chuẩn:
     $$\text{Age (năm)} = \text{Rings} + 1.5$$
3. **Quy tắc xử lý tính toàn vẹn vật lý (Physical Consistency):**
   - **Trong dữ liệu thô gốc:** Tồn tại 4 mẫu bất thường (2 mẫu `Height = 0`, 1 mẫu `Height = 1.13`, 1 mẫu `Diameter > Length`).
   - **Trong dữ liệu huấn luyện & phục vụ dự đoán:** Toàn bộ 4 mẫu bất thường đã được lọc bỏ có đối chiếu khoa học tại [`reports/removed_anomalies.csv`](../reports/removed_anomalies.csv). Do đó, dữ liệu đưa vào mô hình (4,173 mẫu) đảm bảo 100% tính toàn vẹn:
     * Tất cả các giá trị đo vật lý đều dương nghiêm ngặt ($> 0$).
     * Đường kính vỏ luôn nhỏ hơn hoặc bằng chiều dài vỏ (`Diameter <= Length`).
     * Không có giá trị ngoại lệ phi sinh học (`Height` dao động an toàn trong khoảng $(0.0, 0.6)$ theo thang UCI, tương ứng $(0, 120)\text{ mm}$).

4. **Quy ước Hệ quy chiếu & Mẫu ngoại lai ngoài phân bố (OOD - Out-of-Distribution):**
   - **Mẫu vật lý (Physical unit system):** Sử dụng đơn vị thực nghiệm milimét (mm) và gam (g). Đây là định dạng đo thực địa ngoài tự nhiên và là định dạng thân thiện cho chuyên gia nhập liệu trên Web UI.
   - **Mẫu chuẩn UCI (UCI normalized system):** Dữ liệu chuẩn lưu trong `abalone.data`, có được bằng công thức: $\text{Giá trị UCI} = \text{Giá trị vật lý (mm, g)} \div 200$. Đây là không gian vector chính thức mà mô hình K-NN học và tính khoảng cách Euclidean.
   - **Mẫu OOD (Out-of-Distribution):** Là các mẫu quan sát nằm ngoài phân bố dữ liệu huấn luyện hoặc vi phạm tính hợp lý vật lý sinh học. Hệ thống tự động kích hoạt cảnh báo OOD khi:
     * Vi phạm logic hình học: $Diameter > Length$, hoặc $Height \le 0$, hoặc $\sum \text{khối lượng thành phần} > Whole\_weight$.
     * Ngoại lai thống kê cực đoan: Bất kỳ thuộc tính nào vượt quá giới hạn $[ \min - 3\sigma, \max + 3\sigma ]$ của quần thể huấn luyện.
     Khi gặp mẫu OOD, hệ thống gắn nhãn cảnh báo để bảo đảm an toàn, không đưa ra kết luận thiếu căn cứ.
