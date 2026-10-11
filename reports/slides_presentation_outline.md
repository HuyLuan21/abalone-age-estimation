# NỘI DUNG SLIDE BÁO CÁO BẢO VỆ ĐỒ ÁN (12 SLIDES)
**Học phần:** Machine Learning / Nhập môn Học máy  
**Đề tài:** Dự đoán tuổi bào ngư bằng mô hình K-Nearest Neighbors Regression (Đề tài 06)  
**Phân công trình bày:** Thành viên phụ trách Mô hình (Model), Dữ liệu & Pipeline (Data Pipeline), Backend API  

---

## SLIDE 1: Trang tiêu đề & Phân công trách nhiệm
- **Tiêu đề:** Ứng dụng K-NN Regression trong ước lượng tuổi bào ngư không xâm lấn
- **Tên nhóm & Thành viên:**
  - **Thành viên 1 (Báo cáo chính phần Kỹ thuật):** Phụ trách Tiền xử lý dữ liệu, Huấn luyện mô hình K-NN, Tối ưu siêu tham số qua CV, Xây dựng Backend API (FastAPI) và Slide báo cáo.
  - **Thành viên 2:** Phụ trách Xây dựng Giao diện người dùng Web (Frontend 3 màn hình) và Biên soạn Báo cáo chuyên đề (Report Word 15–25 trang).
- **Mã nguồn dự án:** GitHub Repository / Branch chính thức.

---

## SLIDE 2: Đặt vấn đề & Ý nghĩa thực tiễn
- **Thách thức sinh học:** Phương pháp truyền thống xác định tuổi bào ngư đòi hỏi nhuộm mẫu mô, cắt mỏng vỏ và đếm vòng dưới kính hiển vi vi thể.
  - *Hạn chế:* Tốn nhiều giờ công, phá hủy mẫu sinh học, không thể áp dụng trên diện rộng trong công nghiệp nuôi trồng.
- **Mục tiêu mô hình:** Ước lượng số vòng vỏ (`Rings`) và tính toán tuổi sinh học ($\text{Tuổi} = \text{Rings} + 1.5$) hoàn toàn dựa trên 8 phép đo hình thái vật lý (Kích thước hình học & Khối lượng cân đo).
- **Tiêu chí chấp nhận:** Mô hình hồi quy K-NN vượt trội rõ rệt so với Baseline (Dummy Median), sai số MAE $\le 1.6$ vòng trên tập dữ liệu kiểm thử độc lập.

---

## SLIDE 3: Khảo sát dữ liệu & Quy tắc xử lý 04 quan sát bất thường
- **Nguồn dữ liệu:** UCI Machine Learning Repository (DOI: 10.24432/C55C7W, Giấy phép CC BY 4.0), gồm 4,177 mẫu gốc.
- **Bản chất thang đo UCI:** Toàn bộ 7 thuộc tính đo lường liên tục đã được UCI chia cho hệ số 200 ($1\text{ đơn vị UCI} = 200\text{ mm}$ hoặc $200\text{ g}$).
- **Quy tắc làm sạch khoa học (Loại bỏ đúng 04 quan sát bất thường TRƯỚC khi chia tập):**
  1. *Dòng 1210:* `Diameter` ($0.375$) > `Length` ($0.185$) — Vi phạm định nghĩa hình học vỏ (chiều dài là trục đo lớn nhất).
  2. *Dòng 1257 & 3996:* `Height == 0.0` — Bào ngư sống không thể có chiều cao vỏ bằng 0 (phi sinh học).
  3. *Dòng 2051:* `Height == 1.130` ($226\text{ mm}$) — Lỗi nhập liệu ngoại lai (cao gấp 10 lần trung bình mẫu và lớn hơn chiều dài).
- **Kết quả:** Tập dữ liệu sạch chuẩn mực gồm **4,173 quan sát** (100% kích thước đo lường strictly positive $> 0$).

---

## SLIDE 4: Chiến lược phân chia dữ liệu & Phòng chống rò rỉ (Zero Data Leakage)
- **Quy trình phân chia độc lập:**
  - Phân chia ngẫu nhiên $80/20$ với `random_state=42`:
    - **Tập Huấn luyện (Train set):** 3,338 mẫu (80%).
    - **Tập Kiểm thử (Test set):** 835 mẫu (20%).
- **Nguyên tắc chống rò rỉ thông tin (Data Leakage Prevention):**
  - Tập Test hoàn toàn bị niêm phong (held-out) trong suốt quá trình tiền xử lý, chuẩn hóa và tìm kiếm siêu tham số.
  - Toàn bộ siêu tham số ($K$, hàm trọng số, bộ chuẩn hóa) được lựa chọn **duy nhất** thông qua 5-Fold Cross-Validation trên 3,338 mẫu của tập Train.
  - Tập Test chỉ được nạp vào đúng một lần duy nhất để ghi nhận báo cáo khách quan.

---

## SLIDE 5: Kiến trúc mô hình K-NN & Thiết kế Pipeline
- **Tại sao chọn K-Nearest Neighbors Regression?**
  - Phù hợp với đặc thù hình thái học liên tục của sinh vật: các cá thể có kích thước và khối lượng tương tự nhau thường có cùng độ tuổi sinh trưởng.
  - Tính giải thích trực quan cao: kết quả ước lượng được diễn giải trực tiếp từ các cá thể láng giềng gần nhất.
- **Thiết kế Scikit-Learn Pipeline khép kín:**
  - `OneHotEncoder(drop='first')` cho biến phân loại `Sex` (`M`, `F`, `I`).
  - Lựa chọn chuẩn hóa: Thử nghiệm đối sánh giữa `StandardScaler` và `Không chuẩn hóa (No Scale)` do các biến hình thái cùng bậc đơn vị.
  - `KNeighborsRegressor`: Sử dụng khoảng cách Euclidean và trọng số khoảng cách (`weights='distance'`).

---

## SLIDE 6: Thực nghiệm Tối ưu hóa siêu tham số (5-Fold Cross-Validation)
- **Không gian siêu tham số tìm kiếm:**
  - $K \in [1, 30]$ (30 giá trị láng giềng).
  - Trọng số: `uniform` (đồng đều) vs `distance` (nghịch đảo khoảng cách).
  - Chuẩn hóa: `No Scale` vs `StandardScaler`.
  - Tổng số cấu hình đánh giá: $30 \times 2 \times 2 = 120$ cấu hình qua $120 \times 5 = 600$ lần huấn luyện.
- **Kết quả thực nghiệm trên Train CV (Tiêu chí tối thiểu hóa CV MAE):**
  - **Cấu hình tối ưu nhất:** `KNN - No Scale` với $K=19$, `weights='distance'` $\rightarrow$ **CV MAE = 1.5022 $\pm$ 0.0427**.
  - **Cấu hình đối sánh (Ablation):** `KNN - StandardScaler` với $K=25$, `weights='distance'` $\rightarrow$ **CV MAE = 1.5275 $\pm$ 0.0443**.
  - *Nhận xét:* Việc sử dụng trọng số theo khoảng cách (`distance`) giúp giảm hiện tượng overfitting ở $K$ nhỏ và cải thiện độ mượt ở $K$ lớn.

---

## SLIDE 7: Kết quả đánh giá trên tập kiểm thử độc lập (Test Set)
- **Bảng so sánh hiệu năng trên 835 mẫu Test:**
  | Mô hình | Test MAE | Test RMSE | Hệ số $R^2$ |
  | :--- | :---: | :---: | :---: |
  | **Median Baseline (Dummy)** | 2.5401 | 3.6095 | -0.0880 |
  | **KNN - StandardScaler ($K=25$)** | 1.5549 | 2.3335 | 0.5453 |
  | **KNN - No Scale ($K=19$, Best Model)** | **1.5223** | **2.2742** | **0.5681** |
- **Đánh giá định lượng:**
  - Mô hình tối ưu cải thiện **40.1% MAE** so với mô hình cơ sở ngẫu nhiên.
  - Giải thích được xấp xỉ **56.8% phương sai** của số vòng vỏ thực tế ($R^2 = 0.5681$).

---

## SLIDE 8: Phân tích sai số sinh học theo nhóm tuổi (Error Analysis)
- **Sai số phân bổ theo 3 giai đoạn trưởng thành:**
  - **Nhóm Trẻ ($\le 8$ vòng):** $\text{MAE} = \mathbf{1.028}$ vòng ($\approx 1.0$ tuổi).
  - **Nhóm Trung niên ($9 - 12$ vòng):** $\text{MAE} = \mathbf{1.004}$ vòng ($\approx 1.0$ tuổi).
  - **Nhóm Già ($> 12$ vòng):** $\text{MAE} = \mathbf{3.892}$ vòng ($\approx 3.9$ tuổi).
- **Lý giải sinh học:**
  - Ở giai đoạn trẻ và trưởng thành, tốc độ phát triển kích thước và khối lượng vỏ diễn ra tuyến tính mạnh mẽ với độ tuổi.
  - Khi bào ngư già đi ($> 12$ tuổi), tốc độ gia tăng kích thước vỏ chững lại (bão hòa sinh trưởng), trong khi vòng vỏ vẫn tiếp tục tích lũy chậm chạp theo năm. Do đó, các biến hình thái không còn phân tách rõ rệt giữa bào ngư 13 tuổi và 18 tuổi.
- **Giải pháp:** Cung cấp khoảng tham chiếu tin cậy và cảnh báo nhóm tuổi già trên hệ thống.

---

## SLIDE 9: Thiết kế Backend API (FastAPI & Dynamic Metadata)
- **Kiến trúc dịch vụ:**
  - Framework: **FastAPI** hiệu năng cao, tài liệu hóa tự động qua Swagger UI (`/docs`).
  - Phân tách rõ ràng giữa Core Predictor, API Routing và Pydantic Schemas.
- **Hỗ trợ đơn vị kép thông minh (Dual-Unit Normalization):**
  - Chấp nhận linh hoạt cả **thang UCI** (ví dụ $Length = 0.455$) hoặc **đơn vị thực tế mm/gam** (ví dụ $Length = 91.0\text{ mm}$).
  - Tự động phát hiện và chuẩn hóa về thang dữ liệu mà mô hình đã học.
- **Các Endpoint cốt lõi:**
  - `GET /api/health`: Kiểm tra trạng thái tải mô hình.
  - `GET /api/model-card`: Trả về siêu dữ liệu mô hình và chỉ số thực nghiệm động.
  - `GET /api/feature-ranges`: Cung cấp biên độ thống kê và đơn vị cho form nhập liệu.
  - `POST /api/predict`: Xử lý dự đoán, tính tuổi, phân nhóm và tìm láng giềng.

---

## SLIDE 10: Quản trị rủi ro & Tính năng Cảnh báo Ngoại lai (OOD Detection)
- **Phát hiện mẫu bất thường (Out-of-Distribution - OOD):**
  - Kiểm tra xem mẫu đầu vào có nằm ngoài phân bố huấn luyện (Length, Diameter, Height, Weights ngoài $[min - 3\sigma, max + 3\sigma]$).
  - Ràng buộc hình học vật lý: Bắt lỗi nếu $Diameter > Length$ hoặc trọng lượng các phần $> Whole\_weight$.
- **Khoảng tham chiếu từ K láng giềng (Confidence Reference Interval):**
  - Trả về danh sách chi tiết các láng giềng gần nhất (khoảng cách Euclidean, số vòng vỏ thực tế).
  - Cung cấp cận trên/cận dưới ($\min$, $\max$, $\text{median}$) của tuổi láng giềng, giúp chuyên gia nuôi trồng có cơ sở tin cậy khi ra quyết định.

---

## SLIDE 11: Đảm bảo chất lượng & Tính tái lập (Reproducibility & CI/CD)
- **Đóng gói và quản lý phiên bản:**
  - Lưu trữ định danh mã băm: MD5 (`a769fd01...`), SHA-256 (`de37cdcd...`).
  - Script tự động tải và kiểm tra toàn vẹn: `python src/download_data.py`.
- **Đồng bộ hóa Artifacts:**
  - Tệp mô hình chuẩn hóa: `models/abalone_knn_pipeline.pkl`.
  - Siêu dữ liệu đồng bộ: `models/model_metadata.json` được sinh tự động sau mỗi lần huấn luyện, loại bỏ hoàn toàn metadata fallback lỗi thời.
- **Bộ kiểm thử tự động (Unit Test Suite):**
  - `app/test_api.py` gồm 8 ca kiểm thử tích hợp (Health check, Model Card, Feature ranges, Predict UCI scale, Predict physical units, OOD detection, Physical constraint violations).
  - Kết quả kiểm thử: **100% Passed (8/8 test cases)**.

---

## SLIDE 12: Tổng kết, Phân công công việc & Định hướng phát triển
- **Tổng kết đóng góp phần Kỹ thuật (Model & Backend):**
  - Xử lý triệt để 4 quan sát bất thường trước khi phân chia dữ liệu, có kiểm chứng khoa học.
  - Quy trình huấn luyện không rò rỉ dữ liệu, tối ưu hóa qua 5-Fold Cross-Validation.
  - Hệ thống API chuẩn hóa, hỗ trợ chuyển đổi đơn vị thực tế, có kiểm thử hồi quy chặt chẽ.
- **Phối hợp nhóm:**
  - Cung cấp API schema, dữ liệu thống kê và mô hình cho thành viên phụ trách Web Frontend (3 màn hình).
  - Cung cấp bảng số liệu thực nghiệm, phân tích lỗi và mã nguồn cho thành viên viết Báo cáo 15–25 trang.
- **Định hướng nâng cấp:**
  - Thử nghiệm các mô hình phi tuyến tính phức tạp hơn (Random Forest, Gradient Boosting) cho nhóm bào ngư già.
  - Thu thập thêm các đặc trưng vi mô của vỏ để cải thiện độ chính xác ở nhóm trên 12 tuổi.
- **Sẵn sàng cho phần Hỏi - Đáp (Q&A Defense)!**
