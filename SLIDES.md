# KỊCH BẢN & NỘI DUNG SLIDE BÁO CÁO BẢO VỆ ĐỒ ÁN
## Project 06: Ước Lượng Tuổi Bào Ngư Bằng Thuật Toán K-NN Regression

> **Thời lượng đề xuất:** Tổng cộng 12 - 15 phút
> * **Phần 1: Trình bày Slide (6 - 7 phút)** - Slide 1 đến Slide 9
> * **Phần 2: Demo trực tiếp sản phẩm Web/API (4 - 5 phút)** - Slide 10
> * **Phần 3: Đạo đức, Giới hạn & Vấn đáp phản biện (3 - 4 phút)** - Slide 11 & Slide 12

---

## SLIDE 1: Trang Tiêu Đề & Giới Thiệu Đồ Án
* **Tiêu đề lớn:** ƯỚC LƯỢNG TUỔI BÀO NGƯ BẰNG K-NEAREST NEIGHBORS REGRESSION
* **Tiêu đề phụ:** Xây dựng Pipeline Machine Learning Chống Rò Rỉ Dữ Liệu & Ứng Dụng Web Dự Đoán Thời Gian Thực
* **Học phần:** Machine Learning Cơ Bản (Project 06)
* **Thành viên nhóm:** [Tên sinh viên 1] - [MSSV 1] & [Tên sinh viên 2] - [MSSV 2]
* **Giảng viên hướng dẫn:** [Tên Giảng viên]
* **Thời gian thực hiện:** 06 Tuần | Kho mã nguồn mở: GitHub (Clean Architecture)

> 🎙️ **Speaker Notes (Lời thoại gợi ý):**
> *"Kính thưa Thầy/Cô và các bạn, hôm nay nhóm chúng em xin đại diện báo cáo kết quả thực hiện Project 06: Ước lượng tuổi bào ngư bằng thuật toán K-Nearest Neighbors Regression. Mục tiêu của nhóm không chỉ là xây dựng một mô hình dự đoán chính xác, mà quan trọng hơn là thiết lập một quy trình thực nghiệm chuẩn mực khoa học, giải quyết triệt để vấn đề rò rỉ dữ liệu và triển khai thành công một sản phẩm web hoàn chỉnh phục vụ nghiên cứu nuôi trồng thực tế."*

---

## SLIDE 2: Bối Cảnh Thực Tiễn, Câu Hỏi Nghiên Cứu & Bộ Dữ Liệu
* **Bài toán sinh học:**
  * Phương pháp truyền thống: Bào ngư phải bị giết mổ, vỏ được cắt lát mỏng, nhuộm màu và đếm số vòng sinh trưởng (`Rings`) dưới kính hiển vi.
  * Hạn chế: Tốn kém thời gian, chi phí cao, mang tính hủy diệt cá thể (không thể áp dụng cho giống thương phẩm).
  * Công thức tính tuổi sinh học: $\text{Tuổi (năm)} = \text{Rings} + 1.5$.
* **Mục tiêu Machine Learning:** 
  * Dự đoán số vòng vỏ (`Rings`) thông qua **8 phép đo vật lý không xâm lấn** (kích thước vỏ và khối lượng các phần).
* **Nguồn dữ liệu & Bản quyền:**
  * Nguồn: UCI Machine Learning Repository (DOI: 10.24432/C55C7W), bản quyền **CC BY 4.0**.
  * Quy mô: 4,177 cá thể, 8 đặc trưng đầu vào (1 phân loại `Sex` [M/F/I], 7 biến liên tục).
* **Kiểm toán chất lượng dữ liệu (Data Auditing):**
  * Không có giá trị thiếu (0 Missing values).
  * Phát hiện và loại bỏ chính xác **04 quan sát bất thường vật lý** trước khi chia tập (2 mẫu chiều cao bằng 0, 1 mẫu lỗi nhập liệu chiều cao 1.13m, 1 mẫu đường kính lớn hơn chiều dài). Dữ liệu sạch còn lại: **4,173 mẫu**.

> 🎙️ **Speaker Notes:**
> *"Trong thực tế ngành nuôi trồng thủy sản, việc đếm vòng vỏ dưới kính hiển vi là quy trình hủy diệt cá thể và tốn rất nhiều công sức. Câu hỏi nghiên cứu của nhóm là: Liệu ta có thể ước lượng chính xác tuổi bào ngư dựa trên các phép đo hình thái không xâm lấn hay không? Nhóm sử dụng tập dữ liệu chuẩn UCI gồm 4,177 cá thể. Trước khi tiến hành bất kỳ bước xử lý nào, nhóm đã kiểm toán và loại bỏ 4 điểm dị biệt vi phạm cấu trúc hình học và sinh học, đưa về tập dữ liệu sạch 4,173 quan sát."*

---

## SLIDE 3: Thiết Kế Phân Chia Dữ Liệu & Quy Trình Chống Rò Rỉ (3-Stage Protocol)
* **Nguyên tắc cốt lõi: Không rò rỉ dữ liệu (Zero Data Leakage).**
* **Phân định 3 tầng rõ ràng (70% Train - 15% Validation - 15% Test):**
  1. **Train Set (70% ~ 2,921 mẫu):** Dùng để huấn luyện và chạy **5-Fold Cross-Validation** tìm siêu tham số tối ưu ($K$, weights, scaling).
  2. **Validation Set (15% ~ 626 mẫu):** Đóng vai trò **Trọng tài độc lập** đấu loại giữa 2 ứng viên xuất sắc nhất (Scaled vs. No Scale) để chốt cấu hình thắng cuộc.
  3. **Refit Protocol (85% ~ 3,547 mẫu):** Sau khi khóa cấu hình, tái huấn luyện toàn bộ Pipeline trên gộp $\text{Train} + \text{Val}$ để mở rộng kho láng giềng k-NN.
  4. **Held-out Test Set (15% ~ 626 mẫu):** **Niêm phong hoàn toàn**, chỉ mở đúng 1 lần duy nhất để kết luận hiệu năng khách quan cuối cùng.
* **Pipeline đóng gói khép kín:**
  * Dùng `ColumnTransformer` lồng trong `Pipeline` của scikit-learn.
  * `OneHotEncoder` cho `Sex`, `StandardScaler` cho 7 biến liên tục; toàn bộ scaler chỉ được `fit` bên trong từng fold huấn luyện.

> 🎙️ **Speaker Notes:**
> *"Điểm cốt lõi giúp nhóm đảm bảo tính trung thực học thuật là quy trình 3 tầng nghiêm ngặt. Thay vì chỉ chia Train/Test đơn thuần dễ dẫn tới việc tinh chỉnh tham số dựa trên Test, nhóm phân tách rõ ràng: Train dùng để tìm siêu tham số qua 5-Fold CV; Validation dùng để quyết định chọn giữa mô hình có Scale hay không Scale; sau đó mới Refit cấu hình đã khóa trên 85% dữ liệu và mở tập Test duy nhất một lần để báo cáo kết quả."*

---

## SLIDE 4: Mô Hình Cơ Sở (Baseline) & Thuật Toán K-NN Regression
* **Mô hình cơ sở (Median Baseline):**
  * Dự đoán giá trị trung vị (`Median Rings`) làm mốc so sánh ngây thơ (*Naive Benchmark*).
  * Trung vị trên tập Train: $9.0$ vòng vỏ.
* **Thuật toán K-Nearest Neighbors Regression:**
  * Nguyên lý: Ước lượng số vòng vỏ của một cá thể dựa trên trung bình có trọng số của $K$ cá thể láng giềng gần nhất trong không gian đặc trưng Euclidean.
  * Vì sao k-NN phù hợp với bài toán này?
    * Không giả định tuyến tính (mối quan hệ giữa kích thước vỏ và tuổi sinh học mang tính phi tuyến).
    * Khả năng giải thích cao (*Explainable AI*): Cho phép truy xuất trực tiếp các cá thể láng giềng thực tế để kiểm chứng.
* **Hàm trọng số khoảng cách:**
  * `uniform`: Mọi láng giềng có quyền biểu quyết ngang nhau.
  * `distance`: Láng giềng càng gần cá thể cần đo thì trọng số đóng góp càng cao ($w_i = 1 / d_i$).

> 🎙️ **Speaker Notes:**
> *"Để đánh giá mô hình k-NN có thực sự học được tri thức hay không, nhóm thiết lập Baseline dự đoán trung vị cố định là 9 vòng. Thuật toán k-NN được lựa chọn nhờ tính trực quan và khả năng bắt được các quan hệ phi tuyến sinh học. Đặc biệt, k-NN cho phép trích xuất chính xác các cá thể tương đồng nhất làm bằng chứng giải thích cho người nuôi trồng."*

---

## SLIDE 5: Thí Nghiệm 1 & 2: 5-Fold CV & Vòng Tuyển Chọn Trên Validation (Ablation Study)
* **Bảng so sánh 5-Fold Cross-Validation trên tập Train (70% ~ 2,921 mẫu):**
  * `No Scale`: Best $K=17$, `weights='distance'`, $\text{CV MAE} = \mathbf{1.5213 \pm 0.0570}$.
  * `StandardScaler`: Best $K=17$, `weights='distance'`, $\text{CV MAE} = 1.5360 \pm 0.0395$.
* **Vòng đấu loại trực tiếp trên tập Validation (15% ~ 626 mẫu):**

| Mô hình | Trạng thái | Validation MAE | Validation RMSE | Validation $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Median Baseline** | Mốc đối sánh | 2.3674 | 3.3157 | -0.0987 |
| **KNN - StandardScaler** | Bị loại | 1.5190 | 2.2281 | 0.5039 |
| **KNN - No Scale** | **Chiến thắng & Khóa** | **1.4564** | **2.1468** | **0.5394** |

* **Hiện tượng bất ngờ & Phân tích nguyên nhân:**
  * Vì sao `No Scale` lại chiến thắng `StandardScaler`?
  * Trong dữ liệu gốc UCI, các tác giả đã chuẩn hóa 7 biến liên tục về cùng một thang đo (đã chia 200). Các biến đo khối lượng (Whole weight, Shell weight) có phương sai tự nhiên phản ánh mật độ tích lũy sinh khối lớn nhất của cá thể. Khi ép `StandardScaler`, ta vô tình kéo dãn các biến thứ yếu, làm loãng tín hiệu phân biệt khoảng cách Euclidean.

> 🎙️ **Speaker Notes:**
> *"Tại vòng đấu loại trên tập Validation, mô hình không scale đã xuất sắc vượt qua StandardScaler với MAE đạt 1.4564 so với 1.5190. Nhóm đã phân tích sâu và nhận thấy: do dữ liệu UCI đã được quy đồng thang đo chia 200 từ trước, phương sai tự nhiên của khối lượng vỏ đóng vai trò chỉ thị rất mạnh cho tuổi bào ngư; việc chuẩn hóa đơn vị thêm một lần nữa vô tình làm giảm độ nhạy này. Do đó, nhóm quyết định khóa cấu hình No Scale K=17 distance."*

---

## SLIDE 6: Thí Nghiệm 3: Đường Cong Siêu Tham Số K & Biểu Đồ Lỗi
* **Đồ thị MAE theo $K$ (Grid search từ $K=1$ đến $30$):**
  * *(Chèn hình ảnh: [`reports/figures/mae_vs_k.png`](file:///c:/Users/Admin/source/repos/ML/abalone-age-estimation/reports/figures/mae_vs_k.png))*
* **Nhận xét đường cong huấn luyện:**
  * Khi $K$ quá nhỏ ($K=1..3$): Sai số rất cao ($\text{MAE} > 1.8$), mô hình bị **Overfitting** do quá nhạy cảm với nhiễu cục bộ.
  * Khi $K$ tăng dần ($K=15..20$): Sai số chạm đáy tối ưu, đạt độ cân bằng Bias-Variance hoàn hảo tại $K=17$.
  * Khi $K > 25$: Sai số bắt đầu tăng nhẹ do **Underfitting** (vùng láng giềng quá rộng làm mất tính đặc trưng cục bộ).
  * Cơ chế `weights='distance'` (đường xanh lá) luôn ổn định và đạt sai số thấp hơn `weights='uniform'` (đường xanh dương) trên mọi giá trị $K$.

> 🎙️ **Speaker Notes:**
> *"Đồ thị trên slide minh họa rõ ràng hiện tượng đánh đổi Bias-Variance: với K quá nhỏ, mô hình overfit; với K quá lớn, mô hình underfit. Giá trị K=17 cùng trọng số khoảng cách distance mang lại điểm cân bằng tối ưu nhất."*

---

## SLIDE 7: Kết Quả Cuối Cùng Trên Tập Test Độc Lập (Held-out Test 15%)
* **Bảng so sánh hiệu năng trên tập Test độc lập (626 mẫu):**

| Mô hình | Huấn luyện trên | Test MAE (vòng) | Test RMSE (vòng) | Test $R^2$ | Mức độ cải thiện |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Median Baseline** | 85% Train + Val | 2.5655 | 3.6592 | -0.0992 | Mốc ngây thơ (Naive Benchmark) |
| **KNN - No Scale (Đã Khóa & Refit)** | **85% Train + Val** | **1.5018** | **2.2567** | **0.5820** | **Giảm 41.5% MAE, R² đạt 58.2%** |

* *(Chèn hình ảnh trực quan: [`reports/figures/actual_vs_predicted.png`](file:///c:/Users/Admin/source/repos/ML/abalone-age-estimation/reports/figures/actual_vs_predicted.png) và [`reports/figures/residual_plot.png`](file:///c:/Users/Admin/source/repos/ML/abalone-age-estimation/reports/figures/residual_plot.png))*
* **Kết luận thực nghiệm:**
  * Mô hình k-NN tối ưu vượt trội hoàn toàn Baseline, giảm hơn $1.06$ vòng sai số tuyệt đối trung bình và nâng hệ số giải thích $R^2$ lên $58.2\%$.
  * Toàn bộ cấu hình siêu tham số ($K=17$, `weights='distance'`, `No Scale`) được đóng băng từ vòng Validation; tập Test được giữ độc lập và chỉ dùng để đưa ra kết luận cuối cùng.

> 🎙️ **Speaker Notes:**
> *"Khi mở tập Test độc lập lần duy nhất, mô hình k-NN sau khi refit trên 85% dữ liệu đạt Test MAE là 1.5018 vòng, giảm hơn 41% sai số so với Baseline. Đồ thị Actual vs Predicted cho thấy các điểm dự đoán bám rất sát đường lý tưởng y=x ở vùng tuổi từ 6 đến 12."*

---

## SLIDE 8: Thí Nghiệm 4: Phân Tích Sai Số Chuyên Sâu Theo 3 Nhóm Tuổi Sinh Học
* **Kết quả phân tích MAE trên 3 nhóm tuổi:**
  * **Nhóm Non (`Young`, $\le 8$ vòng):** $\text{MAE} = \mathbf{1.033}$ vòng.
  * **Nhóm Trung bình (`Middle`, $9 - 12$ vòng):** $\text{MAE} = \mathbf{0.956}$ vòng.
  * **Nhóm Già (`Old`, $> 12$ vòng):** $\text{MAE} = \mathbf{3.649}$ vòng.
* **Giải thích nguyên nhân sinh học (Biological Insights):**
  1. **Ở giai đoạn non và trưởng thành ($\le 12$ vòng):** Bào ngư sinh trưởng mạnh mẽ, kích thước vỏ và khối lượng tỷ lệ thuận chặt chẽ với thời gian sống $\rightarrow$ Sai số cực thấp dưới 1 vòng vỏ.
  2. **Ở giai đoạn già ($> 12$ vòng):** Bào ngư bước vào giai đoạn bão hòa sinh trưởng; kích thước vỏ gần như ngừng tăng trong khi các vòng vỏ vẫn tiếp tục xếp lớp $\rightarrow$ Hiện tượng nén không gian đặc trưng khiến k-NN bị đánh giá non tuổi hơn thực tế.
  3. Mật độ mẫu già trong tự nhiên rất khan hiếm (chỉ chiếm ~19% tập Test).

> 🎙️ **Speaker Notes:**
> *"Phân tích lỗi theo nhóm tuổi là một đóng góp rất giá trị của đề tài: mô hình đạt độ chính xác cực cao ở cá thể non và trung bình với sai số chỉ dưới 1 vòng. Tuy nhiên ở nhóm già trên 12 vòng, sai số tăng lên 3.65 vòng. Điều này hoàn toàn khớp với quy luật sinh học: bào ngư già hầu như ngừng tăng trưởng kích thước vỏ, khiến các phép đo hình thái không còn phản ánh tuyến tính số tuổi."*

---

## SLIDE 9: Kiến Trúc Ứng Dụng Web & REST API Serving
* **Kiến trúc hệ thống Fullstack hiện đại:**
  * **Backend:** Python + FastAPI + Scikit-Learn Pipeline + Pydantic.
  * **Frontend:** React 19 + TypeScript + Vite + Recharts + Lucide Icons.
  * **Tách biệt Serving & Training:** API chỉ nạp artifact `.pkl` đã lưu, suy luận siêu tốc (< 10ms/request), không huấn luyện lại lúc runtime.
* **Các tính năng nổi bật đáp ứng Rubric:**
  1. **Hỗ trợ 2 hệ đơn vị:** Nhập theo chuẩn UCI ($0.455$) hoặc mm/gam thực tế ($91.0\text{ mm}$), tự động quy đổi hai chiều.
  2. **Giải thích bằng láng giềng (Explainability):** Trả về danh sách $K$ cá thể láng giềng gần nhất kèm khoảng tham chiếu tương đồng sinh học (Min/Max/Median).
  3. **Bộ lọc phát hiện mẫu ngoài miền (OOD Guardrail):** Cảnh báo khi người dùng nhập dữ liệu phi lý (ví dụ: Thịt + Vỏ > Khối lượng toàn phần; Đường kính > Chiều dài; Chiều cao $\le 0$).

> 🎙️ **Speaker Notes:**
> *"Để đưa mô hình vào phục vụ thực tế, nhóm đã phát triển trọn vẹn ứng dụng web fullstack. Điểm đặc biệt của hệ thống là khả năng giải thích dự đoán bằng cách hiển thị các láng giềng gần nhất, và bộ lọc bảo vệ OOD giúp phát hiện ngay các phép đo sai lệch hoặc mâu thuẫn sinh học."*

---

## SLIDE 10: Kịch Bản Demo Trực Tiếp Sản Phẩm (Live Demo - 4 Phút)
* **Màn hình 1: Giới thiệu & Phạm vi**
  * Trình bày mục tiêu bài toán, thẻ dữ liệu và bảng thông tin bản quyền CC BY 4.0.
* **Màn hình 2: Thao tác Dự đoán Tuổi & Láng giềng**
  * *Test Case 1 (Mẫu bình thường):* Nhập kích thước bào ngư trưởng thành $\rightarrow$ Hệ thống dự đoán $10.5$ tuổi (Nhóm Middle), hiển thị khoảng tham chiếu và 17 láng giềng tương đồng.
  * *Test Case 2 (Chuyển đổi đơn vị):* Bật chế độ mm/gam thực tế $\rightarrow$ Hệ thống tự động chia 200 mượt mà.
  * *Test Case 3 (Bẫy ngoài miền OOD):* Nhập mẫu có khối lượng thịt + vỏ lớn hơn toàn bộ cá thể $\rightarrow$ Hệ thống lập tức hiển thị cảnh báo đỏ vi phạm sinh học, không âm thầm thay thế giá trị.
* **Màn hình 3: Dashboard Thực Nghiệm & Model Card**
  * Trực quan hóa 3 thí nghiệm bắt buộc bằng biểu đồ động Recharts, hiển thị đầy đủ thông số siêu dữ liệu runtime từ endpoint `/api/model-card`.

> 🎙️ **Speaker Notes:**
> *(Chuyển sang trình duyệt thao tác trực tiếp trên giao diện [http://localhost:8000/](http://localhost:8000/) theo đúng 3 kịch bản trên).*

---

## SLIDE 11: Đạo Đức Học Thuật, Giới Hạn Mô Hình & Hướng Phát Triển
* **Sử dụng AI có trách nhiệm (Responsible AI):**
  * Mô hình phục vụ mục đích **sàng lọc nhanh hỗ trợ người nuôi**, tuyệt đối không tự ý thay thế quy trình kiểm định sinh học tiêu chuẩn.
* **Giới hạn kỹ thuật:**
  * Độ tin cậy giảm ở cá thể già ($> 12$ vòng).
  * Thuật toán k-NN phụ thuộc vào khoảng cách Euclidean, chi phí bộ nhớ tỉ lệ với kích thước tập dữ liệu.
* **Hướng phát triển:**
  * Bổ sung khoảng bất định bằng phương pháp Conformal Prediction.
  * Thu thập thêm dữ liệu hình ảnh mặt cắt vỏ để kết hợp đa phương thức (Multimodal Learning).

> 🎙️ **Speaker Notes:**
> *"Nhóm luôn ý thức rõ ranh giới đạo đức của mô hình: đây là công cụ hỗ trợ sàng lọc chứ không thay thế quy trình kính hiển vi. Hướng phát triển tiếp theo là nghiên cứu thêm phương pháp Conformal Prediction để đưa ra khoảng tin cậy chặt chẽ hơn cho từng mẫu dự đoán."*

---

## SLIDE 12: Bộ Câu Hỏi Vấn Đáp Phản Biện Trọng Tâm (Q&A Preparation)

### ❓ Câu 1: Vì sao MAE lại dễ giải thích và phù hợp hơn RMSE trong bài toán này?
* **Trả lời:** Đơn vị của MAE cùng bậc với biến mục tiêu (`vòng vỏ` hoặc `năm tuổi`). Khi nói "mô hình có MAE = 1.0 vòng", người nuôi trồng hiểu ngay là trung bình dự đoán lệch 1 năm tuổi. Ngược lại, RMSE phạt bình phương sai số lớn, rất nhạy cảm với nhóm bào ngư già (vốn có sai số tự nhiên lớn do sinh học), làm méo mó đánh giá năng lực của mô hình ở đại đa số cá thể bình thường.

### ❓ Câu 2: Biến phân loại `Sex` (M, F, I) được đưa vào tính khoảng cách k-NN như thế nào?
* **Trả lời:** `Sex` được mã hóa One-Hot thành 3 cột nhị phân $[0, 1]$. Khi tính khoảng cách Euclidean giữa 2 cá thể: nếu cùng giới tính thì khoảng cách phân loại bằng 0; nếu khác giới tính thì đóng góp khoảng cách bằng $\sqrt{1^2 + 1^2} = \sqrt{2} \approx 1.414$. Điều này giúp phân biệt rõ ràng giữa bào ngư con chưa trưởng thành (`Infant`) và bào ngư đã thành thục sinh dục (`M`/`F`).

### ❓ Câu 3: Làm thế nào nhận biết và xử lý dự đoán ngoài miền (Out-of-Distribution)?
* **Trả lời:** Hệ thống kiểm soát 3 lớp:
  1. Kiểm tra miền biên thống kê: So sánh đầu vào với $[Min, Max]$ học được từ tập Train.
  2. Kiểm tra bất biến sinh học: Đường kính không được vượt quá chiều dài; tổng khối lượng các bộ phận không vượt quá khối lượng toàn thân.
  3. Khi phát hiện OOD: Hệ thống vẫn trả về dự đoán nhưng kèm theo cảnh báo rõ ràng trên giao diện web để người dùng cảnh giác, không âm thầm thay thế dữ liệu.

### ❓ Câu 4: Vì sao bước Refit trên 85% Train + Val không gây rò rỉ dữ liệu tập Test?
* **Trả lời:** Vì toàn bộ các quyết định về cấu hình (chọn $K=17$, `weights='distance'`, `No Scale`) đều đã được **chốt cố định** từ trước đó tại tập Validation. Bước Refit hoàn toàn không tinh chỉnh thêm bất kỳ siêu tham số nào, chỉ nạp thêm dữ liệu để làm dày không gian láng giềng k-NN. Tập Test 15% vẫn được niêm phong độc lập và chỉ được mở ra đánh giá sau khi mô hình refit đã đóng gói xong.

---
*Bản quyền tài liệu thuộc về Nhóm tác giả Project 06 - Machine Learning 2026.*
