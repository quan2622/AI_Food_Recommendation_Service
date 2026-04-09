# Mô hình gợi ý lai ghép — đối chiếu mã nguồn và bản mô tả đã chỉnh sửa

Tài liệu này đối chiếu bản mô tả lý thuyết ban đầu với triển khai trong `recommendation-service/` (Python/FastAPI), rồi đưa **nội dung đã chỉnh sửa** phản ánh đúng hệ thống hiện tại.

---

## 1. Các điểm cần chỉnh sửa so với bản mô tả gốc

| # | Nội dung trong tài liệu gốc | Thực tế trong code | Gợi ý chỉnh sửa tài liệu |
|---|----------------------------|--------------------|-------------------------|
| 1 | Ba thành phần độc lập: `Score_content`, `Score_cf`, `Score_nutri` rồi cộng có trọng số | Không có `Score_nutri` tách riêng. Phần “dinh dưỡng” được gộp vào **điểm nội dung**: cosine giữa vector **dinh dưỡng còn lại trong ngày** và vector dinh dưỡng món, nhân hệ số mục tiêu (`goal_multiplier`), `meal_affinity`, và boost theo `nutrition_priority`. | Bỏ hoặc đổi thành “điểm nội dung dinh dưỡng-lệch bữa”, không tách score dinh dưỡng thứ ba. |
| 2 | Content-based: vector người dùng `p_u` = trung bình có trọng số món đã thích; cosine `p_u` với `f_i`; hoặc `α·cos(taste) + γ·cos(nutri)` với α=0.3, γ=0.7 | Vector “người dùng” ở tầng content là **`remaining_nutrition`** (calories, protein, carbs, fat, fiber còn lại so với mục tiêu), không phải embedding sở thích từ lịch sử thích. Không có tách sector taste/nutri với α, γ cố định. | Mô tả đúng: tương tự cosine giữa **ngân sách dinh dưỡng còn lại** và **profile dinh dưỡng món**; có thể nhắn thêm hệ số mục tiêu và ưu tiên macro (HIGH_PROTEIN, …). |
| 3 | Lọc cộng tác “nâng cao”: Matrix Factorization `R ≈ P·Qᵀ`, ALS/BPR, time-aware | CF thực tế là **user-based** từ lịch sử implicit: tìm “neighbor” có cùng món đã ăn, điểm tương tự `SUM(log(1+freq))`, rồi tổng hợp trọng số món ứng viên từ neighbor. **Không** có MF/ALS/BPR trong repo. | Đổi tiêu đề thành CF dựa neighbor / implicit; MF/ALS ghi là **hướng mở rộng** (README cũng ghi nhận). |
| 4 | Chuẩn hóa min–max cho từng loại score trước khi kết hợp tổng thể | Min–max áp dụng cho **điểm CF thô** trong `load_candidate_collaborative_scores` → `_normalize_scores` trên tập ứng viên. Content và các thành phần khác **không** qua min–max theo batch trước khi cộng; điểm cuối chỉ `clip` 0–1. | Chỉ rõ điều kiện chuẩn hóa: CF đã normalize theo ứng viên; hybrid không dùng công thức min–max đầy đủ như sơ đồ lý thuyết. |
| 5 | Công thức hai bước: `Score_pref = ρ·content' + (1-ρ)·cf'` rồi `Score_final = w1·pref + w2·nutri'` | Một bước: `α·content + β·cf + δ·popular + ε·profile − γ·repeat_penalty`, với `α, β, γ, δ, ε` **động** theo `total_logs`, có/không CF, `nutrition_priority`, `goal_type`. | Thay bằng công thức và bảng trọng số động (xem mục 2 dưới). |
| 6 | Vector đặc trưng món Việt: sector bữa, khẩu vị, vùng miền, macro chi tiết | DB: category, nutrition, `meal_affinity` từ thống kê bữa, allergen, popularity. **Không** có vector khẩu vị/vùng miền trong scoring. | Giảm phạm vi: đặc trưng dùng trong điểm số = dinh dữ liệu + affinity bữa + category (profile); taste/region là **mở rộng dữ liệu**. |
| 7 | Các tính năng: thực đơn ngày, món Việt theo budget, meal alternatives | **Một API** gợi ý (`/v1/recommendations`) với filter/score/rerank; không có module riêng “alternative meal” trong tên — có thể map ở tầng sản phẩm. | Ghi rõ phạm vi triển khai hiện tại là endpoint gợi ý chung; use case chi tiết là mô tả nghiệp vụ, không phải từng service. |
| 8 | `ranking/scorer.weighted_score` | Hàm **không được** gọi trong pipeline chính (chỉ là utility). | Không trích dẫn làm công thức chính thức. |

---

## 2. Nội dung mô tả đã chỉnh sửa (khớp triển khai hiện tại)

### 2.1. Tổng quan kiến trúc

Hệ thống dùng **hybrid scoring** kết hợp:

- **Điểm nội dung (nutrition-aware):** đo mức độ “khớp” giữa **dinh dưỡng còn lại trong ngày** của người dùng và **vector dinh dưỡng của món** (cosine), điều chỉnh theo **mục tiêu cân nặng** (`goal_type`), **độ phù hợp bữa** (`meal_affinity`), và **ưu tiên macro** (`nutrition_priority`).
- **Điểm cộng tác (implicit, neighbor-based):** người dùng tương tự được suy ra từ lịch sử món đã ăn; điểm món ứng viên từ hành vi neighbor, rồi **chuẩn hóa min–max** trên tập ứng viên.
- **Điểm phổ biến:** từ `popularity_count` (chuẩn hóa gần 0–1).
- **Điểm profile theo danh mục:** tần suất người dùng chọn theo `category` trong lịch sử.
- **Hệ phạt lặp lại:** giảm điểm món đã ăn lặp lại nhiều lần (repeat penalty).

Điểm cuối:

\[
\text{final\_score} = \alpha \cdot s_{\text{content}} + \beta \cdot s_{\text{cf}} + \delta \cdot s_{\text{popular}} + \varepsilon \cdot s_{\text{profile}} - \gamma \cdot s_{\text{repeat}}
\]

sau đó **cắt** vào \([0, 1]\).

Trọng số \(\alpha, \beta, \gamma, \delta, \varepsilon\) **không cố định**: phụ thuộc số log bữa (`total_logs`), việc có điểm CF khác 0 (`has_cf`), `nutrition_priority`, và `goal_type` (ví dụ `STRICT_DIET` tăng nhẹ trọng số nội dung, giảm CF).

### 2.2. Điểm nội dung (content-based thực tế)

- **Vector trái:** `remaining_nutrition = [calories, protein, carbs, fat, fiber]` còn lại so với mục tiêu ngày.
- **Vector phải:** `food.nutrition` cùng độ dài.
- **Cosine similarity** giữa hai vector → `base_score` (trường hợp đặc biệt khi cả hai norm ~0 có thể gán sàn nhỏ).
- **Nhân** `goal_multiplier(goal_type, food)` (ví dụ giảm cân: ưu tiên giảm calo, tăng chất xơ/protein tương đối).
- **Nhân** `meal_affinity[meal_type]` (thống kê từ lịch sử: món ăn thường xuất hiện ở bữa nào).
- **Cộng** `priority_boost` khi `nutrition_priority` khác `BALANCED`: đưa multiplier từ macro (protein/carbs/fat/fiber) vào dạng cộng có kiểm soát.

Không dùng cosine riêng “khẩu vị” vs “dinh dưỡng” với hệ số α = 0.3, γ = 0.7; toàn bộ phần “sở thích dinh dưỡng trong ngày” nằm trong một pipeline cosine + goal + bữa + priority.

### 2.3. Lọc cộng tác (collaborative — triển khai hiện tại)

- **Dữ liệu:** implicit từ `meal_items` / `daily_logs` (tần suất món).
- **Neighbor:** người dùng khác có tập món trùng với lịch sử gần đây của `u`; điểm tương tự dựa trên `log(1 + freq)` và số món chung.
- **Điểm ứng viên:** tổng trọng số từ neighbor cho từng `food_id` trong danh sách ứng viên.
- **Chuẩn hóa:** min–max trên tập điểm thô của các ứng viên (nếu min = max thì gán 1.0).

Đây là **CF dựa user-neighbor**, không phải Matrix Factorization. MF/ALS/pgvector là hướng mở rộng đã ghi trong README.

### 2.4. Chuẩn hóa và cold-start

- **CF:** đã min–max theo batch ứng viên (xem trên).
- **Cold-start / ít dữ liệu:** khi không có CF (`has_cf = false`), `beta = 0` và `alpha` cao hơn; khi có CF và nhiều log, `beta` tăng dần. Điều này tương đương tinh thần “ρ cao khi ít dữ liệu hành vi” nhưng **không** dùng một tham số ρ đơn lẻ trong công thức.

### 2.5. Xếp hạng và tái sắp xếp

- Sắp xếp theo `final_score` giảm dần.
- **MMR (Maximal Marginal Relevance)** trên top ứng viên: cân bằng điểm relevance và độ đa dạng (cosine giữa vector dinh dưỡng của các món).
- **Chèn món mới:** ưu tiên một slot cho món tạo gần đây (nếu đủ điều kiện).
- **Fallback:** nếu tất cả điểm ≤ 0, chuyển sang chiến lược **popular fallback**.

### 2.6. Đặc trưng dữ liệu món (phạm vi thực tế)

- Dinh dưỡng: calories, protein, carbs, fat, fiber (từ `food_nutrition_profiles` / `food_nutrition_values`).
- Danh mục, mô tả, ảnh; allergen qua ingredient.
- `meal_affinity` và `popularity_count` học từ lịch sử bữa ăn.
- Không có trong scoring: vector “độ mặn/cay/vùng miền” — nếu cần, phải bổ sung schema và pipeline.

### 2.7. Tham số API liên quan

- `meal_type`: `MEAL_BREAKFAST` | `MEAL_LUNCH` | `MEAL_DINNER` | `MEAL_SNACK` — lọc theo ngưỡng `meal_affinity`.
- `nutrition_priority`: `BALANCED | HIGH_PROTEIN | HIGH_CARBS | HIGH_FAT | HIGH_FIBER`.
- `exclude_food_ids`, `limit`, `meal_affinity_threshold`, v.v.

---

## 3. Tham chiếu mã nguồn

- Hybrid & trọng số động: `recommendation-service/app/recommenders/hybrid.py`
- Pipeline: `recommendation-service/app/services/recommendation_service.py`
- CF + normalize: `recommendation-service/app/db/repositories/recommendation_repository.py` (`load_candidate_collaborative_scores`, `_normalize_scores`)
- README phạm vi Phase 1 / gap: `recommendation-service/README.md`

---

*Tài liệu được đồng bộ với triển khai tại thời điểm chỉnh sửa; khi thay đổi thuật toán, cập nhật lại mục 1–2 cho khớp.*
