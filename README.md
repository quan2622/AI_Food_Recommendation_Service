# AI Food Recommendation Service

Dịch vụ gợi ý món ăn theo ngữ cảnh dinh dưỡng, thuộc dự án *AI-Powered Personalized Nutrition Tracking and Analysis Application*. Dịch vụ gợi ý các món phù hợp với ngân sách dinh dưỡng còn lại trong ngày, mục tiêu sức khỏe, bữa ăn và thói quen của người dùng.

Tài liệu này mô tả bài toán, cơ sở lý thuyết và mô hình gợi ý lai ghép (hybrid) của dịch vụ `recommendation-service/` (Python/FastAPI). Nội dung khớp với mã nguồn hiện tại. Hướng dẫn cài đặt, chạy và cấu hình xem [recommendation-service/README.md](recommendation-service/README.md).

---

## 1. Bài toán

Cho tập người dùng $\mathcal{U}$ và tập món ăn $\mathcal{I}$. Với mỗi người dùng $u$, hệ thống biết:

- mục tiêu dinh dưỡng trong ngày (`goal_type`: giảm cân, tăng cân, duy trì, chế độ chặt) và lượng đã nạp;
- dị ứng, lịch sử chọn món, bữa ăn hiện tại (sáng/trưa/tối/phụ);
- ưu tiên macro do người dùng chọn (`nutrition_priority`).

Mỗi món $i$ có hồ sơ dinh dưỡng (calories, protein, carbs, fat, fiber), danh mục, allergen và thống kê sử dụng.

**Mục tiêu:** xây dựng hàm điểm $S(u,i)$ để xếp hạng các món ứng viên, sao cho danh sách gợi ý vừa **khớp ngân sách dinh dưỡng còn lại và bữa ăn**, vừa **phù hợp sở thích/hành vi cộng đồng**, vừa **đa dạng**.

**Đặc thù dữ liệu:**

- Phản hồi ngầm (implicit): chỉ có tín hiệu "đã ăn món X, bao nhiêu lần", không có rating.
- Ràng buộc cứng: loại món chứa dị ứng, món không hợp bữa, món thiếu dữ liệu dinh dưỡng.
- Người dùng mới ít lịch sử (cold-start) nên mô hình phải dựa nhiều vào nội dung dinh dưỡng.

---

## 2. Luồng xử lý (pipeline)

1. **Nạp ngữ cảnh người dùng:** mục tiêu, dinh dưỡng đã nạp hôm nay, dị ứng, số lần lặp món, tổng log, tần suất danh mục.
2. **Nạp ứng viên:** metadata món, dinh dưỡng, allergen, `meal_affinity`.
3. **Lọc cứng:** dị ứng, ngưỡng `meal_affinity`, thiếu dinh dưỡng.
4. **Chấm điểm:** nội dung, cộng tác, phổ biến, profile danh mục, phạt lặp.
5. **Lai ghép** các điểm thành `final_score`.
6. **Tái xếp hạng:** MMR (đa dạng), giới hạn theo danh mục, chèn món mới.
7. **Fallback:** nếu mọi điểm ≤ 0, chuyển sang gợi ý theo độ phổ biến.

---

## 3. Các thành phần điểm

### 3.1. Điểm nội dung (content-based, nhận biết dinh dưỡng)

Phía "nhu cầu" không phải embedding sở thích, mà là **vector dinh dưỡng còn lại trong ngày**
$\mathbf{r}_u = [\text{cal}, \text{pro}, \text{carb}, \text{fat}, \text{fib}]$, so với vector dinh dưỡng của món $\mathbf{n}_i$ bằng cosine:

$$
s^{\cos}(u,i) = \max\Bigl(0, \min\bigl(1, \tfrac{\mathbf{r}_u^\top \mathbf{n}_i}{\lVert \mathbf{r}_u \rVert \cdot \lVert \mathbf{n}_i \rVert}\bigr)\Bigr)
$$

Điểm nền $s^{\text{base}} = s^{\cos}$. Nếu $s^{\cos}=0$ và $\sum \mathbf{r}_u = 0$ (đã đủ chỉ tiêu) thì gán $s^{\text{base}}=0{,}3$ để không triệt tiêu toàn bộ ứng viên.

Điểm nội dung tổng hợp:

$$
s_{\text{content}} = \text{clip}_{[0,1]}\bigl(s^{\text{base}} \cdot g(u,i) \cdot a(i,m) + b_{\text{prio}}(u,i)\bigr)
$$

| Thành phần | Ý nghĩa |
|---|---|
| $g(u,i)$ – `goal_multiplier` | Điều chỉnh theo mục tiêu (tối thiểu 0,25). Giảm cân: phạt calo cao, thưởng protein/chất xơ. Tăng cân: thưởng calo/protein. Chế độ chặt: phạt mạnh calo và fat, thưởng protein/chất xơ. |
| $a(i,m)$ – `meal_affinity` | Mức độ món $i$ thường được dùng ở bữa $m$, học từ lịch sử (mặc định 0,25 khi thiếu). |
| $b_{\text{prio}}$ – priority boost | Khi `nutrition_priority` ≠ `BALANCED`: $0{,}4\cdot (m_{\text{prio}}-1)$, với $m_{\text{prio}} = 1 + 0{,}5\cdot\text{macro-norm}$ của macro được ưu tiên (protein/50, carbs/80, fat/35, fiber/14, cắt ở 1). |

### 3.2. Điểm cộng tác (collaborative filtering, neighbor-based)

- **Dữ liệu:** tần suất món $f_{uj}$ từ `meal_items` / `daily_logs` (implicit).
- **Độ tương đồng người dùng:** $\text{sim}(u,v) \propto \sum_{j \in \mathcal{I}_u \cap \mathcal{I}_v} \log(1+f_{uj})$; lấy top láng giềng $\mathcal{N}(u)$.
- **Điểm thô:** $\tilde{s}_{\text{cf}}(u,i)=\sum_{v\in\mathcal{N}(u)} \text{sim}(u,v)\cdot h_{vi}$, với $h_{vi}=\log(1+\text{số lần } v \text{ ăn } i)$.
- **Chuẩn hóa min–max** trên tập ứng viên; nếu max = min thì gán 1.

Đây là CF dựa user-neighbor, dễ giải thích ("nhiều người có thói quen giống bạn chọn món này"). **Không** dùng Matrix Factorization/ALS/BPR; đó là hướng mở rộng.

### 3.3. Tín hiệu bổ sung

- **Phổ biến:** $s_{\text{pop}}(i)=\min(1,\ c_i/20)$, với $c_i$ = `popularity_count`.
- **Profile danh mục:** $s_{\text{prof}}(u,i)=\min(1,\ N_{u,cat(i)}/5)$, với $N_{u,cat(i)}$ = số lần $u$ chọn danh mục của $i$ (0 nếu thiếu danh mục).
- **Phạt lặp:** $p_{\text{rep}}(u,i)=1-0{,}85^{n_{ui}}$, với $n_{ui}$ = số lần chọn món gần đây (0 → 0; 3 → ≈0,39).

---

## 4. Mô hình lai ghép (Hybrid)

$$
S_{\text{final}} = \text{clip}_{[0,1]}\bigl(\alpha  s_{\text{content}} + \beta  s_{\text{cf}} + \delta  s_{\text{pop}} + \varepsilon  s_{\text{prof}} - \gamma  p_{\text{rep}}\bigr)
$$

Trọng số **động**, phụ thuộc số log bữa ăn (`total_logs`) và việc có điểm CF khác 0 (`has_cf`):

| Điều kiện | $\alpha$ | $\beta$ | $\gamma$ |
|---|---|---|---|
| Không CF, `total_logs` < 10 | 0,90 | 0 | 0,05 |
| Không CF, 10–60 | 0,85 | 0 | 0,10 |
| Không CF, > 60 | 0,80 | 0 | 0,15 |
| Có CF, < 10 | 0,85 | 0,05 | 0,05 |
| Có CF, 10–60 | 0,65 | 0,20 | 0,10 |
| Có CF, > 60 | 0,50 | 0,35 | 0,10 |

Giá trị cơ sở: $\delta=0{,}06$, $\varepsilon=0{,}07$. Sau đó điều chỉnh:

- **`nutrition_priority` ≠ BALANCED:** $\alpha$ tăng +0,30 (HIGH_PROTEIN, HIGH_FIBER) hoặc +0,25 (HIGH_CARBS, HIGH_FAT), tối đa 0,90; $\beta$ giảm 0,85 × mức tăng (tối thiểu 0,03); $\delta$ và $\varepsilon$ giảm 0,02 (sàn 0,04 và 0,05).
- **`goal_type` = STRICT_DIET:** $\alpha$ +0,10 (tối đa 0,95), $\beta$ −0,10, $\delta$ và $\varepsilon$ −0,02.
- Cuối cùng $\alpha,\beta,\delta,\varepsilon$ được **chuẩn hóa để tổng bằng 1**; $\gamma$ trừ riêng.

**Luận giải:** người dùng mới (ít log, chưa có CF) dựa chủ yếu vào dinh dưỡng và bữa ăn, tránh phụ thuộc cộng đồng. Khi dữ liệu hành vi tăng, trọng số CF tăng dần. Ưu tiên macro hoặc chế độ chặt đẩy trọng số về phía khớp dinh dưỡng.

---

## 5. Tái xếp hạng

- Sắp xếp theo $S_{\text{final}}$ giảm dần.
- **MMR** (Maximal Marginal Relevance) để tránh danh sách đồng nhất về dinh dưỡng:

$$
\text{MMR}(i)=\lambda  S_{\text{final}}(u,i)-(1-\lambda)\max_{j\in\mathcal{S}}\cos(\mathbf{n}_i,\mathbf{n}_j)
$$

  với $\mathcal{S}$ là tập món đã chọn; $\lambda$ lớn ưu tiên điểm gốc, nhỏ ưu tiên đa dạng (ví dụ $\lambda=0{,}7$).
- **Diversity cap** theo danh mục và **chèn món mới** (món tạo gần đây) vào khoảng vị trí thứ 3 để tăng khám phá.

---

## 6. API

| Endpoint | Mô tả |
|---|---|
| `GET /health` | Kiểm tra dịch vụ |
| `GET /v1/recommendations` | Lấy gợi ý |
| `POST /v1/recommendations/query` | Lấy gợi ý (body JSON) |
| `POST /v1/feedback` | Gửi phản hồi |

Tham số chính của `GET /v1/recommendations`: `user_id`, `meal_type` (sáng/trưa/tối/phụ), `current_time`, `limit`, `exclude_food_ids`, `meal_affinity_threshold`, `nutrition_priority` (`BALANCED | HIGH_PROTEIN | HIGH_CARBS | HIGH_FAT | HIGH_FIBER`).

Chi tiết cài đặt, cách chạy và biến môi trường xem [recommendation-service/README.md](recommendation-service/README.md).

---

## 7. Phạm vi, hạn chế và hướng mở rộng

**Chưa có trong triển khai hiện tại:**

- Vector khẩu vị / vùng miền trong tính điểm (cần bổ sung schema và pipeline).
- Matrix Factorization / ALS / BPR (cần pgvector hoặc hạ tầng huấn luyện).
- Nhiều module riêng như "thực đơn ngày" hay "món thay thế": hiện chỉ có một endpoint gợi ý chung.
- `targetFiber` trong `nutrition_goals` và nguồn calo đốt cháy (TDEE động) cần bổ sung từ schema bên ngoài.
- Cache hiện ở bộ nhớ trong; production nên dùng Redis.

**Đánh giá:** với phản hồi ngầm nên dùng Recall@K, NDCG@K, Hit Rate hoặc A/B test, không dùng RMSE.

---

## 8. Tham chiếu mã nguồn

| Nội dung | File |
|---|---|
| Hybrid và trọng số động | `recommendation-service/app/recommenders/hybrid.py` |
| Pipeline gợi ý | `recommendation-service/app/services/recommendation_service.py` |
| CF và chuẩn hóa | `recommendation-service/app/db/repositories/recommendation_repository.py` (`load_candidate_collaborative_scores`, `_normalize_scores`) |

Khi thay đổi thuật toán, cập nhật lại mục 3–5 cho khớp.
