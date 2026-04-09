# Cơ sở lý thuyết và mô hình gợi ý món ăn theo ngữ cảnh dinh dưỡng

Tài liệu này trình bày **cơ sở lý thuyết** các kỹ thuật gợi ý liên quan và **mô hình lai ghép** áp dụng trong hệ thống quản lý dinh dưỡng — phù hợp đưa vào **luận văn** (định nghĩa, ký hiệu, công thức, luận giải). Mô hình cụ thể phản ánh triển khai trong dịch vụ gợi ý (content + collaborative dựa neighbor + popularity + profile theo danh mục + phạt lặp + tái xếp hạng đa dạng).

---

## 1. Bài toán gợi ý trong ngữ cảnh dinh dưỡng

### 1.1. Định nghĩa bài toán

Cho tập người dùng \(\mathcal{U}\) và tập món ăn (vật phẩm) \(\mathcal{I}\). Mỗi người \(u \in \mathcal{U}\) có **ngữ cảnh ngày** gồm: mục tiêu dinh dưỡng, phần đã nạp, dị ứng, lịch sử chọn món, bữa ăn hiện tại (sáng/trưa/tối/phụ). Mỗi món \(i \in \mathcal{I}\) có vector đặc trưng (ít nhất là profile dinh dưỡng và metadata).

**Mục tiêu:** với mỗi cặp \((u, i)\) trong tập ứng viên sau lọc, xây dựng hàm điểm \(S(u,i) \in \mathbb{R}\) sao cho sắp xếp giảm dần theo \(S\) cho ra danh sách gợi ý **vừa phù hợp sở thích/hành vi cộng đồng, vừa khớp ngân sách dinh dưỡng còn lại và ràng buộc bữa ăn**.

### 1.2. Đặc thù dữ liệu

- **Phản hồi ngầm (implicit feedback):** đa số tín hiệu là “đã ghi nhận ăn món”, tần suất — không bắt buộc có điểm đánh giá rõ ràng.
- **Ràng buộc cứng:** loại trừ món chứa dị ứng; có thể loại món không phù hợp bữa (ngưỡng độ gắn bữa).
- **Mục tiêu sức khỏe:** giảm cân / tăng cân / duy trì / chế độ chặt — cần điều chỉnh điểm theo macro và năng lượng.

---

## 2. Lọc dựa trên nội dung (Content-Based Filtering) — nền tảng lý thuyết

### 2.1. Ý tưởng chung

Lọc nội dung dựa trên giả thuyết: **sở thích của người dùng có thể biểu diễn trong cùng không gian đặc trưng với món ăn**. Cách phổ biến:

- Xây dựng vector đặc trưng món \(\mathbf{f}_i\).
- Xây dựng vector hồ sơ người dùng \(\mathbf{p}_u\) (từ lịch sử thích, hoặc từ mục tiêu).
- Điểm tương thích: độ đo tương đồng, thường là **cosine similarity**:

\[
\mathrm{cos}(\mathbf{a}, \mathbf{b}) =
\frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}
\]

Nếu \(\|\mathbf{a}\|_2 = 0\) hoặc \(\|\mathbf{b}\|_2 = 0\), đặt \(\mathrm{cos} = 0\) (hoặc xử lý riêng theo ngữ cảnh).

### 2.2. Mô hình nội dung trong hệ thống dinh dưỡng (vector “ngân sách còn lại”)

Thay vì chỉ \(\mathbf{p}_u\) từ lịch sử “thích”, hệ thống dùng vector **dinh dưỡng còn lại trong ngày** làm phía “nhu cầu” để so khớp với vector dinh dưỡng của món.

**Ký hiệu:**

- \(\mathbf{r}_u = [r^{\mathrm{cal}}, r^{\mathrm{pro}}, r^{\mathrm{carb}}, r^{\mathrm{fat}}, r^{\mathrm{fib}}]^\top\) — lượng còn lại (calories, protein, carbs, fat, fiber).
- \(\mathbf{n}_i\) — vector dinh dưỡng cùng chiều của món \(i\) (theo đơn vị thống nhất với dữ liệu nguồn, ví dụ trên khẩu phần hoặc chuẩn hóa theo 100g tùy schema).

**Điểm nền (cosine):**

\[
s^{\mathrm{cos}}(u,i) = \mathrm{cos}(\mathbf{r}_u, \mathbf{n}_i) = \max\left(0,\ \min\left(1,\ \frac{\mathbf{r}_u^\top \mathbf{n}_i}{\|\mathbf{r}_u\|_2 \|\mathbf{n}_i\|_2}\right)\right).
\]

**Trường hợp biên:** nếu \(s^{\mathrm{cos}} = 0\) và \(\sum_k r_{u,k} = 0\) (không còn “room” dinh dưỡng theo vector đích), có thể gán điểm nền cố định nhỏ \(s_0\) (ví dụ \(0{,}3\)) để tránh triệt tiêu toàn bộ ứng viên — phản ánh triển khai thực tế.

**Hệ số mục tiêu cân nặng** \(g(u,i) \in [g_{\min}, +\infty)\) — điều chỉnh theo `goal_type` (ví dụ giảm cân: giảm trọng số calo cao, tăng nhẹ protein/chất xơ; tăng cân: tăng calo/protein). Dạng tổng quát:

\[
g(u,i) = \max\left(g_{\min},\ 1 + \Delta_{\mathrm{goal}}(u,i)\right),
\]

với \(\Delta_{\mathrm{goal}}\) là các hạng tử phụ thuộc macro đã chuẩn hóa (calories, protein, fat, fiber trong ngưỡng \([0,1]\)).

**Độ phù hợp bữa ăn** \(a(i, m) \in [0,1]\): xác suất/thống kê món \(i\) được dùng cho bữa \(m\) (sáng/trưa/tối/phụ), học từ lịch sử hoặc gán mặc định khi thiếu dữ liệu.

**Ưu tiên macro do người dùng chọn** (`nutrition_priority`): với mỗi loại ưu tiên (ví dụ HIGH_PROTEIN), định nghĩa hệ số nhân \(m_{\mathrm{prio}}(u,i) \ge 1\) phụ thuộc mức macro tương ứng của món (sau khi chuẩn hóa). Trong triển khai, phần tăng cường được đưa về **dạng cộng có kiểm soát**:

\[
b_{\mathrm{prio}}(u,i) = 0{,}4 \cdot \bigl(m_{\mathrm{prio}}(u,i) - 1\bigr) \quad \text{khi ưu tiên} \neq \text{BALANCED}.
\]

**Điểm nội dung tổng hợp:**

\[
s_{\mathrm{content}}(u,i) = \mathrm{clip}_{[0,1]}\Bigl(
  s^{\mathrm{base}}(u,i) \cdot g(u,i) \cdot a(i,m) + b_{\mathrm{prio}}(u,i)
\Bigr),
\]

trong đó \(s^{\mathrm{base}} = s^{\mathrm{cos}}\) hoặc \(s_0\) theo quy tắc biên nêu trên, và \(\mathrm{clip}_{[0,1]}(x) = \max(0, \min(1,x))\).

**Luận giải:** cosine giữa \(\mathbf{r}_u\) và \(\mathbf{n}_i\) thể hiện mức độ **cùng hướng** giữa “phần còn thiếu” và “phần món cung cấp” trong không gian dinh dưỡng — phù hợp bài toán “lấp đầy” ngân sách còn lại. Các nhân tố \(g\), \(a\), \(b_{\mathrm{prio}}\) lần lượt đưa **mục tiêu sức khỏe**, **ngữ cảnh bữa**, và **ưu tiên macro** vào cùng một điểm scalar.

---

## 3. Lọc cộng tác (Collaborative Filtering) — nền tảng lý thuyết

### 3.1. Ma trận tương tác và phản hồi ngầm

Xây dựng ma trận \(\mathbf{R} \in \mathbb{R}^{|\mathcal{U}| \times |\mathcal{I}|}\) với \(R_{ui}\) là cường độ tương tác (tần suất, hoặc nhị phân “đã ăn”). Ma trận thường **thưa**.

### 3.2. Các hướng cổ điển (tham chiếu lý thuyết)

- **Lọc cộng tác dựa mô hình:** phân rã ma trận, ví dụ **Matrix Factorization** tìm \(\mathbf{P} \in \mathbb{R}^{|\mathcal{U}| \times k}\), \(\mathbf{Q} \in \mathbb{R}^{|\mathcal{I}| \times k}\) sao cho \( \mathbf{R} \approx \mathbf{P}\mathbf{Q}^\top \), dự đoán \(\hat{r}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i\). Thích hợp dữ liệu lớn, học nhân tố tiềm ẩn.
- **Hạng và tối ưu implicit:** ALS cho implicit, **BPR** cho pairwise ranking — phổ biến khi không có rating rõ ràng.

Các hướng trên là **chuẩn mực lý thuyết**; hệ thống có thể mở rộng theo hướng này khi tăng quy mô dữ liệu và hạ tầng huấn luyện.

### 3.3. Mô hình cộng tác theo neighbor và phản hồi ngầm (triển khai điển hình)

**Ý tưởng:** tìm tập người dùng “láng giềng” \(\mathcal{N}(u)\) có lịch sử món **tương đồng** với \(u\), sau đó tổng hợp tín hiệu trên các món ứng viên \(i\) từ hành vi của láng giềng.

**Độ tương đồng người dùng–người dùng** có thể dựa trên tập món chung và tần suất, ví dụ dạng tổng log:

\[
\mathrm{sim}(u,v) \propto \sum_{j \in \mathcal{I}_{u} \cap \mathcal{I}_{v}} \log(1 + f_{uj}),
\]

với \(f_{uj}\) là tần suất người dùng \(u\) đã chọn món \(j\). Chọn top láng giềng theo \(\mathrm{sim}\) và số món chung.

**Điểm thô cho món \(i\):**

\[
\tilde{s}_{\mathrm{cf}}(u,i) = \sum_{v \in \mathcal{N}(u)} \mathrm{sim}(u,v) \cdot h_{vi},
\]

trong đó \(h_{vi}\) phản ánh mức độ láng giềng \(v\) tương tác với món \(i\) (ví dụ \(\log(1 + \text{số lượng})\)).

**Chuẩn hóa min–max trên tập ứng viên** \(\mathcal{C}_u\) (các món cần xếp hạng sau lọc):

\[
s_{\mathrm{cf}}(u,i) =
\begin{cases}
1, & \text{nếu } \max_j \tilde{s}_{\mathrm{cf}}(u,j) = \min_j \tilde{s}_{\mathrm{cf}}(u,j),\\[6pt]
\dfrac{\tilde{s}_{\mathrm{cf}}(u,i) - \min_j \tilde{s}_{\mathrm{cf}}(u,j)}{\max_j \tilde{s}_{\mathrm{cf}}(u,j) - \min_j \tilde{s}_{\mathrm{cf}}(u,j)}, & \text{ngược lại.}
\end{cases}
\]

**Luận giải:** cách tiếp cận neighbor-based **giải thích được** (món được nhiều người có sở thích giống \(u\) chọn), phù hợp giai đoạn đầu; chuẩn hóa giúp \(s_{\mathrm{cf}}\) nằm \([0,1]\) và so sánh được với các thành phần khác trong hybrid.

---

## 4. Tín hiệu bổ sung: độ phổ biến và hồ sơ theo danh mục

### 4.1. Điểm phổ biến

\[
s_{\mathrm{pop}}(i) = \min\left(1,\ \frac{c_i}{C_{\max}}\right),
\]

với \(c_i\) là số lần món \(i\) xuất hiện trong dữ liệu tổng hợp (ví dụ \(C_{\max} = 20\)).

### 4.2. Điểm profile theo danh mục

Cho tần suất người dùng \(u\) chọn các danh mục trong lịch sử. Với món \(i\) thuộc danh mục \(cat(i)\):

\[
s_{\mathrm{prof}}(u,i) = \min\left(1,\ \frac{N_{u,cat(i)}}{N_{\max}}\right),
\]

ví dụ \(N_{\max} = 5\). Nếu thiếu tên danh mục, điểm có thể bằng \(0\).

---

## 5. Phạt lặp lại (diversity theo thời gian)

Để giảm đề xuất lặp một món quá thường xuyên, định nghĩa **mức phạt** tăng theo số lần đã chọn món \(i\) trong cửa sổ gần đây \(n_{ui}\):

\[
p_{\mathrm{rep}}(u,i) = \mathrm{clip}_{[0,1]}\bigl(1 - \rho^{\,n_{ui}}\bigr),
\]

với \(0 < \rho < 1\) (ví dụ \(\rho = 0{,}85\)). Khi \(n_{ui} = 0\) thì \(p_{\mathrm{rep}} = 0\); khi \(n_{ui}\) lớn, \(p_{\mathrm{rep}} \to 1\).

---

## 6. Mô hình lai ghép (Hybrid Recommender): tổng hợp điểm

### 6.1. Công thức tổng quát

Kết hợp các thành phần đã chuẩn hóa về thang \([0,1]\) (hoặc có giới hạn rõ) bằng **trọng số không âm** và trừ phạt lặp:

\[
S_{\mathrm{hyb}}(u,i) =
\alpha\, s_{\mathrm{content}}(u,i)
+ \beta\, s_{\mathrm{cf}}(u,i)
+ \delta\, s_{\mathrm{pop}}(i)
+ \varepsilon\, s_{\mathrm{prof}}(u,i)
- \gamma\, p_{\mathrm{rep}}(u,i).
\]

**Điểm cuối** sau cắt ngưỡng:

\[
S_{\mathrm{final}}(u,i) = \mathrm{clip}_{[0,1]}\bigl(S_{\mathrm{hyb}}(u,i)\bigr).
\]

Các hệ số \(\alpha,\beta,\gamma,\delta,\varepsilon\) phụ thuộc **độ “trưởng thành” dữ liệu** (số bản ghi lịch sử), **có/không có** tín hiệu CF khác không, **ưu tiên macro**, và **kiểu mục tiêu** (ví dụ chế độ chặt làm tăng trọng số nội dung dinh dưỡng, giảm CF).

### 6.2. Nguyên lý điều chỉnh trọng số (minh họa)

- **Ít lịch sử hoặc không có CF:** \(\beta \approx 0\), \(\alpha\) lớn — ưu tiên **nội dung dinh dưỡng + ngữ cảnh bữa**, giảm phụ thuộc cộng đồng (cold-start hành vi).
- **Nhiều lịch sử và có CF:** tăng \(\beta\) — khai thác **tập láng giềng**.
- **Ưu tiên macro cụ thể (khác BALANCED):** tăng \(\alpha\), giảm \(\beta\) nhẹ — nhấn mạnh **khớp macro** so với signal CF.
- **Mục tiêu chế độ ăn chặt:** có thể tăng thêm \(\alpha\) và giảm \(\beta\) để bảo vệ ràng buộc dinh dưỡng.

Việc gán cụ thể \(\alpha,\ldots,\varepsilon\) có thể trình bày dưới dạng **bảng theo khoảng `total_logs` và cờ `has_cf`**, phù hợp phụ lục luận văn.

### 6.3. Vai trò từng thành phần (tóm tắt)

| Thành phần | Vai trò trong luận giải |
|------------|-------------------------|
| \(s_{\mathrm{content}}\) | Khớp **ngân sách dinh dưỡng còn lại**, mục tiêu cân nặng, bữa ăn, ưu tiên macro |
| \(s_{\mathrm{cf}}\) | Khai thác **hành vi tương tự** trong cộng đồng (implicit) |
| \(s_{\mathrm{pop}}\) | Giảm **cold-start món** — đưa món được dùng phổ biến |
| \(s_{\mathrm{prof}}\) | Cá nhân hóa theo **danh mục** đã chọn nhiều |
| \(p_{\mathrm{rep}}\) | Tránh **lặp** gợi ý gây nhàm chán |

---

## 7. Tái xếp hạng: đa dạng bằng MMR

Sau khi có \(S_{\mathrm{final}}\), danh sách top-\(L\) đầu có thể **quá đồng nhất** về dinh dưỡng. Dùng **Maximal Marginal Relevance (MMR)** để cân bằng độ liên quan và độ khác biệt giữa các món đã chọn.

Cho tập đã chọn \(\mathcal{S}\), ứng viên tiếp theo \(i\), tham số \(\lambda \in [0,1]\):

\[
\mathrm{MMR}(i) = \lambda\, S_{\mathrm{final}}(u,i) - (1-\lambda)\, \max_{j \in \mathcal{S}} \mathrm{sim}_{\mathrm{nutr}}(i,j),
\]

với \(\mathrm{sim}_{\mathrm{nutr}}(i,j)\) là cosine giữa vector dinh dưỡng của hai món. Chọn \(i\) cực đại \(\mathrm{MMR}\) lặp cho đủ \(K\) món (ví dụ \(\lambda = 0{,}7\)).

**Luận giải:** \(\lambda\) lớn → ưu tiên điểm gốc; \(\lambda\) nhỏ → ưu tiên **đa dạng dinh dưỡng** trong danh sách hiển thị.

Có thể kết hợp **chen món mới** (ưu tiên món tạo gần đây) ở một vị trí cố định trong top-\(K\) để tăng khám phá.

---

## 8. Luồng xử lý tổng thể (pipeline)

1. **Nạp ngữ cảnh** \(u\): mục tiêu, đã nạp trong ngày, dị ứng, lịch sử lặp, tổng log, tần suất danh mục.
2. **Nạp ứng viên** \(i\) và metadata dinh dưỡng.
3. **Lọc cứng:** dị ứng, ngưỡng phù hợp bữa, loại món thiếu dinh dưỡng tối thiểu.
4. **Tính** \(s_{\mathrm{content}}\), \(s_{\mathrm{cf}}\) (và chuẩn hóa CF), \(s_{\mathrm{pop}}\), \(s_{\mathrm{prof}}\), \(p_{\mathrm{rep}}\).
5. **Lai ghép** → \(S_{\mathrm{final}}\).
6. **Sắp xếp** → **MMR** (và quy tắc món mới nếu có).
7. **Fallback:** nếu mọi điểm không hợp lệ hoặc quá thấp, chuyển chiến lược gợi ý theo **độ phổ biến**.

---

## 9. Nhận xét về phạm vi lý thuyết và hướng mở rộng

- **Vector chỉ dinh dưỡng** ở tầng content: phù hợp quản lý calo/macro; **mở rộng** có thể thêm chiều khẩu vị, vùng miền (one-hot / embedding) vào \(\mathbf{f}_i\) và \(\mathbf{p}_u\).
- **CF neighbor-based** mở rộng tốt với **MF/ALS** khi dữ liệu và hạ tầng cho phép.
- **Đánh giá offline/online:** RMSE chỉ phù hợp khi có rating; với implicit nên dùng **Recall@K**, **NDCG@K**, **HR**, hoặc kiểm thử A/B trên tương tác thực.

---

## 10. Ký hiệu tham chiếu nhanh

| Ký hiệu | Ý nghĩa |
|--------|---------|
| \(u, i\) | Người dùng, món ăn |
| \(\mathbf{r}_u\) | Vector dinh dưỡng **còn lại** trong ngày |
| \(\mathbf{n}_i\) | Vector dinh dưỡng của món |
| \(s_{\mathrm{content}}\) | Điểm nội dung (cosine + điều chỉnh) |
| \(s_{\mathrm{cf}}\) | Điểm cộng tác (neighbor, đã chuẩn hóa) |
| \(s_{\mathrm{pop}}, s_{\mathrm{prof}}\) | Phổ biến, profile danh mục |
| \(p_{\mathrm{rep}}\) | Phạt lặp |
| \(S_{\mathrm{final}}\) | Điểm lai ghép sau cắt ngưỡng |

---

*Tài liệu phục vụ trình bày trong luận văn; có thể bổ sung hình vẽ pipeline, bảng tham số \(\alpha,\beta,\gamma,\delta,\varepsilon\) theo đúng bảng triển khai, và phụ lục chứng minh tính chất của cosine hoặc MMR nếu giảng viên hướng dẫn yêu cầu.*
