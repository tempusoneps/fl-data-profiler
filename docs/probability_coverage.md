# Probability Coverage & Quantile Matrix Ranking (`probability_coverage` / `coverage`)

Module `probability_coverage` (có thể gọi bằng `coverage` hoặc `probabilitycoverage`) phân tích mối quan hệ giữa **1 Feature $\times$ 1 Label**:
1. Chia đặc trưng thành các nhóm phân vị theo thứ hạng qua `qcut` ($N$ quantiles, mặc định 20 bins).
2. Tạo bảng chéo `crosstab(bins, label)` và chuẩn hóa theo hàng để chuyển đổi từ `count` $\to$ `percent` ($P(Y=c \mid \text{Bin}_k) \times 100\%$).
3. Lấy ngưỡng xác suất tối thiểu $\text{min\_x}$ (`min_probability`, ví dụ $55\%$) từ cấu hình và đếm số lượng ô/bin thỏa mãn $P \ge \text{min\_x}$.
4. **Sắp xếp tất cả các đặc trưng theo chiều đặc trưng nào có nhiều ô/bin vượt ngưỡng $\text{min\_x}$ nhất** (Qualified Bins count).

Module này chuyên dùng để tìm kiếm các đặc trưng có **vùng bao phủ xác suất cao rộng lớn** (High-Probability Coverage Region) trên phân vị, tránh tình trạng phụ thuộc vào một điểm dị biệt (outlier) đơn lẻ.

---

## 1. Nguyên lý Toán học & Quy trình Tính toán

### 1.1. Phân vị Đặc trưng (Quantile Discretization)
Với mỗi đặc trưng số $F$, các giá trị được gán nhãn phân vị đều tần số từ $1$ đến $K$ (mặc định $K = 20$ bins, mỗi bin chiếm khoảng $5\%$ mẫu):
$$\text{Bin} = \text{qcut}(\text{rank}(F), q=K, \text{labels}=False) + 1 \in \{1, \dots, K\}$$

### 1.2. Bảng Chéo Crosstab & Chuyển đổi Count $\to$ Percent
Với mỗi phân vị $k \in \{1, \dots, K\}$ và nhãn lớp $c \in \mathcal{C}$:
* **Số đếm mẫu trong bin**: $N_k = \sum_{c} \text{Count}(k, c)$
* **Số sự kiện xảy ra lớp $c$**: $E_{k, c} = \text{Count}(k, c)$
* **Xác suất điều kiện (Percent %)**:
  $$P(Y = c \mid \text{Bin}_k) = \frac{E_{k, c}}{N_k} \times 100\%$$
* **Tỷ lệ nền (Base Rate)**: $P(Y = c) = \frac{\sum_k E_{k, c}}{\sum_k N_k} \times 100\%$
* **Hệ số Đòn bẩy (Lift)**: $\text{Lift}_{k, c} = \frac{P(Y = c \mid \text{Bin}_k)}{P(Y = c)}$

### 1.3. Điều kiện Ô/Bin Đạt Chuẩn (Qualified Bin Criteria)
Một phân vị $k$ của đặc trưng $F$ đối với nhãn lớp $c$ được xem là đạt chuẩn nếu:
1. **Ngưỡng Xác suất tối thiểu**: $P(Y = c \mid \text{Bin}_k) \ge \text{min\_probability}$ (mặc định $\ge 55.0\%$).
2. **Hỗ trợ mẫu tối thiểu**: $N_k \ge \text{min\_support}$ (mặc định $\ge 20$ mẫu).
3. **Ngưỡng Lift tối thiểu**: $\text{Lift}_{k, c} \ge \text{min\_lift}$ (mặc định $\ge 1.0\text{x}$).

$$\mathbb{I}_{\text{qual}}(k, c) = \mathbb{I}\left(P(Y=c \mid \text{Bin}_k) \ge \text{min\_probability} \land N_k \ge \text{min\_support} \land \text{Lift}_{k, c} \ge \text{min\_lift}\right)$$

### 1.4. Các Chỉ số Đánh giá Độ phủ (Coverage Metrics)
* **Số lượng Bin Đạt Chuẩn (`qualified_bins`)**:
  $$\text{Qualified Bins} = \sum_{k=1}^{K} \mathbb{I}_{\text{qual}}(k, c) \in [0, K]$$
* **Tỷ lệ Phủ Phân vị (`bin_coverage_pct`)**:
  $$\text{Bin Coverage (\%)} = \frac{\text{Qualified Bins}}{K} \times 100\%$$
* **Tỷ lệ Phủ Mẫu Dữ liệu (`sample_coverage_pct`)**:
  $$\text{Sample Coverage (\%)} = \frac{\sum_{k \in \text{Qual}} N_k}{N_{\text{total}}} \times 100\%$$
* **Xác suất Bình quân trong Vùng Phủ (`weighted_qualified_prob`)**:
  $$\overline{P}_{\text{qual}} = \frac{\sum_{k \in \text{Qual}} E_{k, c}}{\sum_{k \in \text{Qual}} N_k} \times 100\%$$
* **Điểm Số Phủ Tổng hợp (`composite_coverage_score`)**:
  $$\text{Score} = \text{Qualified Bins} \times \text{Mean Lift} \times \sqrt{\frac{\text{Sample Coverage (\%)}}{100}}$$

---

## 2. Thứ tự Sắp xếp (Ranking Order)

Tất cả các ma trận phân vị đặc trưng $(F \times \text{Target})$ được xếp hạng theo thứ tự ưu tiên:
1. **`qualified_bins` (Giảm dần - Primary)**: Đặc trưng nào có nhiều bin đạt xác suất $P \ge \text{min\_probability}$ nhất sẽ được xếp lên đầu.
2. **`sample_coverage_pct` (Giảm dần - Secondary)**: Nếu cùng số bin, ưu tiên đặc trưng có lượng mẫu rơi vào vùng đạt chuẩn nhiều hơn.
3. **`weighted_qualified_prob` (Giảm dần - Tertiary)**: Ưu tiên đặc trưng có xác suất bình quân trong vùng cao hơn.

---

## 3. Cấu hình Tham số (`config.default.json`)

```json
"probability_coverage": {
  "n_quantiles": 20,
  "min_probability": 0.55,
  "min_support": 20,
  "min_lift": 1.0,
  "top_features": 25,
  "min_feature_unique_values": 20,
  "max_label_classes": 50
}
```

* `n_quantiles`: Số phân vị chia nhỏ đặc trưng (mặc định 20 bins).
* `min_probability`: Ngưỡng xác suất tối thiểu $\text{min\_x}$ để bin được tính là đạt chuẩn (ví dụ `0.55` cho $55\%$ hoặc `0.60` cho $60\%$).
* `min_support`: Số lượng mẫu tối thiểu trong mỗi bin.
* `min_lift`: Hệ số đòn bẩy xác suất tối thiểu so với base rate.
* `top_features`: Số lượng đặc trưng hàng đầu được đưa vào biểu đồ trực quan và báo cáo chi tiết.
* `min_feature_unique_values`: Ngưỡng số lượng giá trị duy nhất tối thiểu của feature (mặc định `20`). Tự động **bỏ qua các feature dạng Categorical / Boolean / Discrete ít giá trị** (như cờ boolean 0/1, mã danh mục...) vì chúng không thể chia 20 quantiles một cách hợp lý.
* `max_label_classes`: Giới hạn số lượng lớp phân loại tối đa của nhãn (mặc định `50`, có thể cấu hình ở cấp module hoặc `global`), giúp tự động bỏ qua các biến số thực liên tục khi không chỉ định `--target`.

---

## 4. Hướng dẫn Sử dụng CLI

```bash
# Chạy với tên lệnh ngắn gọn
uv run fldataprofiler fit datasets/selected_feature.parquet datasets/label.csv \
  --module coverage \
  --target allow_entry

# Chạy với tên đầy đủ trên nhãn cụ thể
uv run fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module probability_coverage \
  --target direction_filter
```

---

## 5. Danh sách Kết quả Đầu ra (Artifacts)

Tất cả báo cáo và biểu đồ được lưu tại `reports/probability_coverage/`:

1. `matrix_coverage_rankings.csv`: Bảng xếp hạng tất cả các ma trận $(F \times Y)$ theo tổng số ô đạt chuẩn $P \ge \text{min\_x}$ giảm dần.
2. `crosstab_matrices_<target>.csv`: Bảng ma trận chéo phân vị $(20 \times C)$ nằm ngang cho từng Target cụ thể (các cột: `feature, Bin, Range, Samples, Class_1 (%), Class_2 (%), ..., Qualified (>min_x)`), 100% sạch và không bị cột NaN.
3. `feature_coverage_scores.csv`: Bảng điểm đánh giá chi tiết theo từng cặp (Feature $\times$ Target $\times$ Class).
4. `quantile_crosstab_probabilities.csv`: Bảng chi tiết toàn bộ các ô phân vị ở dạng dọc (long-format) kèm xác suất %, count, base rate, lift và cờ `is_qualified`.
5. `probability_coverage_distribution.png`: Biểu đồ phân phối xác suất phân vị của top đặc trưng, tô viền nổi bật cho các ô đạt chuẩn và đường gióng ngưỡng nét đứt đỏ.
6. `summary.json`: Metadata tổng hợp và top ma trận có độ phủ lớn nhất.
7. `report.md`: Báo cáo tổng hợp dạng Markdown chuẩn GitHub với đầy đủ các bảng ma trận crosstab %.
8. `report.html`: Báo cáo giao diện web tương tác kèm bảng tra cứu ma trận tô màu vàng nổi bật và biểu đồ trực quan.
