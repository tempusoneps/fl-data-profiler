# 2D Joint Probability Heatmap (`probability_2d` / `probability2d`)

Module `probability_2d` (có thể gọi bằng `probability2d` hoặc `probability_2d`) phân tích phân phối xác suất có điều kiện đồng thời trên lưới phân vị $10 \times 10$ ($F_1 \times F_2$) cho từng cặp đặc trưng đối với các nhãn phân loại nhị phân và đa lớp.

Module này chuyên dùng để **khai phá các tương tác phi tuyến (Synergy Interactions)** và **trích xuất tự động các vùng quyết định tối ưu (Sweet Spot Decision Rules)** có xác suất thắng vượt trội so với từng đặc trưng đơn lẻ.

---

## 1. Nguyên lý Toán học & Chỉ số Đo lường

### 1.1. Lưới Phân vị Đồng thời $10 \times 10$ (Joint Quantile Grid)
Với mỗi cặp đặc trưng $(F_1, F_2)$, dữ liệu được chia thành 10 bins theo hạng cho mỗi trục:
$$\text{Bin}_1 = \text{qcut}(\text{rank}(F_1), q=10, \text{labels}=False) + 1 \in \{1, \dots, 10\}$$
$$\text{Bin}_2 = \text{qcut}(\text{rank}(F_2), q=10, \text{labels}=False) + 1 \in \{1, \dots, 10\}$$
Mỗi ô (cell) $(i, j)$ đại diện cho khoảng $1\%$ số lượng mẫu dữ liệu.

### 1.2. Xác suất Điều kiện Đồng thời & Lift
Với mỗi nhãn lớp $c \in \mathcal{C}$ và ô $(i, j)$:
$$P(Y = c \mid F_1 \in \text{Bin}_i, F_2 \in \text{Bin}_j) = \frac{\sum_{k \in \text{Cell}_{i, j}} \mathbb{I}(Y_k = c)}{N_{i, j}}$$

* **Biên độ xác suất 2D ($\Delta P_{2D}$)**:
  $$\Delta P_{2D} = \max_{i, j} P(Y=c \mid \text{Cell}_{i, j}) - \min_{i, j} P(Y=c \mid \text{Cell}_{i, j})$$
* **Hệ số Hợp lực / Độ lợi Tương tác (Synergy Gain $\Delta_{\text{synergy}}$)**:
  $$\Delta_{\text{synergy}} = \Delta P_{2D} - \max(\Delta P_{F_1}, \Delta P_{F_2})$$
  $\Delta_{\text{synergy}} > 0$ chứng minh việc kết hợp 2 đặc trưng tạo ra một lợi thế xác suất lớn hơn đáng kể so với việc chỉ dùng đơn lẻ một trong hai đặc trưng.
* **Information Value Đồng thời (2D IV)**:
  $$\text{IV}_{2D} = \sum_{i=1}^{10} \sum_{j=1}^{10} \left(P(\text{Cell}_{i, j} \mid Y=c) - P(\text{Cell}_{i, j} \mid Y \neq c)\right) \times \ln\left(\frac{P(\text{Cell}_{i, j} \mid Y=c) + \epsilon}{P(\text{Cell}_{i, j} \mid Y \neq c) + \epsilon}\right)$$
* **Tỷ lệ Đòn bẩy Xác suất (Lift Ratio)**:
  $$\text{Lift}_{i, j} = \frac{P(Y=c \mid \text{Cell}_{i, j})}{P(Y=c)}$$

### 1.3. Trích xuất Luật Điểm Ngọt (Sweet Spot Decision Rules)
Tự động tìm ô $(i^*, j^*)$ đạt xác suất cực đại (kèm điều kiện mẫu tối thiểu $N \ge 20$) và chuyển hóa thành luật IF-THEN định lượng:
$$\text{IF } a \le F_1 \le b \text{ AND } c \le F_2 \le d \implies P(Y = \text{Target Class}) = p^* \text{ (Lift: } \lambda\text{x)}$$

---

## 2. Hướng dẫn Sử dụng CLI

```bash
# Chạy với tên lệnh ngắn gọn
uv run fldataprofiler fit datasets/selected_feature.parquet datasets/label.csv \
  --module probability2d \
  --target allow_entry

# Chạy với tên đầy đủ
uv run fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module probability_2d \
  --target allow_entry
```

---

## 3. Danh sách Kết quả Đầu ra (Artifacts)

Tất cả báo cáo và biểu đồ được lưu tại `reports/probability_2d/`:

1. `pair_probability_scores.csv`: Bảng xếp hạng tất cả các cặp đặc trưng theo $IV_{2D}$, $\Delta P_{2D}$, Synergy Gain, Sweet Spot Probability, Lift và Luật IF-THEN.
2. `cell_conditional_probabilities.csv`: Bảng chi tiết 100 ô $(10 \times 10)$ cho từng cặp đặc trưng gồm tọa độ bin, cận giá trị thực tế, số lượng mẫu, xác suất $P(Y=c \mid \text{Cell})$, WoE, Lift và Entropy.
3. `probability_2d_heatmaps.png`: Ma trận nhiệt 2D (Heatmaps) trực quan hóa các cặp tương tác mạnh nhất, làm nổi bật ô "Sweet Spot" bằng khung viền đỏ.
4. `summary.json`: Metadata tổng hợp, top cặp đặc trưng và danh sách luật Sweet Spot.
5. `report.md`: Báo cáo tổng hợp dạng Markdown chuẩn GitHub.
6. `report.html`: Báo cáo giao diện web tương tác kèm biểu đồ nhiệt 2D và bảng luật giao dịch có thể copy trực tiếp.
