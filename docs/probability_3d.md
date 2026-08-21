# 3D Joint Probability & Sweet Spots (`probability_3d` / `probability3d`)

Module `probability_3d` (có thể gọi bằng `probability3d` hoặc `probability_3d`) phân tích phân phối xác suất có điều kiện đồng thời trên không gian lưới phân vị 3 chiều $5 \times 5 \times 5$ ($F_1 \times F_2 \times F_3 = 125$ hyper-voxels) cho từng bộ 3 đặc trưng đối với các nhãn phân loại nhị phân và đa lớp.

Module này chuyên dùng để **khai phá các tương tác phi tuyến 3 chiều (3-Way High-Confluence Synergy)** và **trích xuất tự động các siêu vùng quyết định tối ưu (3D Hyper-Voxel Sweet Spot Rules)** có xác suất thắng vượt trội so với từng đặc trưng đơn lẻ hoặc cặp 2D.

---

## 1. Nguyên lý Toán học & Chỉ số Đo lường

### 1.1. Lưới Phân vị Đồng thời 3 Chiều $5 \times 5 \times 5$ (3D Joint Quantile Grid)
Với mỗi bộ 3 đặc trưng $(F_1, F_2, F_3)$, dữ liệu được chia thành 5 bins theo hạng cho mỗi trục:
$$\text{Bin}_1 = \text{qcut}(\text{rank}(F_1), q=5, \text{labels}=False) + 1 \in \{1, \dots, 5\}$$
$$\text{Bin}_2 = \text{qcut}(\text{rank}(F_2), q=5, \text{labels}=False) + 1 \in \{1, \dots, 5\}$$
$$\text{Bin}_3 = \text{qcut}(\text{rank}(F_3), q=5, \text{labels}=False) + 1 \in \{1, \dots, 5\}$$
Mỗi khối (voxel) $(i, j, k)$ đại diện cho khoảng $1/125 = 0.8\%$ số lượng mẫu dữ liệu (trung bình $\approx 400$ mẫu/khối trên tập $50,000$ dòng, đảm bảo tính ổn định thống kê cao).

### 1.2. Xác suất Điều kiện Đồng thời 3D & Hệ số Hợp lực (3D Synergy Gain)
Với mỗi nhãn lớp $c \in \mathcal{C}$ và khối $(i, j, k)$:
$$P(Y = c \mid F_1 \in \text{Bin}_i, F_2 \in \text{Bin}_j, F_3 \in \text{Bin}_k) = \frac{\sum_{m \in \text{Voxel}_{i, j, k}} \mathbb{I}(Y_m = c)}{N_{i, j, k}}$$

* **Biên độ xác suất 3D ($\Delta P_{3D}$)**:
  $$\Delta P_{3D} = \max_{i, j, k} P(Y=c \mid \text{Voxel}_{i, j, k}) - \min_{i, j, k} P(Y=c \mid \text{Voxel}_{i, j, k})$$
* **Hệ số Hợp lực 3D (3D Synergy Gain $\Delta_{\text{synergy}}$)**:
  $$\Delta_{\text{synergy}} = \Delta P_{3D} - \max(\Delta P_{F_1}, \Delta P_{F_2}, \Delta P_{F_3})$$
* **Information Value Đồng thời 3D (3D IV)**:
  $$\text{IV}_{3D} = \sum_{i=1}^5 \sum_{j=1}^5 \sum_{k=1}^5 \left(P(\text{Voxel}_{i, j, k} \mid Y=c) - P(\text{Voxel}_{i, j, k} \mid Y \neq c)\right) \times \ln\left(\frac{P(\text{Voxel}_{i, j, k} \mid Y=c) + \epsilon}{P(\text{Voxel}_{i, j, k} \mid Y \neq c) + \epsilon}\right)$$
* **Tỷ lệ Đòn bẩy Xác suất (Lift Ratio)**:
  $$\text{Lift}_{i, j, k} = \frac{P(Y=c \mid \text{Voxel}_{i, j, k})}{P(Y=c)}$$

### 1.3. Trích xuất Luật Điểm Ngọt 3D (3D Sweet Spot Decision Rules)
Tự động tìm khối $(i^*, j^*, k^*)$ đạt xác suất cực đại (kèm điều kiện mẫu hỗ trợ tối thiểu $N \ge 20$) và chuyển hóa thành luật IF-THEN 3 điều kiện:
$$\text{IF } a \le F_1 \le b \text{ AND } c \le F_2 \le d \text{ AND } e \le F_3 \le f \implies P(Y = \text{Target Class}) = p^* \text{ (Lift: } \lambda\text{x, Support: } N\text{ bars)}$$

---

## 2. Hướng dẫn Sử dụng CLI

```bash
# Chạy với tên lệnh ngắn gọn
uv run fldataprofiler fit datasets/selected_feature.parquet datasets/label.csv \
  --module probability3d \
  --target allow_entry

# Chạy với tên đầy đủ
uv run fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module probability_3d \
  --target allow_entry
```

---

## 3. Danh sách Kết quả Đầu ra (Artifacts)

Tất cả báo cáo và biểu đồ được lưu tại `reports/probability_3d/`:

1. `triplet_probability_scores.csv`: Bảng xếp hạng tất cả các bộ 3 đặc trưng theo $IV_{3D}$, $\Delta P_{3D}$, Synergy Gain, Sweet Spot Probability, Lift và Luật IF-THEN 3D.
2. `voxel_conditional_probabilities.csv`: Bảng chi tiết 125 khối $(5 \times 5 \times 5)$ cho từng bộ 3 đặc trưng gồm tọa độ bin 3D, cận giá trị thực tế, số lượng mẫu, xác suất $P(Y=c \mid \text{Voxel})$, WoE, Lift và Shannon Entropy.
3. `probability_3d_heatmaps.png`: Biểu đồ cắt lát ma trận nhiệt 2D qua 5 phân vị trục $Z$, làm nổi bật siêu khối "Sweet Spot" bằng khung viền đỏ.
4. `summary.json`: Metadata tổng hợp, top bộ 3 đặc trưng và danh sách luật Sweet Spot 3D.
5. `report.md`: Báo cáo tóm tắt định dạng Markdown.
6. `report.html`: Bảng điều khiển tương tác trực quan với các thẻ KPI, bảng tra cứu và bộ lọc luật Sweet Spot.
