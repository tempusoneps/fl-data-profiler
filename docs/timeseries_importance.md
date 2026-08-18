# Tầm quan trọng Tổng hợp Chuỗi Thời gian (`timeseries_importance`)

Module `timeseries_importance` kết hợp đa chiều các phương pháp đánh giá đặc trưng chuỗi thời gian (Information Coefficient, Permutation Drop, Mutual Information và Correlation Support) thành một **Điểm Số Tổng Hợp Chuẩn Hóa (Unified Combined Score)**.

---

## 1. Mục đích & Ứng dụng

- **Đánh giá Đa Tiêu chí (Multi-criteria Scoring)**: Tránh việc phụ thuộc vào một phương pháp đơn lẻ (như chỉ nhìn vào IC hoặc chỉ nhìn vào Feature Importance của Tree Model).
- **Tăng Độ Vững chắc (Robustness)**: Một feature chỉ được xếp hạng cao nếu nó đồng thời thể hiện tốt ở cả khía cạnh tương quan chuỗi thời gian, độ bất định thông tin và đóng góp vào mô hình học máy.
- **Bộ lọc Feature Cuối cùng (Final Feature Selection Gate)**: Thích hợp làm bước sàng lọc chốt danh sách đặc trưng đưa vào hệ thống giao dịch tự động.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Thu thập Điểm Thành phần (Component Scoring)**:
   - **IC Score**: Tính toán Pearson IC và Spearman Rank IC qua các folds walk-forward.
   - **Permutation Drop**: Đo lường mức sụt giảm hiệu năng mô hình Random Forest trên out-of-fold samples.
   - **Mutual Information**: Điểm tương quan phi tuyến.
2. **Chuẩn hóa & Kết hợp (Normalization & Aggregation)**:
   - Các điểm thành phần được đưa về cùng thang đo chuẩn hóa.
   - Điểm tổng hợp được tính toán dựa trên trọng số trung bình:
     $$\text{Combined Score} = \frac{1}{M} \sum_{k=1}^M \text{Normalized Component Score}_k$$
3. **Xếp hạng Ưu tiên**:
   - Xếp hạng giảm dần theo `combined_score` và `valid_folds`.

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy tính toán tổng hợp Time-Series Importance
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module timeseries_importance

# Chỉ định nhãn mục tiêu và giới hạn dữ liệu
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module timeseries_importance \
  --target allow_entry \
  --limit 25000

# Chỉ định thư mục xuất kết quả
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module timeseries_importance \
  --output-dir reports/ts_importance_run
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `feature_scores.csv` | CSV | Bảng xếp hạng toàn bộ các feature theo điểm số tổng hợp `combined_score`. |
| `top_features.csv` | CSV | Danh sách Top 50 features xuất sắc nhất toàn diện. |
| `component_scores.csv` | CSV | Chi tiết điểm số từng thành phần (IC, Permutation, MI) của từng feature. |
| `summary.json` | JSON | Metadata tổng kết kết quả đánh giá. |
| `report.md` / `report.html` | Báo cáo | Báo cáo Markdown và giao diện HTML hiển thị bảng tổng hợp. |

### Các Cột trong `feature_scores.csv`:
- `feature`: Tên đặc trưng.
- `label`: Tên cột nhãn mục tiêu.
- `combined_score`: Điểm số tổng hợp đa chiều ($0.0 \to 1.0$).
- `mean_abs_score`: Giá trị trung bình tuyệt đối của các điểm thành phần.
- `component_count`: Số lượng phương pháp thành phần đóng góp vào điểm số.
- `valid_folds`: Số lượng folds thời gian hợp lệ.
- `samples`: Tổng số mẫu dữ liệu đánh giá.

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Chọn Lọc Đặc trưng Cho Production**: Khuyến nghị chọn các đặc trưng nằm trong Top 20-30 của bảng `top_features.csv`.
- **Phân tích Sâu Thành phần**: Xem xét `component_scores.csv` để hiểu rõ thế mạnh của từng feature:
  - Feature mạnh về IC: Phù hợp làm tín hiệu trực tiếp (Linear Alpha).
  - Feature mạnh về Permutation/MI nhưng IC thấp: Phù hợp làm biến điều kiện lọc (Regime/Volatility Filter).\n