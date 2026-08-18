# Thống kê Mô tả & Tương quan Đặc trưng (`statistics`)

Module `statistics` thực hiện phân tích thống kê định lượng chi tiết, đo lường mối quan hệ tuyến tính (Pearson correlation) và mối quan hệ phân vị (quantile profile) giữa từng đặc trưng (feature) và từng nhãn mục tiêu (label).

---

## 1. Mục đích & Ứng dụng

- **Khám phá Tương quan Cơ bản**: Nhanh chóng xác định đặc trưng nào có tương quan tuyến tính mạnh nhất với nhãn mục tiêu.
- **Phân tích Phân vị Label (Quantile Means)**: Quan sát xem giá trị trung bình của feature thay đổi như thế nào khi chia nhãn thành các phân vị từ thấp đến cao (10-quantiles).
- **Sàng lọc Nhanh (Fast Filtering)**: Loại bỏ các feature không có bất kỳ tương quan nào với mục tiêu trước khi bước vào các pipeline học máy nặng.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Hệ số Tương quan Tuyến tính Pearson**:
   $$r_{X, Y} = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum (X_i - \bar{X})^2 \sum (Y_i - \bar{Y})^2}}$$
   - Xếp hạng theo giá trị tuyệt đối $|r_{X, Y}|$ từ cao đến thấp.
2. **Phân tích Phân vị (Quantile Means Profile)**:
   - Chia nhãn $Y$ thành các bins phân vị (deciles).
   - Tính giá trị trung bình của feature $X$ trong từng bin của $Y$ để kiểm tra tính đơn điệu.
3. **Thống kê Hình dạng Dữ liệu (Dataset Profile)**:
   - Đếm số mẫu khả dụng, số giá trị hợp lệ sau khi loại bỏ `NaN`.

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy phân tích thống kê mặc định
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module statistics

# Chạy cho nhãn cụ thể
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module statistics --target allow_entry

# Giới hạn số dòng để phân tích nhanh
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module statistics --limit 10000
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `feature_label_correlations.csv` | CSV | Bảng tương quan Pearson và trị tuyệt đối tương quan giữa từng cặp (Feature, Label). |
| `feature_profile.csv` | CSV | Thống kê mô tả (Mean, Std, Min, Max, Nulls) cho toàn bộ tập feature. |
| `label_profile.csv` | CSV | Thống kê mô tả cho toàn bộ tập label. |
| `feature_label_correlation_heatmap.png` | Biểu đồ | Heatmap ma trận tương quan giữa top features và labels. |
| `statistics_summary.json` | JSON | File JSON tổng hợp metadata và top 10 tương quan mạnh nhất. |
| `report.md` / `report.html` | Báo cáo | Báo cáo chi tiết định dạng Markdown và HTML. |

### Các Cột trong `feature_label_correlations.csv`:
- `feature`: Tên đặc trưng.
- `label`: Tên cột nhãn.
- `pearson_correlation`: Hệ số tương quan Pearson ($[-1, 1]$).
- `abs_correlation`: Giá trị tuyệt đối của tương quan ($[0, 1]$).
- `samples`: Số lượng mẫu hợp lệ được dùng để tính toán.

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Tương quan tuyến tính cao ($|r| > 0.15$)**: Trong dữ liệu tài chính (financial time-series), hệ số tương quan $> 0.1$ đã là một tín hiệu rất đáng chú ý.
- **Lưu ý về Tương quan Phi tuyến**: Tương quan Pearson $= 0$ không đồng nghĩa feature và label độc lập; có thể tồn tại quan hệ phi tuyến (ví dụ hình parabol $Y = X^2$). Hãy kết hợp với module `mutual_information` hoặc `scipy` để kiểm định toàn diện.\n