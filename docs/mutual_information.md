# Điểm Thông tin Hỗ tương (`mutual_information`)

Module `mutual_information` sử dụng lý thuyết thông tin (Information Theory) để đo lường mức độ phụ thuộc phi tuyến (Non-linear Dependency) giữa từng đặc trưng và nhãn mục tiêu, khắc phục nhược điểm của các thước đo tương quan tuyến tính cổ điển.

---

## 1. Mục đích & Ứng dụng

- **Phát hiện Quan hệ Phi tuyến (Non-linear Relationships)**: Phát hiện các mẫu hình quan hệ phức tạp (như hình chữ U, hàm sin, hoặc ngưỡng kích hoạt) mà hệ số tương quan Pearson/Spearman bỏ sót.
- **Model-Agnostic Feature Selection**: Đánh giá sức mạnh dự báo độc lập với bất kỳ thuật toán học máy cụ thể nào.
- **Hỗ trợ Cả Phân loại & Hồi quy**: Tự động áp dụng `mutual_info_classif` cho biến nhãn phân loại và `mutual_info_regression` cho biến nhãn liên tục.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Định nghĩa Thông tin Hỗ tương**:
   $$I(X; Y) = \iint p(x, y) \log \frac{p(x, y)}{p(x)p(y)} \, dx \, dy$$
   - $I(X; Y) = 0$ khi và chỉ khi $X$ và $Y$ độc lập thống kê hoàn toàn.
   - $I(X; Y) > 0$: $X$ chứa thông tin giúp giảm độ bất định (entropy) của $Y$.
2. **Thuật toán Ước lượng**:
   - Sử dụng phương pháp ước lượng phi tham số k-Nearest Neighbors (k-NN) theo thuật toán Kraskov-Stögbauer-Grassberger (KSG).
3. **Tiền xử lý & Điền khuyết thiếu**:
   - Tự động chuẩn hóa ma trận số và điền giá trị median cho các ô khuyết thiếu trước khi ước lượng.

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy đánh giá Mutual Information
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module mutual_information

# Chỉ định cột target
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module mutual_information --target allow_entry

# Giới hạn số dòng và thư mục kết quả
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module mutual_information \
  --limit 25000 \
  --output-dir reports/mi_output
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `feature_scores.csv` | CSV | Toàn bộ danh sách features xếp hạng theo điểm Mutual Information giảm dần. |
| `top_features.csv` | CSV | Top 50 features có giá trị thông tin hỗ tương cao nhất. |
| `summary.json` | JSON | Metadata tổng kết số dòng và danh sách top feature. |
| `report.md` / `report.html` | Báo cáo | Báo cáo dạng Markdown và HTML tương tác. |

### Các Cột trong `feature_scores.csv`:
- `feature`: Tên đặc trưng.
- `label`: Tên cột nhãn.
- `score_name`: Tên phương pháp ước lượng (`mutual_info_classif` hoặc `mutual_info_regression`).
- `score`: Điểm Mutual Information (đơn vị nats, giá trị $\ge 0$).
- `samples`: Số lượng mẫu hợp lệ được dùng để ước lượng.

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Ý nghĩa Điểm MI**:
  - `score == 0`: Feature không chứa bất kỳ thông tin nào về nhãn.
  - `score > 0.05`: Feature có mối liên hệ đáng kể với biến mục tiêu.
  - `score > 0.15`: Feature có mức độ phụ thuộc rất mạnh mẽ.
- **So sánh với Pearson Correlation**: Nếu một feature có tương quan Pearson gần $0$ nhưng điểm Mutual Information cao, đây là tín hiệu cho thấy sự tồn tại của quan hệ phi tuyến mạnh mẽ mà các mô hình cây quyết định (Decision Trees / GBDT) có thể khai thác rất tốt.\n