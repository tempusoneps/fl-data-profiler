# Phân tích Khám phá Dữ liệu (`eda`)

Module `eda` (Exploratory Data Analysis) cung cấp cái nhìn toàn diện về cấu trúc dữ liệu, tỷ lệ khuyết thiếu (missing values), phân phối thống kê, kiểu dữ liệu và ma trận tương quan giữa các đặc trưng (features) và nhãn (labels).

---

## 1. Mục đích & Ứng dụng

- **Data Health Check**: Phát hiện kịp thời các cột dữ liệu rỗng, cột có phương sai bằng 0 (hằng số), hoặc tỷ lệ giá trị bất thường.
- **Phân loại Đặc trưng**: Tự động phân loại biến thành nhóm số học (numeric) và biến phân loại (categorical/string).
- **Phát hiện Đa cộng tuyến (Multicollinearity)**: Nhận diện các cặp đặc trưng có tương quan tuyến tính quá cao thông qua ma trận tương quan nhiệt (correlation heatmap).

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Thống kê Khuyết thiếu (Missingness)**:
   - Tính toán số lượng `null/NaN/None` và tỷ lệ phần trăm khuyết thiếu cho từng cột.
2. **Thống kê Biến Số (Numeric Profiling)**:
   - Tính toán Count, Mean, Std, Min, 25%, Median (50%), 75%, Max, Skewness, Kurtosis.
3. **Thống kê Biến Phân loại (Categorical Profiling)**:
   - Đếm số lượng giá trị duy nhất (unique count), giá trị xuất hiện nhiều nhất (mode) và tần suất của mode.
4. **Ma trận Tương quan Tương hỗ (Correlation Heatmaps)**:
   - Tính ma trận tương quan Pearson giữa các numeric features và giữa các numeric labels.

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy phân tích EDA cơ bản
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module eda

# Phân tích với dữ liệu giới hạn 20,000 dòng đầu tiên
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module eda --limit 20000

# Chỉ định thư mục lưu báo cáo EDA
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module eda --output-dir reports/eda_summary
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `dataset_overview.csv` | CSV | Bảng tóm tắt số dòng, số cột, kiểu dữ liệu, dung lượng bộ nhớ của feature và label. |
| `columns_profile.csv` | CSV | Danh sách tất cả các cột, kiểu dữ liệu suy diễn, tỷ lệ null và số lượng giá trị duy nhất. |
| `missingness.csv` | CSV | Bảng xếp hạng các cột có tỷ lệ missing value từ cao xuống thấp. |
| `numeric_summary.csv` | CSV | Thống kê mô tả chi tiết cho toàn bộ biến số (Mean, Std, Min, Max, Skew, Kurtosis). |
| `categorical_summary.csv` | CSV | Thống kê mô tả cho toàn bộ biến định danh/phân loại. |
| `feature_correlation_heatmap.png` | Biểu đồ | Heatmap biểu diễn tương quan giữa các đặc trưng đầu vào. |
| `label_correlation_heatmap.png` | Biểu đồ | Heatmap biểu diễn tương quan giữa các nhãn mục tiêu. |
| `report.md` / `report.html` | Báo cáo | Báo cáo Markdown và giao diện HTML tương tác. |
| `summary.json` | JSON | Metadata tổng kết số lượng dòng và trạng thái dữ liệu. |

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Xử lý Missing Data**: Nếu một cột có tỷ lệ missing $> 30\%$, cân nhắc loại bỏ hoặc áp dụng kỹ thuật điền khuyết thiếu phù hợp (Forward-fill cho time-series hoặc Median imputation).
- **Biến có Skewness/Kurtosis cao**: Các đặc trưng có độ lệch (skewness) lớn $(>3)$ hoặc kurtosis cao nên được chuẩn hóa qua hàm $\log(1+x)$ hoặc Box-Cox trước khi đưa vào các mô hình tuyến tính.
- **Tương quan quá cao giữa các Feature ($>0.95$)**: Dấu hiệu redundancy (dư thừa thông tin), nên lọc bớt bằng module `mrmr` hoặc `feature_interactions`.\n