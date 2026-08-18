# Chỉ số Thông tin Chuỗi Thời gian (`information_coefficient`)

Module `information_coefficient` thực hiện tính toán hệ số tương quan thông tin (Information Coefficient - IC) theo phương pháp **Walk-Forward Validation** (cuộn cửa sổ thời gian) để đo lường độ tin cậy và sự ổn định của đặc trưng đối với nhãn mục tiêu theo thời gian.

---

## 1. Mục đích & Ứng dụng

- **Kiểm tra Out-of-Sample (OOS) Nghiêm ngặt**: Đánh giá IC trên các fold tương lai chưa từng xuất hiện trong quá khứ nhằm chống look-ahead bias và rò rỉ dữ liệu (data leakage).
- **Đo lường Sự Ổn định (Consistency)**: Kiểm tra xem feature có duy trì khả năng sinh lời đều đặn qua các chu kỳ thị trường khác nhau hay chỉ bùng nổ trong một khoảng thời gian ngắn.
- **Phù hợp Chuỗi Thời gian Tài chính**: Thay thế cách tính tương quan toàn bộ mẫu (in-sample) truyền thống vốn dễ gây ngộ nhận về hiệu năng.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Phân tách Walk-Forward (Walk-Forward Splits)**:
   - Dữ liệu được chia thành các folds liên tiếp theo thời gian:
     - `min_train_size`: Kích thước tối thiểu ban đầu (mặc định 100 mẫu).
     - `test_size`: Kích thước tập kiểm định mỗi fold (mặc định 50 mẫu).
     - `step_size`: Bước nhảy dịch chuyển cửa sổ (mặc định 50 mẫu).
     - `max_folds`: Tối đa 20 folds liên tiếp.
2. **Tính toán IC cho Từng Fold**:
   - **Pearson IC**: Đo lường mối liên hệ tuyến tính giữa feature và label trên tập test fold.
   - **Rank IC (Spearman)**: Đo lường mối liên hệ thứ bậc (đơn điệu), ít bị ảnh hưởng bởi giá trị ngoại lai (outliers).
3. **Tổng hợp Thống kê Đa Fold (Aggregation)**:
   - `mean_score`: Giá trị IC trung bình qua tất cả các folds.
   - `mean_abs_score`: Độ lớn trung bình tuyệt đối $| \text{IC} |$.
   - `score_std`: Độ biến động (độ lệch chuẩn) của IC qua các folds.
   - `positive_fold_ratio`: Tỷ lệ phần trăm số folds có $\text{IC} > 0$.

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy tính toán Information Coefficient
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module information_coefficient

# Chỉ định nhãn mục tiêu cụ thể
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module information_coefficient --target allow_entry

# Lưu kết quả vào thư mục riêng
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module information_coefficient \
  --output-dir reports/ic_scoring_run
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `feature_scores.csv` | CSV | Bảng tổng hợp chỉ số IC trung bình, độ lệch chuẩn và tỷ lệ fold dương cho toàn bộ features. |
| `top_features.csv` | CSV | Danh sách Top 50 features có `mean_abs_score` cao nhất. |
| `fold_scores.csv` | CSV | Điểm số IC chi tiết của từng feature trên từng fold thời gian riêng biệt. |
| `summary.json` | JSON | Metadata lần chạy và thông tin tóm tắt. |
| `report.md` / `report.html` | Báo cáo | Báo cáo Markdown và giao diện HTML tương tác. |

### Các Cột trong `feature_scores.csv`:
- `feature`: Tên đặc trưng.
- `label`: Tên cột nhãn.
- `score_name`: Tên chỉ số (`pearson_ic` hoặc `rank_ic`).
- `mean_score`: Điểm IC trung bình qua các folds.
- `mean_abs_score`: Điểm tuyệt đối trung bình $| \text{IC} |$.
- `score_std`: Độ lệch chuẩn IC giữa các folds.
- `valid_folds`: Số lượng folds hợp lệ có thể tính toán.
- `positive_fold_ratio`: Tỷ lệ folds đạt điểm dương ($0.0 \to 1.0$).
- `samples`: Tổng số mẫu kiểm định qua tất cả các folds.

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Đặc trưng Tốt**:
  - `mean_abs_score` $\ge 0.03 - 0.05$.
  - `positive_fold_ratio` $\ge 0.70$ (hoặc $\le 0.30$ với feature nghịch biến ổn định).
  - `score_std` nhỏ (thể hiện sự ổn định qua mọi chu kỳ thị trường).
- **Tránh Đặc trưng Không Ổn định**: Nếu `mean_abs_score` cao nhưng `score_std` quá lớn và `positive_fold_ratio` quanh $0.50$, đặc trưng này chỉ hoạt động tốt ngẫu nhiên ở 1 vài fold và không đáng tin cậy khi giao dịch live.\n