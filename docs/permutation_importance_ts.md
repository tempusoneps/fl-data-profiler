# Tầm quan trọng Xáo trộn Chuỗi Thời gian (`permutation_importance_ts`)

Module `permutation_importance_ts` đo lường tầm quan trọng của đặc trưng bằng phương pháp hoán vị (Permutation Feature Importance) kết hợp mô hình Random Forest trên cấu trúc cửa sổ trượt chuỗi thời gian (**Time-Series Walk-Forward Folds**).

---

## 1. Mục đích & Ứng dụng

- **Đo lường Mức Độ Tổn thất Thực tế (Drop in Performance)**: Đánh giá mô hình bị giảm bao nhiêu điểm Accuracy hoặc $R^2$ khi thông tin của một đặc trưng bị phá vỡ hoàn toàn bằng cách xáo trộn (shuffle).
- **Chống Bias của Gini Importance**: Tránh hiện tượng cây quyết định ưu tiên sai lệch cho các biến liên tục có độ phân giải cao (high cardinality features).
- **Đánh giá trên Dữ liệu Kiểm định Tương lai (Out-of-Fold Evaluation)**: Quá trình hoán vị chỉ thực hiện trên tập Test Fold để đảm bảo tính khách quan.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Huấn luyện Mô hình trên Từng Fold**:
   - Trên mỗi fold thời gian $(Train_t, Test_t)$, huấn luyện mô hình `RandomForestClassifier` hoặc `RandomForestRegressor`.
   - Tính toán điểm số cơ sở (Baseline Score: `accuracy` cho bài toán phân loại hoặc $R^2$ cho bài toán hồi quy) trên $Test_t$.
2. **Xáo trộn Đặc trưng (Feature Permutation)**:
   - Với mỗi đặc trưng $F_j$, hoán vị ngẫu nhiên thứ tự các giá trị của $F_j$ trên $Test_t$, giữ nguyên toàn bộ các đặc trưng khác.
   - Dự báo lại và tính điểm số sau hoán vị $\text{Score}_{\text{permuted}}$.
3. **Tính toán Mức Sụt giảm (Permutation Drop)**:
   $$\text{Drop}(F_j) = \text{Baseline Score} - \text{Score}_{\text{permuted}}$$
   - $\text{Drop} > 0$: Đặc trưng có đóng góp tích cực giúp mô hình dự đoán chính xác hơn.
   - $\text{Drop} \le 0$: Đặc trưng không quan trọng hoặc làm nhiễu mô hình.
4. **Tổng hợp qua Nhiều Folds**:
   - Tính trung bình `mean_score` và tỷ lệ fold mang lại đóng góp dương.

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy Permutation Importance cho Time-Series
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module permutation_importance_ts

# Chỉ định nhãn mục tiêu và giới hạn số dòng
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module permutation_importance_ts \
  --target allow_entry \
  --limit 20000

# Chỉ định thư mục xuất báo cáo
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module permutation_importance_ts \
  --output-dir reports/perm_ts_run
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `feature_scores.csv` | CSV | Bảng xếp hạng mức độ sụt giảm hiệu năng trung bình của từng feature qua các folds. |
| `top_features.csv` | CSV | Top 50 đặc trưng quan trọng nhất. |
| `fold_scores.csv` | CSV | Điểm baseline, permutation drop và correlation support của từng feature trên từng fold cụ thể. |
| `summary.json` | JSON | Metadata tổng hợp lần chạy. |
| `report.md` / `report.html` | Báo cáo | Báo cáo chi tiết định dạng Markdown và HTML tương tác. |

### Các Cột trong `feature_scores.csv`:
- `feature`: Tên đặc trưng.
- `label`: Tên cột nhãn.
- `score_name`: Tên chỉ số (`permutation_importance`).
- `mean_score`: Điểm tầm quan trọng tổng hợp trung bình.
- `mean_abs_score`: Độ lớn trung bình của mức sụt giảm hiệu năng.
- `score_std`: Độ lệch chuẩn của tầm quan trọng giữa các folds.
- `valid_folds`: Số lượng folds thời gian hợp lệ.
- `positive_fold_ratio`: Tỷ lệ folds mà việc xáo trộn feature này làm giảm hiệu năng mô hình.
- `samples`: Tổng số mẫu kiểm định.

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Feature Cốt lõi (Core Alpha Drivers)**: Là các đặc trưng có `mean_score` cao và `positive_fold_ratio` $\ge 0.80$. Khi thiếu các biến này, hiệu năng mô hình sụt giảm nghiêm trọng.
- **Feature Nhiễu (Noisy Features)**: Các đặc trưng có `mean_score` $\approx 0$ hoặc âm có thể loại bỏ ngay lập tức để mô hình tinh gọn và tăng tốc độ suy luận.\n