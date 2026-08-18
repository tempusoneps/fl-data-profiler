# Tầm quan trọng Đặc trưng LightGBM (`lightgbm`)

Module `lightgbm` sử dụng framework LightGBM (Light Gradient Boosting Machine) của Microsoft với thuật toán Histogram-based và Leaf-wise tree growth để tính toán tầm quan trọng đặc trưng với tốc độ cực nhanh và khả năng xử lý dữ liệu lớn vượt trội.

---

## 1. Mục đích & Ứng dụng

- **Đánh giá Siêu Tốc trên Dữ liệu Lớn**: LightGBM nhanh hơn XGBoost từ 5-10 lần khi làm việc với hàng trăm đặc trưng và hàng triệu dòng dữ liệu tick/orderbook.
- **Đánh giá Importance theo Split & Gain**: Phân tích chi tiết mức độ đóng góp của từng đặc trưng vào việc tối ưu hóa hàm mục tiêu.
- **Hỗ trợ Dữ liệu Chuỗi Thời gian**: Xử lý tốt các đặc trưng phân loại (categorical) và giá trị thiếu (missing values) trực tiếp mà không cần one-hot encoding cồng kềnh.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Thuật toán Tăng trưởng Cây theo Lá (Leaf-wise Splitting)**:
   - Tối ưu hóa việc giảm sai số nhanh hơn so với cách tăng trưởng theo tầng (depth-wise) truyền thống.
2. **Trích xuất Tầm quan trọng (Feature Importance)**:
   - `importance_type="gain"`: Tổng mức giảm entropy hoặc sai số hồi quy mà feature mang lại.
   - `importance_type="split"`: Tần suất feature được sử dụng tại các nút chia nhánh.
3. **Xếp hạng và Xuất Báo cáo**:
   - Xếp hạng theo Gain tuyệt đối và chuẩn hóa về tỷ lệ phần trăm đóng góp ($0 \to 100\%$).

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy đánh giá tầm quan trọng LightGBM
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module lightgbm

# Chỉ định nhãn mục tiêu cụ thể
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module lightgbm --target allow_entry

# Lưu kết quả vào thư mục riêng
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module lightgbm \
  --output-dir reports/lightgbm_run
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `feature_scores.csv` | CSV | Bảng xếp hạng toàn bộ các đặc trưng theo điểm số tầm quan trọng LightGBM. |
| `top_features.csv` | CSV | Top 50 đặc trưng có điểm Gain cao nhất. |
| `summary.json` | JSON | Metadata tổng kết và thời gian thực thi. |
| `report.md` / `report.html` | Báo cáo | Báo cáo chi tiết dạng Markdown và HTML tương tác. |

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Sàng lọc Nhanh cho Dataset Lớn**: Khi dataset có $> 100,000$ dòng và $> 200$ cột, hãy chạy `lightgbm` đầu tiên để lọc ra Top 50 đặc trưng trước khi thực hiện các phân tích tính toán nặng khác như `shap` hay `permutation_importance_ts`.\n