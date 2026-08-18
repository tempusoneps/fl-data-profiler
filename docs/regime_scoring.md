# Phân tích Đặc trưng theo Chế độ Thị trường (`regime_scoring`)

Module `regime_scoring` phân đoạn dữ liệu chuỗi thời gian thành các **Chế độ Thị trường (Market Regimes)** khác nhau (ví dụ: Biến động cao/thấp, Xu hướng tăng/giảm) và đánh giá sức mạnh dự báo của đặc trưng bên trong từng chế độ riêng biệt.

---

## 1. Mục đích & Ứng dụng

- **Phát hiện Tín hiệu Đặc thù theo Chế độ (Regime-Specific Alpha)**: Nhận diện các đặc trưng chỉ hoạt động hiệu quả trong thị trường biến động mạnh (High Volatility) hoặc chỉ hoạt động trong thị trường đi ngang (Sideways).
- **Phòng ngừa Bẫy Tín hiệu Bình quân (Averaging Trap)**: Tránh việc loại bỏ một tín hiệu xuất sắc trong thời kỳ khủng hoảng chỉ vì nó hoạt động kém trong thời kỳ thị trường êm ả.
- **Xây dựng Hệ thống Giao dịch Đa Chế độ (Regime-Switching Models)**: Cung cấp danh sách feature tối ưu cho từng nhánh mô hình chuyên biệt.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Phân đoạn Thị trường (Regime Segmentation)**:
   - Dữ liệu được chia thành các phân đoạn thời gian dựa trên các chỉ báo trạng thái hoặc các folds biến động.
2. **Tính toán Hiệu năng Từng Phân đoạn**:
   - Trên mỗi chế độ/phân đoạn thời gian, module tính toán hệ số tương quan thông tin (IC) và độ suy giảm dự báo của từng đặc trưng.
3. **Tổng hợp Thống kê**:
   - Đo lường mức độ biến thiên của điểm số giữa các chế độ thị trường khác nhau.

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy đánh giá Regime Scoring
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module regime_scoring

# Chỉ định nhãn mục tiêu cụ thể
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module regime_scoring --target allow_entry

# Lưu báo cáo vào thư mục riêng
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module regime_scoring \
  --output-dir reports/regime_scoring_run
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `feature_scores.csv` | CSV | Bảng tổng hợp điểm số trung bình và độ lệch chuẩn của features qua các chế độ thị trường. |
| `top_features.csv` | CSV | Top 50 features có sức mạnh dự báo tốt nhất trên đa chế độ. |
| `fold_scores.csv` | CSV | Điểm số chi tiết của từng feature trong từng chế độ/fold thị trường cụ thể. |
| `summary.json` | JSON | Metadata tổng kết lần chạy. |
| `report.md` / `report.html` | Báo cáo | Báo cáo Markdown và HTML tổng kết chi tiết. |

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **All-Weather Features**: Các đặc trưng có điểm số dương ổn định trên mọi chế độ (`positive_fold_ratio` $\ge 0.85$) nên được sử dụng làm đặc trưng cốt lõi cho mọi mô hình.
- **Regime Specialists**: Các đặc trưng có điểm rất cao ở một số fold nhưng âm ở các fold khác cần được đưa vào kèm theo biến điều kiện chế độ thị trường (Conditional Gate).\n