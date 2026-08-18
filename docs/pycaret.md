# Tự động hóa Pipeline Học máy PyCaret (`pycaret`)

Module `pycaret` tích hợp thư viện mã nguồn mở Low-code Machine Learning PyCaret, tự động hóa toàn diện các bước chuẩn bị dữ liệu, so sánh đối đầu hàng chục thuật toán phân loại/hồi quy và tổng hợp bảng xếp hạng đặc trưng.

---

## 1. Mục đích & Ứng dụng

- **So sánh Đa Thuật toán Đồng thời (Multi-model Benchmark)**: Tự động huấn luyện và so sánh đồng thời hơn 15 thuật toán khác nhau (Logistic Regression, Decision Trees, Random Forest, AdaBoost, Gradient Boosting, LightGBM, CatBoost, Extra Trees...).
- **Quy trình Chuẩn hóa Khép kín (End-to-End Pipeline)**: Tự động xử lý các vấn đề dữ liệu thực tế như mất cân bằng nhãn (imbalance handling), chuẩn hóa thang đo và trích xuất đặc trưng.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Khởi tạo Môi trường (Setup & Preprocessing)**:
   - Tự động mã hóa biến, chuẩn hóa dữ liệu và phân tách tập kiểm định chéo (Cross-Validation Folds).
2. **So sánh Mô hình Đối đầu (`compare_models`)**:
   - Đánh giá toàn diện các mô hình dựa trên nhiều chỉ số: Accuracy, AUC, Recall, Precision, F1, Kappa, MCC.
3. **Trích xuất Tầm quan trọng**:
   - Trích xuất điểm tầm quan trọng từ mô hình có điểm số trung bình cao nhất.

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy AutoML PyCaret
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module pycaret

# Chỉ định nhãn mục tiêu cụ thể
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module pycaret --target allow_entry

# Lưu kết quả vào thư mục riêng
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module pycaret \
  --output-dir reports/pycaret_run
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `scores.csv` | CSV | Bảng so sánh hiệu năng chi tiết giữa các thuật toán được thử nghiệm. |
| `importance.csv` | CSV | Bảng xếp hạng đặc trưng theo mô hình tốt nhất. |
| `summary.json` | JSON | Metadata tổng kết lần chạy và thông tin mô hình chiến thắng. |
| `report.md` / `report.html` | Báo cáo | Báo cáo Markdown và HTML trực quan. |

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Chọn Kiến trúc Thuật toán Thích hợp**: Dựa vào `scores.csv` của PyCaret để xác định họ mô hình nào (Linear, Tree-based, hay Ensemble) tương thích tốt nhất với bản chất phân phối của dữ liệu hiện tại.\n