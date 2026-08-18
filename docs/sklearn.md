# Đánh giá Mô hình Cơ bản Scikit-Learn (`sklearn`)

Module `sklearn` cung cấp pipeline huấn luyện và đánh giá mô hình học máy chuẩn mực từ thư viện Scikit-Learn (Ridge Regression cho bài toán hồi quy và SGDClassifier cho bài toán phân loại nhị phân/đa lớp).

---

## 1. Mục đích & Ứng dụng

- **Mô hình Chuẩn đối sánh (Standard ML Baseline)**: Tạo benchmark hiệu năng chuẩn mực trước khi chuyển sang các mô hình phức tạp hơn (XGBoost/AutoML).
- **Đánh giá Hiệu năng Toàn diện**: Tính toán đầy đủ các chỉ số Accuracy, ROC-AUC, Precision, Recall, F1, RMSE, MAE, $R^2$ trên tập test độc lập.
- **Phát hiện Overfitting Nhanh**: So sánh trực tiếp giữa điểm số trên tập Train và tập Test.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Pipeline Tiền xử lý (Preprocessing Pipeline)**:
   - Điền khuyết thiếu bằng `SimpleImputer(strategy="median")`.
   - Chuẩn hóa thang đo bằng `StandardScaler()`.
2. **Chia Tập Dữ liệu**:
   - Phân chia tập Train (70%) và Test (30%) theo thứ tự tuần tự hoặc phân tầng (stratified).
3. **Huấn luyện Mô hình**:
   - **Classification**: `SGDClassifier(loss="log_loss", penalty="l2")` hoặc Logistic Regression.
   - **Regression**: `Ridge(alpha=1.0)`.
4. **Trích xuất Tầm quan trọng (Importance Extraction)**:
   - Chuẩn hóa vector trọng số hệ số $|\beta_i|$ để xếp hạng mức độ đóng góp của từng biến.

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy mô hình Sklearn baseline
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module sklearn

# Chỉ định nhãn mục tiêu
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module sklearn --target allow_entry

# Xuất báo cáo ra thư mục riêng
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module sklearn \
  --output-dir reports/sklearn_baseline
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `scores.csv` | CSV | Bảng tổng hợp các chỉ số hiệu năng (Train/Test Accuracy, ROC-AUC, F1, R2, RMSE) cho từng target. |
| `importance.csv` | CSV | Danh sách các feature xếp hạng theo độ lớn tầm quan trọng trọng số. |
| `summary.json` | JSON | Metadata tổng kết mô hình và chỉ số nổi bật. |
| `report.md` / `report.html` | Báo cáo | Báo cáo Markdown và HTML tổng kết chi tiết. |

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Đánh giá Benchmark**: Nếu mô hình XGBoost hay AutoML sau này không vượt trội hơn mô hình Sklearn Linear Baseline quá 5% điểm ROC-AUC, dữ liệu có thể chủ yếu mang đặc tính quan hệ tuyến tính đơn giản.\n