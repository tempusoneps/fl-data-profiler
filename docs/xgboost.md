# Mô hình Gradient Boosting XGBoost (`xgboost` / `xgboost-numeric`)

Module `xgboost` cung cấp pipeline phân tích và đánh giá dữ liệu mạnh mẽ sử dụng thuật toán Extreme Gradient Boosting (XGBoost) tiên tiến nhất. Module hỗ trợ cả bài toán phân loại (XGBClassifier) và bài toán hồi quy (XGBRegressor), phân tích chi tiết tầm quan trọng theo Gain/Weight/Cover và sinh các biểu đồ trực quan chuyên sâu.

---

## 1. Mục đích & Ứng dụng

- **Mô hình Học máy Chủ lực (State-of-the-art GBDT Baseline)**: Đánh giá khả năng dự báo tối đa của tập đặc trưng khi được khai thác bởi mô hình cây quyết định tăng cường độ dốc.
- **Phát hiện Mối quan hệ Phi tuyến & Tương tác Phức tạp**: Cây quyết định tự động nắm bắt các điểm bùng nổ tín hiệu, ngưỡng kích hoạt và tương tác đa biến.
- **Phân tích Chi tiết Từng Class (Per-class Performance)**: Đánh giá Precision, Recall, F1-score riêng biệt cho từng nhãn trong bài toán đa lớp hoặc mất cân bằng nhãn.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Tiền xử lý Dữ liệu**:
   - Tự động mã hóa nhãn phân loại bằng `LabelEncoder`.
   - Lọc và xử lý đặc trưng số, tự động quản lý giá trị `NaN` bằng cơ chế phân nhánh mặc định tối ưu của XGBoost.
2. **Cấu hình Huấn luyện Tối ưu**:
   - `n_estimators = 100`, `max_depth = 4 - 6`, `learning_rate = 0.05 - 0.1`, `subsample = 0.8`.
   - Chia tập Train (70%) và Test (30%) giữ nguyên trật tự thời gian.
3. **Các Thước đo Tầm quan trọng (Feature Importance)**:
   - **Gain (Khuyến nghị chính)**: Mức độ cải thiện độ chính xác (giảm loss) trung bình mà feature mang lại qua tất cả các điểm chia nhánh (splits).
   - **Weight (Frequency)**: Số lần feature được chọn để chia nhánh.
   - **Cover**: Số lượng mẫu dữ liệu chịu ảnh hưởng bởi các điểm chia nhánh liên quan đến feature.
4. **Sinh Biểu đồ Trực quan Hóa**:
   - Biểu đồ tầm quan trọng đặc trưng (Feature Importance Bar Chart).
   - Ma trận nhầm lẫn (Confusion Matrix Heatmap) cho bài toán phân loại.
   - Biểu đồ Scatter Plot Giá trị Thực tế vs Dự báo (Actual vs Predicted) cho bài toán hồi quy.

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy mô hình XGBoost đầy đủ
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module xgboost

# Hoặc dùng alias 'xgboost-numeric'
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module xgboost-numeric

# Chỉ định nhãn mục tiêu và giới hạn số dòng
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module xgboost \
  --target allow_entry \
  --limit 50000 \
  --output-dir reports/xgboost_run
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `scores.csv` | CSV | Bảng tổng hợp hiệu năng mô hình (Train/Test Accuracy, ROC-AUC, PR-AUC, F1, Log-Loss, R2, RMSE). |
| `importance.csv` | CSV | Danh sách đặc trưng xếp hạng theo điểm Gain Importance từ cao xuống thấp. |
| `per_class_metrics.csv` | CSV | Bảng chi tiết Precision, Recall, F1 và Support cho từng class nhãn. |
| `xgboost_importance_*.png` | Biểu đồ | Biểu đồ thanh ngang hiển thị Top 20 đặc trưng có Gain cao nhất. |
| `confusion_matrix_*.png` | Biểu đồ | Đồ thị ma trận nhầm lẫn chuẩn hóa theo tỷ lệ phần trăm. |
| `regression_pred_vs_actual.png` | Biểu đồ | Đồ thị so sánh đường dự báo và thực tế cho bài toán hồi quy. |
| `summary.json` | JSON | Metadata tổng kết và cấu hình chạy. |
| `report.md` / `report.html` | Báo cáo | Báo cáo Markdown và HTML đầy đủ chi tiết. |

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Ưu tiên Chỉ số Gain hơn Weight**: Weight thường bị thiên lệch về các biến liên tục biến động nhiều; chỉ số **Gain** mới phản ánh đúng giá trị thông tin thực tế giúp giảm sai số dự báo.
- **Phát hiện Overfitting**: So sánh `train_accuracy` (hoặc `train_roc_auc`) với `test_accuracy` (hoặc `test_roc_auc`). Nếu điểm Train là $0.98$ mà Test chỉ là $0.55$, mô hình đang bị overfit nặng, cần giảm `max_depth` hoặc tăng tham số phạt `reg_lambda`, `reg_alpha`.\n