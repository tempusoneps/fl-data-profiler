# Phân tích Tín hiệu Đơn lẻ & Tổng hợp (`signal_analysis`)

Module `signal_analysis` cung cấp công cụ đánh giá chuyên sâu cho các đặc trưng dạng tín hiệu (Trading Signals). Module thực hiện đánh giá hiệu năng của từng tín hiệu đơn lẻ, đo lường ma trận dư thừa/tương quan giữa các tín hiệu (Signal Redundancy), và đánh giá sức mạnh khi kết hợp toàn bộ tín hiệu trong một mô hình tổng hợp (Combined Signal Model).

---

## 1. Mục đích & Ứng dụng

- **Đánh giá Tín hiệu Giao dịch (Single Signal Evaluation)**: Đo lường chính xác các chỉ số Accuracy, ROC-AUC, Precision, Recall, F1-score và PR-AUC của từng tín hiệu.
- **Kiểm soát Dư thừa Tín hiệu (Signal Redundancy Analysis)**: Phát hiện các tín hiệu bị trùng lặp thông tin hoặc sao chép lẫn nhau để tinh gọn danh mục tín hiệu.
- **Tối ưu hóa Mô hình Tín hiệu Tổng hợp (Combined Alpha)**: Sử dụng mô hình Gradient Boosting (XGBoost) để đánh giá tầm quan trọng thực tế của từng tín hiệu khi hoạt động trong môi trường đa tín hiệu.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Nhận diện Tín hiệu (Signal Detection)**:
   - Ưu tiên tự động nhận diện các cột có tiền tố/hậu tố `signal` trong tên, hoặc sử dụng toàn bộ tập đặc trưng số hợp lệ.
2. **Đánh giá Tín hiệu Đơn lẻ (Single Signal Model)**:
   - Với mỗi tín hiệu, huấn luyện mô hình phân loại đơn biến (hoặc hồi quy) để đánh giá khả năng dự đoán nhãn mục tiêu trên tập test độc lập.
3. **Phân tích Dư thừa Tín hiệu (Redundancy Matrix)**:
   - Tính toán ma trận tương quan cặp đôi giữa các tín hiệu. Cặp tín hiệu có hệ số tương quan $|r| \ge 0.85$ được đánh dấu là có nguy cơ dư thừa cao.
4. **Mô hình Tín hiệu Kết hợp (Combined XGBoost Model)**:
   - Huấn luyện mô hình XGBoost đa biến trên toàn bộ tập tín hiệu.
   - Tính toán điểm tầm quan trọng (Gain Feature Importance) của từng tín hiệu trong mô hình chung.

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy phân tích tín hiệu đầy đủ
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module signal_analysis

# Chỉ định nhãn mục tiêu
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module signal_analysis --target allow_entry

# Xuất báo cáo ra thư mục riêng
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module signal_analysis \
  --output-dir reports/signal_analysis_report
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `single_signal_scores.csv` | CSV | Bảng đánh giá chi tiết từng tín hiệu đơn lẻ (Accuracy, ROC-AUC, F1, PR-AUC). |
| `signal_redundancy.csv` | CSV | Danh sách các cặp tín hiệu có mức độ tương quan và trùng lặp cao. |
| `combined_signal_importance.csv` | CSV | Điểm tầm quan trọng (Gain Importance) của từng tín hiệu trong mô hình tổng hợp. |
| `top_single_signals.png` | Biểu đồ | Biểu đồ cột xếp hạng top tín hiệu đơn lẻ tốt nhất theo ROC-AUC / Accuracy. |
| `combined_signal_importance.png` | Biểu đồ | Biểu đồ thanh ngang thể hiện đóng góp của từng tín hiệu trong mô hình tổng hợp. |
| `signal_redundancy_heatmap.png` | Biểu đồ | Heatmap ma trận tương quan giữa các tín hiệu hàng đầu. |
| `summary.json` | JSON | Metadata và danh sách các tín hiệu chủ lực. |
| `report.md` / `report.html` | Báo cáo | Báo cáo tổng hợp chi tiết dạng Markdown và HTML. |

### Các Cột trong `single_signal_scores.csv`:
- `signal`: Tên cột tín hiệu.
- `label`: Tên nhãn mục tiêu.
- `accuracy`: Độ chính xác phân loại trên tập test.
- `roc_auc`: Diện tích dưới đường cong ROC.
- `f1_score`: Điểm F1-Score cân bằng giữa Precision và Recall.
- `pr_auc`: Diện tích dưới đường cong Precision-Recall (rất quan trọng với nhãn mất cân bằng).
- `samples`: Số lượng mẫu đánh giá.

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Phát hiện Tín hiệu "Ngụy Alpha"**: Một tín hiệu có điểm `single_signal_score` cao nhưng điểm trong `combined_signal_importance` lại bằng 0 nghĩa là thông tin của nó đã bị bao hàm hoàn toàn bởi các tín hiệu khác trong hệ thống.
- **Loại bỏ Tín hiệu Dư thừa**: Dựa vào `signal_redundancy.csv`, chỉ giữ lại tín hiệu có ROC-AUC cao nhất trong nhóm các tín hiệu có tương quan $> 0.85$.\n