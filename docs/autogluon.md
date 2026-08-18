# Tự động hóa Máy học với AutoGluon (`autogluon`)

Module `autogluon` tích hợp framework AutoML hàng đầu của Amazon (AutoGluon-Tabular) để tự động huấn luyện, tối ưu siêu tham số, xếp chồng đa tầng (Multi-layer Stacking) và đánh giá độ quan trọng đặc trưng trên tập hợp nhiều kiến trúc mô hình khác nhau.

---

## 1. Mục đích & Ứng dụng

- **Sức Mạnh Dự Báo Tối Đa (Ensemble SOTA)**: Kết hợp sức mạnh của hàng loạt mô hình hàng đầu (LightGBM, CatBoost, XGBoost, Random Forest, Extra Trees, Neural Networks / MLP, FastAI).
- **Tự Động Hóa Toàn Bộ Pipeline**: Tự động xử lý kiểu dữ liệu, điền khuyết thiếu, encoding và tối ưu hóa hàm mất mát phù hợp với cấu trúc nhãn.
- **Leaderboard So sánh Đa Mô hình**: Bảng xếp hạng chi tiết hiệu năng của từng kiến trúc mô hình riêng biệt.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Multi-layer Stacking & Ensembling**:
   - AutoGluon không chỉ chọn 1 mô hình tốt nhất mà tự động xây dựng mô hình xếp chồng (Stacking Ensemble) nhiều tầng giúp đẩy hiệu năng lên mức cao nhất.
2. **Đo lường Tầm quan trọng Đặc trưng (Permutation Importance)**:
   - Tính toán mức sụt giảm hiệu năng của mô hình Ensemble tối ưu khi hoán vị từng đặc trưng trên tập dữ liệu kiểm định độc lập.

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy AutoML AutoGluon
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module autogluon

# Chỉ định nhãn mục tiêu và giới hạn số dòng
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module autogluon \
  --target allow_entry \
  --limit 25000

# Chỉ định thư mục xuất báo cáo
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module autogluon \
  --output-dir reports/autogluon_run
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `scores.csv` | CSV | Bảng tổng hợp điểm số hiệu năng (Accuracy, ROC-AUC, F1, RMSE) của mô hình Ensemble tốt nhất. |
| `importance.csv` | CSV | Bảng xếp hạng tầm quan trọng đặc trưng được tính toán bởi AutoGluon. |
| `summary.json` | JSON | Metadata tổng hợp mô hình và cấu hình thực thi. |
| `report.md` / `report.html` | Báo cáo | Báo cáo Markdown và HTML chi tiết. |

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Trần Hiệu năng Dữ liệu (Performance Ceiling)**: Điểm số của AutoGluon thường đại diện cho mức trần hiệu năng tối đa có thể đạt được với tập đặc trưng hiện tại.
- Nếu điểm số của AutoGluon vẫn thấp (ví dụ ROC-AUC $\approx 0.52$), vấn đề nằm ở chất lượng đặc trưng (Feature Quality), cần quay lại bước sinh đặc trưng thay vì cố gắng tinh chỉnh siêu tham số.\n