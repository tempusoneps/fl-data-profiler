# Tự động hóa Máy học Tiết kiệm Tài nguyên FLAML (`flaml`)

Module `flaml` tích hợp thư viện Fast and Lightweight AutoML (FLAML) của Microsoft Research, được thiết kế để tìm ra mô hình và bộ siêu tham số tối ưu với chi phí tính toán và thời gian tối thiểu.

---

## 1. Mục đích & Ứng dụng

- **Tối ưu Hóa Chi phí Tính toán (Cost-Frugal Hyperparameter Optimization)**: Tìm kiếm siêu tham số thông minh theo phương pháp CFO (Cost-Frugal Optimization) giúp hội tụ nhanh hơn nhiều lần so với Bayesian Optimization truyền thống.
- **Thử nghiệm Nhanh (Fast Prototyping)**: Rất phù hợp khi cần chạy thử nghiệm trên máy tính cá nhân hoặc môi trường không có GPU mạnh.
- **Tập Trung vào Các Thuật Toán Cây Nhẹ**: Tự động so sánh LightGBM, XGBoost, Random Forest và Extra Trees.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Không gian Tìm kiếm Tăng dần (Iterative Search Strategy)**:
   - Bắt đầu thử nghiệm từ các mô hình nhỏ, số lượng cây ít và mở rộng dần không gian tìm kiếm khi phát hiện vùng siêu tham số tiềm năng.
2. **Đánh giá Tầm quan trọng Đặc trưng**:
   - Trích xuất tầm quan trọng đặc trưng từ mô hình estimator tối ưu được lựa chọn.

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy AutoML FLAML
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module flaml

# Chỉ định nhãn mục tiêu và giới hạn số dòng
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module flaml \
  --target allow_entry \
  --limit 30000

# Chỉ định thư mục lưu kết quả
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module flaml \
  --output-dir reports/flaml_run
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `scores.csv` | CSV | Bảng điểm số hiệu năng của mô hình tốt nhất tìm được bởi FLAML. |
| `importance.csv` | CSV | Danh sách đặc trưng xếp hạng theo độ quan trọng của mô hình chiến thắng. |
| `summary.json` | JSON | Metadata tổng kết thuật toán chiến thắng và tham số tối ưu. |
| `report.md` / `report.html` | Báo cáo | Báo cáo Markdown và HTML tổng kết. |

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- FLAML là lựa chọn lý tưởng khi bạn muốn có kết quả AutoML nhanh chóng trong vòng 1-2 phút mà vẫn đảm bảo độ chính xác tiệm cận các framework nặng nề.\n