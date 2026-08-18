# Hồi quy Tuyến tính Phạt Lasso & Ridge (`regularized_linear`)

Module `regularized_linear` sử dụng các kỹ thuật hồi quy tuyến tính có thành phần chuẩn hóa/phạt trọng số (L1 Lasso, L2 Ridge, ElasticNet) từ `scikit-learn` để đánh giá tầm quan trọng và triệt tiêu các đặc trưng gây nhiễu.

---

## 1. Mục đích & Ứng dụng

- **Thu nhỏ Hệ số & Lọc Biến (Feature Shrinkage & Sparsity)**: Lasso (L1) tự động ép các hệ số của biến không quan trọng về đúng bằng 0, tạo ra mô hình thưa (sparse model).
- **Chống Đa cộng tuyến (Handling Multicollinearity)**: Ridge (L2) thu nhỏ hệ số của các biến tương quan mạnh, giúp mô hình ổn định và tránh sai số lớn.
- **Baseline Tuyến tính Nhanh & Nhẹ**: Cung cấp thước đo tầm quan trọng tuyến tính với chi phí tính toán cực thấp.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Chuẩn hóa Đầu vào (Standardization)**:
   - Dữ liệu được đưa qua `StandardScaler` (Mean = 0, Std = 1) để đảm bảo hình phạt được áp dụng công bằng lên tất cả các biến có đơn vị đo khác nhau.
2. **Hàm Mục tiêu Tối ưu**:
   - **Lasso (L1)**: $\min_{\beta} \|Y - X\beta\|_2^2 + \alpha \|\beta\|_1$
   - **Ridge (L2)**: $\min_{\beta} \|Y - X\beta\|_2^2 + \alpha \|\beta\|_2^2$
3. **Điểm Tầm quan trọng (Importance Score)**:
   - Xếp hạng đặc trưng theo giá trị tuyệt đối của hệ số chuẩn hóa $|\hat{\beta}_j|$.

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy hồi quy phạt regularized linear
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module regularized_linear

# Chỉ định nhãn mục tiêu cụ thể
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module regularized_linear --target allow_entry

# Lưu kết quả vào thư mục riêng
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module regularized_linear \
  --output-dir reports/reg_linear_output
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `feature_scores.csv` | CSV | Danh sách đặc trưng xếp hạng theo độ lớn hệ số chuẩn hóa tuyệt đối $|\hat{\beta}|$. |
| `top_features.csv` | CSV | Top 50 đặc trưng có hệ số tác động lớn nhất. |
| `summary.json` | JSON | Metadata tổng kết mô hình. |
| `report.md` / `report.html` | Báo cáo | Báo cáo Markdown và HTML tương tác. |

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Hệ số bằng 0**: Các đặc trưng có hệ số bị triệt tiêu về 0 bởi L1 penalty có thể an tâm loại bỏ khỏi dataset.
- **Dấu của Hệ số ($\beta > 0$ hoặc $\beta < 0$)**: Thể hiện chiều hướng tác động thuận hay nghịch của feature lên nhãn mục tiêu.\n