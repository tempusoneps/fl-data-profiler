# Chọn lọc Đặc trưng Toàn diện Boruta (`boruta`)

Module `boruta` áp dụng thuật toán **Boruta Feature Selection** (dựa trên mô hình Random Forest) để tìm ra **Toàn bộ các đặc trưng có liên quan thực sự (All-relevant Feature Selection)**, thay vì chỉ tìm một tập con tối thiểu không dư thừa (Minimal-optimal).

---

## 1. Mục đích & Ứng dụng

- **Loại bỏ Yếu tố May rủi (Shadow Feature Contrast)**: So sánh độ quan trọng của từng đặc trưng thật với các "đặc trưng bóng ma" (Shadow Features - được sinh ra bằng cách xáo trộn ngẫu nhiên dữ liệu gốc).
- **Phân loại Rõ Ràng 3 Trạng thái**:
  - **Confirmed (Chắc chắn quan trọng)**: Vượt trội hơn hẳn so với đặc trưng ngẫu nhiên tốt nhất với mức ý nghĩa thống kê cao.
  - **Tentative (Nghi vấn/Cần xem xét)**: Nằm ở vùng ranh giới, cần thêm dữ liệu để khẳng định.
  - **Rejected (Loại bỏ)**: Kém hơn hoặc bằng các biến bóng ma ngẫu nhiên.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Nhân đôi Dataset với Shadow Features**:
   - Với mỗi đặc trưng $X_j$, tạo ra một bản sao bóng ma $\tilde{X}_j$ bằng cách xáo trộn ngẫu nhiên các hàng của $X_j$.
2. **Huấn luyện Random Forest**:
   - Huấn luyện mô hình Random Forest trên dataset mở rộng chứa cả biến thật và biến bóng ma.
   - Tính điểm Z-score của độ quan trọng (Feature Importance) cho toàn bộ các biến.
3. **Tìm Ngưỡng Bóng Ma Cao Nhất (Max Shadow Z-score - $Z_{\text{max-shadow}}$)**:
   - Xác định giá trị Z-score cao nhất trong số tất cả các đặc trưng bóng ma.
4. **Kiểm định Thống kê Nhị thức (Binomial Hypothesis Test)**:
   - Qua nhiều vòng lặp (iterations), đếm số lần $Z(X_j) > Z_{\text{max-shadow}}$.
   - Áp dụng phân phối nhị thức để gắn nhãn `Confirmed` nếu số lần vượt trội có ý nghĩa thống kê ($p < 0.01$), hoặc `Rejected` nếu ngược lại.

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy thuật toán Boruta Feature Selection
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module boruta

# Chỉ định nhãn mục tiêu cụ thể
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module boruta --target allow_entry

# Giới hạn số dòng và thư mục kết quả
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module boruta \
  --limit 20000 \
  --output-dir reports/boruta_output
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `boruta_features.csv` | CSV | Bảng danh sách tất cả các đặc trưng kèm phân loại trạng thái (`Confirmed`, `Tentative`, `Rejected`) và điểm Z-score. |
| `scores.csv` | CSV | Hiệu năng mô hình Random Forest được sử dụng trong quá trình chạy Boruta. |
| `summary.json` | JSON | Metadata tổng hợp số lượng biến Confirmed / Rejected. |
| `report.md` / `report.html` | Báo cáo | Báo cáo Markdown và HTML tổng kết trực quan. |

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Chiến lược Chọn Biến**:
  - Giữ lại toàn bộ các biến có trạng thái **`Confirmed`**.
  - Loại bỏ hoàn toàn các biến có trạng thái **`Rejected`** vì chúng không mang lại giá trị nào tốt hơn một chuỗi số ngẫu nhiên.
  - Với các biến **`Tentative`**, có thể chạy thêm vòng lặp hoặc kết hợp với module `mutual_information` để quyết định.\n