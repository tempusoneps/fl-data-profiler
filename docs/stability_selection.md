# Chọn lọc Biến Ổn định qua Lấy mẫu Lặp (`stability_selection`)

Module `stability_selection` áp dụng phương pháp **Stability Selection** (Meinshausen & Bühlmann) kết hợp kỹ thuật lấy mẫu lặp ngẫu nhiên (Subsampling/Bootstrap) và mô hình hồi quy phạt (Randomized Lasso / Logistic Regression) để tìm ra tập đặc trưng thực sự bền vững và không phụ thuộc vào một tập mẫu dữ liệu ngẫu nhiên cụ thể.

---

## 1. Mục đích & Ứng dụng

- **Giải quyết Vấn đề Không Ổn định của Lasso (Lasso Instability)**: Khi có nhiều biến tương quan, Lasso thông thường sẽ chọn ngẫu nhiên 1 biến và loại bỏ các biến còn lại. Stability Selection khắc phục triệt để nhược điểm này.
- **Kiểm soát Tỷ lệ Phát hiện Sai (False Discovery Rate - FDR)**: Đảm bảo các biến được chọn có xác suất là biến thật cao và kiểm soát chặt chẽ số lượng biến rác lọt vào.
- **Đánh giá Độ Bền vững Qua Nhiều Mẫu**: Đo lường tần suất một biến được mô hình lựa chọn qua hàng trăm lần lấy mẫu con.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Lấy mẫu Phân đoạn Lặp lại (Subsampling)**:
   - Thực hiện $B$ lần lấy mẫu ngẫu nhiên (mặc định 100 lần), mỗi lần lấy $50\%$ số dòng dữ liệu.
2. **Mô hình Hồi quy Phạt Ngẫu nhiên (Randomized Regularization)**:
   - Trên mỗi mẫu con, huấn luyện mô hình Lasso / L1 Logistic Regression với hệ số phạt được nhân ngẫu nhiên với một trọng số trong khoảng $[\alpha, 1]$.
   - Ghi nhận xem biến $F_j$ có hệ số $\ne 0$ hay không.
3. **Xác suất Lựa chọn Ổn định (Selection Probability $\Pi$)**:
   $$\Pi(F_j) = \frac{\text{Số lần } F_j \text{ có hệ số khác } 0}{\text{Tổng số lần lấy mẫu } B}$$
4. **Ngưỡng Ổn định (Stability Threshold $\pi_{\text{threshold}}$)**:
   - Các biến có $\Pi(F_j) \ge 0.60 - 0.75$ được xác nhận là biến ổn định (Stable Features).

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy Stability Selection
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module stability_selection

# Chỉ định nhãn mục tiêu và giới hạn số dòng
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module stability_selection \
  --target allow_entry \
  --limit 20000

# Chỉ định thư mục xuất kết quả
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module stability_selection \
  --output-dir reports/stability_selection_run
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `feature_scores.csv` | CSV | Toàn bộ danh sách đặc trưng xếp hạng theo xác suất ổn định $\Pi$ giảm dần. |
| `top_features.csv` | CSV | Top 50 đặc trưng có độ ổn định cao nhất. |
| `summary.json` | JSON | Metadata tổng kết số lượng biến vượt ngưỡng ổn định. |
| `report.md` / `report.html` | Báo cáo | Báo cáo Markdown và HTML tổng kết chi tiết. |

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Xác suất Ổn định $\Pi \ge 0.80$**: Đặc trưng cực kỳ vững chắc, bắt buộc phải có trong mô hình giao dịch.
- **Xác suất $\Pi \le 0.30$**: Đặc trưng không ổn định, thường chỉ xuất hiện do nhiễu ở một số giai đoạn thị trường ngắn hạn, nên loại bỏ.\n