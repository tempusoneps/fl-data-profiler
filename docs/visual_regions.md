# Trích xuất Vùng Quyết định & Luật 2D (`visual_regions`)

Module `visual_regions` tự động quét không gian đặc trưng 2 chiều $(F_1, F_2)$, phân chia thành lưới phân vị (quantile grid), đánh giá độ thuần khiết (purity) của từng ô lưới và hợp nhất các ô liền kề để sinh ra các **Luật Hộp Quyết định (2D Bounding Box Rules)** có thể diễn giải trực tiếp.

---

## 1. Mục đích & Ứng dụng

- **Sinh Luật Giao dịch Tự động (Rule Generation)**: Chuyển đổi các đặc trưng số phức tạp thành các điều kiện dạng văn bản: `IF Feature_1 IN [a, b] AND Feature_2 IN [c, d] THEN Label = 1`.
- **Trực quan hóa Không gian Quyết định**: Xác định chính xác "vùng an toàn" hoặc "vùng kích hoạt tín hiệu" trên đồ thị 2D.
- **White-box Explainability**: Cung cấp luật logic rõ ràng, minh bạch, dễ dàng tích hợp vào bot giao dịch tự động mà không cần chạy mô hình ML phức tạp khi live.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Rời rạc hóa Phân vị (Quantile Binning)**:
   - Mỗi đặc trưng số $F$ được chia thành $N$ khoảng phân vị (mặc định $N=5$ bins: Q1, Q2, Q3, Q4, Q5).
2. **Sàng lọc Ứng viên 1D (1D Candidate Scoring)**:
   - Đánh giá độ tinh khiết đơn biến của từng đặc trưng và chọn ra top ứng viên sáng giá nhất.
3. **Đánh giá Lưới 2D (2D Grid Purity Evaluation)**:
   - Tạo lưới $5 \times 5 = 25$ ô cho mỗi cặp $(F_1, F_2)$.
   - Tính tỷ lệ nhãn mục tiêu trong từng ô:
     $$\text{Purity}(C_{i, j}) = \frac{\text{Count}(\text{Target Class in } C_{i, j})}{\text{Total Samples in } C_{i, j}}$$
4. **Hợp nhất Vùng Liền kề (Contiguous Region Merging)**:
   - Tìm và gộp các ô lân cận có cùng nhãn chiếm ưu thế và độ thuần khiết vượt ngưỡng $(\ge 0.70)$ thành một hình chữ nhật bao phủ (Bounding Box Rule).
   - Tính toán chỉ số Coverage (độ phủ mẫu) và Purity (độ chính xác) của luật.

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy trích xuất luật visual 2D regions
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module visual_regions

# Chỉ định nhãn mục tiêu và giới hạn số dòng
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module visual_regions \
  --target allow_entry \
  --limit 20000

# Chỉ định thư mục xuất kết quả
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module visual_regions \
  --output-dir reports/rules_2d_output
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `rules_2d.csv` | CSV | Bảng danh sách tất cả các luật 2D được trích xuất kèm điều kiện biên, độ thuần khiết và độ phủ. |
| `summary.json` | JSON | Metadata tổng kết số lượng luật tìm thấy cho từng nhãn. |
| `report.md` / `report.html` | Báo cáo | Báo cáo chi tiết định dạng Markdown và HTML hiển thị danh sách luật dễ đọc. |

### Các Cột trong `rules_2d.csv`:
- `label`: Tên cột nhãn mục tiêu.
- `target_class`: Giá trị class được dự báo (ví dụ: `1` hoặc `True`).
- `feature_1` / `feature_2`: Hai đặc trưng tạo nên vùng quyết định 2D.
- `f1_min` / `f1_max`: Ngưỡng giá trị cận dưới và cận trên của đặc trưng thứ nhất.
- `f2_min` / `f2_max`: Ngưỡng giá trị cận dưới và cận trên của đặc trưng thứ hai.
- `purity`: Tỷ lệ mẫu thuộc class mục tiêu trong vùng này ($0.0 \to 1.0$).
- `coverage`: Số lượng mẫu và tỷ lệ mẫu trong toàn dataset rơi vào vùng này.
- `rule_text`: Biểu thức logic hoàn chỉnh dạng text để áp dụng vào code (ví dụ: `feature_A in [1.2, 3.5] and feature_B in [10.0, 25.0]`).

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Quy tắc chọn Luật Tốt**:
  - $\text{Purity} \ge 0.75$: Tỷ lệ chính xác cao.
  - $\text{Coverage} \ge 2\% - 5\%$: Đủ số lượng mẫu thực tế để tín hiệu không bị quá hiếm (tránh overfit vào vài điểm nhiễu).
- **Tích hợp vào Hệ thống**: Có thể copy trực tiếp cột `rule_text` đưa vào bộ điều kiện kích hoạt chiến lược (Trade Filter Rule Engine).\n