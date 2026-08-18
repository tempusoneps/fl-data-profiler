# Giải thích Đóng góp Đặc trưng bằng SHAP Values (`shap`)

Module `shap` áp dụng lý thuyết trò chơi cộng tác Shapley Values từ kinh tế học (SHapley Additive exPlanations) kết hợp thuật toán tối ưu TreeSHAP trên mô hình XGBoost để bóc tách chính xác mức độ và hướng tác động (cực dương hoặc cực âm) của từng đặc trưng lên từng dự báo cá biệt.

---

## 1. Mục đích & Ứng dụng

- **Giải thích Mô hình Hộp đen (Black-box Explainability)**: Chuyển đổi mô hình GBDT phức tạp thành một hệ thống giải thích minh bạch, hiểu rõ lý do tại sao mô hình đưa ra tín hiệu Mua/Bán tại mỗi thời điểm.
- **Tính Nhất quán (Consistency & Fair Attribution)**: Đảm bảo nếu một đặc trưng đóng góp nhiều hơn cho dự báo thì giá trị SHAP của nó luôn lớn hơn (khắc phục hoàn toàn nhược điểm thiếu nhất quán của Split/Weight Importance).
- **Đo lường Mean Absolute SHAP**: Đánh giá độ lớn tác động trung bình của từng đặc trưng trên toàn bộ không gian mẫu.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Nguyên lý Giá trị Shapley**:
   - Đo lường đóng góp biên của đặc trưng $j$ qua tất cả các tập con đặc trưng kết hợp có thể có $\mathcal{S} \subseteq \mathcal{F} \setminus \{j\}$:
     $$\phi_j = \sum_{\mathcal{S} \subseteq \mathcal{F} \setminus \{j\}} \frac{|\mathcal{S}|! (|\mathcal{F}| - |\mathcal{S}| - 1)!}{|\mathcal{F}|!} \left( f(\mathcal{S} \cup \{j\}) - f(\mathcal{S}) \right)$$
2. **Thuật toán TreeSHAP Tối ưu**:
   - Sử dụng thuật toán TreeSHAP của Lundberg et al. với độ phức tạp $\mathcal{O}(TLD^2)$ (với $T$ là số cây, $L$ là số lá, $D$ là độ sâu), cho phép tính toán SHAP chính xác cho hàng chục ngàn mẫu trong vài giây.
3. **Chỉ số Tầm quan trọng Toàn cục (Global SHAP Importance)**:
   $$\text{Mean Abs SHAP}_j = \frac{1}{N} \sum_{i=1}^N |\phi_j^{(i)}|$$

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy phân tích SHAP Value
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module shap

# Chỉ định nhãn mục tiêu cụ thể
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module shap --target allow_entry

# Giới hạn số dòng giải thích và xuất kết quả
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module shap \
  --limit 20000 \
  --output-dir reports/shap_analysis
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `shap_importance.csv` | CSV | Bảng xếp hạng đặc trưng theo giá trị trung bình tuyệt đối Mean Absolute SHAP. |
| `scores.csv` | CSV | Hiệu năng của mô hình nền tảng (XGBoost) được dùng để tính SHAP values. |
| `summary.json` | JSON | Metadata tổng kết số dòng được giải thích và danh sách top feature. |
| `report.md` / `report.html` | Báo cáo | Báo cáo Markdown và HTML tổng kết chi tiết. |

### Các Cột trong `shap_importance.csv`:
- `label`: Tên cột nhãn mục tiêu.
- `task`: Loại bài toán (`classification` hoặc `regression`).
- `feature`: Tên đặc trưng.
- `mean_abs_shap`: Độ lớn tác động trung bình của đặc trưng lên đầu ra mô hình ($|\phi|$).
- `explain_rows`: Số lượng dòng dữ liệu được đưa vào tính toán SHAP.

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Thước đo Đáng Tin cậy Nhất**: Mean Absolute SHAP được cộng đồng Data Science và Quant xem là tiêu chuẩn vàng (Gold Standard) để xếp hạng mức độ quan trọng của đặc trưng trong các mô hình cây.
- **Loại bỏ Biến Không Ảnh hưởng**: Các biến có $\text{mean\_abs\_shap} \approx 0$ hoàn toàn không có tác động nào lên quyết định dự báo của mô hình.\n