# Mô hình Kinh tế Lượng & Hồi quy OLS Statsmodels (`statsmodels`)

Module `statsmodels` áp dụng các mô hình kinh tế lượng chính thống (Ordinary Least Squares - OLS và Generalized Linear Models) từ thư viện `statsmodels` để phân tích các hệ số tác động, khoảng tin cậy 95%, và kiểm định đa biến giữa features và labels.

---

## 1. Mục đích & Ứng dụng

- **Phân tích Tác động Độc lập (Multivariate Attribution)**: Đo lường tác động biên của một feature sau khi đã kiểm soát ảnh hưởng của các feature khác.
- **Độ tin cậy của Hệ số (Confidence Intervals & t-stats)**: Cung cấp khoảng tin cậy $[2.5\%, 97.5\%]$ cho từng trọng số hồi quy.
- **Chỉ số Đánh giá Mô hình (AIC, BIC, Adjusted $R^2$)**: So sánh mức độ phù hợp và mức độ phạt độ phức tạp của mô hình để phòng tránh overfitting.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Lọc Đặc trưng Tiền xử lý (Feature Pre-selection)**:
   - Sàng lọc top $K$ features có tương quan cao nhất với target để xây dựng mô hình OLS tránh hiện tượng ma trận kỳ dị (singular matrix).
2. **Ước lượng OLS**:
   $$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k + \epsilon$$
3. **Thống kê Đánh giá**:
   - $R^2$ và Adjusted $R^2$: Tỷ lệ phương sai của biến mục tiêu được giải thích bởi mô hình.
   - $F$-statistic & $p$-value: Đánh giá xem toàn bộ tập feature có cùng bằng 0 hay không.
   - $t$-statistic & $p$-value cho từng $\beta_i$: Đánh giá mức độ đóng góp riêng rẽ của từng biến.
   - Akaike Information Criterion (AIC) và Bayesian Information Criterion (BIC).

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy phân tích mô hình kinh tế lượng statsmodels
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module statsmodels

# Chạy cho nhãn cụ thể
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module statsmodels --target allow_entry

# Lưu kết quả vào thư mục riêng
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module statsmodels --output-dir reports/statsmodels_run
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `scores.csv` | CSV | Bảng đánh giá tổng thể mô hình cho từng target: $R^2$, Adj $R^2$, F-stat, AIC, BIC. |
| `coefficients.csv` | CSV | Bảng chi tiết hệ số hồi quy $\beta$, sai số chuẩn (std err), $t$-stat, $p$-value, và khoảng tin cậy 95%. |
| `summary.json` | JSON | Metadata tổng hợp kết quả mô hình và danh sách biến quan trọng. |
| `report.md` / `report.html` | Báo cáo | Báo cáo Markdown và HTML tổng kết chi tiết. |

### Các Cột trong `coefficients.csv`:
- `label`: Tên cột nhãn mục tiêu.
- `feature`: Tên đặc trưng đầu vào (kèm hằng số `const`).
- `coef`: Hệ số hồi quy ước lượng $\hat{\beta}$.
- `std_err`: Sai số chuẩn của hệ số ước lượng.
- `t_statistic`: Giá trị thống kê $t = \hat{\beta} / \text{std\_err}$.
- `p_value`: Mức ý nghĩa thống kê của hệ số ($p < 0.05$ thể hiện biến có tác động thực sự).
- `ci_lower`: Cận dưới khoảng tin cậy 95% ($2.5\%$).
- `ci_upper`: Cận trên khoảng tin cậy 95% ($97.5\%$).

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Khoảng tin cậy không chứa giá trị 0**: Nếu khoảng $[\text{ci\_lower}, \text{ci\_upper}]$ nằm hoàn toàn về phía dương $(>0)$ hoặc âm $(<0)$, biến đó tác động vững chắc lên nhãn mục tiêu.
- **Hiện tượng Đa cộng tuyến (Multicollinearity)**: Nếu $R^2$ của mô hình cao nhưng $p$-value của các biến đơn lẻ đều lớn ($>0.05$), các features đang bị trùng lặp thông tin lẫn nhau. Cần loại bỏ các feature tương quan cao bằng module `mrmr`.\n