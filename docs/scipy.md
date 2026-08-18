# Kiểm định Thống kê Giả thuyết SciPy (`scipy`)

Module `scipy` áp dụng các phương pháp kiểm định thống kê toán học phi tham số và tham số từ thư viện SciPy để xác định mức ý nghĩa thống kê ($p$-value) và kích thước hiệu ứng (effect size) giữa các đặc trưng và nhãn mục tiêu.

---

## 1. Mục đích & Ứng dụng

- **Kiểm định Nghiêm ngặt (Hypothesis Testing)**: Đảm bảo mối liên hệ giữa feature và label không phải do may rủi hay ngẫu nhiên ($p < 0.05$).
- **Phù hợp Mọi Kiểu Dữ liệu**:
  - Biến liên tục vs Biến liên tục: Pearson, Spearman, Kendall's Tau.
  - Biến liên tục vs Biến nhị phân/phân loại: Two-sample t-test, Mann-Whitney U, ANOVA F-test, Kruskal-Wallis.
  - Biến phân loại vs Biến phân loại: Chi-Square Independence Test, Cramer's V.
- **Kiểm định Tương tác Cặp Đôi (2-Feature Linear Test)**: Đánh giá khả năng giải thích biến mục tiêu khi kết hợp 2 features trong mô hình hồi quy tuyến tính.

---

## 2. Phương pháp & Nguyên lý Tính toán

Module tự động phát hiện kiểu dữ liệu của cặp $(X, Y)$ và áp dụng bài kiểm định tương ứng:

| Loại Cặp $(X, Y)$ | Phép Kiểm định Áp dụng | Chỉ số Thống kê | Kích thước Hiệu ứng (Effect Size) |
| :--- | :--- | :--- | :--- |
| **Continuous vs Continuous** | Pearson / Spearman Rank / Kendall Tau | $r$, $\rho$, $\tau$ | $|r|$ hoặc $|\rho|$ |
| **Continuous vs Binary Label** | Two-sample t-test & Mann-Whitney U | $t$-statistic / $U$ | Cohen's $d$ |
| **Continuous vs Multi-class** | One-way ANOVA & Kruskal-Wallis | $F$-statistic / $H$ | $\eta^2$ (Eta-squared) |
| **Categorical vs Categorical** | Chi-Square Test of Independence | $\chi^2$-statistic | Cramer's $V$ |

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy kiểm định thống kê SciPy
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module scipy

# Chỉ định nhãn mục tiêu và giới hạn mẫu
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module scipy --target allow_entry --limit 30000

# Chỉ định thư mục xuất kết quả
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module scipy --output-dir reports/scipy_tests
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `pairwise.csv` | CSV | Bảng kết quả kiểm định thống kê chi tiết cho từng cặp (Feature, Label). |
| `two_feature.csv` | CSV | Bảng kiểm định mô hình kết hợp 2 features dự đoán nhãn mục tiêu. |
| `summary.json` | JSON | Metadata tổng hợp số lượng kiểm định đạt mức ý nghĩa thống kê $p < 0.05$. |
| `report.md` / `report.html` | Báo cáo | Báo cáo Markdown và HTML tổng kết các phát hiện quan trọng. |

### Các Cột trong `pairwise.csv`:
- `feature`: Tên đặc trưng.
- `label`: Tên nhãn.
- `test`: Tên bài kiểm định được thực hiện (ví dụ: `spearman`, `mann_whitney_u`, `chi_square`, `anova`).
- `statistic`: Giá trị thống kê tính toán ($t, F, \chi^2, \rho$).
- `p_value`: Mức xác suất p-value.
- `effect_size`: Kích thước tác động chuẩn hóa ($[0, 1]$ hoặc Cohen's $d$).
- `effect_name`: Tên thước đo effect size (ví dụ: `cohens_d`, `cramers_v`, `abs_corr`).
- `samples`: Số lượng mẫu tham gia kiểm định.
- `feature_type` / `label_type`: Kiểu dữ liệu nhận diện (`continuous`, `binary`, `categorical`).

---

## 5. Hướng dẫn Đọc hiểu & Phân tích Chỉ số

- **Mức ý nghĩa Thống kê ($p$-value)**:
  - $p < 0.001$: Mối quan hệ có ý nghĩa thống kê cực kỳ rõ nét.
  - $p < 0.05$: Đạt chuẩn bác bỏ giả thuyết vô hiệu $H_0$.
  - $p \ge 0.05$: Chưa đủ bằng chứng chứng minh sự liên hệ giữa feature và label.
- **Kích thước Hiệu ứng (Cohen's $d$ / Cramer's $V$)**:
  - Cohen's $d \ge 0.2$: Hiệu ứng nhỏ.
  - Cohen's $d \ge 0.5$: Hiệu ứng trung bình.
  - Cohen's $d \ge 0.8$: Hiệu ứng lớn, tính phân tách nhãn rất mạnh.\n