# Phân tích Factor Alphalens (`alphalens`)

Module `alphalens` thực hiện phân tích hiệu suất nhân tố (Alpha Factor Tearsheet Analysis) theo tiêu chuẩn định lượng định hướng bởi thư viện Alphalens của Quantopian. Module đánh giá sức mạnh dự báo của các đặc trưng (features/factors) đối với lợi suất tương lai (forward returns) trên nhiều khung thời gian (horizons).

---

## 1. Mục đích & Ứng dụng

- **Alpha Mining & Factor Research**: Đánh giá xem một chỉ báo kỹ thuật hay đặc trưng số có mang lại tín hiệu sinh lời thực sự hay không.
- **Phân tầng Quantile (Quantile Stratification)**: Kiểm tra tính đơn điệu (monotonicity) của lợi nhuận khi chia factor thành $N$ nhóm phân vị (mặc định 5 quantiles).
- **Phân tích Phân rã Tín hiệu (IC Decay)**: Đo lường độ trễ và thời gian suy giảm hiệu quả của tín hiệu theo các mốc thời gian $t+1, t+5, t+15, t+60$.
- **Chiến lược Long-Short Spread**: Ước tính tỷ suất sinh lời và Sharpe ratio của danh mục Long Quantile cao nhất (Q5) và Short Quantile thấp nhất (Q1).

---

## 2. Phương pháp & Nguyên lý Tính toán

### 2.1. Xác định Lợi suất Kỳ hạn (Forward Returns)
- Nếu dataset có cột giá (`Close`, `close_price`, `price`), module tự động tính forward return:
  $$\text{fwd\_ret}_h(t) = \frac{\text{Price}_{t+h} - \text{Price}_t}{\text{Price}_t}$$
  với các horizons mặc định $h \in \{1, 5, 15, 60\}$.
- Nếu không có cột giá, module tự động nhận diện các cột target liên tục trong file `label` để làm forward return mục tiêu.

### 2.2. Các Chỉ số Định lượng Chính
1. **Information Coefficient (IC)**: Hệ số tương quan Rank Spearman giữa giá trị Factor tại thời điểm $t$ và Forward Return tại thời điểm $t+h$.
2. **Information Ratio (IR)**:
   $$\text{IR} = \frac{\text{Mean}(\text{IC})}{\text{Std}(\text{IC})}$$
3. **Monotonicity Score**: Hệ số tương quan Spearman giữa thứ tự Quantile $(1, 2, \dots, 5)$ và Lợi suất trung bình của từng Quantile. Điểm tuyệt đối gần 1.0 thể hiện tính phân tầng lý tưởng.
4. **Long-Short Spread**: Chênh lệch lợi suất kỳ vọng giữa nhóm quantile cao nhất và thấp nhất:
   $$\text{Spread} = \bar{R}_{Q5} - \bar{R}_{Q1}$$
5. **Cumulative Return of Long-Short**: Đường cong lợi nhuận tích lũy giả định tái cân bằng theo chu kỳ.

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy phân tích Alphalens cơ bản
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module alphalens

# Chạy với giới hạn số dòng dữ liệu
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module alphalens --limit 50000

# Chỉ định cột target cụ thể và thư mục lưu kết quả
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module alphalens \
  --target fwd_ret_5 \
  --output-dir reports/alphalens_run
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

Thư mục kết quả (mặc định `reports/alphalens/` hoặc đường dẫn tùy chọn) bao gồm:

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `factor_metrics.csv` | CSV | Bảng tổng hợp chỉ số cho từng feature: Mean IC, IC Std, IR, p-value, Monotonicity, Long-Short Spread. |
| `quantile_returns.csv` | CSV | Lợi suất trung bình theo từng phân vị Quantile (Q1 đến Q5) theo từng horizon. |
| `ic_decay.png` | Biểu đồ | Đồ thị thể hiện mức độ suy giảm IC theo các mốc thời gian horizons. |
| `quantile_returns.png` | Biểu đồ | Biểu đồ cột thể hiện phân bổ lợi suất trung bình qua các nhóm phân vị. |
| `cumulative_spread.png` | Biểu đồ | Đường cong lợi nhuận tích lũy của chiến lược Long Q5 - Short Q1. |
| `summary.json` | JSON | Metadata lần chạy và danh sách Top Factor hiệu quả nhất. |
| `report.md` / `report.html` | Báo cáo | Báo cáo hoàn chỉnh dạng Markdown và HTML tương tác. |

### Các Cột trong `factor_metrics.csv`:
- `factor`: Tên đặc trưng/nhân tố.
- `horizon`: Khung thời gian dự báo ($h=1, 5, 15, 60$).
- `mean_ic`: IC trung bình (Spearman Rank IC).
- `ic_std`: Độ lệch chuẩn của chuỗi IC.
- `information_ratio`: Tỷ số thông tin $\text{IR} = \text{mean\_ic} / \text{ic\_std}$.
- `p_value`: Mức ý nghĩa thống kê của kiểm định IC khác 0.
- `positive_ic_ratio`: Tỷ lệ phần trăm chu kỳ mà IC mang dấu dương ($>0$).
- `monotonicity_score`: Độ phân tầng tuyến tính giữa các quantiles.
- `long_short_spread`: Lợi suất chênh lệch giữa Quantile cao nhất và thấp nhất.

---

## 5. Hướng dẫn Đọc hiểu & Phân tích Chỉ số

- **Đánh giá Mean IC**:
  - $| \text{Mean IC} | \ge 0.05$: Tín hiệu alpha rất mạnh trong giao dịch tần suất trung/cao.
  - $0.02 \le | \text{Mean IC} | < 0.05$: Tín hiệu tốt, có thể đưa vào mô hình tổng hợp (composite alpha).
  - $| \text{Mean IC} | < 0.01$: Tín hiệu yếu hoặc nhiễu ngẫu nhiên.
- **Đánh giá Information Ratio (IR)**:
  - $\text{IR} \ge 0.5$: Tín hiệu ổn định cao qua thời gian.
  - $\text{IR} \ge 1.0$: Alpha đặc biệt xuất sắc.
- **Đánh giá Monotonicity**:
  - Điểm gần $+1.0$: Giá trị factor càng cao, lợi nhuận càng lớn (Thuận).
  - Điểm gần $-1.0$: Giá trị factor càng cao, lợi nhuận càng âm (Nghịch - có thể đảo ngược tín hiệu).
  - Điểm quanh $0.0$: Mối quan hệ phi tuyến phức tạp hoặc không có tính phân tách rõ ràng.\n