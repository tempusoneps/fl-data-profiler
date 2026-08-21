# Kelly Criterion & Position Sizing Profiling (`probability_kellycriterion`)

Module `probability_kellycriterion` đo lường sức mạnh dự báo của từng đặc trưng dưới lăng kính **Tối ưu hóa Phân bổ Vốn & Kỳ vọng Lợi nhuận (Kelly Criterion & Expected Value)** qua 20 phân vị (quantiles). Module giúp xác định chính xác các vùng giá trị mang lại lợi thế toán học (Positive Edge), tính toán tỷ lệ đi tiền tối ưu (Full Kelly, Half-Kelly) và mức kỳ vọng lợi nhuận ($EV$) trên từng lệnh.

---

## 1. Nguyên lý Toán học & Các Chỉ số Đo lường

### 1.1. Tỷ lệ Thắng Có điều kiện (Conditional Win Probability)
Dữ liệu đặc trưng được chia thành 20 phân vị đều mẫu $\text{Bin}_k$ ($k=1 \dots 20$). Với mỗi nhãn mục tiêu (Class $c$):
$$p_k = P(Y = c \mid \text{Bin}_k) = \frac{\sum_{i \in \text{Bin}_k} \mathbb{I}(Y_i = c)}{N_k}, \quad q_k = 1 - p_k$$

### 1.2. Xác suất Hòa vốn (Breakeven Win Rate)
Với tỷ lệ Win/Loss Payoff giả định $b = \frac{\text{Lợi nhuận trung bình}}{\text{Thua lỗ trung bình}}$ (Mặc định $b = 1.5$, tức $R:R = 1.5:1$):
$$p_{\text{breakeven}} = \frac{1}{1 + b} = \frac{1}{1 + 1.5} = 40\%$$

### 1.3. Kỳ vọng Lợi nhuận Toán học (Expected Value - EV)
Kỳ vọng lợi nhuận trên mỗi đơn vị rủi ro ($1R$) tại từng bin:
$$\text{EV}_k = p_k \cdot b - q_k \cdot 1.0 = (1 + b) p_k - 1$$
* $\text{EV}_k > 0$: Vùng có lợi thế toán học dương (Lãi kỳ vọng).
* $\text{EV}_k \le 0$: Vùng kỳ vọng âm (Chắc chắn thua lỗ về dài hạn nếu vào lệnh).

### 1.4. Công thức Kelly Criterion & Phân bổ Vốn Phân số (Fractional Kelly)
* **Full Kelly Fraction ($f^*$ - Tỷ lệ vốn tối ưu tối đa hóa tốc độ tăng trưởng vốn kép)**:
  $$f^*_k = \frac{p_k \cdot b - q_k}{b} = p_k - \frac{1 - p_k}{b} = \frac{\text{EV}_k}{b}$$
* **Half-Kelly (Khuyên dùng trong Thực chiến)**:
  $$f^*_{\text{half}, k} = \max\left(0, \frac{1}{2} f^*_k\right)$$
  * *Lý do*: Full Kelly tối đa hóa lợi nhuận lý thuyết nhưng có mức sụt giảm tài khoản (Drawdown) rất sâu khi gặp chuỗi biến động. Half-Kelly giúp giữ $75\%$ tốc độ tăng trưởng vốn nhưng giảm hơn $50\%$ mức độ biến động và rủi ro sụt giảm tài khoản.
* **Quarter-Kelly**:
  $$f^*_{\text{quarter}, k} = \max\left(0, \frac{1}{4} f^*_k\right)$$

### 1.5. Tốc độ Tăng trưởng Vốn Kỳ vọng (Expected Log Growth Rate)
$$g_k = p_k \ln(1 + b \cdot f^*_k) + q_k \ln(1 - f^*_k) \quad (\text{với } f^*_k > 0)$$

### 1.6. Khuyến nghị Hành động Thực chiến (Action Recommendation)
Dựa trên giá trị $f^*_k$ tại từng bin:
* `STRONG_BET`: $f^*_k \ge 15\%$ (Lợi thế cực lớn, tăng size lệnh tối đa).
* `MODERATE_BET`: $5\% \le f^*_k < 15\%$ (Lợi thế tốt, vào lệnh size chuẩn).
* `SMALL_BET`: $0\% < f^*_k < 5\%$ (Lợi thế nhỏ, thăm dò).
* `AVOID_NO_BET`: $f^*_k \le 0\%$ (Không có edge, cấm tuyệt đối vào lệnh).

---

## 2. Hướng dẫn Sử dụng CLI

```bash
# Chạy phân tích Kelly Criterion trên tất cả các nhãn (R:R mặc định 1.5:1)
uv run fldataprofiler fit datasets/VN30F1M_5m.csv datasets/label.csv --module probability_kellycriterion

# Chạy trên nhãn mục tiêu cụ thể không subsampling
uv run fldataprofiler fit datasets/VN30F1M_5m.csv datasets/label.csv \
  --module probability_kellycriterion \
  --target allow_entry \
  --full
```

---

## 3. Danh sách Kết quả Đầu ra (Artifacts)

Báo cáo và số liệu thống kê được lưu tại `reports/probability_kellycriterion/`:

1. `kelly_probability_scores.csv`: Bảng xếp hạng feature theo $\text{Kelly Rank Score}$, $\text{Max Kelly } f^*$, $\text{Max Half-Kelly}$, $\text{Max EV}$, $\text{Kelly Spread}$, số lượng bin có lợi thế dương.
2. `quantile_kelly_probabilities.csv`: Bảng chi tiết 20 bins cho từng feature gồm xác suất thắng, tỷ lệ Kelly $f^*$, Half-Kelly, Quarter-Kelly, kỳ vọng lợi nhuận $EV$, và khuyến nghị hành động (`STRONG_BET`, `MODERATE_BET`, `AVOID_NO_BET`).
3. `kelly_distribution.png`: Biểu đồ trực quan kết hợp cột tỷ lệ đi tiền Kelly, đường Half-Kelly, đường xác suất thắng và đường chuẩn hòa vốn (Breakeven Line).
4. `summary.json`: Metadata tổng hợp và top feature theo Kelly Rank Score.
5. `report.md`: Báo cáo tổng quan dạng Markdown.
6. `report.html`: Báo cáo HTML tương tác.
