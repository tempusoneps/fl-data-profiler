# Phân tích Xác suất Tín hiệu Rời rạc & Chẩn đoán Bẫy Thị trường (`signal_analysis`)

Module `signal_analysis` cung cấp công cụ định lượng chuyên sâu dành riêng cho các biến tín hiệu giao dịch (Trading Signals từ thư viện `signalx`). Module đánh giá phân phối xác suất có điều kiện rời rạc của từng trạng thái tín hiệu (`buy`, `sell`, `hold`, `none`), đo lường độ nâng xác suất (Lift), khoảng tin cậy Bayes 95%, và chẩn đoán phân rã dạng sai số thị trường (Whipsaw & Reversal Trap Diagnosis).

---

## 1. Mục đích & Ứng dụng

- **Ma trận Xác suất Có điều kiện Rời rạc (Discrete Probability Matrix)**: Đo lường chính xác $P(\text{Target} = \text{Class} \mid \text{Signal} = \text{State})$ cho mọi trạng thái và mọi lớp nhãn (kể cả nhãn đa lớp như `allow_entry`, `price_shape` hay nhãn 2 lớp như `direction_filter`).
- **Độ Nâng Xác suất & Khoảng Tin cậy Bayes (Lift & 95% Bayesian CI)**: Tính toán chỉ số Lift khuếch đại xác suất và áp dụng phân phối Beta-Binomial (tiên nghiệm Jeffreys) để loại bỏ bẫy ảo tưởng xác suất trên các tín hiệu kích hoạt quá ít mẫu.
- **Chẩn đoán Bẫy Sideway & Bẫy Đảo chiều (Whipsaw & Reversal Trap Diagnosis)**: Phân rã 100% các lần phát tín hiệu thành:
  - `% True Alpha (Win)`: Bắt trúng sóng xu hướng mục tiêu.
  - `% Sideway Trap (Whipsaw)`: Bị kẹt trong vùng tích lũy/sideway, tốn phí giao dịch.
  - `% Reversal Trap (Counter-trend)`: Bị dính cú lừa đảo chiều ngược hướng cực kỳ nguy hiểm.
- **Xếp hạng Lợi thế Thực (Clean Directional Edge)**: Đo lường $\text{Clean Edge} = \text{True Alpha \%} - \text{Reversal Trap \%}$ và Tỷ số Rủi ro Ngược chiều ($\text{Adverse Risk Ratio}$) để tìm ra các tín hiệu đáng tin cậy nhất.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Chuẩn hóa 4 Trạng thái Tín hiệu (Canonical States)**:
   - Toàn bộ giá trị tín hiệu dạng chuỗi (`"buy"`, `"sell"`, `"hold"`, `"none"`) hoặc số (`1`, `-1`, `0`) được chuẩn hóa về 4 trạng thái chuẩn.
2. **Ma trận Xác suất Rời rạc & Thông tin Phân tách (Probability & WoE/IV)**:
   - Với mỗi cặp `(signal, state)` và `target_class`:
     $$\text{Conditional Prob} = \frac{N_{s, C}}{N_s}, \quad \text{Lift} = \frac{\text{Conditional Prob}}{P(C)}$$
   - Khoảng tin cậy Bayes 95%: Sử dụng `scipy.stats.beta.ppf` với tham số $(\alpha = N_{s, C} + 0.5, \beta = N_s - N_{s, C} + 0.5)$.
   - Tính toán Weight of Evidence (WoE) và đóng góp Information Value (IV).
3. **Phân rã Hành vi & Chẩn đoán Bẫy Thị trường**:
   - Khi tín hiệu kích hoạt `buy` hoặc `sell`, ánh xạ kết quả thực tế của nến thành 3 kịch bản:
     - **True Alpha**: Đúng chiều mục tiêu (`Yes - Buy` cho lệnh Buy, `Yes - Sell` cho lệnh Sell).
     - **Sideway Trap**: Rơi vào vùng nén dao động (`No - Sideway`, `narrow_range`).
     - **Reversal Trap**: Rơi vào sóng đảo chiều ngược hướng (`Yes - Sell` cho lệnh Buy, `Yes - Buy` cho lệnh Sell).
   - $\text{Adverse Risk Ratio} = \frac{\text{Reversal Trap \%}}{\text{True Alpha \%}}$ (Tỷ số $> 1.0$ cảnh báo tín hiệu độc hại thường xuyên dẫn đến đợt sập ngược hướng).

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy phân tích tín hiệu trên datasets/signal.csv và datasets/label.csv
fldataprofiler fit datasets/signal.csv datasets/label.csv --module signal_analysis

# Chỉ định nhãn mục tiêu cụ thể
fldataprofiler fit datasets/signal.csv datasets/label.csv --module signal_analysis --target allow_entry

# Xuất báo cáo ra thư mục riêng
fldataprofiler fit datasets/signal.csv datasets/label.csv \
  --module signal_analysis \
  --output-dir reports/signal_analysis_report
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `signal_probability_matrix.csv` | CSV | Ma trận đầy đủ xác suất có điều kiện, Lift, 95% Bayesian CI, WoE và IV cho từng cặp `(signal, state, target_class)`. |
| `signal_trap_diagnosis.csv` | CSV | Bảng chẩn đoán bẫy thị trường: True Alpha %, Sideway Trap %, Reversal Trap %, Clean Edge, Adverse Risk Ratio. |
| `top_clean_signals.csv` | CSV | Bảng lọc các tín hiệu sạch nhất có số lần kích hoạt đủ lớn ($\ge 20$) sắp xếp theo Clean Edge giảm dần. |
| `signal_trap_distribution.png` | Biểu đồ | Biểu đồ thanh ngang xếp chồng phân bố True Alpha vs Bẫy Sideway vs Bẫy Đảo chiều của Top 15 tín hiệu. |
| `top_signal_probabilities.png` | Biểu đồ | Biểu đồ thanh ngang thể hiện xác suất có điều kiện cao nhất kèm thanh sai số khoảng tin cậy Bayes 95%. |
| `summary.json` | JSON | Metadata, Top Clean Signals, Top Whipsaw Signals, Highest Reversal Risk Signals. |
| `report.md` / `report.html` | Báo cáo | Báo cáo chi tiết định dạng Markdown và giao diện Web HTML tương tác. |

### Các Cột trong `signal_trap_diagnosis.csv`:
- `signal_name`: Tên cột tín hiệu (ví dụ: `STA007_signal`).
- `signal_state`: Trạng thái kích hoạt (`buy` hoặc `sell`).
- `trigger_count`: Tổng số lần trạng thái này xuất hiện.
- `trigger_pct`: Tỷ lệ phần trăm xuất hiện trên tổng số nến.
- `true_alpha_pct`: Tỷ lệ % nến bắt trúng sóng xu hướng mục tiêu.
- `sideway_trap_pct`: Tỷ lệ % nến bị kẹt trong bẫy đi ngang sideway.
- `reversal_trap_pct`: Tỷ lệ % nến bị lừa vào sóng đảo chiều ngược hướng.
- `vol_trap_pct`: Tỷ lệ % nến rơi vào trạng thái cấm vào lệnh do biến động rủi ro cao.
- `clean_edge`: Lợi thế định hướng sạch ($\text{True Alpha \%} - \text{Reversal Trap \%}$).
- `adverse_risk_ratio`: Tỷ số rủi ro ngược chiều ($\text{Reversal Trap} / \text{True Alpha}$).

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị Thực chiến

1. **Lựa chọn Tín hiệu Vào lệnh (Trigger Signals)**:
   - Ưu tiên các tín hiệu có **`clean_edge > +10.0%`** và **`adverse_risk_ratio < 0.70`** (như `STA007_signal`, `VOL020_signal`, `VOL009_signal`).
2. **Cảnh báo Tín hiệu Độc hại (Toxic Falling Knives)**:
   - Tuyệt đối loại bỏ hoặc đảo ngược các tín hiệu có **`adverse_risk_ratio > 1.30`** (như `MOM017_signal`, `MOM027_signal`, `MOM004_signal` khi bắt đáy). Các tín hiệu này tạo ra số lần sập ngược chiều nhiều hơn hẳn số lần thắng.
3. **Kết hợp Bộ lọc Xu hướng cho Tín hiệu Bị Bẫy Sideway**:
   - Những tín hiệu có **`sideway_trap_pct > 25%`** (như `VLM008_signal`) bắt buộc phải kết hợp thêm bộ lọc xu hướng vĩ mô (như `TRD014_signal` trên EMA200 hoặc Choppiness Index) để triệt tiêu nhiễu sideway trước khi kích hoạt lệnh.
