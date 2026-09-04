# Phân tích Xác suất Tín hiệu Rời rạc, Chẩn đoán Bẫy Thị trường & Độ Ổn định Nhiều Năm (`signal_analysis`)

Module `signal_analysis` cung cấp công cụ định lượng chuyên sâu dành riêng cho các biến tín hiệu giao dịch (Trading Signals từ thư viện `signalx`). Module đánh giá phân phối xác suất có điều kiện rời rạc của từng trạng thái tín hiệu (`buy`, `sell`, `hold`, `none`), đo lường độ nâng xác suất (Lift), khoảng tin cậy Bayes 95%, chẩn đoán phân rã dạng sai số thị trường (Whipsaw & Reversal Trap Diagnosis), và kiểm định độ ổn định lợi thế định hướng qua từng năm dương lịch (Multi-Year Stability & Consistency Analysis).

---

## 1. Mục đích & Ứng dụng

- **Ma trận Xác suất Có điều kiện Rời rạc (Discrete Probability Matrix)**: Đo lường chính xác $P(\text{Target} = \text{Class} \mid \text{Signal} = \text{State})$ cho mọi trạng thái và mọi lớp nhãn (kể cả nhãn đa lớp như `allow_entry`, `price_shape` hay nhãn 2 lớp như `direction_filter`).
- **Độ Nâng Xác suất & Khoảng Tin cậy Bayes (Lift & 95% Bayesian CI)**: Tính toán chỉ số Lift khuếch đại xác suất và áp dụng phân phối Beta-Binomial (tiên nghiệm Jeffreys) để loại bỏ bẫy ảo tưởng xác suất trên các tín hiệu kích hoạt quá ít mẫu.
- **Chẩn đoán Bẫy Sideway & Bẫy Đảo chiều (Whipsaw & Reversal Trap Diagnosis)**: Phân rã 100% các lần phát tín hiệu thành:
  - `% True Alpha (Win)`: Bắt trúng sóng xu hướng mục tiêu.
  - `% Sideway Trap (Whipsaw)`: Bị kẹt trong vùng tích lũy/sideway, tốn phí giao dịch.
  - `% Reversal Trap (Counter-trend)`: Bị dính cú lừa đảo chiều ngược hướng cực kỳ nguy hiểm.
  - `% Volatility / Lockout Trap`: Rơi vào vùng cấm vào lệnh do biến động lớn.
- **Xếp hạng Lợi thế Thực (Clean Directional Edge)**: Đo lường $\text{Clean Edge} = \text{True Alpha \%} - \text{Reversal Trap \%}$ và Tỷ số Rủi ro Ngược chiều ($\text{Adverse Risk Ratio}$) để tìm ra các tín hiệu đáng tin cậy nhất.
- **Phân tích Độ Ổn định & Tính Nhất quán Nhiều Năm (Multi-Year Stability & Consistency)**: Bóc tách Clean Edge qua từng năm dương lịch (calendar years), đo lường tỷ lệ nhất quán ($\text{Consistency \%}$), lợi thế trung bình ($\text{Mean Clean Edge}$), rủi ro kịch bản xấu nhất ($\text{Worst Year Edge}$), độ biến động lợi thế ($\text{Edge Volatility}$) và phân hạng mức độ ổn định (`STABLE_ALPHA`, `MODERATE_STABLE`, `ERRATIC / UNSTABLE`) nhằm loại trừ triệt để bẫy "siêu sao một mùa" (one-hit wonders).

---

## 2. Phương pháp & Nguyên lý Tính toán

### 2.1. Chuẩn hóa 4 Trạng thái Tín hiệu (Canonical States)
Toàn bộ giá trị tín hiệu dạng chuỗi (`"buy"`, `"sell"`, `"hold"`, `"none"`) hoặc số (`1`, `-1`, `0`) được chuẩn hóa tự động về 4 trạng thái chuẩn:
- `buy`: Lệnh mua / vị thế Long.
- `sell`: Lệnh bán / vị thế Short.
- `hold`: Trạng thái nắm giữ / trung lập.
- `none`: Không có hành động / nằm ngoài thị trường.

### 2.2. Ma trận Xác suất Rời rạc & Thông tin Phân tách (Probability & WoE/IV)
Với mỗi cặp `(signal, state)` và `target_class`:
$$\text{Conditional Prob} = \frac{N_{s, C}}{N_s}, \quad \text{Lift} = \frac{\text{Conditional Prob}}{P(C)}$$
- **Khoảng tin cậy Bayes 95%**: Sử dụng phân phối Beta-Binomial với tiên nghiệm Jeffreys (`scipy.stats.beta.ppf`):
  $$\alpha = N_{s, C} + 0.5, \quad \beta = N_s - N_{s, C} + 0.5$$
- Tính toán Weight of Evidence (WoE) và đóng góp Information Value (IV) cho từng trạng thái rời rạc.

### 2.3. Phân rã Hành vi & Chẩn đoán Bẫy Thị trường
Khi tín hiệu kích hoạt `buy` hoặc `sell`, ánh xạ kết quả thực tế của nến thành các kịch bản:
- **True Alpha**: Đúng chiều mục tiêu (`Yes - Buy` cho lệnh Buy, `Yes - Sell` cho lệnh Sell).
- **Sideway Trap**: Rơi vào vùng nén dao động (`No - Sideway`, `narrow_range`, `flat`).
- **Reversal Trap**: Rơi vào sóng đảo chiều ngược hướng (`Yes - Sell` cho lệnh Buy, `Yes - Buy` cho lệnh Sell).
- **Volatility / Lockout Trap**: Rơi vào vùng rủi ro cực đoan (`none`, `lockout`, `skip`).

Chỉ số rủi ro ngược chiều:
$$\text{Adverse Risk Ratio} = \frac{\text{Reversal Trap \%}}{\text{True Alpha \%}}$$
Tỷ số $> 1.0$ cảnh báo tín hiệu độc hại thường xuyên dẫn đến đợt cắt lỗ ngược chiều nặng nề.

### 2.4. Phân tích Độ Ổn định & Tính Nhất quán Nhiều Năm (Multi-Year Stability & Consistency)
Một tín hiệu có Clean Edge tổng thể cao có thể chỉ do ăn may trong một năm bùng nổ duy nhất nhưng lại lỗ trong toàn bộ các năm còn lại. Để loại bỏ hiện tượng này, module phân tích chuỗi thời gian theo từng năm dương lịch:

1. **Phân tách Dữ liệu theo Năm Dương lịch**:
   Dựa vào Datetime Index hoặc cột thời gian (`Date`, `timestamp`, `time`), dữ liệu được nhóm theo từng năm dương lịch $\mathcal{Y} = \{y_1, y_2, \dots, y_M\}$.
2. **Ngưỡng Dữ liệu Tối thiểu (Insufficient Data Threshold)**:
   Để đảm bảo ý nghĩa thống kê, mỗi năm yêu cầu tối thiểu $N_{\text{min}} = 20$ lần kích hoạt tín hiệu (`trigger_count >= 20`):
   - Nếu $\text{trigger\_count} \ge 20$: Trạng thái là `valid`. Các chỉ số `clean_edge` và `adverse_risk_ratio` được tính toán bình thường.
   - Nếu $\text{trigger\_count} < 20$: Trạng thái là `insufficient_data`. Giá trị `clean_edge` và `adverse_risk_ratio` được đặt thành `None` (hiển thị `N/A (<20)` trên biểu đồ và báo cáo), đồng thời **loại trừ** khỏi tập các năm đánh giá độ ổn định $\mathcal{Y}_{\text{eval}}$.
3. **Lợi thế Sạch theo Năm (Yearly Clean Edge)**:
   Với mỗi năm $y \in \mathcal{Y}_{\text{eval}}$:
   $$E_y = \text{True Alpha \%}_y - \text{Reversal Trap \%}_y$$
4. **Tỷ số Nhất quán (Consistency %)**:
   Tỷ lệ phần trăm số năm giữ được lợi thế sạch dương ($E_y > 0$) trên tổng số năm đủ dữ liệu đánh giá:
   $$\text{Consistency \%} = \frac{\sum_{y \in \mathcal{Y}_{\text{eval}}} \mathbb{I}(E_y > 0)}{|\mathcal{Y}_{\text{eval}}|} \times 100\%$$
5. **Lợi thế Trung bình, Kịch bản Xấu nhất & Tốt nhất**:
   $$\overline{E} = \frac{1}{|\mathcal{Y}_{\text{eval}}|} \sum_{y \in \mathcal{Y}_{\text{eval}}} E_y$$
   $$E_{\text{worst}} = \min_{y \in \mathcal{Y}_{\text{eval}}} E_y, \quad E_{\text{best}} = \max_{y \in \mathcal{Y}_{\text{eval}}} E_y$$
6. **Độ Biến động Lợi thế (Edge Volatility)**:
   Đo lường độ biến động phân tán của Clean Edge qua các năm bằng độ lệch chuẩn hiệu chỉnh mẫu ($|\mathcal{Y}_{\text{eval}}| > 1$):
   $$\sigma_E = \sqrt{\frac{1}{|\mathcal{Y}_{\text{eval}}| - 1} \sum_{y \in \mathcal{Y}_{\text{eval}}} (E_y - \overline{E})^2}$$
7. **Hệ thống Phân hạng Độ Ổn định (Stability Grades)**:
   - 🟢 **`STABLE_ALPHA`**: Khi $|\mathcal{Y}_{\text{eval}}| > 0$, $\text{Consistency} \ge 75\%$ và $\overline{E} > 0\%$. Tín hiệu giữ vững lợi thế dương trong ít nhất 3/4 số năm và có trung bình lợi thế dương. Đây là các tín hiệu có tính bền bỉ vượt trội qua các chu kỳ thị trường.
   - 🟡 **`MODERATE_STABLE`**: Khi $|\mathcal{Y}_{\text{eval}}| > 0$, $\text{Consistency} \ge 50\%$ và $\overline{E} > 0\%$. Tín hiệu giữ được lợi thế dương trong ít nhất 1/2 số năm và có trung bình lợi thế dương.
   - 🔴 **`ERRATIC / UNSTABLE`**: Các trường hợp còn lại ($\text{Consistency} < 50\%$, hoặc $\overline{E} \le 0\%$, hoặc không có năm nào đủ mẫu $|\mathcal{Y}_{\text{eval}}| = 0$). Tín hiệu biến động thất thường, có nguy cơ cao là kết quả ngẫu nhiên hoặc bẫy overfit.

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
| `signal_trap_diagnosis.csv` | CSV | Bảng chẩn đoán bẫy thị trường tổng hợp: True Alpha %, Sideway Trap %, Reversal Trap %, Clean Edge, Adverse Risk Ratio toàn thời gian. |
| `top_clean_signals.csv` | CSV | Bảng lọc các tín hiệu sạch nhất có số lần kích hoạt $\ge 20$, xếp theo Clean Edge giảm dần, bổ sung trực tiếp `consistency_pct` và `worst_year_edge`. |
| `signal_yearly_stability.csv` | CSV | Bảng phân rã chi tiết hiệu suất theo từng năm dương lịch cho từng cặp `(signal, state, year)` kèm số lượng kích hoạt và cờ trạng thái dữ liệu. |
| `signal_stability_ranking.csv` | CSV | Bảng xếp hạng độ ổn định đa năm tổng hợp: Overall Clean Edge, số năm đánh giá, số năm dương, Consistency %, Mean Edge, Worst/Best Edge, Edge Volatility và Stability Grade. |
| `signal_yearly_stability.png` | Biểu đồ | Ma trận nhiệt (Heatmap Matrix) thể hiện Clean Edge qua từng năm của Top tín hiệu sạch, sử dụng dải màu phân kỳ RdYlGn và đánh dấu `N/A (<20)` cho năm thiếu mẫu. |
| `signal_trap_distribution.png` | Biểu đồ | Biểu đồ thanh ngang xếp chồng phân bố True Alpha vs Bẫy Sideway vs Bẫy Đảo chiều của Top 15 tín hiệu. |
| `top_signal_probabilities.png` | Biểu đồ | Biểu đồ thanh ngang thể hiện xác suất có điều kiện cao nhất kèm thanh sai số khoảng tin cậy Bayes 95%. |
| `summary.json` | JSON | Metadata, Top Clean Signals, Top Stable Signals, Years Analyzed, Yearly Stability Records, Top Whipsaw Signals, Highest Reversal Risk Signals. |
| `report.md` / `report.html` | Báo cáo | Báo cáo chi tiết định dạng Markdown và giao diện Web HTML tương tác tích hợp đầy đủ bảng xếp hạng độ ổn định nhiều năm. |

### 4.1. Các Cột trong `signal_trap_diagnosis.csv`:
- `signal_name`: Tên cột tín hiệu (ví dụ: `STA007_signal`).
- `signal_state`: Trạng thái kích hoạt (`buy` hoặc `sell`).
- `trigger_count`: Tổng số lần trạng thái này xuất hiện.
- `trigger_pct`: Tỷ lệ phần trăm xuất hiện trên tổng số nến.
- `true_alpha_pct`: Tỷ lệ % nến bắt trúng sóng xu hướng mục tiêu.
- `sideway_trap_pct`: Tỷ lệ % nến bị kẹt trong bẫy đi ngang sideway.
- `reversal_trap_pct`: Tỷ lệ % nến bị lừa vào sóng đảo chiều ngược hướng.
- `vol_trap_pct`: Tỷ lệ % nến rơi vào trạng thái cấm vào lệnh do biến động rủi ro cao.
- `clean_edge`: Lợi thế định hướng sạch toàn thời gian ($\text{True Alpha \%} - \text{Reversal Trap \%}$).
- `adverse_risk_ratio`: Tỷ số rủi ro ngược chiều toàn thời gian ($\text{Reversal Trap} / \text{True Alpha}$).

### 4.2. Các Cột trong `top_clean_signals.csv`:
Bảng này chứa các cột giống `signal_trap_diagnosis.csv` nhưng được lọc với `trigger_count >= 20` và xếp theo `clean_edge` giảm dần, đồng thời được làm giàu thêm 2 trường quan trọng:
- `consistency_pct`: Tỷ lệ % số năm giữ vững Clean Edge dương ($E_y > 0$).
- `worst_year_edge`: Mức Clean Edge trong năm dương lịch có kết quả tệ nhất.

### 4.3. Các Cột trong `signal_yearly_stability.csv`:
- `signal_name`: Tên cột tín hiệu.
- `signal_state`: Trạng thái kích hoạt (`buy` hoặc `sell`).
- `year`: Năm dương lịch đánh giá (ví dụ: `2021`, `2022`, ... hoặc `"All"` nếu không có thông tin năm).
- `trigger_count`: Số lần trạng thái này xuất hiện trong năm đó.
- `true_alpha_pct`: Tỷ lệ % nến bắt trúng sóng xu hướng mục tiêu trong năm.
- `sideway_trap_pct`: Tỷ lệ % nến bị kẹt trong bẫy đi ngang sideway trong năm.
- `reversal_trap_pct`: Tỷ lệ % nến bị lừa vào sóng đảo chiều ngược hướng trong năm.
- `vol_trap_pct`: Tỷ lệ % nến rơi vào trạng thái cấm vào lệnh trong năm.
- `clean_edge`: Lợi thế định hướng sạch trong năm. Được gán giá trị số khi $\text{trigger\_count} \ge 20$; gán `None` khi thiếu dữ liệu.
- `adverse_risk_ratio`: Tỷ số rủi ro ngược chiều trong năm. Được gán giá trị số khi $\text{trigger\_count} \ge 20$; gán `None` khi thiếu dữ liệu.
- `status`: Cờ kiểm định mẫu: `'valid'` ($\ge 20$ triggers) hoặc `'insufficient_data'` ($< 20$ triggers).

### 4.4. Các Cột trong `signal_stability_ranking.csv`:
- `signal_name`: Tên cột tín hiệu.
- `signal_state`: Trạng thái kích hoạt (`buy` hoặc `sell`).
- `overall_clean_edge`: Lợi thế định hướng sạch tính trên toàn bộ dữ liệu mẫu (All-time Clean Edge).
- `years_evaluated`: Tổng số năm dương lịch có đủ dữ liệu kiểm định ($\ge 20$ triggers).
- `positive_years`: Số năm dương lịch đạt Clean Edge dương ($E_y > 0$).
- `consistency_pct`: Tỷ lệ phần trăm số năm duy trì lợi thế dương ($\text{positive\_years} / \text{years\_evaluated} \times 100\%$).
- `mean_clean_edge`: Trung bình cộng Clean Edge qua các năm có dữ liệu hợp lệ.
- `worst_year_edge`: Clean Edge trong năm xấu nhất lịch sử.
- `best_year_edge`: Clean Edge trong năm tốt nhất lịch sử.
- `edge_volatility`: Độ lệch chuẩn hiệu chỉnh mẫu của Clean Edge giữa các năm.
- `stability_grade`: Phân hạng độ ổn định (`STABLE_ALPHA`, `MODERATE_STABLE`, `ERRATIC / UNSTABLE`).

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị Thực chiến

### 5.1. Lựa chọn Tín hiệu Vào lệnh (Trigger Signals)
- Ưu tiên các tín hiệu có **`clean_edge > +10.0%`** và **`adverse_risk_ratio < 0.70`** (như `STA007_signal`, `VOL020_signal`, `VOL009_signal`).

### 5.2. Cảnh báo Tín hiệu Độc hại (Toxic Falling Knives)
- Tuyệt đối loại bỏ hoặc đảo ngược các tín hiệu có **`adverse_risk_ratio > 1.30`** (như `MOM017_signal`, `MOM027_signal`, `MOM004_signal` khi bắt đáy). Các tín hiệu này tạo ra số lần sập ngược chiều nhiều hơn hẳn số lần thắng.

### 5.3. Kết hợp Bộ lọc Xu hướng cho Tín hiệu Bị Bẫy Sideway
- Những tín hiệu có **`sideway_trap_pct > 25%`** (như `VLM008_signal`) bắt buộc phải kết hợp thêm bộ lọc xu hướng vĩ mô (như `TRD014_signal` trên EMA200 hoặc Choppiness Index) để triệt tiêu nhiễu sideway trước khi kích hoạt lệnh.

### 5.4. Nhận diện & Loại trừ Bẫy "Siêu sao Một mùa" (Robust Signals vs One-Hit Wonders)
Trong nghiên cứu định lượng, một trong những cạm bẫy nguy hiểm nhất là chọn nhầm tín hiệu "ăn may theo mùa":
- **Hiện tượng Siêu sao Một mùa (One-Hit Wonder)**:
  - Một tín hiệu có thể đạt `overall_clean_edge = +18%` rất ấn tượng, nhưng khi bóc tách theo từng năm: năm 2021 đạt $+48\%$ (nhờ sóng thị trường tăng điên cuồng), nhưng sang 2022 lại sập $-12\%$, 2023 đạt $-4\%$, 2024 đạt $-1\%$.
  - Nếu chỉ nhìn vào bảng tổng hợp toàn thời gian, bạn sẽ bị đánh lừa bởi một mùa thắng lớn trong quá khứ. Đưa tín hiệu này vào giao dịch thực chiến sẽ đối mặt với sụt giảm vốn nghiêm trọng khi điều kiện thị trường không còn lý tưởng.
- **Tín hiệu Bền vững Thực thụ (Robust Alpha)**:
  - Một tín hiệu có thể có `overall_clean_edge = +11%` (thấp hơn về mặt con số tuyệt đối), nhưng phân rã theo năm: $2021: +12\%$, $2022: +8\%$, $2023: +10\%$, $2024: +13\%$.
  - Tín hiệu này đạt $\text{Consistency} = 100\%$, $\text{Worst Year Edge} = +8\%$, $\text{Edge Volatility} = 2.1\%$, xếp hạng `STABLE_ALPHA`. Đây chính là "chân ái" của các hệ thống giao dịch tự động.

#### Quy trình 4 Bước Lọc Tín hiệu Bền vững:
1. **Bước 1 - Lọc Hạng Ổn định**: Bắt buộc chọn tín hiệu đạt **`stability_grade == 'STABLE_ALPHA'`** ($\text{Consistency} \ge 75\%$ và $\text{Mean Clean Edge} > 0\%$).
2. **Bước 2 - Kiểm soát Kịch bản Tệ nhất**: Yêu cầu **`worst_year_edge > 0%`** (hoặc tối thiểu không âm sâu, ví dụ $> -3.0\%$). Nếu trong năm tồi tệ nhất mà tín hiệu vẫn bảo toàn vốn hoặc chỉ lỗ nhẹ, hệ thống sẽ tránh được hiện tượng sụt giảm tài sản không thể hồi phục (irreversible drawdown).
3. **Bước 3 - Ưu tiên Độ Biến động Lợi thế Thấp**: Chọn các tín hiệu có **`edge_volatility <= 10.0%`**. Độ biến động lợi thế thấp bảo chứng cho việc lợi thế định hướng không phụ thuộc vào các cú giật giá cá biệt.
4. **Bước 4 - Kiểm tra Độ Dài Chu kỳ**: Chỉ tin tưởng các tín hiệu có **`years_evaluated >= 3`** năm hợp lệ ($\ge 20$ triggers/năm) để đảm bảo tín hiệu đã trải qua trọn vẹn cả chu kỳ Tăng trưởng (Bull), Suy thoái (Bear) và Đi ngang (Sideway).
