# Probability Drift & Alpha Stability Profiling (`probability_drift`)

Module `probability_drift` đánh giá **độ ổn định xác suất theo chuỗi thời gian (Time-Series Probability Stability)**, **độ trôi dạt phân phối (Population Stability Index - PSI)**, **hiện tượng lật ngược tín hiệu (Regime / Monotonicity Flips)** và **tốc độ suy giảm nhân tố (IV Alpha Decay)** qua các cửa sổ thời gian tuần tự (Chronological Time Folds).

---

## 1. Nguyên lý Toán học & Phương pháp Đo lường

### 1.1. Chia Cửa sổ Thời gian Tuần tự (Chronological Folds)
Tập dữ liệu được chia thành $K$ folds theo thứ tự thời gian ($K = 5$ folds mặc định):
$$\mathcal{D} = \{\mathcal{F}_1, \mathcal{F}_2, \dots, \mathcal{F}_K\}$$
Giúp so sánh trực tiếp hành vi dự báo của feature giữa các chu kỳ thị trường khác nhau (Bull, Bear, Sideway, High/Low Volatility).

### 1.2. Chỉ số Trôi dạt Phân phối (Population Stability Index - PSI)
Đo lường mức độ thay đổi hình dạng phân phối của Feature qua 20 Quantile Bins giữa từng Fold $\mathcal{F}_t$ và phân phối chuẩn toàn bộ dữ liệu $\mathcal{F}_{\text{ref}}$:
$$\text{PSI}_t = \sum_{k=1}^{20} \left(Q_{t,k} - Q_{\text{ref},k}\right) \times \ln\left(\frac{Q_{t,k} + \epsilon}{Q_{\text{ref},k} + \epsilon}\right)$$
* $\text{PSI} < 0.10$: 🟢 Phân phối **Ổn định (Stable)**, không bị trôi dạt dữ liệu.
* $0.10 \le \text{PSI} \le 0.25$: 🟡 Trôi dạt **Trung bình (Moderate Drift)**, cần theo dõi.
* $\text{PSI} > 0.25$: 🔴 Trôi dạt **Nghiêm trọng (Significant Drift)**, feature không dừng (non-stationary).

### 1.3. Laplace-Smoothed Weight of Evidence (WoE) & Information Value (IV)
Để tránh hiện tượng $IV$ bị thổi phồng vô cực khi một số bin ở các fold nhỏ không có mẫu sự kiện ($Events = 0$), module áp dụng Laplace Smoothing với tham số $\alpha = 0.5$:
$$p_e = \frac{\text{events}_{t,k} + 0.5}{\text{total\_events}_t + 0.5 \times B}, \quad p_{ne} = \frac{\text{non\_events}_{t,k} + 0.5}{\text{total\_non\_events}_t + 0.5 \times B}$$
$$\text{WoE}_{t,k} = \ln\left(\frac{p_e}{p_{ne}}\right), \quad IV_t = \sum_{k=1}^{B} (p_e - p_{ne}) \times \text{WoE}_{t,k}$$

### 1.4. Tỷ số Ổn định Thông tin (Information Stability Ratio)
$$\text{Stability Ratio} = \frac{\overline{IV}}{\sigma(IV) + \epsilon}$$
Tỷ số Signal-to-Noise của sức mạnh dự báo. Tỷ số càng cao chứng minh feature duy trì sức mạnh dự báo đồng đều qua mọi giai đoạn thời gian.

### 1.5. Hiện tượng Lật ngược Tính Đơn điệu (Monotonicity Regime Flips)
Đo lường hệ số tương quan hạng Spearman $\rho_{\text{mono}, t}$ giữa thứ tự bin $k \in \{1 \dots 20\}$ và xác suất $P_t(Y=c \mid \text{Bin}_k)$:
* Nếu $\rho_{\text{mono}}$ đổi dấu đáng kể giữa 2 folds liên tiếp (ví dụ: Fold 1 có $\rho = +0.85$, sang Fold 2 thành $\rho = -0.70$), feature bị **Regime Flip** — nguy hiểm khi giao dịch thực tế vì chiến lược mua/bán sẽ bị phản tác dụng.

---

## 2. Phân loại Trạng thái Ổn định (Drift Status)

| Trạng thái | Điều kiện Phân loại | Khuyến nghị Sử dụng |
| :--- | :--- | :--- |
| 🟢 **STABLE** | $\text{Max PSI} < 0.10$ & $\text{Flips} = 0$ & $\text{Stability Ratio} \ge 1.5$ | Feature cực kỳ vững chắc, ưu tiên hàng đầu đưa vào mô hình giao dịch. |
| 🟡 **MODERATE_DRIFT** | $0.10 \le \text{Max PSI} \le 0.25$ hoặc $\text{Flips} \le 1$ hoặc $\text{Stability Ratio} \ge 0.8$ | Chấp nhận được, nên kết hợp cơ chế thích nghi (adaptive / rolling retraining). |
| 🔴 **HIGH_DRIFT** | $\text{Max PSI} > 0.25$ hoặc $\text{Flips} \ge 2$ hoặc $\text{Stability Ratio} < 0.8$ | Rủi ro cao, dễ bị overfit hoặc gãy tín hiệu khi thị trường đổi pha. |

---

## 3. Hướng dẫn Sử dụng CLI

```bash
# Chạy đánh giá độ trôi dạt và ổn định xác suất trên toàn bộ nhãn
uv run fldataprofiler fit datasets/feature.parquet datasets/label.csv --module probability_drift

# Chỉ định nhãn mục tiêu cụ thể
uv run fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module probability_drift \
  --target allow_entry
```

---

## 4. Danh sách Kết quả Đầu ra (Artifacts)

Báo cáo và các tệp dữ liệu được lưu tại `reports/probability_drift/`:

1. `feature_drift_scores.csv`: Bảng tổng hợp các chỉ số ổn định toàn diện (`drift_status`, `mean_iv`, `stability_ratio`, `monotonicity_flips`, `max_psi`, `prob_curve_drift`, `iv_trend_slope`).
2. `fold_probability_metrics.csv`: Chi tiết số đo từng Fold thời gian (`fold`, `iv`, `prob_spread`, `monotonicity`, `psi`, `base_rate`, `curve_drift`).
3. `quantile_drift_probabilities.csv`: Bảng phân vị $20 \times K$ bins ghi nhận xác suất có điều kiện $P_t(c \mid \text{Bin}_k)$ ở từng fold đối chiếu với Overall.
4. `probability_drift_charts.png`: Đồ thị so sánh trực quan đa đường (multi-line) của các Fold thời gian so với đường chuẩn Overall.
5. `summary.json`: Metadata kỹ thuật, danh sách Top Stable Features và Top Drifting Features.
6. `report.md`: Báo cáo tóm tắt Markdown chuẩn GitHub.
7. `report.html`: Dashboard Web HTML tương tác kèm KPI cards và bảng dữ liệu có thể tìm kiếm/sắp xếp.
