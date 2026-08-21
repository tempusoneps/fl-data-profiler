# Bayesian Probability & Quantile Profiling (`probability_bayes`)

Module `probability_bayes` phân tích phân phối xác suất có điều kiện theo trường phái **Xác suất Bayes (Bayesian Probability)**, sử dụng phân phối tiên nghiệm liên hợp (Conjugate Priors: Beta-Binomial & Dirichlet-Multinomial) kết hợp cơ chế co Bayes (Bayesian Shrinkage / m-estimate), tính toán khoảng tin cậy hậu nghiệm (95% Credible Intervals) và tỷ số bằng chứng Bayes (Bayes Factor $BF_{10}$) qua 20 phân vị (quantiles) của từng đặc trưng.

---

## 1. Nguyên lý Toán học & Các Chỉ số Đo lường

### 1.1. Chia Phân vị Đều Mẫu (Rank-based Quantile Binning)
Để đảm bảo mỗi bin luôn chứa chính xác $5\%$ số lượng mẫu dữ liệu ($N_k \approx 0.05 N$):
$$\text{ranks} = \text{rank}(X, \text{method='first'})$$
$$\text{Bin}(X) = \text{qcut}(\text{ranks}, q=20, \text{labels}=False) + 1 \in \{1, 2, \dots, 20\}$$

### 1.2. Phân phối Tiên nghiệm Cơ sở (Prior Base Rate & Conjugate Priors)
* **Tỷ lệ Cơ sở toàn tập (Global Base Rate)**:
  $$P_0(Y = c) = \frac{\sum_{i=1}^N \mathbb{I}(Y_i = c)}{N}$$
* **Trọng số Tiên nghiệm (Prior Strength $m$)**: Mặc định $m = 10.0$ (tương đương 10 quan sát giả lập / pseudo-counts theo phân phối nền).
* **Tham số Tiên nghiệm (Prior Parameters)**:
  $$\alpha_{0, c} = m \cdot P_0(Y = c), \quad \beta_{0, c} = m \cdot (1 - P_0(Y = c))$$

### 1.3. Xác suất Hậu nghiệm Bayes & Co Bayes (Bayesian Posterior & Shrinkage)
Với mỗi bin $k \in \{1, \dots, 20\}$ có tổng số mẫu $N_k$ và số biến cố $N_{k, c} = \sum_{i \in \text{Bin}_k} \mathbb{I}(Y_i = c)$:
* **Kỳ vọng Hậu nghiệm Bayes (Posterior Mean Probability)**:
  $$P_{\text{Bayes}}(Y = c \mid \text{Bin}_k) = \frac{N_{k, c} + \alpha_{0, c}}{N_k + m}$$
  * *Tự động triệt tiêu overfitting ở mẫu nhỏ*: Khi bin có ít mẫu ($N_k \to 0$), $P_{\text{Bayes}} \to P_0(c)$ (Base Rate). Khi có nhiều mẫu ($N_k \to \infty$), $P_{\text{Bayes}} \to \frac{N_{k,c}}{N_k}$ (Frequentist).
  * *Đảm bảo $P_{\text{Bayes}} \in (0, 1)$ strictly*, giải quyết triệt để lỗi chia cho 0 hay $\ln(0)$.

### 1.4. Khoảng Tin cậy Hậu nghiệm 95% (95% Bayesian Credible Intervals)
Từ phân phối hậu nghiệm $\text{Beta}(\alpha_{\text{post}}, \beta_{\text{post}})$ với $\alpha_{\text{post}} = N_{k, c} + \alpha_{0, c}$ và $\beta_{\text{post}} = (N_k - N_{k, c}) + \beta_{0, c}$:
$$\text{CI}_{\text{lower}} = \text{Beta}^{-1}(0.025; \alpha_{\text{post}}, \beta_{\text{post}})$$
$$\text{CI}_{\text{upper}} = \text{Beta}^{-1}(0.975; \alpha_{\text{post}}, \beta_{\text{post}})$$
$$\text{CI}_{\text{width}} = \text{CI}_{\text{upper}} - \text{CI}_{\text{lower}}$$
Độ rộng $\text{CI}_{\text{width}}$ đo lường trực tiếp mức độ bất định (Uncertainty) của xác suất tại từng bin.

### 1.5. Tỷ số Bằng chứng Bayes (Bayes Factor $BF_{10}$)
Kiểm định mức độ bằng chứng thống kê giữa giả thuyết $\mathcal{H}_1$ (bin có phân phối riêng biệt khác Base Rate) so với $\mathcal{H}_0$ (bin ngẫu nhiên sinh ra từ Base Rate):
$$\ln \text{BF}_{10, k} = \ln \left[\frac{\text{B}(N_{k, c} + \alpha_{0, c}, N_k - N_{k, c} + \beta_{0, c})}{\text{B}(\alpha_{0, c}, \beta_{0, c})}\right] - \left[ N_{k, c} \ln P_0(c) + (N_k - N_{k, c}) \ln (1 - P_0(c)) \right]$$
* $\ln \text{BF} < 0$: Ủng hộ giả thuyết không có tín hiệu ($\mathcal{H}_0$).
* $0 \le \ln \text{BF} < 1.1$ ($BF < 3$): Bằng chứng yếu/ngẫu nhiên.
* $1.1 \le \ln \text{BF} < 2.3$ ($3 \le BF < 10$): Bằng chứng vừa phải (Moderate).
* $2.3 \le \ln \text{BF} < 4.6$ ($10 \le BF < 100$): Bằng chứng mạnh (Strong).
* $\ln \text{BF} \ge 4.6$ ($BF \ge 100$): Bằng chứng áp đảo (Decisive).

### 1.6. Bayesian Weight of Evidence (WoE) & Information Value (IV)
* **Bayesian WoE**:
  $$\text{Bayes WoE}_k = \ln\left(\frac{P_{\text{Bayes}}(Y=c \mid \text{Bin}_k)}{1 - P_{\text{Bayes}}(Y=c \mid \text{Bin}_k)} \cdot \frac{1 - P_0(c)}{P_0(c)}\right)$$
* **Bayesian Information Value (Bayes IV)**:
  $$\text{Bayes IV} = \sum_{k=1}^{20} \left(P(X \in \text{Bin}_k \mid Y=c) - P(X \in \text{Bin}_k \mid Y \neq c)\right) \times \text{Bayes WoE}_k$$

---

## 2. Hướng dẫn Sử dụng CLI

```bash
# Chạy phân tích Bayesian probability trên tất cả các nhãn
uv run fldataprofiler fit datasets/feature.parquet datasets/label.csv --module probability_bayes

# Phân tích trên nhãn cụ thể không subsampling
uv run fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module probability_bayes \
  --target allow_entry \
  --full
```

---

## 3. Danh sách Kết quả Đầu ra (Artifacts)

Báo cáo và số liệu thống kê được lưu tại `reports/probability_bayes/`:

1. `bayes_probability_scores.csv`: Bảng xếp hạng feature theo $\text{Bayes IV}$, $\Delta P_{\text{Bayes}}$, Monotonicity, Mean $\ln(\text{BF}_{10})$, Mean CI Width và Entropy.
2. `quantile_bayes_probabilities.csv`: Bảng chi tiết 20 bins gồm cận $[val\_min, val\_max]$, số mẫu, $P_{\text{raw}}$, $P_{\text{Bayes}}$, $95\%$ CI $[\text{lower}, \text{upper}]$, $\ln(\text{BF}_{10})$, $\text{Bayes WoE}$ và Entropy.
3. `bayes_probability_distribution.png`: Biểu đồ trực quan đường cong xác suất hậu nghiệm Bayes, dải sai số $95\%$ Credible Interval, cột mờ $P_{\text{raw}}$ và đường Base Rate.
4. `summary.json`: Metadata tổng hợp và top feature theo Bayes Information Value.
5. `report.md`: Báo cáo tổng quan dạng Markdown.
6. `report.html`: Báo cáo HTML tương tác với đồ thị và bảng điểm.
