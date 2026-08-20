# Probability & Quantile Profiling (`probability`)

Module `probability` phân tích phân phối xác suất có điều kiện và sức mạnh thông tin của từng đặc trưng (feature) qua các phân vị (mặc định: 20 quantiles / 20 bins) đối với các nhãn mục tiêu nhị phân và đa lớp (multi-class categorical labels).

---

## 1. Nguyên lý Toán học & Các Chỉ số Đo lường

### 1.1. Chia Phân vị Đều Mẫu (Rank-based Quantile Binning)
Để đảm bảo mỗi bin luôn chứa chính xác $5\%$ số lượng mẫu dữ liệu (không bị dồn nén do phân phối lệch hay crash do trùng lặp giá trị):
$$\text{ranks} = \text{rank}(X, \text{method='first'})$$
$$\text{Bin}(X) = \text{qcut}(\text{ranks}, q=20, \text{labels}=False) + 1 \in \{1, 2, \dots, 20\}$$

### 1.2. Xác suất Có điều kiện (Conditional Probability)
Với mỗi nhãn lớp $c \in \mathcal{C}$ và bin $k \in \{1, \dots, 20\}$:
$$P(Y = c \mid \text{Bin}_k) = \frac{\sum_{i \in \text{Bin}_k} \mathbb{I}(Y_i = c)}{N_k}$$

* **Tỷ lệ Cơ sở (Base Rate / Prior)**: $P(Y = c) = \frac{\sum \mathbb{I}(Y_i = c)}{N}$.
* **Biên độ Xác suất (Probability Spread $\Delta P$)**:
  $$\Delta P = \max_{k} P(Y=c \mid \text{Bin}_k) - \min_{k} P(Y=c \mid \text{Bin}_k)$$
* **Độ Đơn điệu (Monotonicity $\rho_{\text{mono}}$)**: Hệ số tương quan hạng Spearman giữa thứ tự bin $k \in \{1, \dots, 20\}$ và chuỗi xác suất $\{P_1, \dots, P_{20}\}$. $|\rho_{\text{mono}}| \approx 1$ thể hiện tín hiệu xác suất tăng/giảm đơn điệu và mượt mà, không bị nhiễu ngẫu nhiên.

### 1.3. Weight of Evidence (WoE) & Information Value (IV)
* **Weight of Evidence** cho từng bin:
  $$\text{WoE}_k = \ln\left(\frac{P(X \in \text{Bin}_k \mid Y=c) + \epsilon}{P(X \in \text{Bin}_k \mid Y \neq c) + \epsilon}\right)$$
* **Information Value (IV)** tổng thể của feature:
  $$\text{IV} = \sum_{k=1}^{20} \left(P(X \in \text{Bin}_k \mid Y=c) - P(X \in \text{Bin}_k \mid Y \neq c)\right) \times \text{WoE}_k$$
  * $IV < 0.02$: Không có khả năng dự báo.
  * $0.02 \le IV < 0.1$: Khả năng dự báo yếu.
  * $0.1 \le IV < 0.3$: Khả năng dự báo trung bình.
  * $0.3 \le IV \le 0.5$: Khả năng dự báo rất mạnh.
  * $IV > 0.5$: Quá mạnh (cần kiểm tra rò rỉ dữ liệu).

### 1.4. Shannon Entropy của Bin
$$H(\text{Bin}_k) = -\sum_{c \in \mathcal{C}} P(Y=c \mid \text{Bin}_k) \log_2 (P(Y=c \mid \text{Bin}_k) + \epsilon)$$
Entropy càng thấp thể hiện độ tinh khiết (purity) và độ chắc chắn của quyết định vào lệnh tại bin đó càng cao.

---

## 2. Hướng dẫn Sử dụng CLI

```bash
# Chạy phân tích xác suất trên tất cả các nhãn
uv run fldataprofiler fit datasets/feature.parquet datasets/label.csv --module probability

# Chỉ phân tích trên nhãn mục tiêu cụ thể
uv run fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module probability \
  --target allow_entry
```

---

## 3. Danh sách Kết quả Đầu ra (Artifacts)

Báo cáo và số liệu thống kê được lưu tại `reports/probability/`:

1. `feature_probability_scores.csv`: Bảng xếp hạng feature theo $IV$, $\Delta P$, Monotonicity, KL Divergence và Mean Entropy.
2. `quantile_conditional_probabilities.csv`: Bảng chi tiết 20 bins cho từng feature gồm cận giá trị $[val\_min, val\_max]$, số lượng mẫu, xác suất $P(Y=c \mid \text{Bin})$, WoE, và Entropy.
3. `probability_distribution.png`: Biểu đồ trực quan thanh xác suất qua 20 bins của top các feature mạnh nhất.
4. `summary.json`: Metadata tổng hợp và top feature theo Information Value.
5. `report.md`: Báo cáo tổng quan dạng Markdown.
6. `report.html`: Báo cáo HTML tương tác.
