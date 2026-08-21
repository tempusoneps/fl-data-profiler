# fl-data-profiling

CLI và Python Toolkit chuyên sâu phục vụ nghiên cứu định lượng (Quantitative Research), phân tích quan hệ và đánh giá sức mạnh dự báo của tập đặc trưng (`features`) đối với các nhãn mục tiêu (`labels`).

Hệ thống hỗ trợ cả định dạng **Parquet** và **CSV**, tự động căn chỉnh mốc thời gian (Time-Series alignment), cung cấp 27 modules phân tích định lượng & Machine Learning, cùng công cụ lọc đặc trưng đa cộng tuyến (`prune`).

---

## 1. Cài đặt Môi trường

Sử dụng [`uv`](https://github.com/astral-sh/uv) để cài đặt môi trường và các gói phụ thuộc:

```bash
# Cài đặt môi trường cơ bản
uv sync

# Cài đặt đầy đủ các gói phát triển & kiểm thử
uv sync --dev
```

---

## 2. Chuẩn bị Dữ liệu Mẫu

Sử dụng kịch bản có sẵn để tự động tải dữ liệu OHLCV mẫu (VN30F1M), tự động sinh nhãn (`labelohlcv`) và trích xuất đặc trưng (`autofcholv`) vào thư mục `datasets/`:

```bash
bash scripts/prepare_datasets.sh
```

Dữ liệu sinh ra bao gồm:
* `datasets/VN30F1M_5m.csv`: File dữ liệu OHLCV 5 phút gốc.
* `datasets/label.csv`: File nhãn phân loại (Classification) và hồi quy (Regression).
* `datasets/feature.parquet`: File đặc trưng kỹ thuật số lượng lớn ở định dạng Parquet.

---

## 3. Chạy Toàn bộ Pipeline Tự động (`run_modules.sh`)

Sử dụng script `scripts/run_modules.sh` để tự động chạy tuần tự các module với đồng hồ bấm giờ, theo dõi tiến độ và xuất bảng tổng kết:

```bash
# 1. Chạy 14 modules Nhanh & Khuyến nghị (Mặc định - hoàn thành trong ~2-4 phút)
bash scripts/run_modules.sh

# 2. Chạy thử nghiệm nhanh giới hạn 1,000 dòng đầu tiên
bash scripts/run_modules.sh --limit 1000

# 3. Chạy toàn bộ 25 modules (bao gồm các module AutoML / nặng tính toán)
bash scripts/run_modules.sh --all

# 4. Chạy chỉ định danh sách các modules cụ thể
bash scripts/run_modules.sh --modules statistics,eda,xgboost,lightgbm,probability

# 5. Chạy bỏ qua một số modules không cần thiết
bash scripts/run_modules.sh --skip-modules kmean,visual_regions
```

---

## 4. Hướng dẫn Chạy Từng Module qua CLI (`fldataprofiler fit`)

Cú pháp lệnh tổng quát:

```bash
fldataprofiler fit <feature_file> <label_file> --module <module_name> [OPTIONS]
```

### 4.1. Phân tích Tín hiệu & Nhân tố Tài chính (Alpha & Factor Analysis)

* **`alphalens`**: Phân tích Tearsheet nhân tố theo chuẩn Alphalens (Forward Returns, 5 Quantiles, IC Decay $t+1 \dots t+60$, Information Ratio, Long-Short Cumulative Returns).
* **`probability`**: Phân tích xác suất có điều kiện 20 Quantiles, Information Value (IV), Weight of Evidence (WoE), Probability Spread, Monotonicity và Shannon Entropy.
* **`probability_2d`** (hoặc `probability2d`): Ma trận nhiệt xác suất kết hợp 2D ($10 \times 10$ quantile grid), đo lường $IV_{2D}$, Synergy Gain và trích xuất Vùng Điểm Ngọt (Sweet Spots).
* **`probability_3d`** (hoặc `probability3d`): Không gian xác suất kết hợp 3D ($5 \times 5 \times 5$ hyper-voxels), 3-Way Synergy Gain và trích xuất Siêu Vùng Điểm Ngọt (3D Sweet Spots).
* **`probability_drift`**: Đánh giá độ ổn định xác suất chuỗi thời gian (Alpha Stability), Population Stability Index (PSI), IV Decay và kiểm tra lật ngược tính đơn điệu (Regime Inversion Flips).
* **`information_coefficient`**: Đánh giá chỉ số IC (Pearson IC & Spearman Rank IC) qua các cửa sổ trượt Walk-Forward.
* **`signal_analysis`**: Đánh giá tín hiệu đơn lẻ (ROC-AUC, PR-AUC, F1), ma trận dư thừa (Redundancy Matrix) và mô hình kết hợp đa tín hiệu.
* **`regime_scoring`**: Đánh giá sức mạnh dự báo của đặc trưng theo từng chế độ thị trường (Trend / Volatility regimes).

```bash
# Alphalens factor tearsheet
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module alphalens

# 20-bin Quantile Conditional Probability & WoE/IV
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module probability --target allow_entry

# 2D Joint Probability Heatmap & Sweet Spots
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module probability_2d --target allow_entry

# 3D Joint Probability & Hyper Sweet Spots
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module probability_3d --target allow_entry

# Time-Series Probability Drift & Alpha Stability
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module probability_drift --target allow_entry

# Information Coefficient chuỗi thời gian
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module information_coefficient

# Trading Signal Analysis
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module signal_analysis

# Phân tích theo Regime thị trường
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module regime_scoring
```

### 4.2. Khám phá Dữ liệu & Thống kê (EDA & Statistics)

* **`eda`**: Báo cáo tổng quan dữ liệu, tỷ lệ khuyết thiếu (missingness), phân phối đặc trưng và ma trận tương quan.
* **`statistics`**: Thống kê mô tả chi tiết, tương quan Pearson và phân vị nhãn.
* **`scipy`**: Kiểm định giả thuyết thống kê nghiêm ngặt ($t$-test, ANOVA $F$, Mann-Whitney U, Chi-square, Cohen's $d$, Cramer's $V$).
* **`statsmodels`**: Mô hình kinh tế lượng OLS & Logit ($\beta$, $t$-stat, $p$-value, khoảng tin cậy 95%, AIC/BIC).

```bash
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module eda
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module statistics
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module scipy
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module statsmodels
```

### 4.3. Phân cụm & Trực quan Vùng Quyết định (Clustering & Rules)

* **`kmean`** (hoặc `kmeans_gpu`): Phân cụm KMeans 2D không giám sát trên từng cặp đặc trưng $(F_1, F_2)$, tự động đo lường độ tinh khiết (purity) và khả năng tách nhãn.
* **`visual_regions`**: Phân chia lưới $10 \times 10$ quantile và tự động sinh tập luật quyết định Bounding Box (`IF F1 IN [...] AND F2 IN [...] THEN Class X`).

```bash
# Phân cụm KMeans CPU
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module kmean --target allow_entry

# Sinh luật vùng quyết định 2D
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module visual_regions --target allow_entry
```

### 4.4. Chọn lọc & Xếp hạng Đặc trưng (Feature Selection & Scoring)

* **`mutual_information`**: Đo lường tương quan phi tuyến bằng Mutual Information.
* **`permutation_importance_ts`**: Tầm quan trọng hoán vị Random Forest trên các fold chuỗi thời gian.
* **`timeseries_importance`**: Điểm số tổng hợp đa tiêu chí (IC + Permutation Drop + MI + Correlation).
* **`mrmr`**: Thuật toán Minimum Redundancy Maximum Relevance (lọc biến tối ưu thông tin, tối thiểu trùng lặp).
* **`stability_selection`**: Chọn lọc biến ổn định qua lấy mẫu lặp nhiều lần (Subsampling).
* **`feature_interactions`**: Tự động sinh và đánh giá các cặp đặc trưng tương tác ($F_1 \times F_2$, $F_1 / F_2$, $F_1 - F_2$).
* **`boruta`**: Thuật toán Boruta lọc toàn bộ biến có liên quan bằng cách đối chiếu với Shadow Features.

```bash
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module mutual_information
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module mrmr
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module feature_interactions
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module boruta
```

### 4.5. Mô hình Học máy Cơ sở (Machine Learning Models)

* **`xgboost`** (hoặc `xgboost-numeric`): Gradient Boosting XGBoost với tầm quan trọng Gain/Weight/Cover, Confusion Matrix và đường hồi quy.
* **`lightgbm`**: GBDT dựa trên Histogram siêu tốc độ với tầm quan trọng Split & Gain.
* **`shap`**: Bóc tách đóng góp của từng biến bằng TreeSHAP & Mean Absolute SHAP values.
* **`sklearn`**: Scikit-Learn linear baseline (SGDClassifier & Ridge Regression).
* **`regularized_linear`**: Hồi quy Lasso (L1) và Ridge (L2) feature shrinkage.

```bash
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module xgboost
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module lightgbm
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module shap
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module regularized_linear
```

### 4.6. Tự động hóa Máy học (AutoML Frameworks)

* **`autogluon`**: Pipeline Multi-layer Stacking Ensemble từ Amazon AutoGluon.
* **`flaml`**: AutoML nhanh và tiết kiệm tài nguyên từ Microsoft Research.
* **`pycaret`**: Low-code AutoML so sánh đối đầu hơn 15 thuật toán máy học.

```bash
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module flaml
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module autogluon
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module pycaret
```

---

## 5. Lọc và Loại bỏ Đặc trưng Đa cộng tuyến (`fldataprofiler prune`)

Công cụ `fldataprofiler prune` hỗ trợ tự động làm sạch tập đặc trưng: loại bỏ cột nhiều null, loại bỏ cột có phương sai thấp (hằng số), giải quyết đa cộng tuyến (|corr| > threshold) và ưu tiên giữ lại các đặc trưng có điểm profiling cao nhất:

```bash
# 1. Lọc cơ bản (Loại bỏ null > 20%, low-variance và đa cộng tuyến |corr| > 0.85)
fldataprofiler prune datasets/feature.parquet

# 2. Tùy chỉnh ngưỡng correlation và file output
fldataprofiler prune datasets/feature.parquet -o datasets/selected_feature.parquet --max-corr 0.80

# 3. Lọc thông minh kết hợp file điểm profiling (ví dụ từ xgboost/lightgbm/mrmr) và lấy Top 30
fldataprofiler prune datasets/feature.parquet \
  --output datasets/selected_feature.parquet \
  --max-corr 0.85 \
  --max-null 0.10 \
  --scores-file reports/xgboost/feature_scores.csv \
  --top-k 30
```

### Các tham số tùy chọn của `prune`:
* `-o, --output <đường_dẫn>`: Đường dẫn file dataset đầu ra (mặc định: `datasets/selected_feature.parquet` hoặc `.csv`).
* `--max-corr <float>`: Ngưỡng tương quan tối đa giữa 2 features (mặc định: `0.85`).
* `--corr-method <pearson|spearman>`: Phương pháp tính tương quan (mặc định: `pearson`).
* `--max-null <float>`: Tỷ lệ khuyết thiếu tối đa cho phép (mặc định: `0.20`).
* `--min-variance <float>`: Ngưỡng phương sai tối thiểu để loại bỏ feature hằng số (mặc định: `0.0`).
* `--scores-file <đường_dẫn>`: File CSV điểm quan trọng để ưu tiên giữ feature tốt hơn khi bị đa cộng tuyến.
* `--score-col <tên_cột>`: Tên cột điểm trong file score (tự động nhận diện nếu để trống).
* `--top-k <N>`: Giới hạn số lượng feature tối đa giữ lại sau khi lọc.
* `--keep-col <tên_cột>`: Tên cột luôn giữ lại không bao giờ bị loại bỏ (có thể truyền nhiều lần).
* `--summary-json <đường_dẫn>`: Đường dẫn lưu file JSON giải trình lý do loại bỏ từng cột (mặc định: `reports/prune_summary.json`).

---

## 6. Các Tham số Tùy chọn CLI cho Lệnh `fit`

| Tùy chọn | Mô tả |
| :--- | :--- |
| `--full` | Chạy phân tích trên **toàn bộ 100% dữ liệu** (vô hiệu hóa cơ chế internal subsampling 20k rows ở các module ML). |
| `--target <cột_nhãn>` | Chỉ định cột nhãn mục tiêu cần phân tích (có thể khai báo nhiều lần). |
| `--limit <N>` | Giới hạn chỉ phân tích $N$ dòng đầu tiên (dùng để kiểm thử nhanh). |
| `--output-dir <thư_mục>` | Chỉ định thư mục lưu trữ báo cáo (mặc định: `reports/`). |
| `--join-key <tên_cột>` | Cột dùng để ghép nối giữa feature và label (mặc định: tự động ghép theo cột thời gian `Date` hoặc index). |

**Ví dụ kết hợp:**
```bash
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module xgboost \
  --target allow_entry \
  --full \
  --output-dir reports/xgboost_full_run
```

---

## 7. Sử dụng qua Python API

Ngoài CLI, bạn có thể gọi trực tiếp các module và engine trong mã nguồn Python:

### Chạy Profiling Module
```python
from pathlib import Path
from fldataprofiler.registry import get_module

# 1. Khởi tạo module từ registry
module = get_module("probability")

# 2. Chạy profiling
result = module.run(
    feature_csv=Path("datasets/feature.parquet"),
    label_csv=Path("datasets/label.csv"),
    output_dir=Path("reports"),
    targets=["allow_entry"],
)

print(f"Báo cáo được lưu tại: {result.report_dir}")
for artifact in result.artifacts:
    print(f"- {artifact}")
```

### Sử dụng Engine Lọc Đặc trưng (`feature_pruner`)
```python
from pathlib import Path
import pandas as pd
from fldataprofiler.feature_pruner import PruneConfig, prune_features, load_scores

# 1. Đọc dữ liệu và điểm số quan trọng
df = pd.read_parquet("datasets/feature.parquet")
scores = load_scores(Path("reports/xgboost/feature_scores.csv"))

# 2. Cấu hình và tiến hành lọc
config = PruneConfig(max_corr=0.85, max_null=0.20, min_variance=0.0, top_k=30)
result = prune_features(df, config=config, scores=scores)

# 3. Lưu dataset sạch đã được chọn lọc
result.df_selected.to_parquet("datasets/selected_feature.parquet", index=False)
print(f"Số lượng đặc trưng giữ lại: {len(result.retained_features)}")
```

---

## 8. Danh mục Kết quả Đầu ra (Artifacts)

Mỗi module khi hoàn thành sẽ tạo ra một thư mục riêng trong `reports/<module_name>/` gồm:

* `report.md`: Báo cáo tổng hợp định dạng Markdown chuẩn GitHub.
* `report.html`: Báo cáo Web HTML tương tác kèm bảng dữ liệu có thể tìm kiếm và sắp xếp.
* `summary.json`: Metadata kỹ thuật và top các chỉ số quan trọng.
* `*.csv`: Các file bảng dữ liệu chi tiết (ví dụ: `feature_scores.csv`, `top_features.csv`, `probability_results.csv`, `sweet_spots.csv`, `kmean_results.csv`).
* `*.png`: Các biểu đồ trực quan hóa (nếu có).

---

## 9. Tài liệu Chi tiết Từng Module

Xem hướng dẫn chi tiết, cơ sở lý thuyết toán học và đặc tả dữ liệu tại thư mục [`docs/`](docs/README.md):

* 📊 **Factor & Signals**: [`alphalens.md`](docs/alphalens.md), [`probability.md`](docs/probability.md), [`probability_2d.md`](docs/probability_2d.md), [`probability_3d.md`](docs/probability_3d.md), [`probability_drift.md`](docs/probability_drift.md), [`information_coefficient.md`](docs/information_coefficient.md), [`signal_analysis.md`](docs/signal_analysis.md), [`regime_scoring.md`](docs/regime_scoring.md)
* 📈 **EDA & Statistics**: [`eda.md`](docs/eda.md), [`statistics.md`](docs/statistics.md), [`scipy.md`](docs/scipy.md), [`statsmodels.md`](docs/statsmodels.md)
* 🔍 **Clustering & Rules**: [`kmean.md`](docs/kmean.md), [`visual_regions.md`](docs/visual_regions.md)
* 🎯 **Feature Selection**: [`mrmr.md`](docs/mrmr.md), [`mutual_information.md`](docs/mutual_information.md), [`permutation_importance_ts.md`](docs/permutation_importance_ts.md), [`timeseries_importance.md`](docs/timeseries_importance.md), [`stability_selection.md`](docs/stability_selection.md), [`feature_interactions.md`](docs/feature_interactions.md), [`boruta.md`](docs/boruta.md)
* 🤖 **Machine Learning**: [`xgboost.md`](docs/xgboost.md), [`lightgbm.md`](docs/lightgbm.md), [`shap.md`](docs/shap.md), [`sklearn.md`](docs/sklearn.md), [`regularized_linear.md`](docs/regularized_linear.md)
* ⚡ **AutoML**: [`autogluon.md`](docs/autogluon.md), [`flaml.md`](docs/flaml.md), [`pycaret.md`](docs/pycaret.md)

