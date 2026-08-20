# fl-data-profiling

CLI tạo báo cáo mô tả mối quan hệ và đánh giá sức dự đoán giữa `feature` (CSV/Parquet) và `label` (CSV/Parquet).

## 1. Cài đặt môi trường

Sử dụng `uv` để cài đặt tự động các phụ thuộc và thư viện:

```bash
uv sync
```

---

## 2. Chuẩn bị dữ liệu mẫu

Sử dụng kịch bản có sẵn để tự động tải dữ liệu OHLCV, sinh nhãn (Label) và trích xuất đặc trưng (Feature) ra thư mục `datasets/`:

```bash
bash scripts/prepare_datasets.sh
```

Dữ liệu sinh ra bao gồm:
* `datasets/VN30F1M_5m.csv`: File dữ liệu OHLCV gốc.
* `datasets/label.csv`: File nhãn phân loại được sinh bởi `labelohlcv`.
* `datasets/feature.parquet`: File đặc trưng trích xuất trực tiếp dạng Parquet bởi `autofcholv`.

---

## 3. Hướng dẫn chạy các Module

Cú pháp lệnh tổng quát:

```bash
fldataprofiler fit <feature_file> <label_file> --module <module_name> [OPTIONS]
```

### 3.1. Phân tích Phân cụm KMeans (`kmean` / `kmeans_gpu`)

Đánh giá khả năng phân tách nhãn bằng thuật toán phân cụm KMeans trên từng cặp đặc trưng số $(F_1, F_2)$. Module tự động lấy dữ liệu tuần tự (không xáo trộn), tự động lọc Top 50 đặc trưng liên quan nhất và tính toán tỉ lệ phần trăm phân cụm đúng (`train_accuracy`, `test_accuracy`):

```bash
# Chạy KMeans CPU
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module kmean

# Chạy chỉ định nhãn mục tiêu quan tâm (giúp chạy cực nhanh)
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module kmean --target allow_entry

# Chạy KMeans GPU (Yêu cầu môi trường cài đặt RAPIDS cuML)
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module kmeans_gpu
```

### 3.2. Phân tích Tổng quan EDA & Thống kê (`eda`, `statistics`, `scipy`)

```bash
# Phân tích tổng quan dữ liệu EDA (Missing value, distribution, heatmap)
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module eda

# Thống kê tương quan cơ bản giữa feature và label
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module statistics

# Kiểm định giả thuyết SciPy (Pearson, Spearman, Chi-square)
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module scipy
```

### 3.3. Đánh giá Feature Importance & Machine Learning Model (`sklearn`, `xgboost`, `shap`, `boruta`, `statsmodels`)

```bash
# Đánh giá tầm quan trọng feature bằng Machine Learning (Ridge / SGD)
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module sklearn

# Gradient Boosting với XGBoost (chỉ sử dụng numeric features)
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module xgboost-numeric

# Giải thích giá trị tác động của từng feature bằng SHAP Value
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module shap

# Chọn lọc đặc trưng quan trọng bằng Boruta (Random Forest shadow features)
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module boruta

# Phân tích mô hình OLS p-value và Confidence Interval
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module statsmodels
```

### 3.4. Các Module Feature Scoring chuyên sâu khác

Các module đánh giá và xếp hạng độ quan trọng đặc trưng:
* `alphalens`: Phân tích Factor Tearsheet theo chuẩn Alphalens (Forward Returns, Quantiles, IC Decay, Information Ratio, Long-Short Spread, Cumulative Returns).
* `information_coefficient`: Tính chỉ số IC (Information Coefficient) chuỗi thời gian.
* `mutual_information`: Điểm tương quan thông tin hỗ tương (Mutual Information).
* `mrmr`: Thuật toán mRMR (Max-Relevance Min-Redundancy) lọc feature tối ưu.
* `lightgbm`: Tầm quan trọng feature sử dụng LightGBM.
* `feature_interactions`: Đánh giá tương tác cặp đặc trưng.
* `timeseries_importance`: Tầm quan trọng feature có xét đến cấu trúc time-series.
* `regime_scoring`: Phân tích theo từng chế độ thị trường (Regime).
* `regularized_linear`: Hồi quy Lasso / Ridge regularization scoring.
* `stability_selection`: Lựa chọn biến ổn định qua lấy mẫu lặp (Subsampling).
* `permutation_importance_ts`: Permutation Importance cho Time-Series.
* `autogluon`, `flaml`, `pycaret`: Đánh giá feature tầm quan trọng qua các bộ thư viện AutoML.

Ví dụ chạy:
```bash
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module alphalens
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module alphalens --limit 10000
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module mutual_information
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module mrmr
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module lightgbm
```

---

## 4. Lọc và Trích xuất Dữ liệu Đặc trưng (`fldataprofiler prune`)

Công cụ `fldataprofiler prune` cho phép tự động loại bỏ các feature kém chất lượng, dư thừa và đa cộng tuyến để trích xuất ra một dataset sạch (`selected_feature.parquet` hoặc `selected_feature.csv`):

```bash
# 1. Lọc cơ bản (Loại bỏ null > 20%, low-variance và tương quan đa cộng tuyến |corr| > 0.85)
fldataprofiler prune datasets/feature.parquet

# 2. Tùy chỉnh ngưỡng correlation và đường dẫn file output
fldataprofiler prune datasets/feature.parquet -o datasets/selected_feature.parquet --max-corr 0.80

# 3. Lọc thông minh kết hợp file điểm profiling (ưu tiên giữ lại feature có điểm cao hơn khi bị trùng lặp) và cắt Top-K
fldataprofiler prune datasets/feature.parquet \
  --output datasets/selected_feature.parquet \
  --max-corr 0.85 \
  --max-null 0.10 \
  --scores-file reports/xgboost/feature_scores.csv \
  --top-k 30
```

### Các tham số tùy chọn của `prune`:
* `-o, --output <đường_dẫn>`: Đường dẫn lưu file dataset đã lọc (mặc định: `datasets/selected_feature.parquet` hoặc `.csv`).
* `--max-corr <float>`: Ngưỡng tương quan tối đa (mặc định: `0.85`).
* `--corr-method <pearson|spearman>`: Phương pháp tính tương quan (mặc định: `pearson`).
* `--max-null <float>`: Tỷ lệ giá trị thiếu tối đa cho phép (mặc định: `0.20`).
* `--min-variance <float>`: Ngưỡng phương sai tối thiểu để loại bỏ feature hằng số (mặc định: `0.0`).
* `--scores-file <đường_dẫn>`: File CSV điểm quan trọng (từ các module profiling) để ưu tiên giữ feature tốt hơn khi đa cộng tuyến.
* `--score-col <tên_cột>`: Tên cột điểm trong file score (tự động nhận diện nếu để trống).
* `--top-k <N>`: Giới hạn số lượng feature tối đa giữ lại.
* `--keep-col <tên_cột>`: Tên cột luôn giữ lại không bao giờ bị loại bỏ (có thể truyền nhiều lần).
* `--summary-json <đường_dẫn>`: Đường dẫn lưu file JSON ghi vết lý do loại bỏ từng cột (mặc định: `reports/prune_summary.json`).

---

## 5. Các Tham số Tùy chọn cho Lệnh `fit` (CLI Options)

* `--full`: Chạy phân tích trên **toàn bộ 100% dữ liệu** mà không thực hiện lấy mẫu rút gọn (vô hiệu hóa cơ chế internal subsampling 20k rows ở các module ML).
* `--target <cột_nhãn>`: Chỉ định cột nhãn cụ thể cần phân tích (có thể truyền nhiều lần).
* `--limit <N>`: Giới hạn chỉ phân tích $N$ dòng đầu tiên của dữ liệu (dùng khi thử nghiệm nhanh).
* `--output-dir <thư_mục>`: Chỉ định thư mục lưu kết quả báo cáo (mặc định: `reports/`).
* `--join-key <tên_cột>`: Chỉ định cột chung dùng để ghép nối dữ liệu giữa feature và label (mặc định: tự động ghép theo mốc thời gian `Date` hoặc chỉ mục dòng).

**Ví dụ kết hợp các tham số:**
```bash
# Chạy full toàn bộ dữ liệu trên module xgboost
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module xgboost \
  --target allow_entry \
  --full \
  --output-dir reports/xgboost_full_run
```

---

## 6. Danh sách Kết quả Đầu ra (Artifacts)

Mỗi module khi hoàn thành sẽ lưu báo cáo và dữ liệu thống kê vào thư mục `reports/<module>/`:

* `report.md`: Báo cáo định dạng Markdown tổng hợp kết quả.
* `report.html`: Báo cáo định dạng HTML tương tác kèm bảng hiển thị dữ liệu.
* `summary.json`: File JSON chứa thông tin metadata và top các chỉ số nổi bật.
* `*.csv`: Các file CSV chi tiết tương ứng từng module (ví dụ: `kmean_results.csv`, `cluster_label_distribution.csv`, `feature_scores.csv`, `top_features.csv`).

---

## 7. Tài liệu Chi tiết Từng Module

Xem hướng dẫn chi tiết, nguyên lý toán học và cấu trúc dữ liệu cho từng module tại thư mục [`docs/`](docs/README.md):

* 📊 **Factor & Signals**: [`alphalens.md`](docs/alphalens.md), [`information_coefficient.md`](docs/information_coefficient.md), [`signal_analysis.md`](docs/signal_analysis.md), [`regime_scoring.md`](docs/regime_scoring.md)
* 📈 **EDA & Statistics**: [`eda.md`](docs/eda.md), [`statistics.md`](docs/statistics.md), [`scipy.md`](docs/scipy.md), [`statsmodels.md`](docs/statsmodels.md)
* 🔍 **Clustering & Rules**: [`kmean.md`](docs/kmean.md), [`visual_regions.md`](docs/visual_regions.md)
* 🎯 **Feature Selection**: [`mrmr.md`](docs/mrmr.md), [`mutual_information.md`](docs/mutual_information.md), [`permutation_importance_ts.md`](docs/permutation_importance_ts.md), [`timeseries_importance.md`](docs/timeseries_importance.md), [`stability_selection.md`](docs/stability_selection.md), [`feature_interactions.md`](docs/feature_interactions.md), [`boruta.md`](docs/boruta.md)
* 🤖 **Machine Learning**: [`xgboost.md`](docs/xgboost.md), [`lightgbm.md`](docs/lightgbm.md), [`shap.md`](docs/shap.md), [`sklearn.md`](docs/sklearn.md), [`regularized_linear.md`](docs/regularized_linear.md)
* ⚡ **AutoML**: [`autogluon.md`](docs/autogluon.md), [`flaml.md`](docs/flaml.md), [`pycaret.md`](docs/pycaret.md)

