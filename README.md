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
*(Hoặc dùng alias `fldataprofier fit ...`)*

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

# Gradient Boosting với XGBoost
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module xgboost

# Giải thích giá trị tác động của từng feature bằng SHAP Value
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module shap

# Chọn lọc đặc trưng quan trọng bằng Boruta (Random Forest shadow features)
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module boruta

# Phân tích mô hình OLS p-value và Confidence Interval
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module statsmodels
```

### 3.4. Các Module Feature Scoring chuyên sâu khác

Các module đánh giá và xếp hạng độ quan trọng đặc trưng:
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
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module mutual_information
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module mrmr
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module lightgbm
```

---

## 4. Các Tham số Tùy chọn (CLI Options)

* `--target <cột_nhãn>`: Chỉ định cột nhãn cụ thể cần phân tích (có thể truyền nhiều lần).
* `--limit <N>`: Giới hạn chỉ phân tích $N$ dòng đầu tiên của dữ liệu (dùng khi thử nghiệm nhanh).
* `--output-dir <thư_mục>`: Chỉ định thư mục lưu kết quả báo cáo (mặc định: `reports/`).
* `--join-key <tên_cột>`: Chỉ định cột chung dùng để ghép nối dữ liệu giữa feature và label (mặc định: tự động ghép theo mốc thời gian `Date` hoặc chỉ mục dòng).

**Ví dụ kết hợp các tham số:**
```bash
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module kmean \
  --target allow_entry \
  --limit 20000 \
  --output-dir reports/kmean_run
```

---

## 5. Danh sách Kết quả Đầu ra (Artifacts)

Mỗi module khi hoàn thành sẽ lưu báo cáo và dữ liệu thống kê vào thư mục `reports/<module>/`:

* `report.md`: Báo cáo định dạng Markdown tổng hợp kết quả.
* `report.html`: Báo cáo định dạng HTML tương tác kèm bảng hiển thị dữ liệu.
* `summary.json`: File JSON chứa thông tin metadata và top các chỉ số nổi bật.
* `*.csv`: Các file CSV chi tiết tương ứng từng module (ví dụ: `kmean_results.csv`, `cluster_label_distribution.csv`, `feature_scores.csv`, `top_features.csv`).
