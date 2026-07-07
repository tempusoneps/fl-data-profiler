# fl-data-profiling

CLI tạo report mô tả mối quan hệ giữa `feature.csv` và `label.csv`.

## Cài đặt

```bash
uv sync
```

## Chạy report

```bash
fldataprofier fit feature.csv label.csv --module statistics
```

EDA module dùng để phân tích tổng quan riêng cả `feature.csv` và `label.csv`:

```bash
fldataprofier fit feature.csv label.csv --module eda
```

SciPy module dùng để kiểm định mối quan hệ feature/label:

```bash
fldataprofier fit feature.csv label.csv --module scipy
```

Sklearn module dùng để đánh giá sức dự đoán và feature importance:

```bash
fldataprofier fit feature.csv label.csv --module sklearn
```

Statsmodels module dùng để xem OLS coefficient, p-value và confidence interval:

```bash
fldataprofier fit feature.csv label.csv --module statsmodels
```

XGBoost module dùng gradient boosting để đánh giá sức dự đoán phi tuyến và feature importance:

```bash
fldataprofier fit feature.csv label.csv --module xgboost
```

SHAP module fit XGBoost và giải thích feature impact bằng mean absolute SHAP value:

```bash
fldataprofier fit feature.csv label.csv --module shap
```

Boruta module dùng RandomForest với shadow features để chọn feature quan trọng:

```bash
fldataprofier fit feature.csv label.csv --module boruta
```

Mặc định output được ghi vào `reports/<module>/`, ví dụ `reports/statistics/` hoặc `reports/scipy/`. Có thể đổi thư mục output hoặc khóa join:

```bash
fldataprofier fit feature.csv label.csv --module statistics --output-dir reports --join-key id
```

CLI cũng đăng ký alias đúng chính tả hơn:

```bash
fldataprofiler fit feature.csv label.csv --module statistics
```

## Artifacts

Module `eda` tạo các file:

- `report.md`
- `report.html`
- `summary.json`
- `dataset_overview.csv`
- `columns_profile.csv`
- `missingness.csv`
- `numeric_summary.csv`
- `categorical_summary.csv`
- `feature_correlation_heatmap.png`
- `label_correlation_heatmap.png`

Module `statistics` tạo các file:

- `report.md`
- `report.html`
- `feature_label_correlation_heatmap.png`
- `statistics_summary.json`
- `feature_profile.csv`
- `label_profile.csv`
- `feature_label_correlations.csv`

Module `scipy` bỏ qua cột `Date` khi đánh giá và tạo các file:

- `report.md`
- `report.html`
- `summary.json`
- `pairwise.csv`
- `two_feature.csv`

Module `sklearn` dùng numeric features trước, fit `Ridge` cho label numeric và `SGDClassifier` cho label categorical/binary:

- `report.md`
- `report.html`
- `summary.json`
- `scores.csv`
- `importance.csv`

Module `statsmodels` fit OLS cho numeric labels, chọn tối đa 25 numeric features có absolute correlation cao nhất cho từng label:

- `report.md`
- `report.html`
- `summary.json`
- `scores.csv`
- `coefficients.csv`

Module `xgboost` dùng numeric features trước, fit `XGBRegressor` cho label numeric và `XGBClassifier` cho label categorical/binary:

- `report.md`
- `report.html`
- `summary.json`
- `scores.csv`
- `importance.csv`

Module `shap` fit XGBoost rồi tính mean absolute SHAP value cho từng feature/label:

- `report.md`
- `report.html`
- `summary.json`
- `scores.csv`
- `shap_importance.csv`

Module `boruta` fit RandomForest nhiều vòng với shadow features, phân loại feature thành `confirmed`, `tentative` hoặc `rejected`:

- `report.md`
- `report.html`
- `summary.json`
- `scores.csv`
- `boruta_features.csv`

## Thiết kế module

Package được tách theo registry để mở rộng:

- `fldataprofier/cli.py`: parse command `fit` và gọi module được chọn.
- `fldataprofier/registry.py`: đăng ký module theo tên.
- `fldataprofier/modules/base.py`: protocol chung cho module profiling.
- `fldataprofier/modules/boruta.py`: module Boruta-style feature selection.
- `fldataprofier/modules/eda.py`: module EDA tổng quan cho feature và label.
- `fldataprofier/modules/statistics.py`: module thống kê đầu tiên.
- `fldataprofier/modules/scipy.py`: module SciPy cho kiểm định feature/label.
- `fldataprofier/modules/shap.py`: module SHAP cho giải thích model XGBoost.
- `fldataprofier/modules/sklearn.py`: module sklearn cho model score và feature importance.
- `fldataprofier/modules/statsmodels.py`: module statsmodels cho OLS inference.
- `fldataprofier/modules/xgboost.py`: module XGBoost cho gradient boosting score và feature importance.

Để thêm module mới, tạo class có method `run(...)` trả về `ModuleResult`, rồi đăng ký instance trong `fldataprofier/registry.py`.
