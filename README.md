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

Mặc định output được ghi vào `reports/<module>/`, ví dụ `reports/statistics/` hoặc `reports/scipy/`. Có thể đổi thư mục output hoặc khóa join:

```bash
fldataprofier fit feature.csv label.csv --module statistics --output-dir reports --join-key id
```

CLI cũng đăng ký alias đúng chính tả hơn:

```bash
fldataprofiler fit feature.csv label.csv --module statistics
```

## Artifacts

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

## Thiết kế module

Package được tách theo registry để mở rộng:

- `fldataprofier/cli.py`: parse command `fit` và gọi module được chọn.
- `fldataprofier/registry.py`: đăng ký module theo tên.
- `fldataprofier/modules/base.py`: protocol chung cho module profiling.
- `fldataprofier/modules/statistics.py`: module thống kê đầu tiên.
- `fldataprofier/modules/scipy.py`: module SciPy cho kiểm định feature/label.
- `fldataprofier/modules/sklearn.py`: module sklearn cho model score và feature importance.
- `fldataprofier/modules/statsmodels.py`: module statsmodels cho OLS inference.

Để thêm module mới, tạo class có method `run(...)` trả về `ModuleResult`, rồi đăng ký instance trong `fldataprofier/registry.py`.
