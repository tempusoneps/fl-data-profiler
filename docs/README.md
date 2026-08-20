# Tài liệu Hướng dẫn Toàn diện các Module trong `fl-data-profiling`

Hệ thống `fl-data-profiling` cung cấp một bộ công cụ định lượng và học máy toàn diện gồm **25 modules** chuyên sâu, phục vụ cho việc phân tích, kiểm định thống kê, khai phá tín hiệu (Alpha Mining) và đánh giá sức mạnh dự báo của tập đặc trưng (`features`) đối với các nhãn mục tiêu (`labels`).

---

## 1. Bản đồ Phân loại các Module (Module Taxonomy)

```
fl-data-profiling
├── 1. Khám phá & Thống kê Cơ bản (EDA & Statistics)
│   ├── eda                      : Phân tích tổng quan, missing values, phân phối & heatmaps
│   ├── statistics               : Thống kê mô tả, tương quan Pearson & phân vị nhãn
│   ├── scipy                    : Kiểm định giả thuyết (t-test, ANOVA, Mann-Whitney, Chi-square)
│   └── statsmodels              : Mô hình kinh tế lượng OLS/Logit, p-value & khoảng tin cậy 95%
│
├── 2. Phân tích Tín hiệu & Nhân tố Tài chính (Alpha & Factor Analysis)
│   ├── alphalens                : Factor tearsheet, Quantile returns, IC decay, Long-Short spread
│   ├── probability              : Phân tích xác suất điều kiện 20 quantiles, Information Value (IV) & WoE
│   ├── probability_2d           : Ma trận nhiệt xác suất kết hợp 2D (10x10 grid), Synergy Gain & Sweet Spots
│   ├── information_coefficient  : Walk-forward IC (Pearson & Spearman Rank IC) chuỗi thời gian
│   ├── signal_analysis          : Đánh giá tín hiệu đơn lẻ, ma trận dư thừa & tín hiệu kết hợp
│   └── regime_scoring           : Đánh giá sức mạnh đặc trưng theo từng chế độ thị trường
│
├── 3. Phân cụm & Trực quan Vùng Quyết định (Clustering & Rules)
│   ├── kmean                    : Phân cụm 2D không giám sát và đánh giá độ phân tách nhãn
│   └── visual_regions           : Trích xuất vùng quyết định 2D và sinh luật dạng Bounding Box
│
├── 4. Chọn lọc & Xếp hạng Đặc trưng (Feature Selection & Scoring)
│   ├── mutual_information       : Điểm thông tin hỗ tương phi tuyến (Mutual Information)
│   ├── permutation_importance_ts: Tầm quan trọng hoán vị Random Forest trên Time-Series folds
│   ├── timeseries_importance    : Điểm số tổng hợp đa tiêu chí chuỗi thời gian
│   ├── mrmr                     : Thuật toán Max-Relevance Min-Redundancy lọc biến tối ưu
│   ├── stability_selection      : Chọn lọc biến ổn định qua lấy mẫu lặp (Subsampling)
│   └── feature_interactions     : Tự động sinh và đánh giá các cặp đặc trưng tương tác
│
├── 5. Mô hình Học máy Cơ sở (Machine Learning Models)
│   ├── regularized_linear       : Hồi quy tuyến tính phạt Lasso (L1) & Ridge (L2)
│   ├── sklearn                  : Baseline Scikit-Learn (SGDClassifier & Ridge)
│   ├── xgboost                  : Gradient Boosting XGBoost (Gain/Weight, Confusion Matrix)
│   ├── lightgbm                 : LightGBM siêu tốc độ theo Gain & Split
│   ├── shap                     : Giải thích đóng góp cá biệt từng biến bằng SHAP Values
│   └── boruta                   : Thuật toán Boruta lọc biến toàn diện với Shadow Features
│
└── 6. Tự động hóa Máy học (AutoML Frameworks)
    ├── autogluon                : Multi-layer Stacking Ensemble từ Amazon AutoGluon
    ├── flaml                    : Fast & Lightweight AutoML từ Microsoft Research
    └── pycaret                  : Pipeline so sánh đối đầu hơn 15 thuật toán máy học
```

---

## 2. Bảng Tra cứu Nhanh Tất cả các Module

| Tên Lệnh CLI (`--module`) | File Tài liệu Chi tiết | Loại Phân tích | Trường hợp Sử dụng Chính |
| :--- | :--- | :--- | :--- |
| `alphalens` | [`alphalens.md`](alphalens.md) | Factor Research | Đánh giá factor tearsheet, IC decay, phân tầng quantile, long-short spread |
| `probability` | [`probability.md`](probability.md) | Probabilistic | Phân tích xác suất điều kiện 20 quantiles, Information Value (IV), WoE, Monotonicity |
| `probability_2d` (hoặc `probability2d`) | [`probability_2d.md`](probability_2d.md) | Joint Probability | Ma trận nhiệt xác suất kết hợp 2D (10x10 grid), Synergy Gain & Sweet Spots |
| `eda` | [`eda.md`](eda.md) | Data Profiling | Kiểm tra missingness, phân phối, kiểu dữ liệu, correlation heatmaps |
| `statistics` | [`statistics.md`](statistics.md) | Descriptive Stats | Tương quan tuyến tính Pearson, thống kê mô tả, quantile profile |
| `scipy` | [`scipy.md`](scipy.md) | Hypothesis Testing | Kiểm định p-value nghiêm ngặt (t-test, ANOVA, Kruskal, Chi-square, effect size) |
| `statsmodels` | [`statsmodels.md`](statsmodels.md) | Econometrics | Mô hình hồi quy OLS/Logit, t-stat, p-value, khoảng tin cậy 95%, AIC/BIC |
| `kmean` (hoặc `kmeans_gpu`) | [`kmean.md`](kmean.md) | Clustering | Phân cụm cặp đặc trưng 2D và đo lường độ chính xác phân tách nhãn |
| `visual_regions` | [`visual_regions.md`](visual_regions.md) | Rule Extraction | Trích xuất vùng quyết định 2D và sinh luật Bounding Box dạng if-then |
| `signal_analysis` | [`signal_analysis.md`](signal_analysis.md) | Signal Evaluation | Đánh giá tín hiệu đơn lẻ, ma trận dư thừa (redundancy) và mô hình kết hợp |
| `information_coefficient` | [`information_coefficient.md`](information_coefficient.md) | Time-Series IC | Tính toán Pearson & Spearman Rank IC qua các cửa sổ trượt Walk-Forward |
| `mutual_information` | [`mutual_information.md`](mutual_information.md) | Information Theory | Đo lường mức độ phụ thuộc phi tuyến độc lập với mô hình |
| `permutation_importance_ts` | [`permutation_importance_ts.md`](permutation_importance_ts.md) | Model Importance | Mức độ suy giảm hiệu năng khi xáo trộn feature trên out-of-fold samples |
| `timeseries_importance` | [`timeseries_importance.md`](timeseries_importance.md) | Unified Scoring | Điểm số tổng hợp đa chiều chuẩn hóa (IC + Permutation + MI + Correlation) |
| `regime_scoring` | [`regime_scoring.md`](regime_scoring.md) | Market Regimes | Đánh giá sức mạnh dự báo của feature trong từng chế độ thị trường |
| `feature_interactions` | [`feature_interactions.md`](feature_interactions.md) | Feature Eng | Tự động sinh và đánh giá các biến tương tác phép nhân, tỷ lệ, hiệu số |
| `mrmr` | [`mrmr.md`](mrmr.md) | Feature Selection | Lọc tập biến tối ưu: Tối đa liên quan với nhãn và tối thiểu trùng lặp giữa các biến |
| `stability_selection` | [`stability_selection.md`](stability_selection.md) | Robust Selection | Xác định biến ổn định vững chắc qua hàng trăm lần lấy mẫu con ngẫu nhiên |
| `regularized_linear` | [`regularized_linear.md`](regularized_linear.md) | Linear Models | Hồi quy Lasso/Ridge phạt trọng số và triệt tiêu biến không quan trọng về 0 |
| `sklearn` | [`sklearn.md`](sklearn.md) | ML Baseline | Baseline chuẩn Scikit-Learn (SGDClassifier / Ridge Regression) |
| `xgboost` (hoặc `xgboost-numeric`) | [`xgboost.md`](xgboost.md) | GBDT Modeling | Gradient Boosting với XGBoost, xếp hạng Gain/Weight, Confusion Matrix |
| `lightgbm` | [`lightgbm.md`](lightgbm.md) | Fast GBDT | Tính toán tầm quan trọng Split & Gain siêu nhanh trên dữ liệu quy mô lớn |
| `shap` | [`shap.md`](shap.md) | Explainable AI | Bóc tách giá trị đóng góp cá biệt từng biến bằng TreeSHAP values |
| `boruta` | [`boruta.md`](boruta.md) | Feature Selection | Tìm toàn bộ biến có liên quan bằng cách đối chiếu với Shadow Features |
| `autogluon` | [`autogluon.md`](autogluon.md) | AutoML | Xếp chồng đa tầng Multi-layer Stacking Ensemble từ Amazon AutoGluon |
| `flaml` | [`flaml.md`](flaml.md) | Light AutoML | Tối ưu hóa siêu tham số tiết kiệm tài nguyên và thời gian từ Microsoft |
| `pycaret` | [`pycaret.md`](pycaret.md) | AutoML Pipeline | Tự động so sánh đối đầu hơn 15 mô hình phân loại và hồi quy |

---

## 3. Cú pháp Lệnh Tổng quát

```bash
# Cú pháp tổng quát
fldataprofiler fit <feature_file> <label_file> --module <module_name> [OPTIONS]
```

### Các Tham số Tùy chọn Quan trọng:
- `--full`: Chạy phân tích trên **toàn bộ 100% dữ liệu** (vô hiệu hóa cơ chế internal subsampling 20k rows ở các module ML).
- `--target <tên_cột_nhãn>`: Chỉ định cột nhãn cụ thể cần phân tích (có thể truyền nhiều lần).
- `--limit <N>`: Giới hạn chỉ lấy $N$ dòng đầu tiên để chạy thử nghiệm nhanh.
- `--output-dir <đường_dẫn>`: Chỉ định thư mục lưu trữ artifacts báo cáo (mặc định: `reports/<module_name>/`).
- `--join-key <tên_cột>`: Chỉ định cột dùng để ghép nối dữ liệu giữa 2 file (mặc định tự động ghép theo thời gian `Date` hoặc index).

---

## 4. Cấu trúc Thư mục Kết quả Tiêu chuẩn (Output Artifacts)

Mỗi module khi hoàn thành đều tạo ra cấu trúc file nhất quán:
- `report.md`: Báo cáo tổng hợp bằng Markdown chuẩn GitHub.
- `report.html`: Báo cáo giao diện web tương tác kèm bảng dữ liệu có thể sắp xếp.
- `summary.json`: Metadata kỹ thuật và danh sách chỉ số nổi bật nhất.
- `*.csv`: Các file bảng dữ liệu chi tiết tương ứng với từng module (ví dụ: `feature_scores.csv`, `top_features.csv`, `kmean_results.csv`, `pairwise.csv`, `importance.csv`).
- `*.png`: Các biểu đồ trực quan hóa (nếu module có hỗ trợ sinh đồ thị).\n