# Phân tích Phân cụm 2D Đặc trưng KMeans (`kmean` / `kmeans_gpu`)

Module `kmean` đánh giá khả năng phân tách nhãn mục tiêu (label separation) bằng thuật toán phân cụm không giám sát KMeans trên từng cặp đặc trưng số $(F_1, F_2)$. Module xác định những cặp đặc trưng nào có thể nhóm dữ liệu thành các cụm thuần nhất (pure clusters) tương ứng với nhãn phân loại.

---

## 1. Mục đích & Ứng dụng

- **Đánh giá Cặp Tín hiệu Trực quan (2D Feature Synergy)**: Tìm ra các cặp chỉ báo bổ trợ cho nhau tốt nhất (ví dụ: RSI kết hợp ADX, hoặc Volume Spike kết hợp Bollinger Band Width).
- **Phân cụm Không Giám sát (Unsupervised Clustering)**: Kiểm tra xem các cụm tự nhiên trong không gian đặc trưng có phản ánh đúng trạng thái thị trường (Long / Short / Neutral) hay không.
- **Đánh giá Độ thuần khiết Phân cụm (Cluster Purity & Accuracy)**: Đo lường độ chính xác phân loại khi gán cụm KMeans về nhãn thực tế bằng thuật toán ghép cặp tối ưu.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Sàng lọc Đặc trưng Tiền đề (Candidate Filtering)**:
   - Tự động lọc Top 50 đặc trưng số có liên quan nhất với nhãn để tránh quá tải tổ hợp cặp $\binom{N}{2}$.
2. **Phân cụm Không Xáo trộn Thứ tự (Time-Order Preserved)**:
   - Dữ liệu được chia thành tập Train (70%) và Test (30%) theo thứ tự thời gian tuần tự (không shuffle) để đảm bảo tính thực tế của time-series.
3. **Thuật toán KMeans & Gán nhãn Tối ưu (Optimal Cluster Mapping)**:
   - Chuẩn hóa đặc trưng bằng `StandardScaler`.
   - Huấn luyện KMeans với $k = \text{số lượng nhãn duy nhất}$.
   - Gán mỗi cụm cluster cho class nhãn chiếm đa số trong cụm đó trên tập train.
4. **Đo lường Hiệu năng (Performance Metrics)**:
   - Tính toán `train_accuracy` và `test_accuracy`.
   - Xếp hạng các cặp đặc trưng $(F_1, F_2)$ theo `test_accuracy` giảm dần.

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy KMeans trên CPU
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module kmean

# Chạy chỉ định nhãn cụ thể (tăng tốc độ chạy)
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module kmean --target allow_entry

# Giới hạn dữ liệu và chỉ định thư mục xuất kết quả
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module kmean \
  --limit 20000 \
  --output-dir reports/kmean_run
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `kmean_results.csv` | CSV | Bảng xếp hạng tất cả các cặp đặc trưng $(F_1, F_2)$ theo độ chính xác phân cụm `test_accuracy`. |
| `cluster_label_distribution.csv` | CSV | Phân bố tỷ lệ từng nhãn thực tế bên trong mỗi cụm cluster cho các cặp đặc trưng tốt nhất. |
| `numeric_features.csv` | CSV | Danh sách các đặc trưng số được đưa vào thử nghiệm phân cụm. |
| `categorical_labels.csv` | CSV | Danh sách các nhãn phân loại mục tiêu được đánh giá. |
| `summary.json` | JSON | Metadata lần chạy và thông tin cặp đặc trưng phân tách nhãn tốt nhất. |
| `report.md` / `report.html` | Báo cáo | Báo cáo Markdown và HTML tương tác hiển thị bảng xếp hạng trực quan. |

### Các Cột trong `kmean_results.csv`:
- `label`: Tên cột nhãn phân loại.
- `feature_1`: Tên đặc trưng thứ nhất trong cặp.
- `feature_2`: Tên đặc trưng thứ hai trong cặp.
- `train_accuracy`: Độ chính xác phân cụm trên tập huấn luyện ($0.0 \to 1.0$).
- `test_accuracy`: Độ chính xác phân cụm trên tập kiểm định ($0.0 \to 1.0$).
- `accuracy_drop`: Mức sụt giảm độ chính xác giữa train và test ($\text{train\_acc} - \text{test\_acc}$), dùng để phát hiện overfitting.
- `samples`: Số lượng mẫu dữ liệu sử dụng.

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Tiêu chuẩn Cặp Đặc trưng Tốt**:
  - `test_accuracy` cao vượt trội so với baseline ngẫu nhiên (ví dụ với bài toán 2 nhãn cân bằng $50/50$, `test_accuracy` $> 0.65$ là rất tốt).
  - `accuracy_drop` nhỏ $(< 0.05)$, thể hiện cấu trúc phân cụm bền vững và không bị overfit theo thời gian.
- **Ứng dụng vào Xây dựng Chiến lược**:
  - Sử dụng các cặp $(F_1, F_2)$ đứng đầu bảng để thiết lập các vùng lọc tín hiệu vào lệnh (Entry Filter Zones) 2 chiều.\n