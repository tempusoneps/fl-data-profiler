# Chọn lọc Đặc trưng Tối ưu mRMR (`mrmr`)

Module `mrmr` áp dụng thuật toán kinh điển **Minimum Redundancy Maximum Relevance (mRMR)** để chọn ra một tập con đặc trưng vừa có mức độ liên quan tối đa với biến mục tiêu (Max-Relevance), vừa có mức độ trùng lặp/dư thừa thông tin tối thiểu giữa các đặc trưng với nhau (Min-Redundancy).

---

## 1. Mục đích & Ứng dụng

- **Chống Dư thừa Thông tin (Feature Redundancy Removal)**: Tránh việc chọn 10 biến cùng đo lường một hiện tượng (ví dụ 10 đường SMA các chu kỳ sát nhau).
- **Tối ưu Hóa Số lượng Biến**: Tìm ra tập đặc trưng nhỏ gọn nhất nhưng chứa lượng thông tin đa dạng và phong phú nhất.
- **Tăng Tốc Độ và Tính Khái Quát của Mô hình**: Giúp các mô hình máy học học nhanh hơn, ít bị nhiễu và hạn chế tối đa overfitting.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Nguyên lý Tối đa Hóa Mức Liên quan (Max-Relevance)**:
   - Ưu tiên các đặc trưng có mức độ tương quan thông tin hỗ tương (Mutual Information) hoặc tương quan thống kê cao nhất với nhãn $Y$:
     $$\text{Relevance}(X_i, Y) = I(X_i; Y)$$
2. **Nguyên lý Tối thiểu Hóa Sự Dư thừa (Min-Redundancy)**:
   - Phạt các đặc trưng có tương quan cao với các đặc trưng đã được chọn vào tập $\mathcal{S}$:
     $$\text{Redundancy}(X_i, \mathcal{S}) = \frac{1}{|\mathcal{S}|} \sum_{X_j \in \mathcal{S}} I(X_i; X_j)$$
3. **Tiêu chuẩn Tối ưu mRMR (Mutual Information Difference - MID)**:
   - Tại mỗi bước tham lam (greedy step), chọn đặc trưng $X^*$ thỏa mãn:
     $$X^* = \arg\max_{X_i \notin \mathcal{S}} \left[ \text{Relevance}(X_i, Y) - \text{Redundancy}(X_i, \mathcal{S}) \right]$$

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy thuật toán chọn lọc mRMR
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module mrmr

# Chỉ định nhãn mục tiêu
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module mrmr --target allow_entry

# Lưu kết quả vào thư mục riêng
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module mrmr \
  --output-dir reports/mrmr_selection
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `feature_scores.csv` | CSV | Bảng xếp hạng thứ tự các feature được chọn bởi thuật toán mRMR kèm điểm số tối ưu. |
| `top_features.csv` | CSV | Top 50 đặc trưng tối ưu nhất theo mRMR. |
| `summary.json` | JSON | Metadata tổng kết danh sách biến tối ưu. |
| `report.md` / `report.html` | Báo cáo | Báo cáo Markdown và HTML tổng hợp. |

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Thứ tự Lựa chọn (Selection Rank)**: Các đặc trưng đứng ở vị trí đầu tiên ($1, 2, 3, \dots, K$) là tập đặc trưng bổ trợ cho nhau hoàn hảo nhất.
- **Số lượng Biến Khuyến nghị**: Thường chỉ cần giữ lại Top 15 - 30 đặc trưng đầu tiên của bảng `top_features.csv` là đủ để đạt $95\% - 98\%$ hiệu năng tối đa của mô hình mà không lo bị overfit.\n