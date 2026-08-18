# Khám phá Tương tác Cặp Đặc trưng (`feature_interactions`)

Module `feature_interactions` tự động phát hiện và sinh ra các đặc trưng tương tác bậc hai (Pairwise Feature Interactions: phép nhân $F_1 \times F_2$, tỷ lệ $F_1 / F_2$, hiệu số $F_1 - F_2$) và đánh giá giá trị thông tin gia tăng của chúng so với các đặc trưng gốc.

---

## 1. Mục đích & Ứng dụng

- **Tự động Hóa Feature Engineering**: Tự động tìm kiếm các chỉ báo kết hợp mới mà con người khó nghĩ ra bằng trực giác thông thường.
- **Mở rộng Không gian Biểu diễn Tuyến tính**: Giúp các mô hình tuyến tính (Linear / Logistic Regression) học được các quan hệ phi tuyến mà không cần chuyển sang mô hình hộp đen phức tạp.
- **Đánh giá Đóng góp Gia tăng (Incremental Value)**: Chỉ giữ lại các đặc trưng tương tác có sức mạnh dự báo vượt trội hơn cả 2 đặc trưng gốc cấu thành nó.

---

## 2. Phương pháp & Nguyên lý Tính toán

1. **Sàng lọc Top Ứng viên Gốc (Base Feature Filtering)**:
   - Chọn ra top $N$ đặc trưng gốc đơn lẻ có tương quan tốt nhất với nhãn để tránh bùng nổ tổ hợp $\mathcal{O}(N^2)$.
2. **Sinh Đặc trưng Tương tác (Interaction Generation)**:
   - **Phép Nhân (Cross-product)**: $I_{\text{mult}} = F_1 \times F_2$ (Mô hình hóa tác động đồng thuận).
   - **Tỷ Lệ (Ratio)**: $I_{\text{ratio}} = F_1 / (F_2 + \epsilon)$ (Mô hình hóa sự mất cân đối / tốc độ tương đối).
   - **Hiệu Số (Spread)**: $I_{\text{diff}} = F_1 - F_2$ (Mô hình hóa độ lệch chuẩn hóa).
3. **Đánh giá & Xếp hạng**:
   - Đánh giá điểm tương quan/thông tin hỗ tương của biến tương tác mới với nhãn mục tiêu và so sánh với điểm của $F_1, F_2$.

---

## 3. Cú pháp Lệnh CLI

```bash
# Chạy khám phá tương tác đặc trưng
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module feature_interactions

# Chỉ định nhãn mục tiêu và giới hạn dữ liệu
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module feature_interactions \
  --target allow_entry \
  --limit 20000

# Chỉ định thư mục xuất kết quả
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module feature_interactions \
  --output-dir reports/feature_interactions_run
```

---

## 4. Cấu trúc Kết quả Đầu ra (Artifacts)

| Tên File | Loại | Mô tả Chi tiết |
| :--- | :--- | :--- |
| `generated_interactions.csv` | CSV | Danh sách tất cả các biến tương tác mới được sinh ra kèm công thức toán học và điểm số. |
| `feature_scores.csv` | CSV | Bảng xếp hạng toàn bộ các biến (gốc + tương tác) theo điểm dự báo. |
| `top_features.csv` | CSV | Top 50 biến hiệu quả nhất sau khi bổ sung tương tác. |
| `summary.json` | JSON | Metadata tổng kết các tương tác vượt trội nhất. |
| `report.md` / `report.html` | Báo cáo | Báo cáo Markdown và HTML tương tác. |

---

## 5. Hướng dẫn Đọc hiểu & Khuyến nghị

- **Tương tác Đắt giá (High-Synergy Interactions)**: Tìm các dòng trong `generated_interactions.csv` có điểm số cao hơn đáng kể ($> 20\%$) so với cả 2 biến gốc $F_1$ và $F_2$.
- **Tích hợp vào Bộ Sinh Đặc trưng**: Thêm công thức của các biến tương tác hàng đầu vào pipeline tiền xử lý dữ liệu trước khi train model chính.\n