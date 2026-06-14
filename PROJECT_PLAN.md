# Hệ hỗ trợ quyết định đặt hàng — Tối ưu tồn kho trong điều kiện không chắc chắn

## 1. Mục tiêu

Xây dựng hệ hỗ trợ quyết định (DSS) giúp người quản lý trả lời câu hỏi: **"Tháng này nên đặt bao nhiêu hàng?"**, dựa trên:

1. **Dự báo nhu cầu** bằng mô hình chuỗi thời gian **RevIN + N-BEATS** (tự xây dựng bằng PyTorch).
2. **Tối ưu tồn kho** bằng mô hình **EOQ** (Economic Order Quantity) mở rộng cho điều kiện nhu cầu không chắc chắn (safety stock + reorder point).
3. **Web app demo** cho phép người dùng nhập tham số chi phí tồn kho và nhận khuyến nghị đặt hàng.

## 2. Dữ liệu

- File: `Data/data_TSI_v2.csv` — **120 quan sát theo tháng** (2015-01 → 2024-12).
- Biến mục tiêu: `Quantity` (lượng tiêu thụ/nhu cầu hàng tháng).
- Biến ngoại sinh (10 biến): `CompetitorQuantity`, `PromotionAmount`, `Construction`, `CPI`, `Exports`, `Imports`, `IPI`, `RegisteredFDI`, `DisbursedFDI`, `RetailSales`.

### Chia dữ liệu — chiến lược 8 : 1 : 1 (theo thời gian, không xáo trộn)

| Tập | Tỷ lệ | Số tháng | Khoảng thời gian |
| --- | --- | --- | --- |
| Train | 80% | 96 | 2015-01 → 2022-12 |
| Validation | 10% | 12 | 2023-01 → 2023-12 |
| Test | 10% | 12 | 2024-01 → 2024-12 |

## 3. Mô hình dự báo: RevIN + N-BEATS (tự xây dựng)

### 3.1. RevIN (Reversible Instance Normalization)

- Chuẩn hóa từng cửa sổ đầu vào (instance) bằng mean/std của chính nó, có tham số affine học được (γ, β).
- Sau khi mạng dự báo xong, **đảo ngược** phép chuẩn hóa để trả về thang đo gốc.
- Mục đích: xử lý **distribution shift** giữa các giai đoạn (dữ liệu 2015 vs 2024 khác phân phối rõ rệt — ví dụ `PromotionAmount` tăng ~100 lần).

### 3.2. N-BEATS — dự báo PHÂN VỊ (quantile)

- Kiến trúc **doubly-residual stacking**: nhiều stack, mỗi stack gồm nhiều block; mỗi block là MLP cho ra `backcast` (trừ khỏi đầu vào còn dư) và `forecast` (cộng dồn vào dự báo cuối).
- **Đầu ra đa phân vị**: thay vì một giá trị điểm, mỗi block xuất `horizon × n_quantiles` giá trị. Mô hình dự báo đồng thời các phân vị **P5, P10, P25, P50, P75, P90, P95** của nhu cầu → cho cả một **phân phối** thay vì một con số.
- **Chống quantile crossing**: ràng buộc đơn điệu P5 ≤ P10 ≤ … ≤ P95 bằng tham số hóa "phân vị thấp nhất + cộng dồn các số dương (softplus)".
- Pipeline: `input window → RevIN.normalize → N-BEATS (đa phân vị) → RevIN.denormalize → các phân vị`.
- Tạo mẫu huấn luyện bằng cửa sổ trượt: input `lookback` tháng → dự báo `horizon` tháng (mặc định horizon = 1, dự báo tháng kế tiếp).

### 3.3. Hàm mất mát & tìm siêu tham số bằng Optuna

- **Pinball loss (quantile loss)** — thước đo "đúng" cho dự báo phân vị, phạt sai số bất đối xứng theo từng mức phân vị. Optuna tối thiểu hóa **pinball loss trung bình trên tập validation**.
- Không gian tìm kiếm:

| Tham số | Khoảng |
| --- | --- |
| `lookback` | 6 – 24 tháng |
| `n_stacks` | 1 – 3 |
| `n_blocks` mỗi stack | 1 – 4 |
| `hidden_size` | 64 – 512 |
| `n_layers` mỗi block | 2 – 4 |
| `learning_rate` | 1e-4 – 1e-2 (log) |
| `batch_size` | 8 / 16 / 32 |
| `dropout` | 0 – 0.3 |
| `revin_affine` | True / False |

- Số trial: ~100, có pruner (MedianPruner) + early stopping theo val loss.
- Sau khi chọn được tham số tốt nhất: huấn luyện lại trên train (+val) rồi đánh giá **một lần duy nhất** trên test.

### 3.4. Đánh giá

- Metrics điểm (trên trung vị P50): **MAE, RMSE, sMAPE, MAPE**; metric phân vị: **Pinball loss**.
- Baseline so sánh: Naive (giá trị tháng trước), Seasonal Naive (cùng tháng năm trước).
- **Hiệu chỉnh (calibration / coverage)**: kiểm tra "khi mô hình nói P90 thì thực tế có đúng ~90% giá trị nằm dưới mức đó không" — bằng chứng cho thấy phân vị đáng tin về mặt xác suất.

## 4. Mô hình tối ưu tồn kho

### 4.1. Newsvendor — ý nghĩa kinh tế của dự báo phân vị

Đây là cầu nối kinh tế then chốt giữa dự báo phân vị và quyết định đặt hàng. Bài toán newsvendor (người bán báo) cho mức tồn kho tối ưu một kỳ:

```text
Q* = F⁻¹(CR),   CR = Cu / (Cu + Co)
```

- `Cu` — **thiệt hại khi THIẾU** 1 đơn vị = lợi nhuận biên bị mất (giá bán − giá vốn).
- `Co` — **thiệt hại khi THỪA** 1 đơn vị = chi phí ôm hàng tồn/ế (lưu kho, vốn đọng, giảm giá).
- `CR` — **tỷ lệ tới hạn**, chính là **mức phân vị tối ưu** để chuẩn bị hàng.
- `F⁻¹` — hàm phân vị nhu cầu, lấy **trực tiếp từ dự báo phân vị của N-BEATS**.

Ý nghĩa: sản phẩm **lãi cao, khó ế** (Cu ≫ Co) → CR cao → chuẩn bị ở phân vị cao (P90+), chấp nhận ôm thêm để hiếm khi hết hàng. Sản phẩm **dễ ế, chi phí tồn lớn** (Co ≫ Cu) → CR thấp → giữ ít (P20–P40), chấp nhận đôi khi hết hàng. Đây là điều một dự báo điểm đơn lẻ **không thể** diễn đạt.

### 4.2. EOQ — cỡ lô đặt hàng cho vận hành dài hạn

```text
Q* = sqrt(2 · D · S / H)
```

- `D` — nhu cầu năm (= 12 × dự báo trung vị P50), `S` — chi phí một lần đặt, `H` — chi phí lưu kho/đơn vị/năm.
- Điểm đặt hàng lại `ROP`, tồn kho an toàn `SS = z·σ·√L` (z lấy theo CR), tổng chi phí `TC = (D/Q)·S + (Q/2 + SS)·H`.

### 4.3. Đầu ra khuyến nghị

- **Mức chuẩn bị tối ưu tháng này** = phân vị P(CR) của nhu cầu (newsvendor) → **đặt thêm bao nhiêu** = mức đó − tồn kho hiện có.
- **Q\*** (cỡ lô), **số lần đặt/năm**, **chu kỳ**, **tổng chi phí** (EOQ) cho vận hành dài hạn.
- So sánh **chi phí thiếu hàng vs tồn dư kỳ vọng** ở mức khuyến nghị, và thiệt hại nếu chỉ chuẩn bị theo mức trung vị.

## 5. Web app demo

- **Công nghệ**: Streamlit (toàn bộ Python). Hai phiên bản cho hai đối tượng:

**`app.py` — bản kỹ thuật**: fan chart phân vị (P5–P95, P25–P75, P50), bảng metrics (MAE/Pinball/MAPE) + bảng calibration, mục newsvendor (nhập Cu, Co → tỷ lệ tới hạn → phân vị mục tiêu, kèm biểu đồ hàm phân vị), mục EOQ (đường cong chi phí), công thức LaTeX.

**`app_business.py` — bản cho người dùng business** (ngôn ngữ đời thường, không thuật ngữ):

1. *"Tháng này bán được bao nhiêu?"* — hiển thị mức hay gặp (P50) kèm khoảng dao động: tháng bán chậm (~P10) đến bán chạy (~P90).
2. *"Tình hình & lãi/lỗ của bạn"* — tồn kho, lead time (theo ngày), và 2 con số kinh tế: **lãi mỗi sản phẩm** (Cu) và **thiệt hại mỗi sản phẩm ế** (Co).
3. *Khuyến nghị* — giải thích bằng lời "lãi cao thì chuẩn bị dư / dễ ế thì giữ ít", đưa ra **mức chuẩn bị hợp lý** và **nên đặt thêm bao nhiêu** (kèm nhắc lead time), và so sánh thiệt hại kỳ vọng nếu chỉ chuẩn bị mức trung bình.

## 6. Cấu trúc thư mục

```text
Hehotroquyetdinh/
├── PROJECT_PLAN.md
├── requirements.txt
├── Data/
│   └── data_TSI_v2.csv
├── src/
│   ├── data_loader.py        # đọc CSV, chia 8/1/1, tạo cửa sổ trượt, Dataset
│   ├── revin.py              # lớp RevIN tự xây dựng
│   ├── nbeats.py             # N-BEATS (Block, Stack, Model) tự xây dựng
│   ├── train.py              # vòng lặp huấn luyện (pinball loss), early stopping
│   ├── tune_optuna.py        # Optuna (pinball), lưu phân vị + coverage
│   ├── evaluate.py           # MAE/RMSE/sMAPE + pinball loss + coverage + baseline
│   └── inventory.py          # newsvendor (phân vị) + EOQ, safety stock, ROP
├── models/
│   ├── best_model.pt         # checkpoint tốt nhất (đa phân vị)
│   └── best_params.json      # siêu tham số + danh sách phân vị + σ
├── results/
│   ├── metrics.json          # metrics + coverage
│   ├── forecast.json         # phân vị test & tương lai cho web app
│   └── forecast_plot.png     # fan chart
└── app/
    ├── app.py                # Streamlit web app (bản kỹ thuật)
    └── app_business.py       # Streamlit web app (bản cho người dùng business)
```

## 7. Lộ trình thực hiện

| Bước | Nội dung | Kết quả |
| --- | --- | --- |
| 1 | Lập kế hoạch dự án (file này) | ✅ PROJECT_PLAN.md |
| 2 | Pipeline dữ liệu: load, chia 8/1/1, sliding window | ✅ `data_loader.py` |
| 3 | Xây dựng RevIN + N-BEATS đa phân vị bằng PyTorch | ✅ `revin.py`, `nbeats.py` |
| 4 | Huấn luyện (pinball loss) + Optuna tuning (100 trial) | ✅ `best_model.pt`, `best_params.json` |
| 5 | Đánh giá test: metrics điểm + pinball + coverage | ✅ `metrics.json` |
| 6 | Mô hình tồn kho: newsvendor (phân vị) + EOQ | ✅ `inventory.py` |
| 7 | Web app Streamlit (bản kỹ thuật + bản business) | ✅ `app/app.py`, `app/app_business.py` |

### Kết quả thực tế (tập test 2024-01 → 2024-12)

| Mô hình | MAE | RMSE | MAPE | Pinball loss |
| --- | --- | --- | --- | --- |
| **RevIN + N-BEATS (P50)** | **3.254** | **4.299** | **0,97%** | **1.027** |
| Naive | 79.188 | 99.343 | 22,76% | — |
| Seasonal Naive | 71.495 | 96.009 | 23,37% | — |

- Tham số tốt nhất (Optuna, 100 trial, objective = pinball loss val): lookback=24, n_stacks=2, n_blocks=2, hidden=512, n_layers=3, lr≈3,1e-4, batch=8, dropout≈0,002, revin_affine=False.
- **Coverage (calibration) trên test**: do tập test chỉ 12 điểm nên độ phân giải thô (mỗi điểm = 8,3%); P50 bao phủ 58%, P75 → 75%, P90 → 83%. Đầu cao (P90/P95) hơi thiếu phủ trên tập nhỏ này — cần thêm dữ liệu test để hiệu chỉnh chắc chắn hơn.
- σ sai số dự báo (std residuals P50): ≈ 4.482 — dùng cho phần safety stock của EOQ.
- Ví dụ dự báo phân vị tháng 2025-01: **P10 = 315.573 · P50 = 319.507 · P90 = 322.998**.

## 8. Cách chạy

```bash
# Cài đặt (đã có sẵn .venv trong dự án)
.venv/bin/pip install -r requirements.txt

# Huấn luyện lại + tuning (tùy chọn, mặc định 100 trial)
.venv/bin/python -m src.tune_optuna 100

# Chạy web app (bản kỹ thuật — đầy đủ metrics, công thức)
.venv/bin/streamlit run app/app.py

# Chạy web app (bản business — ngôn ngữ đơn giản, tập trung ra quyết định)
.venv/bin/streamlit run app/app_business.py
```

## 9. Môi trường

- Python ≥ 3.10
- Thư viện chính: `torch`, `optuna`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `streamlit`, `scipy`.
