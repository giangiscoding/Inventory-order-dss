# Hệ hỗ trợ quyết định đặt hàng — Tối ưu tồn kho trong điều kiện không chắc chắn

Đồ án xây dựng một **hệ hỗ trợ quyết định (DSS)** trả lời câu hỏi *"Tháng này nên đặt bao nhiêu hàng?"*, gồm:

1. **Dự báo nhu cầu đa phân vị** bằng **RevIN + N-BEATS** (tự xây dựng bằng PyTorch) — cho cả *phân phối* nhu cầu thay vì một con số.
2. **So sánh công bằng 5 kiến trúc** dự báo (MLP, DLinear, TSMixer, NHITS, N-BEATS) — mỗi mô hình được Optuna tinh chỉnh riêng.
3. **Tối ưu tồn kho** bằng **Newsvendor** (tỷ lệ tới hạn) + **EOQ** (cỡ lô, safety stock, điểm đặt lại).
4. **Hai web app Streamlit** (bản kỹ thuật + bản kinh doanh) cho khuyến nghị đặt hàng.

## 1. Dữ liệu

- `Data/data_TSI_v2.csv` — **120 quan sát theo tháng** (2015-01 → 2024-12).
- Mục tiêu: `Quantity`. Ngoại sinh (10 biến): `CompetitorQuantity`, `PromotionAmount`, `Construction`, `CPI`, `Exports`, `Imports`, `IPI`, `RegisteredFDI`, `DisbursedFDI`, `RetailSales`.

### Chia dữ liệu — 7 : 1 : 2 (theo thời gian, không xáo trộn)

| Tập | Tỷ lệ | Số tháng | Khoảng thời gian |
| --- | --- | --- | --- |
| Train | 70% | 84 | 2015-01 → 2021-12 |
| Validation | 10% | 12 | 2022-01 → 2022-12 |
| Test | 20% | 24 | 2023-01 → 2024-12 |

Tập test 24 tháng (thay vì 8:1:1 thông thường) để đánh giá ổn định hơn, đặc biệt cho calibration.

## 2. Mô hình dự báo: RevIN + N-BEATS

- **RevIN** (Reversible Instance Normalization): chuẩn hóa từng cửa sổ theo mean/std của chính nó, đảo ngược khi xuất → chống *distribution shift* (vd. `PromotionAmount` tăng ~90 lần 2015→2024).
- **N-BEATS** đa phân vị: doubly-residual stacking; mỗi block xuất `horizon × n_quantiles`. Dự báo đồng thời **P5, P10, P25, P50, P75, P90, P95**; ràng buộc đơn điệu (base + cộng dồn softplus) chống *quantile crossing*.
- Huấn luyện bằng **pinball loss**; cửa sổ trượt `lookback` tháng → 1 tháng kế tiếp.

## 3. So sánh công bằng các kiến trúc

**Mỗi mô hình được Optuna tinh chỉnh riêng** (100 trial/mô hình) — điều kiện then chốt để so sánh khách quan. MLP/NHITS/N-BEATS chạy **đơn biến**; DLinear/TSMixer chạy **đa biến** (`Quantity` + 6 ngoại sinh tương quan cao nhất: IPI, Imports, RetailSales, DisbursedFDI, Exports, CompetitorQuantity).

Kết quả trên test 24 tháng (trung bình ± độ lệch chuẩn qua 3 hạt giống):

| Mô hình | MAE | MAPE | Pinball | Tham số |
| --- | --- | --- | --- | --- |
| **N-BEATS** | **3.894** | **1,12%** | **1.145** | 1,9 tr |
| DLinear (đa biến) | 4.854 | 1,42% | 1.498 | 2,4k |
| TSMixer (đa biến) | 5.648 | 1,75% | 1.615 | 13k |
| NHITS | 6.937 | 2,09% | 1.977 | 343k |
| MLP | 7.188 | 2,18% | 2.234 | 37k |

→ **N-BEATS tốt nhất rõ rệt trên mọi chỉ số** → chọn làm mô hình triển khai. (`compare_models.py` với cấu hình cố định chung được giữ làm tham khảo.)

## 4. Mô hình triển khai & đánh giá

Bộ siêu tham số N-BEATS tốt nhất (Optuna): `lookback=24, n_stacks=2, n_blocks=3, hidden=384, n_layers=3, lr≈1,95e-4, batch=16, dropout≈0,069, revin_affine=True`.

| Mô hình | MAE | RMSE | MAPE | Pinball |
| --- | --- | --- | --- | --- |
| **RevIN + N-BEATS (P50)** | **4.074** | **5.090** | **1,15%** | **1.114** |
| Naive | 80.045 | 95.211 | 23,59% | — |
| Seasonal Naive | 72.276 | 96.971 | 21,85% | — |

- Vượt baseline Naive ~**20 lần** về MAE.
- **Calibration** (coverage) trên test: kiểm tra "khi nói P90 thì ~90% thực tế có nằm dưới không"; mô hình hơi thận trọng (bao phủ vượt mức nhẹ).
- **Backtest gốc trượt** (rolling-origin, 36 tháng 2022-2024) cho ước lượng đáng tin & thận trọng hơn: **MAE ≈ 5.851, MAPE ≈ 1,77%, Pinball ≈ 1.818**.
- Ví dụ dự báo phân vị 2025-01: **P10 = 316.766 · P50 = 322.017 · P90 = 328.602**.

## 5. Tối ưu tồn kho

### Newsvendor — ý nghĩa kinh tế của dự báo phân vị

```text
Q* = F⁻¹(CR),   CR = Cu / (Cu + Co)
```

`Cu` = thiệt hại khi **thiếu** 1 đơn vị (lãi mất); `Co` = thiệt hại khi **thừa** 1 đơn vị (tồn/ế); `CR` = tỷ lệ tới hạn = **mức phân vị tối ưu**; `F⁻¹` lấy trực tiếp từ dự báo phân vị. Lãi cao/khó ế (Cu≫Co) → chuẩn bị phân vị cao; dễ ế (Co≫Cu) → giữ ít. Đây là điều dự báo điểm không làm được.

### EOQ — cỡ lô cho vận hành dài hạn

`Q* = √(2·D·S/H)`, điểm đặt lại `ROP`, tồn kho an toàn `SS = z·σ·√L`, tổng chi phí `TC = (D/Q)·S + (Q/2 + SS)·H`.

## 6. Web app (Streamlit)

- **`app/app.py`** — bản kỹ thuật: fan chart phân vị, bảng metrics + calibration, newsvendor & EOQ tương tác, công thức LaTeX.
- **`app/app_business.py`** — bản kinh doanh: luồng 3 bước, ngôn ngữ đời thường ("tháng này bán bao nhiêu?" → nhập lãi/lỗ → khuyến nghị đặt thêm).

## 7. Cấu trúc thư mục

```text
Inventory-order-dss/
├── README.md
├── requirements.txt
├── Data/data_TSI_v2.csv
├── src/
│   ├── data_loader.py        # đọc CSV, chia 7/1/2, cửa sổ trượt (đơn + đa biến)
│   ├── revin.py              # RevIN
│   ├── nbeats.py             # N-BEATS đa phân vị
│   ├── models.py             # MLP, DLinear, TSMixer, NHITS (+ phiên bản đa biến)
│   ├── train.py              # huấn luyện, pinball loss, early stopping
│   ├── tune_all.py           # Optuna RIÊNG từng mô hình (so sánh công bằng)
│   ├── deploy_nbeats.py      # triển khai N-BEATS tốt nhất → forecast.json
│   ├── compare_hparams.py    # top-5 bộ siêu tham số N-BEATS
│   ├── backtest.py           # backtest gốc trượt (đánh giá robust)
│   ├── make_report_figures.py# sinh hình fig1 + res1-5
│   ├── compare_models.py     # so sánh cấu hình cố định (tham khảo)
│   ├── tune_optuna.py        # Optuna cho N-BEATS (legacy)
│   ├── evaluate.py           # MAE/RMSE/MAPE + pinball + coverage + baseline
│   └── inventory.py          # newsvendor + EOQ
├── models/                   # best_model.pt, best_params.json
├── results/                  # metrics, forecast, comparison, tune_all, hparam_top5, backtest (JSON)
├── report/                   # report.tex/pdf, slides.tex/pdf, figures/
└── app/                      # app.py, app_business.py
```

## 8. Cách chạy

```bash
.venv/bin/pip install -r requirements.txt

python -m src.tune_all 100 3       # Optuna riêng từng mô hình (so sánh công bằng)
python -m src.deploy_nbeats        # triển khai N-BEATS tốt nhất → forecast.json
python -m src.make_report_figures  # sinh hình kết quả (fig1 + res1-5)
python -m src.compare_hparams 100 5 # top-5 siêu tham số N-BEATS
python -m src.backtest 36 12       # backtest gốc trượt 36 tháng

streamlit run app/app.py           # web app (bản kỹ thuật)
streamlit run app/app_business.py  # web app (bản kinh doanh)
```

## 9. Môi trường

- Python ≥ 3.10; thư viện chính: `torch`, `optuna`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `streamlit`, `scipy`.
