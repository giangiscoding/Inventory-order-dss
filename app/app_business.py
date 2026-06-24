"""Tro ly dat hang (ban kinh doanh) — ngon ngu don gian, giao dien dashboard.

Chay:  streamlit run app/app_business.py
"""
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ui  # noqa: E402
from src.inventory import newsvendor  # noqa: E402

ui.setup("Trợ lý đặt hàng", "🛒", layout="wide")


def money(x: float) -> str:
    if abs(x) >= 1e9:
        return f"{x/1e9:,.1f} tỷ đ".replace(",", ".")
    if abs(x) >= 1e6:
        return f"{x/1e6:,.0f} triệu đ".replace(",", ".")
    return f"{x:,.0f} đ".replace(",", ".")


def qty(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


@st.cache_data
def load_forecast():
    with open(ROOT / "results" / "forecast.json") as f:
        return json.load(f)


fc = load_forecast()
QUANTILES = fc["quantiles"]
MEDIAN_IDX = fc["median_idx"]
fut_q = np.array(fc["future_quantiles"])
fut_months = fc["future_months"]


def q_index(level: float) -> int:
    return int(np.argmin(np.abs(np.array(QUANTILES) - level)))


with st.sidebar:
    st.markdown("## 🛒 Trợ lý đặt hàng")
    st.caption("Dành cho quản lý cửa hàng")
    st.divider()
    st.markdown("**Cách dùng**")
    st.caption("1️⃣ Chọn tháng & xem dự kiến bán")
    st.caption("2️⃣ Nhập tồn kho và lãi/lỗ")
    st.caption("3️⃣ Nhận khuyến nghị đặt thêm")

ui.topbar("Trợ lý đặt hàng tháng này",
          "Tháng này bán bao nhiêu? · Nên chuẩn bị bao nhiêu? · Cần đặt thêm bao nhiêu?")

# ============ BUOC 1 ============
with st.container(border=True):
    ui.section("Bước 1 — Tháng này dự kiến bán bao nhiêu?")
    plan_month = st.selectbox("Chọn tháng cần lập kế hoạch", fut_months, index=0)
    idx = fut_months.index(plan_month)
    q_vals = list(fut_q[idx])
    p10, p50, p90 = q_vals[q_index(0.1)], q_vals[MEDIAN_IDX], q_vals[q_index(0.9)]

    k1, k2, k3 = st.columns(3)
    k1.metric("📈 Hay gặp nhất", qty(p50), help="Mức bán dễ xảy ra nhất (trung vị).")
    k2.metric("🔻 Tháng bán chậm", qty(p10), help="10% khả năng bán dưới mức này.")
    k3.metric("🔺 Tháng bán chạy", qty(p90), help="10% khả năng bán trên mức này.")

    band = pd.DataFrame({"Tháng": pd.to_datetime(fut_months, format="%Y-%m"),
                         "lo": fut_q[:, q_index(0.1)], "hi": fut_q[:, q_index(0.9)],
                         "mid": fut_q[:, MEDIAN_IDX]})
    area = alt.Chart(band).mark_area(opacity=0.22, color=ui.BRAND).encode(
        x=alt.X("Tháng:T", title=None), y=alt.Y("lo:Q", title="Số lượng",
                                                scale=alt.Scale(zero=False)), y2="hi:Q")
    mid = alt.Chart(band).mark_line(color=ui.BRAND, strokeWidth=2.5).encode(x="Tháng:T", y="mid:Q")
    st.altair_chart(ui.style_chart((area + mid).properties(height=230)), use_container_width=True)
    st.caption("👉 Nhu cầu luôn dao động: đặt ít thì sợ hết, đặt nhiều thì sợ ế. "
               "Bước 3 sẽ chọn mức hợp lý nhất theo lãi/lỗ của bạn.")
    demand_p50 = st.number_input("Có gì đặc biệt tháng này (khuyến mãi lớn…)? Chỉnh mức hay gặp tại đây:",
                                 min_value=1.0, value=float(round(p50)), step=1000.0)
    q_vals_adj = [v + (demand_p50 - p50) for v in q_vals]

# ============ BUOC 2 ============
with st.container(border=True):
    ui.section("Bước 2 — Tình hình & lãi/lỗ của bạn")
    c1, c2 = st.columns(2)
    with c1:
        current_stock = st.number_input("📦 Trong kho còn bao nhiêu?", 0.0, value=100_000.0, step=5_000.0)
        on_order = st.number_input("🚚 Hàng đã đặt chưa về?", 0.0, value=0.0, step=5_000.0)
        lead_days = st.number_input("⏱️ Đặt xong mấy ngày thì về kho?", 1, value=30, step=1)
    with c2:
        profit_per_unit = st.number_input("💰 Bán 1 sản phẩm LÃI bao nhiêu? (đ)", 1.0,
                                          value=30_000.0, step=5_000.0,
                                          help="Lãi gộp = giá bán − giá vốn. Hết hàng = mất khoản này.")
        stockout_penalty = st.number_input("💔 Hết hàng — thiệt hại THÊM mỗi sp? (đ)", 0.0,
                                           value=0.0, step=5_000.0,
                                           help="Mất uy tín, khách bỏ đi… Hàng dễ thay thế → để 0.")
        waste_per_unit = st.number_input("📉 1 sản phẩm Ế cuối tháng thiệt hại? (đ)", 1.0,
                                         value=4_000.0, step=1_000.0,
                                         help="Kho bãi, vốn đọng, giảm giá. Hàng để lâu → nhỏ.")
    shortage_cost = profit_per_unit + stockout_penalty

# ============ BUOC 3 ============
ui.section("Bước 3 — Khuyến nghị của hệ thống")
if st.button("👉 Cho tôi biết nên làm gì", type="primary", use_container_width=True):
    position = current_stock + on_order
    nv = newsvendor(QUANTILES, q_vals_adj, shortage_cost, waste_per_unit, current_position=position)
    cr_pct = nv.critical_ratio * 100

    if nv.order_quantity > 0:
        ui.reco_banner("order", f"🔴 Nên đặt thêm {qty(nv.order_quantity)} sản phẩm",
                       f"Hiện có {qty(position)} sp, chưa đủ mức chuẩn bị hợp lý "
                       f"({qty(nv.target_stock)}). Đặt sớm vì hàng cần ~{int(lead_days)} ngày mới về.")
    else:
        ui.reco_banner("hold", "🟢 Chưa cần đặt thêm",
                       f"Đang có {qty(position)} sp, đã đủ mức chuẩn bị hợp lý "
                       f"({qty(nv.target_stock)}) cho tháng này.")

    st.write("")
    if stockout_penalty > 0:
        short_desc = (f"thiếu 1 sp thiệt hại {money(shortage_cost)} "
                      f"(lãi mất {money(profit_per_unit)} + ảnh hưởng {money(stockout_penalty)})")
    else:
        short_desc = f"thiếu 1 sp mất khoản lãi {money(shortage_cost)}"
    if nv.critical_ratio >= 0.5:
        reason = (f"Bạn cho biết {short_desc}, trong khi ôm 1 sp ế chỉ mất {money(waste_per_unit)}. "
                  f"Thiệt hại khi thiếu lớn hơn ⟹ nên **chuẩn bị dư một chút** để không lỡ doanh thu.")
    else:
        reason = (f"Ôm 1 sp ế mất {money(waste_per_unit)}, trong khi {short_desc}. "
                  f"Rủi ro ế lớn hơn ⟹ nên **giữ ít lại**, chấp nhận đôi lúc hết hàng.")

    with st.container(border=True):
        st.markdown(f"#### 💡 {reason}")
        st.markdown(f"Mức chuẩn bị hợp lý: đủ cho kịch bản bán tới **{qty(nv.target_stock)} sản phẩm** "
                    f"(đủ hàng trong khoảng **{cr_pct:.0f}%** số tháng).")

        dense_p = np.linspace(0.001, 0.999, 999)
        dense_d = np.interp(dense_p, QUANTILES, q_vals_adj)

        def eloss(stock):
            return (np.clip(dense_d - stock, 0, None).mean() * shortage_cost
                    + np.clip(stock - dense_d, 0, None).mean() * waste_per_unit)

        extra = eloss(demand_p50) - eloss(nv.target_stock)
        m1, m2, m3 = st.columns(3)
        m1.metric("Mức chuẩn bị khuyến nghị", qty(nv.target_stock))
        m2.metric("Cần đặt thêm", qty(nv.order_quantity))
        m3.metric("Nếu chỉ theo mức 'hay gặp'", qty(demand_p50),
                  delta=f"tốn thêm {money(extra)}/tháng", delta_color="inverse")
        st.caption("Chuẩn bị đúng mức 'hay gặp' nghe hợp lý nhưng về lâu dài tốn hơn — "
                   "mức khuyến nghị đã cân nhắc cả lãi lẫn rủi ro ế.")
else:
    st.info("Điền thông tin ở Bước 1–2 rồi bấm nút để nhận khuyến nghị.")
