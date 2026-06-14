"""Web app demo (ban ky thuat): He ho tro quyet dinh dat hang dua tren du bao
PHAN VI (RevIN+N-BEATS) va toi uu ton kho (newsvendor + EOQ).

Chay:  streamlit run app/app.py
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

from src.inventory import InventoryInputs, cost_curve, newsvendor, optimize  # noqa: E402

st.set_page_config(page_title="DSS Dat hang toi uu", page_icon="📦", layout="wide")


@st.cache_data
def load_forecast():
    with open(ROOT / "results" / "forecast.json") as f:
        return json.load(f)


fc = load_forecast()
QUANTILES = fc["quantiles"]
QLABELS = [f"P{int(round(q * 100))}" for q in QUANTILES]
MEDIAN_IDX = fc["median_idx"]


def q_index(level: float) -> int:
    return int(np.argmin(np.abs(np.array(QUANTILES) - level)))


st.title("📦 Hệ hỗ trợ quyết định đặt hàng")
st.caption(
    "Dự báo **phân vị nhu cầu** bằng RevIN + N-BEATS (pinball loss, Optuna, chia 8/1/1) "
    "→ tối ưu tồn kho theo mô hình **newsvendor** (tỷ lệ tới hạn) và **EOQ**."
)

# ====== 1. Du bao phan vi ======
st.header("1️⃣ Dự báo nhu cầu — dạng phân vị")

hist = pd.DataFrame({
    "Tháng": pd.to_datetime(fc["history_months"], format="%Y-%m"),
    "Giá trị": fc["history"], "Loại": "Thực tế",
})
fut_q = np.array(fc["future_quantiles"])           # (12, Q)
fut_months = pd.to_datetime(fc["future_months"], format="%Y-%m")
band_df = pd.DataFrame({
    "Tháng": fut_months,
    "p05": fut_q[:, 0], "p95": fut_q[:, -1],
    "p25": fut_q[:, q_index(0.25)], "p75": fut_q[:, q_index(0.75)],
    "p50": fut_q[:, MEDIAN_IDX],
})

base_hist = alt.Chart(hist).mark_line(color="#4682b4").encode(
    x=alt.X("Tháng:T", title=None),
    y=alt.Y("Giá trị:Q", title="Nhu cầu (đơn vị)", scale=alt.Scale(zero=False)),
)
band_outer = alt.Chart(band_df).mark_area(opacity=0.15, color="#dc143c").encode(
    x="Tháng:T", y="p05:Q", y2="p95:Q")
band_inner = alt.Chart(band_df).mark_area(opacity=0.30, color="#dc143c").encode(
    x="Tháng:T", y="p25:Q", y2="p75:Q")
median_line = alt.Chart(band_df).mark_line(color="#dc143c", strokeDash=[5, 4]).encode(
    x="Tháng:T", y="p50:Q")
st.altair_chart((band_outer + band_inner + base_hist + median_line).properties(height=340),
                width="stretch")
st.caption("Vùng đỏ đậm: khoảng P25–P75 (50% khả năng). Vùng đỏ nhạt: P5–P95 (90% khả năng). "
           "Đường nét đứt: trung vị P50.")

m = fc["metrics"]["RevIN+NBEATS"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE / trung vị (test)", f"{m['MAE']:,.0f}")
c2.metric("Pinball loss (test)", f"{m['PinballLoss']:,.0f}")
c3.metric("MAPE / trung vị (test)", f"{m['MAPE']:.2f}%")
c4.metric("σ sai số dự báo/tháng", f"{fc['sigma_demand_monthly']:,.0f}")

with st.expander("Kiểm tra hiệu chỉnh (calibration) & so sánh baseline"):
    cov = fc["coverage"]
    cov_df = pd.DataFrame({
        "Mức phân vị": [f"P{int(round(float(k) * 100))}" for k in cov],
        "Lý tưởng (= mức)": [float(k) for k in cov],
        "Thực tế bao phủ": [cov[k] for k in cov],
    })
    st.caption("Cột 'thực tế bao phủ' = tỷ lệ giá trị thật ≤ phân vị dự báo trên tập test. "
               "Càng gần cột lý tưởng thì mô hình càng đáng tin về mặt xác suất.")
    st.dataframe(cov_df.style.format({"Lý tưởng (= mức)": "{:.2f}", "Thực tế bao phủ": "{:.2f}"}),
                 width="stretch")
    base_df = pd.DataFrame({k: v for k, v in fc["metrics"].items()}).T
    st.dataframe(base_df.style.format("{:,.2f}"), width="stretch")

# ====== 2. Thong so ======
st.header("2️⃣ Nhập thông số")

month_options = fc["future_months"]
plan_month = st.selectbox("Tháng lập kế hoạch", month_options, index=0)
idx = month_options.index(plan_month)
q_vals = list(fut_q[idx])
demand_p50 = q_vals[MEDIAN_IDX]
st.info(
    f"Nhu cầu dự báo tháng **{plan_month}** — "
    f"P10 = **{q_vals[q_index(0.1)]:,.0f}**, "
    f"P50 = **{demand_p50:,.0f}**, "
    f"P90 = **{q_vals[q_index(0.9)]:,.0f}** đơn vị."
)

with st.form("inv_form"):
    st.markdown("**Tham số kinh tế (cho mô hình newsvendor)**")
    e1, e2, e3 = st.columns(3)
    with e1:
        underage = st.number_input(
            "Cu — thiệt hại khi THIẾU 1 đơn vị (lợi nhuận mất, đ)",
            min_value=1.0, value=30_000.0, step=5_000.0,
            help="Thường = lợi nhuận biên = giá bán − giá vốn (cộng thêm thiệt hại uy tín nếu có).")
    with e2:
        overage = st.number_input(
            "Co — thiệt hại khi THỪA 1 đơn vị (tồn/ế, đ)",
            min_value=1.0, value=4_000.0, step=1_000.0,
            help="Chi phí ôm 1 đơn vị dư trong kỳ: lưu kho, vốn đọng, hao hụt/giảm giá.")
    with e3:
        current_stock = st.number_input("Tồn kho hiện tại (đơn vị)", min_value=0.0,
                                        value=100_000.0, step=5_000.0)
        on_order = st.number_input("Hàng đã đặt đang về (đơn vị)", min_value=0.0,
                                   value=0.0, step=5_000.0)

    st.markdown("**Tham số EOQ (cho cỡ lô đặt hàng dài hạn)**")
    b1, b2, b3 = st.columns(3)
    with b1:
        ordering_cost = st.number_input("Chi phí một lần đặt hàng S (đ)", min_value=1.0,
                                        value=5_000_000.0, step=100_000.0)
    with b2:
        holding_cost = st.number_input("Chi phí lưu kho H (đ/đơn vị/năm)", min_value=0.01,
                                       value=48_000.0, step=1_000.0,
                                       help="Mặc định ≈ Co × 12 nếu Co là chi phí tồn theo tháng.")
    with b3:
        lead_time = st.number_input("Lead time (tháng)", min_value=0.1, value=1.0, step=0.5)

    submitted = st.form_submit_button("🔍 Tính toán phương án đặt hàng", width="stretch")

# ====== 3. Ket qua ======
if submitted:
    position = current_stock + on_order
    nv = newsvendor(QUANTILES, q_vals, underage, overage, current_position=position)

    st.header(f"3️⃣ Kết quả — tháng {plan_month}")

    # ---- Newsvendor: y nghia kinh te ----
    st.subheader("🎯 Mức chuẩn bị tối ưu theo kinh tế (newsvendor)")
    cr_pct = nv.critical_ratio * 100
    st.markdown(
        f"Tỷ lệ tới hạn **CR = Cu / (Cu + Co) = {underage:,.0f} / ({underage:,.0f} + {overage:,.0f}) "
        f"= {nv.critical_ratio:.3f}**. "
        f"Vì vậy mức tồn kho tối ưu cho tháng này là **phân vị P{cr_pct:.0f}** của nhu cầu — "
        f"tức **{nv.target_stock:,.0f} đơn vị**."
    )
    if nv.critical_ratio >= 0.5:
        st.markdown(
            f"➡️ Lãi khi bán (Cu) **lớn hơn** thiệt hại khi tồn (Co), nên nên **chuẩn bị dư** — "
            f"chấp nhận ôm thêm hàng để hiếm khi bị hết. Chuẩn bị ở mức cao hơn trung vị "
            f"({nv.target_stock - nv.p50:,.0f} đơn vị trên P50)."
        )
    else:
        st.markdown(
            f"➡️ Thiệt hại khi tồn (Co) **lớn hơn** lãi khi bán (Cu), nên nên **giữ ít** — "
            f"chấp nhận đôi khi hết hàng để tránh ôm hàng ế. Chuẩn bị ở mức thấp hơn trung vị "
            f"({nv.p50 - nv.target_stock:,.0f} đơn vị dưới P50)."
        )

    if nv.order_quantity > 0:
        st.success(f"### ✅ ĐẶT THÊM **{nv.order_quantity:,.0f} đơn vị** để đạt mức chuẩn bị "
                   f"{nv.target_stock:,.0f} (hiện có {position:,.0f}).")
    else:
        st.info(f"### ⏸️ CHƯA CẦN ĐẶT — tồn kho hiện tại ({position:,.0f}) đã ≥ mức chuẩn bị "
                f"tối ưu ({nv.target_stock:,.0f}).")

    # Bieu do ham phan vi (inverse CDF) + diem CR
    qcurve = pd.DataFrame({"Xác suất (phân vị)": QUANTILES, "Nhu cầu": q_vals})
    curve = alt.Chart(qcurve).mark_line(point=True, color="#2c6fae").encode(
        x=alt.X("Xác suất (phân vị):Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Nhu cầu:Q", scale=alt.Scale(zero=False)))
    cr_point = alt.Chart(pd.DataFrame({"x": [nv.critical_ratio], "y": [nv.target_stock]})).mark_point(
        size=160, color="red", filled=True).encode(x="x:Q", y="y:Q")
    cr_rule = alt.Chart(pd.DataFrame({"x": [nv.critical_ratio]})).mark_rule(
        color="red", strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart((curve + cr_rule + cr_point).properties(height=280), width="stretch")
    st.caption(f"Hàm phân vị nhu cầu. Điểm đỏ tại xác suất CR = {nv.critical_ratio:.3f} "
               f"cho mức tồn kho tối ưu {nv.target_stock:,.0f}.")

    nk1, nk2, nk3 = st.columns(3)
    nk1.metric("Phân vị mục tiêu", f"P{cr_pct:.0f}")
    nk2.metric("Chi phí thiếu hàng kỳ vọng", f"{nv.expected_understock_cost:,.0f} đ")
    nk3.metric("Chi phí tồn dư kỳ vọng", f"{nv.expected_overstock_cost:,.0f} đ")

    # ---- EOQ: co lo dat hang ----
    st.divider()
    st.subheader("📦 Cỡ lô đặt hàng tối ưu (EOQ) cho vận hành dài hạn")
    inp = InventoryInputs(
        demand_monthly=demand_p50, sigma_monthly=fc["sigma_demand_monthly"],
        ordering_cost=ordering_cost, holding_cost_annual=holding_cost,
        lead_time_months=lead_time, service_level=nv.critical_ratio,
        current_stock=current_stock, on_order=on_order)
    res = optimize(inp)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Q* — cỡ lô tối ưu", f"{res.eoq:,.0f}")
    k2.metric("Số lần đặt/năm", f"{res.orders_per_year:.1f}")
    k3.metric("Chu kỳ đặt hàng", f"{res.cycle_months:.2f} tháng")
    k4.metric("TỔNG chi phí/năm", f"{res.total_annual_cost:,.0f} đ")

    qs, ordering_c, holding_c, total = cost_curve(inp)
    cost_df = pd.concat([
        pd.DataFrame({"Q": qs, "Chi phí (đ/năm)": ordering_c, "Thành phần": "Chi phí đặt hàng"}),
        pd.DataFrame({"Q": qs, "Chi phí (đ/năm)": holding_c, "Thành phần": "Chi phí lưu kho"}),
        pd.DataFrame({"Q": qs, "Chi phí (đ/năm)": total, "Thành phần": "Tổng chi phí"}),
    ])
    cost_chart = alt.Chart(cost_df).mark_line().encode(
        x=alt.X("Q:Q", title="Lượng đặt hàng Q (đơn vị)"),
        y=alt.Y("Chi phí (đ/năm):Q"), color="Thành phần:N")
    rule = alt.Chart(pd.DataFrame({"Q": [res.eoq]})).mark_rule(
        color="red", strokeDash=[5, 4]).encode(x="Q:Q")
    st.altair_chart((cost_chart + rule).properties(height=300), width="stretch")
    st.caption(f"Đường đỏ: Q* = {res.eoq:,.0f} — cỡ lô có tổng chi phí đặt + lưu kho thấp nhất.")

    with st.expander("Chi tiết công thức"):
        st.latex(r"\text{CR} = \frac{C_u}{C_u + C_o} = "
                 rf"\frac{{{underage:,.0f}}}{{{underage:,.0f} + {overage:,.0f}}} = {nv.critical_ratio:.3f}")
        st.latex(rf"Q^*_{{\text{{newsvendor}}}} = F^{{-1}}(\text{{CR}}) = P{cr_pct:.0f} = {nv.target_stock:,.0f}")
        st.latex(r"Q^*_{\text{EOQ}} = \sqrt{\frac{2DS}{H}} = "
                 rf"\sqrt{{\frac{{2 \times {res.annual_demand:,.0f} \times {ordering_cost:,.0f}}}{{{holding_cost:,.0f}}}}} = {res.eoq:,.0f}")
