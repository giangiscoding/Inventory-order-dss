"""Dashboard ky thuat: He ho tro quyet dinh dat hang (RevIN+N-BEATS + ton kho (r,q)).

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

import ui  # noqa: E402
from src.inventory import InventoryInputs, cost_curve, newsvendor, optimize  # noqa: E402

ui.setup("OrderSense — DSS Tồn kho", "📦", layout="wide")


def qty(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def money(x: float) -> str:
    if abs(x) >= 1e9:
        return f"{x/1e9:,.2f} tỷ".replace(",", ".")
    if abs(x) >= 1e6:
        return f"{x/1e6:,.1f} tr".replace(",", ".")
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
m = fc["metrics"]["RevIN+NBEATS"]


def q_index(level: float) -> int:
    return int(np.argmin(np.abs(np.array(QUANTILES) - level)))


# ============ Sidebar ============
with st.sidebar:
    st.markdown("## 📦 OrderSense")
    st.caption("Hệ hỗ trợ quyết định đặt hàng & tồn kho")
    st.divider()
    view = st.radio("Điều hướng", ["Tổng quan", "Lập kế hoạch đặt hàng",
                                   "Hiệu suất mô hình"], label_visibility="collapsed")
    st.divider()
    st.caption("Mô hình: RevIN + N-BEATS (đa phân vị)")
    st.caption(f"Dữ liệu: {fc['history_months'][0]} → {fc['history_months'][-1]}")

ui.topbar("OrderSense", "Dự báo nhu cầu đa phân vị → tối ưu tồn kho theo chính sách (r, q)")

# =====================================================================
# VIEW 1 — TONG QUAN
# =====================================================================
if view == "Tổng quan":
    p10, p50, p90 = fut_q[0, q_index(0.1)], fut_q[0, MEDIAN_IDX], fut_q[0, q_index(0.9)]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Dự báo {fut_months[0]} (P50)", qty(p50))
    k2.metric("Khoảng P10–P90", f"{qty(p10)} – {qty(p90)}")
    k3.metric("MAPE mô hình (test)", f"{m['MAPE']:.2f}%")
    k4.metric("σ sai số / tháng", qty(fc["sigma_demand_monthly"]))

    st.write("")
    with st.container(border=True):
        ui.section("Dự báo nhu cầu 12 tháng tới (fan chart phân vị)")
        hist = pd.DataFrame({"Tháng": pd.to_datetime(fc["history_months"], format="%Y-%m"),
                             "Giá trị": fc["history"]}).tail(36)
        fm = pd.to_datetime(fut_months, format="%Y-%m")
        band = pd.DataFrame({"Tháng": fm, "p05": fut_q[:, 0], "p95": fut_q[:, -1],
                             "p25": fut_q[:, q_index(0.25)], "p75": fut_q[:, q_index(0.75)],
                             "p50": fut_q[:, MEDIAN_IDX]})
        outer = alt.Chart(band).mark_area(opacity=0.14, color=ui.BAND).encode(
            x=alt.X("Tháng:T", title=None), y=alt.Y("p05:Q", title="Nhu cầu (đơn vị)",
                                                    scale=alt.Scale(zero=False)), y2="p95:Q")
        inner = alt.Chart(band).mark_area(opacity=0.28, color=ui.BAND).encode(
            x="Tháng:T", y="p25:Q", y2="p75:Q")
        hline = alt.Chart(hist).mark_line(color="#475569", strokeWidth=2).encode(
            x="Tháng:T", y=alt.Y("Giá trị:Q", scale=alt.Scale(zero=False)))
        mline = alt.Chart(band).mark_line(color=ui.BRAND, strokeWidth=2.5).encode(
            x="Tháng:T", y="p50:Q")
        st.altair_chart(ui.style_chart((outer + inner + hline + mline).properties(height=340)),
                        use_container_width=True)
        st.caption("Xám: lịch sử thực tế · xanh đậm: trung vị P50 · vùng xanh: P25–P75 (50%) "
                   "và P5–P95 (90%).")

    c1, c2 = st.columns([2, 1])
    with c1.container(border=True):
        ui.section("Chi tiết dự báo 12 tháng")
        tbl = pd.DataFrame({
            "Tháng": fut_months,
            "P10": [qty(v) for v in fut_q[:, q_index(0.1)]],
            "P50 (hay gặp)": [qty(v) for v in fut_q[:, MEDIAN_IDX]],
            "P90": [qty(v) for v in fut_q[:, q_index(0.9)]],
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True, height=330)
    with c2.container(border=True):
        ui.section("Mô hình đề xuất")
        st.metric("MAE (test)", qty(m["MAE"]))
        st.metric("Pinball loss (test)", qty(m["PinballLoss"]))
        st.caption("N-BEATS tinh chỉnh Optuna, vượt baseline Naive ~20 lần về MAE.")

# =====================================================================
# VIEW 2 — LAP KE HOACH DAT HANG
# =====================================================================
elif view == "Lập kế hoạch đặt hàng":
    with st.container(border=True):
        ui.section("Thông số kế hoạch")
        plan_month = st.selectbox("Tháng lập kế hoạch", fut_months, index=0)
        idx = fut_months.index(plan_month)
        q_vals = list(fut_q[idx])
        demand_p50 = q_vals[MEDIAN_IDX]
        st.caption(f"Nhu cầu dự báo {plan_month}: P10 = {qty(q_vals[q_index(0.1)])} · "
                   f"P50 = {qty(demand_p50)} · P90 = {qty(q_vals[q_index(0.9)])}")
        with st.form("plan"):
            st.markdown("**Chi phí & tồn kho**")
            e1, e2, e3 = st.columns(3)
            underage = e1.number_input("Cu — thiếu 1 đv (đ)", 1.0, value=30_000.0, step=5_000.0,
                                       help="Lợi nhuận mất khi thiếu hàng = giá bán − giá vốn.")
            overage = e2.number_input("Co — thừa 1 đv (đ)", 1.0, value=4_000.0, step=1_000.0,
                                      help="Chi phí ôm 1 đơn vị tồn/ế trong kỳ.")
            current_stock = e3.number_input("Tồn kho hiện tại", 0.0, value=100_000.0, step=5_000.0)
            b1, b2, b3 = st.columns(3)
            on_order = b1.number_input("Hàng đang về", 0.0, value=0.0, step=5_000.0)
            ordering_cost = b2.number_input("Chi phí đặt S (đ)", 1.0, value=5_000_000.0, step=100_000.0)
            holding_cost = b3.number_input("Lưu kho H (đ/đv/năm)", 0.01, value=48_000.0, step=1_000.0)
            lead_time = st.slider("Thời gian chờ L (tháng)", 0.5, 6.0, 2.0, 0.5)
            go = st.form_submit_button("🔍 Tính phương án đặt hàng", use_container_width=True)

    if go:
        position = current_stock + on_order
        nv = newsvendor(QUANTILES, q_vals, underage, overage, current_position=position)
        res = optimize(InventoryInputs(
            demand_monthly=demand_p50, sigma_monthly=fc["sigma_demand_monthly"],
            ordering_cost=ordering_cost, holding_cost_annual=holding_cost,
            lead_time_months=lead_time, service_level=nv.critical_ratio,
            current_stock=current_stock, on_order=on_order))
        cr_pct = nv.critical_ratio * 100

        if nv.order_quantity > 0:
            ui.reco_banner("order", f"🔴 Nên đặt thêm {qty(nv.order_quantity)} đơn vị",
                           f"Đưa tồn kho lên mức chuẩn bị {qty(nv.target_stock)} "
                           f"(hiện có {qty(position)}). Đặt sớm vì lead time {lead_time:g} tháng.")
        else:
            ui.reco_banner("hold", "🟢 Chưa cần đặt thêm",
                           f"Tồn kho hiện tại ({qty(position)}) đã ≥ mức chuẩn bị tối ưu "
                           f"({qty(nv.target_stock)}).")

        st.write("")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Mức phục vụ (CR)", f"{cr_pct:.0f}%", help="Tỷ lệ tới hạn = mức phân vị mục tiêu.")
        k2.metric("Tồn kho an toàn (SS)", qty(res.safety_stock))
        k3.metric("Điểm đặt lại (ROP)", qty(res.reorder_point))
        k4.metric("Cỡ lô EOQ (Q*)", qty(res.eoq))

        c1, c2 = st.columns(2)
        with c1.container(border=True):
            ui.section("Mức chuẩn bị theo kinh tế (Newsvendor)")
            qcurve = pd.DataFrame({"p": QUANTILES, "Nhu cầu": q_vals})
            line = alt.Chart(qcurve).mark_line(point=True, color=ui.BRAND).encode(
                x=alt.X("p:Q", title="Phân vị", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Nhu cầu:Q", scale=alt.Scale(zero=False)))
            pt = alt.Chart(pd.DataFrame({"x": [nv.critical_ratio], "y": [nv.target_stock]})).mark_point(
                size=170, color=ui.WARN, filled=True).encode(x="x:Q", y="y:Q")
            rule = alt.Chart(pd.DataFrame({"x": [nv.critical_ratio]})).mark_rule(
                color=ui.WARN, strokeDash=[4, 4]).encode(x="x:Q")
            st.altair_chart(ui.style_chart((line + rule + pt).properties(height=260)),
                            use_container_width=True)
            st.caption(f"Mức chuẩn bị = P{cr_pct:.0f} = {qty(nv.target_stock)} · "
                       f"thiếu KV {money(nv.expected_understock_cost)}đ · "
                       f"tồn KV {money(nv.expected_overstock_cost)}đ.")
        with c2.container(border=True):
            ui.section("Cỡ lô & tổng chi phí (EOQ)")
            qs, oc, hc, tot = cost_curve(InventoryInputs(
                demand_monthly=demand_p50, sigma_monthly=fc["sigma_demand_monthly"],
                ordering_cost=ordering_cost, holding_cost_annual=holding_cost,
                lead_time_months=lead_time, service_level=nv.critical_ratio))
            cdf = pd.concat([
                pd.DataFrame({"Q": qs, "Chi phí": oc, "Loại": "Đặt hàng"}),
                pd.DataFrame({"Q": qs, "Chi phí": hc, "Loại": "Lưu kho"}),
                pd.DataFrame({"Q": qs, "Chi phí": tot, "Loại": "Tổng"})])
            cc = alt.Chart(cdf).mark_line().encode(
                x=alt.X("Q:Q", title="Cỡ lô Q"), y=alt.Y("Chi phí:Q", title="đ/năm"),
                color=alt.Color("Loại:N", scale=alt.Scale(
                    domain=["Đặt hàng", "Lưu kho", "Tổng"],
                    range=["#94A3B8", "#38BDF8", ui.BRAND])))
            r2 = alt.Chart(pd.DataFrame({"Q": [res.eoq]})).mark_rule(
                color=ui.WARN, strokeDash=[5, 4]).encode(x="Q:Q")
            st.altair_chart(ui.style_chart((cc + r2).properties(height=260)),
                            use_container_width=True)
            st.caption(f"Q* = {qty(res.eoq)} · {res.orders_per_year:.1f} lần/năm · "
                       f"tổng chi phí {money(res.total_annual_cost)}đ/năm.")

        with st.expander("Công thức (r, q)"):
            st.latex(r"\text{CR}=\frac{C_u}{C_u+C_o}="
                     rf"{nv.critical_ratio:.3f}\;\Rightarrow\;z=\Phi^{{-1}}(\text{{CR}})={res.z_score:.2f}")
            st.latex(rf"\text{{SS}}=z\,\sigma\sqrt{{L}}={qty(res.safety_stock)}\quad"
                     rf"r=\mu_D L+\text{{SS}}={qty(res.reorder_point)}")
            st.latex(r"Q^*=\sqrt{\tfrac{2DS}{H}}=" rf"{qty(res.eoq)}")
    else:
        st.info("Nhập thông số rồi bấm **Tính phương án đặt hàng**.")

# =====================================================================
# VIEW 3 — HIEU SUAT MO HINH
# =====================================================================
else:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("MAE (test)", qty(m["MAE"]))
    k2.metric("RMSE (test)", qty(m["RMSE"]))
    k3.metric("MAPE (test)", f"{m['MAPE']:.2f}%")
    k4.metric("Pinball loss (test)", qty(m["PinballLoss"]))

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        ui.section("So sánh với baseline")
        base = pd.DataFrame(fc["metrics"]).T[["MAE", "RMSE", "MAPE"]]
        base.index = ["RevIN+N-BEATS" if i == "RevIN+NBEATS" else i for i in base.index]
        st.dataframe(base.style.format("{:,.0f}", subset=["MAE", "RMSE"]).format(
            "{:.2f}", subset=["MAPE"]), use_container_width=True)
        st.caption("N-BEATS vượt Naive/Seasonal Naive với biên độ lớn.")
    with c2.container(border=True):
        ui.section("Hiệu chỉnh phân vị (calibration)")
        cov = fc["coverage"]
        cov_df = pd.DataFrame({"q": [float(k) for k in cov],
                               "Lý tưởng": [float(k) for k in cov],
                               "Thực tế": [cov[k] for k in cov]})
        ideal = alt.Chart(cov_df).mark_line(color="#94A3B8", strokeDash=[4, 4]).encode(
            x=alt.X("q:Q", title="Mức phân vị"), y=alt.Y("Lý tưởng:Q", title="Bao phủ"))
        real = alt.Chart(cov_df).mark_line(point=True, color=ui.BRAND).encode(x="q:Q", y="Thực tế:Q")
        st.altair_chart(ui.style_chart((ideal + real).properties(height=260)),
                        use_container_width=True)
        st.caption("Đường xanh gần đường chấm (lý tưởng) ⟹ phân vị đáng tin.")
