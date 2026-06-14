"""Web app cho nguoi dung BUSINESS — ngon ngu don gian, tap trung ra quyet dinh.
Dung du bao PHAN VI + logic kinh te newsvendor (lai vs thiet hai ton).

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

from src.inventory import newsvendor  # noqa: E402

st.set_page_config(page_title="Trợ lý đặt hàng", page_icon="🛒", layout="centered")


def fmt_money(x: float) -> str:
    if abs(x) >= 1e9:
        return f"{x / 1e9:,.1f} tỷ đ"
    if abs(x) >= 1e6:
        return f"{x / 1e6:,.0f} triệu đ"
    return f"{x:,.0f} đ"


def fmt_qty(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


@st.cache_data
def load_forecast():
    with open(ROOT / "results" / "forecast.json") as f:
        return json.load(f)


fc = load_forecast()
QUANTILES = fc["quantiles"]
MEDIAN_IDX = fc["median_idx"]


def q_index(level: float) -> int:
    return int(np.argmin(np.abs(np.array(QUANTILES) - level)))


st.title("🛒 Trợ lý đặt hàng tháng này")
st.markdown(
    "Trả lời nhanh **3 câu hỏi**: tháng này *bán được bao nhiêu*, "
    "*nên chuẩn bị bao nhiêu hàng cho hợp lý*, và *bây giờ cần đặt thêm bao nhiêu*."
)

# ============ BUOC 1: Du kien ban duoc bao nhieu ============
st.header("Bước 1 — Tháng này dự kiến bán được bao nhiêu?")

month_options = fc["future_months"]
plan_month = st.selectbox("Chọn tháng cần lập kế hoạch", month_options, index=0)
idx = month_options.index(plan_month)
q_vals = list(np.array(fc["future_quantiles"])[idx])
p10 = q_vals[q_index(0.1)]
p50 = q_vals[MEDIAN_IDX]
p90 = q_vals[q_index(0.9)]

st.success(
    f"### 📈 Tháng {plan_month}: nhiều khả năng bán khoảng **{fmt_qty(p50)}** sản phẩm\n"
    f"Nhưng nhu cầu luôn dao động — hệ thống dự báo:\n"
    f"- Tháng **bán chậm** (10% khả năng): có thể chỉ bán ~**{fmt_qty(p10)}**\n"
    f"- Tháng **bán chạy** (10% khả năng): có thể lên tới ~**{fmt_qty(p90)}**\n\n"
    f"👉 Đây chính là điều khiến việc đặt hàng khó: đặt ít thì sợ hết, đặt nhiều thì sợ ế. "
    f"Hệ thống sẽ giúp bạn chọn mức **hợp lý nhất theo tiền lãi/lỗ** ở Bước 3."
)

with st.expander("Xem khoảng dao động nhu cầu 12 tháng tới"):
    fut_q = np.array(fc["future_quantiles"])
    band = pd.DataFrame({
        "Tháng": pd.to_datetime(fc["future_months"], format="%Y-%m"),
        "Thấp (P10)": fut_q[:, q_index(0.1)],
        "Cao (P90)": fut_q[:, q_index(0.9)],
        "Hay gặp (P50)": fut_q[:, MEDIAN_IDX],
    })
    area = alt.Chart(band).mark_area(opacity=0.25, color="#4a90d9").encode(
        x=alt.X("Tháng:T", title=None), y="Thấp (P10):Q", y2="Cao (P90):Q")
    mid = alt.Chart(band).mark_line(color="#2c6fae").encode(x="Tháng:T", y="Hay gặp (P50):Q")
    st.altair_chart((area + mid).properties(height=240), width="stretch")
    st.caption("Vùng xanh: khoảng nhu cầu thường gặp (80% khả năng nằm trong đây).")

demand_p50 = st.number_input(
    "Nếu bạn biết tháng này có gì đặc biệt (khuyến mãi lớn…), chỉnh mức bán hay gặp tại đây:",
    min_value=1.0, value=float(round(p50)), step=1000.0,
)
# Dich chuyen ca dai phan vi theo muc nguoi dung chinh
shift = demand_p50 - p50
q_vals_adj = [v + shift for v in q_vals]

# ============ BUOC 2: Thong tin cua ban ============
st.header("Bước 2 — Cho biết tình hình & lãi/lỗ của bạn")

c1, c2 = st.columns(2)
with c1:
    current_stock = st.number_input("📦 Trong kho đang còn bao nhiêu sản phẩm?",
                                    min_value=0.0, value=100_000.0, step=5_000.0)
    on_order = st.number_input("🚚 Lô hàng đã đặt mà chưa về? (số lượng)",
                               min_value=0.0, value=0.0, step=5_000.0)
    lead_days = st.number_input("⏱️ Đặt hàng xong mấy ngày thì về tới kho?",
                                min_value=1, value=30, step=1)
with c2:
    profit_per_unit = st.number_input(
        "💰 Bán được 1 sản phẩm thì LÃI bao nhiêu? (đ)",
        min_value=1.0, value=30_000.0, step=5_000.0,
        help="Lãi gộp mỗi sản phẩm = giá bán − giá vốn. Nếu hết hàng, đây là số tiền lãi bạn mất.")
    stockout_penalty = st.number_input(
        "💔 Hết hàng — thiệt hại THÊM mỗi sản phẩm (ngoài lãi đã mất)? (đ)",
        min_value=0.0, value=0.0, step=5_000.0,
        help="Mức ảnh hưởng khi thiếu hàng: mất uy tín, khách bỏ đi mua nơi khác, phải nhập gấp giá cao… "
             "Hàng dễ thay thế → để 0. Khách khó tính/đơn quan trọng → đặt cao.")
    waste_per_unit = st.number_input(
        "📉 1 sản phẩm bị TỒN/Ế cuối tháng thì THIỆT HẠI bao nhiêu? (đ)",
        min_value=1.0, value=4_000.0, step=1_000.0,
        help="Tiền ôm hàng dư: kho bãi, vốn đọng, giảm giá/hư hỏng. Hàng để được lâu → số này nhỏ.")

# Tong thiet hai khi THIEU 1 san pham (underage cost Cu) = lai mat + anh huong them
shortage_cost = profit_per_unit + stockout_penalty

# ============ BUOC 3: Ket qua ============
st.header("Bước 3 — Khuyến nghị của hệ thống")

if st.button("👉 Cho tôi biết nên làm gì", type="primary", width="stretch"):
    position = current_stock + on_order
    nv = newsvendor(QUANTILES, q_vals_adj, shortage_cost, waste_per_unit,
                    current_position=position)
    cr_pct = nv.critical_ratio * 100

    # Mo ta thiet hai khi thieu hang (gop lai mat + anh huong them neu co)
    if stockout_penalty > 0:
        short_desc = (f"thiếu 1 sản phẩm thiệt hại **{fmt_money(shortage_cost)}** "
                      f"(lãi mất {fmt_money(profit_per_unit)} + ảnh hưởng {fmt_money(stockout_penalty)})")
    else:
        short_desc = f"thiếu 1 sản phẩm mất khoản lãi **{fmt_money(shortage_cost)}**"

    # ---- Giai thich kinh te bang ngon ngu doi thuong ----
    if nv.critical_ratio >= 0.5:
        reason = (
            f"Bạn cho biết {short_desc}, "
            f"trong khi ôm 1 sản phẩm ế chỉ mất **{fmt_money(waste_per_unit)}**. "
            f"Thiệt hại khi thiếu lớn hơn rủi ro ế ⟹ nên **chuẩn bị dư một chút** để đừng bỏ lỡ doanh thu."
        )
    else:
        reason = (
            f"Ôm 1 sản phẩm ế mất **{fmt_money(waste_per_unit)}**, "
            f"trong khi {short_desc}. "
            f"Rủi ro ế lớn hơn ⟹ nên **giữ ít lại**, chấp nhận đôi lúc hết hàng còn hơn ôm hàng tồn."
        )

    st.markdown(f"#### 💡 {reason}")
    st.markdown(
        f"Mức chuẩn bị hợp lý nhất là đủ hàng cho **kịch bản bán tới {fmt_qty(nv.target_stock)} "
        f"sản phẩm** (tức bạn sẽ đủ hàng trong khoảng **{cr_pct:.0f}%** số tháng)."
    )

    # ---- Cau tra loi chinh ----
    if nv.order_quantity > 0:
        st.error(
            f"# 🔴 NÊN ĐẶT THÊM: {fmt_qty(nv.order_quantity)} sản phẩm\n"
            f"Hiện trong kho + hàng đang về mới có **{fmt_qty(position)}** sản phẩm, "
            f"chưa đủ mức chuẩn bị hợp lý (**{fmt_qty(nv.target_stock)}**). "
            f"Đặt thêm để yên tâm cho cả tháng — nhớ đặt sớm vì hàng cần **~{lead_days} ngày** mới về tới kho."
        )
    else:
        st.success(
            f"# 🟢 CHƯA CẦN ĐẶT THÊM\n"
            f"Trong kho + hàng đang về đã có **{fmt_qty(position)}** sản phẩm, "
            f"đủ vượt mức chuẩn bị hợp lý (**{fmt_qty(nv.target_stock)}**) cho tháng này."
        )

    # ---- Hau qua neu chuan bi sai ----
    st.subheader("Nếu chuẩn bị quá ít hoặc quá nhiều thì sao?")
    target = nv.target_stock
    # So sanh chi phi ky vong: chuan bi theo P50 vs theo khuyen nghi
    dense_p = np.linspace(0.001, 0.999, 999)
    dense_d = np.interp(dense_p, QUANTILES, q_vals_adj)

    def expected_loss(stock):
        short = np.clip(dense_d - stock, 0, None).mean() * shortage_cost
        over = np.clip(stock - dense_d, 0, None).mean() * waste_per_unit
        return short + over

    loss_reco = expected_loss(target)
    loss_p50 = expected_loss(demand_p50)
    cc1, cc2 = st.columns(2)
    cc1.metric("Chuẩn bị theo khuyến nghị", fmt_qty(target),
               help="Mức cân bằng tối ưu giữa rủi ro hết hàng và rủi ro ế.")
    cc2.metric("Nếu chỉ chuẩn bị mức 'hay gặp'", fmt_qty(demand_p50),
               delta=f"thiệt hại thêm {fmt_money(loss_p50 - loss_reco)}/tháng",
               delta_color="inverse")
    st.caption(
        "Chỉ chuẩn bị đúng mức bán 'hay gặp' nghe hợp lý, nhưng vì nhu cầu hay dao động, "
        "về lâu dài cách đó tốn kém hơn so với mức khuyến nghị đã cân nhắc cả lãi lẫn rủi ro ế."
    )

else:
    st.info("Điền thông tin ở trên rồi bấm nút để nhận khuyến nghị.")
