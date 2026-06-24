"""Thanh phan giao dien dung chung cho 2 web app (phong cach dashboard quan ly).

Cung cap: bang mau, CSS chung, header bar, the KPI, tieu de section, badge, va
mot theme bieu do Altair gon gang.
"""
import altair as alt
import streamlit as st

# --- Bang mau thuong hieu ---
BRAND = "#2563EB"
BRAND_DARK = "#1D4ED8"
INK = "#0F172A"
MUTED = "#64748B"
LINE = "#E2E8F0"
GOOD = "#16A34A"
WARN = "#DC2626"
BAND = "#2563EB"

_CSS = """
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2.2rem; max-width: 1200px;}
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

/* KPI: bien st.metric thanh the */
div[data-testid="stMetric"]{
  background:#fff; border:1px solid #E2E8F0; border-radius:14px;
  padding:14px 18px; box-shadow:0 1px 3px rgba(15,23,42,.06);
}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{
  font-size:1.55rem; font-weight:800; color:#0F172A;}
div[data-testid="stMetric"] label{color:#64748B; font-weight:600;}

/* the (st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:#fff; border-radius:16px; border:1px solid #E2E8F0 !important;
  box-shadow:0 1px 3px rgba(15,23,42,.05);
}

/* nut */
.stButton>button, .stFormSubmitButton>button{
  border-radius:10px; font-weight:700; border:0;
  background:#2563EB; color:#fff; padding:.55rem 1rem;}
.stButton>button:hover, .stFormSubmitButton>button:hover{filter:brightness(1.08);}

/* sidebar toi kieu admin */
section[data-testid="stSidebar"]{background:#0B1220; border-right:1px solid #1E293B;}
section[data-testid="stSidebar"] *{color:#CBD5E1;}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{color:#fff;}

/* header bar */
.topbar{background:linear-gradient(95deg,#2563EB,#1D4ED8);
  color:#fff; padding:18px 24px; border-radius:16px; margin-bottom:20px;
  box-shadow:0 6px 20px rgba(37,99,235,.25);}
.topbar .brand{font-size:1.5rem; font-weight:800; letter-spacing:-.4px;}
.topbar .sub{opacity:.92; margin-top:2px; font-size:.95rem;}

/* tieu de section */
.sect{display:flex; align-items:center; gap:.5rem; margin:.3rem 0 .2rem;
  font-size:1.15rem; font-weight:700; color:#0F172A;}
.sect .bar{width:5px; height:1.15rem; background:#2563EB; border-radius:3px;}

/* badge */
.badge{display:inline-block; padding:.18rem .6rem; border-radius:999px;
  font-size:.78rem; font-weight:700;}
.badge.ok{background:#DCFCE7; color:#166534;}
.badge.warn{background:#FEE2E2; color:#991B1B;}

/* the khuyen nghi lon */
.reco{border-radius:16px; padding:20px 22px; margin:8px 0;
  color:#fff; box-shadow:0 8px 24px rgba(2,6,23,.18);}
.reco.order{background:linear-gradient(95deg,#DC2626,#B91C1C);}
.reco.hold{background:linear-gradient(95deg,#16A34A,#15803D);}
.reco .big{font-size:1.7rem; font-weight:800;}
.reco .small{opacity:.95; margin-top:4px;}

h1{font-weight:800; letter-spacing:-.5px;}
</style>
"""


def setup(page_title: str, page_icon: str, layout: str = "wide") -> None:
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
    st.markdown(_CSS, unsafe_allow_html=True)


def topbar(brand: str, sub: str) -> None:
    st.markdown(
        f'<div class="topbar"><div class="brand">{brand}</div>'
        f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="sect"><span class="bar"></span>{title}</div>',
                unsafe_allow_html=True)


def reco_banner(kind: str, headline: str, detail: str) -> None:
    st.markdown(
        f'<div class="reco {kind}"><div class="big">{headline}</div>'
        f'<div class="small">{detail}</div></div>', unsafe_allow_html=True)


def style_chart(ch: alt.Chart) -> alt.Chart:
    return (ch.configure_view(strokeOpacity=0)
            .configure_axis(grid=True, gridColor="#EEF2F7", domain=False,
                            labelColor=MUTED, titleColor=MUTED, tickColor="#EEF2F7")
            .configure_legend(labelColor=INK, titleColor=MUTED))
