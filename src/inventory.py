"""Mo hinh toi uu ton kho: EOQ + ton kho an toan trong dieu kien khong chac chan.

Quy uoc don vi: nhu cau theo THANG (tu mo hinh du bao), lead time theo THANG,
chi phi luu kho H theo don vi/NAM.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class InventoryInputs:
    demand_monthly: float        # nhu cau du bao trung binh / thang
    sigma_monthly: float         # do lech chuan nhu cau thang (tu sai so du bao)
    ordering_cost: float         # S — chi phi mot lan dat hang
    holding_cost_annual: float   # H — chi phi luu kho 1 don vi / nam
    lead_time_months: float      # L — thoi gian cho hang (thang)
    service_level: float         # muc phuc vu, vd 0.95
    current_stock: float = 0.0   # ton kho hien tai
    on_order: float = 0.0        # luong da dat dang ve


@dataclass
class InventoryResult:
    eoq: float                   # Q* — luong dat toi uu moi lan
    safety_stock: float          # SS
    reorder_point: float         # ROP
    z_score: float
    annual_demand: float
    orders_per_year: float
    cycle_months: float          # chu ky dat hang (thang)
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_cost: float
    should_order_now: bool
    order_quantity_now: float    # khuyen nghi dat ngay bao nhieu


def optimize(inp: InventoryInputs) -> InventoryResult:
    D = inp.demand_monthly * 12.0
    S = inp.ordering_cost
    H = inp.holding_cost_annual

    if D <= 0 or S <= 0 or H <= 0:
        raise ValueError("D, S, H phai duong.")

    q_star = float(np.sqrt(2.0 * D * S / H))

    z = float(norm.ppf(inp.service_level))
    ss = max(0.0, z * inp.sigma_monthly * np.sqrt(inp.lead_time_months))
    rop = inp.demand_monthly * inp.lead_time_months + ss

    orders_per_year = D / q_star
    cycle_months = 12.0 / orders_per_year
    annual_ordering = orders_per_year * S
    annual_holding = (q_star / 2.0 + ss) * H
    total_cost = annual_ordering + annual_holding

    inventory_position = inp.current_stock + inp.on_order
    should_order = inventory_position <= rop
    # Dat den muc ROP + Q* (order-up-to), toi thieu la Q*
    order_now = max(q_star, rop + q_star - inventory_position) if should_order else 0.0

    return InventoryResult(
        eoq=q_star,
        safety_stock=ss,
        reorder_point=rop,
        z_score=z,
        annual_demand=D,
        orders_per_year=orders_per_year,
        cycle_months=cycle_months,
        annual_ordering_cost=annual_ordering,
        annual_holding_cost=annual_holding,
        total_annual_cost=total_cost,
        should_order_now=should_order,
        order_quantity_now=order_now,
    )


@dataclass
class NewsvendorResult:
    critical_ratio: float        # CR = Cu / (Cu + Co) — cung la muc phan vi toi uu
    underage_cost: float         # Cu — thiet hai khi THIEU 1 don vi (lai bi mat)
    overage_cost: float          # Co — thiet hai khi THUA 1 don vi (ton/e)
    target_stock: float          # muc ton kho muc tieu cho ky = phan vi CR cua nhu cau
    order_quantity: float        # can dat them = max(0, target - vi the hien tai)
    p50: float                   # nhu cau trung vi (de doi chieu)
    expected_understock_cost: float  # chi phi thieu hang ky vong tai target_stock
    expected_overstock_cost: float   # chi phi ton du ky vong tai target_stock


def quantile_at(level: float, quantile_levels, quantile_values) -> float:
    """Noi suy gia tri nhu cau tai mot muc phan vi bat ky tu cac phan vi da co."""
    return float(np.interp(level, quantile_levels, quantile_values))


def newsvendor(quantile_levels, quantile_values, underage_cost: float, overage_cost: float,
               current_position: float = 0.0) -> NewsvendorResult:
    """Mo hinh nguoi ban bao (newsvendor) — bai toan dat hang mot ky voi nhu cau
    ngau nhien.

    Y nghia kinh te cot loi: muc ton kho toi uu chinh la PHAN VI cua phan phoi
    nhu cau tai ty le toi han
        CR = Cu / (Cu + Co)
    voi Cu = thiet hai khi thieu 1 don vi (loi nhuan bi mat),
        Co = thiet hai khi thua 1 don vi (chi phi ton/e).
    San pham lai cao (Cu lon) -> chuan bi du o phan vi cao (P90+); san pham de
    e/chi phi ton cao (Co lon) -> chuan bi o phan vi thap hon.
    """
    if underage_cost <= 0 or overage_cost <= 0:
        raise ValueError("Cu va Co phai duong.")
    cr = underage_cost / (underage_cost + overage_cost)
    levels = np.asarray(quantile_levels, dtype=float)
    values = np.asarray(quantile_values, dtype=float)
    target = float(np.interp(cr, levels, values))

    # Chi phi thieu/thua ky vong tai target_stock, uoc luong tu cac phan vi
    # (tich phan thang qua luoi phan vi).
    dense_p = np.linspace(0.001, 0.999, 999)
    dense_d = np.interp(dense_p, levels, values)
    short = np.clip(dense_d - target, 0, None).mean() * underage_cost
    over = np.clip(target - dense_d, 0, None).mean() * overage_cost

    return NewsvendorResult(
        critical_ratio=cr,
        underage_cost=underage_cost,
        overage_cost=overage_cost,
        target_stock=target,
        order_quantity=max(0.0, target - current_position),
        p50=float(np.interp(0.5, levels, values)),
        expected_understock_cost=float(short),
        expected_overstock_cost=float(over),
    )


def cost_curve(inp: InventoryInputs, q_min_ratio: float = 0.3, q_max_ratio: float = 3.0, n: int = 200):
    """Duong cong tong chi phi nam theo Q, de ve bieu do minh hoa."""
    res = optimize(inp)
    qs = np.linspace(res.eoq * q_min_ratio, res.eoq * q_max_ratio, n)
    D, S, H = res.annual_demand, inp.ordering_cost, inp.holding_cost_annual
    ordering = D / qs * S
    holding = (qs / 2.0 + res.safety_stock) * H
    return qs, ordering, holding, ordering + holding
