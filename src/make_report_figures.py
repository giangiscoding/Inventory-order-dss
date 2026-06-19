"""Tai tao cac figure phu thuoc vao cach chia du lieu (split) cho report.

Doc results/forecast.json (do tune_optuna sinh) va ve lai:
  - fig1_series_split : chuoi + vung train/val/test
  - res1_test_forecast: fan chart phan vi tren tap test
  - res2_fanchart     : lich su gan + test + du bao tuong lai (fan)
  - res3_calibration  : calibration tren tap test
  - res4_cost_curve   : duong chi phi Newsvendor cho thang tuong lai dau tien
  - res5_comparison   : MAE/MAPE RevIN+NBEATS vs baseline

Chay:  python -m src.make_report_figures
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_loader import ROOT, load_series, split_indices

RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "report" / "figures"
CU, CO = 30000.0, 4000.0          # cau truc chi phi vi du cho duong cost curve


def _months(strs):
    return pd.to_datetime(strs, format="%Y-%m")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "forecast.json") as f:
        fc = json.load(f)

    quantiles = np.array(fc["quantiles"])
    qi = {q: i for i, q in enumerate(fc["quantiles"])}

    # ---- fig1: chuoi + vung train/val/test ----
    y, months = load_series()
    train_end, val_end = split_indices(len(y))
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(months, y, color="steelblue", lw=1.2)
    ax.axvspan(months[0], months[train_end - 1], color="tab:blue", alpha=0.08,
               label=f"Train ({months[0].year}-{months[train_end-1].year}, n={train_end})")
    ax.axvspan(months[train_end], months[val_end - 1], color="tab:orange", alpha=0.18,
               label=f"Validation ({months[train_end].year}, n={val_end-train_end})")
    ax.axvspan(months[val_end], months[len(y) - 1], color="tab:red", alpha=0.15,
               label=f"Test ({months[val_end].year}-{months[len(y)-1].year}, n={len(y)-val_end})")
    ax.set_xlabel("Tháng"); ax.set_ylabel("Số lượng")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig1_series_split.png", dpi=130); plt.close(fig)

    # ---- res1: fan chart tren tap test ----
    tm = _months(fc["test_months"])
    actual = np.array(fc["test_actual"])
    tq = np.array(fc["test_pred_quantiles"])           # (N, Q)
    p10, p25, p50 = tq[:, qi[0.1]], tq[:, qi[0.25]], tq[:, qi[0.5]]
    p75, p90 = tq[:, qi[0.75]], tq[:, qi[0.9]]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(tm, p10, p90, color="crimson", alpha=0.15, label="P10–P90")
    ax.fill_between(tm, p25, p75, color="crimson", alpha=0.28, label="P25–P75")
    ax.plot(tm, p50, color="crimson", lw=1.8, label="Dự báo P50")
    ax.plot(tm, actual, "k-o", lw=1.6, ms=4, label="Thực tế")
    ax.set_xlabel("Tháng"); ax.set_ylabel("Quantity")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "res1_test_forecast.png", dpi=130); plt.close(fig)

    # ---- res2: lich su gan + test + du bao tuong lai ----
    hist_m = _months(fc["history_months"])
    hist = np.array(fc["history"])
    fut_m = _months(fc["future_months"])
    fq = np.array(fc["future_quantiles"])              # (12, Q)
    k = 36                                             # so thang lich su hien thi
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(hist_m[-k:], hist[-k:], color="steelblue", lw=1.3, label="Lịch sử")
    ax.plot(tm, actual, color="black", lw=1.6, label="Thực tế (test)")
    ax.fill_between(fut_m, fq[:, qi[0.05]], fq[:, qi[0.95]], color="crimson", alpha=0.15, label="P5–P95")
    ax.fill_between(fut_m, fq[:, qi[0.25]], fq[:, qi[0.75]], color="crimson", alpha=0.28, label="P25–P75")
    ax.plot(fut_m, fq[:, qi[0.5]], color="crimson", lw=1.8, label="Dự báo P50")
    ax.axvline(months[val_end], color="gray", ls=":", lw=1)
    ax.axvline(hist_m[-1], color="gray", ls=":", lw=1)
    ax.set_xlabel("Tháng"); ax.set_ylabel("Quantity")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout(); fig.savefig(FIG_DIR / "res2_fanchart.png", dpi=130); plt.close(fig)

    # ---- res3: calibration ----
    cov = fc["coverage"]                               # {"0.05":..., ...}
    actual_cov = [cov[f"{q:.2f}"] for q in fc["quantiles"]]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot([0, 1], [0, 1], "b--", lw=1.2, label="Lý tưởng")
    ax.plot(quantiles, actual_cov, "r-o", lw=1.6, label="Thực tế (test)")
    ax.set_xlabel("Mức phân vị danh nghĩa"); ax.set_ylabel("Tỷ lệ bao phủ thực tế")
    ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(FIG_DIR / "res3_calibration.png", dpi=130); plt.close(fig)

    # ---- res4: duong chi phi Newsvendor (thang tuong lai dau tien) ----
    qvals = fq[0]
    cr = CU / (CU + CO)
    dense_p = np.linspace(0.001, 0.999, 999)
    dense_d = np.interp(dense_p, quantiles, qvals)
    grid = np.linspace(qvals[0], qvals[-1], 200)
    short = np.array([CU * np.clip(dense_d - s, 0, None).mean() for s in grid])
    over = np.array([CO * np.clip(s - dense_d, 0, None).mean() for s in grid])
    total = short + over
    q_star = float(np.interp(cr, quantiles, qvals))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(grid, short, color="tab:red", lw=1.4, label="Chi phí thiếu hàng")
    ax.plot(grid, over, color="tab:blue", lw=1.4, label="Chi phí tồn dư")
    ax.plot(grid, total, color="black", lw=2.0, label="Tổng chi phí kỳ vọng")
    ax.axvline(q_star, color="green", ls="--", lw=1.5, label=f"$Q^*$ (CR={cr:.3f})")
    ax.set_xlabel("Mức chuẩn bị hàng"); ax.set_ylabel("Chi phí kỳ vọng (đồng)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "res4_cost_curve.png", dpi=130); plt.close(fig)

    # ---- res5: MAE/MAPE RevIN+NBEATS vs baseline ----
    m = fc["metrics"]
    names = ["RevIN+NBEATS", "Naive", "SeasonalNaive"]
    labels = ["RevIN+\nN-BEATS", "Naive", "Seasonal\nNaive"]
    colors = ["#dd8452", "#9aa0a6", "#9aa0a6"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, metric, title in [(axes[0], "MAE", "MAE (đơn vị)"), (axes[1], "MAPE", "MAPE (%)")]:
        vals = [m[n][metric] for n in names]
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_title(title); ax.set_ylabel(title)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}" if metric == "MAE" else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "res5_comparison.png", dpi=130); plt.close(fig)

    print("Da tai tao: fig1_series_split, res1_test_forecast, res2_fanchart, "
          "res3_calibration, res4_cost_curve, res5_comparison")


if __name__ == "__main__":
    main()
