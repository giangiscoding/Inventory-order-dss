"""Backtest goc truot (rolling-origin / time-series CV) cho N-BEATS trien khai.

Khac phuc van de tap test qua nho: thay vi 1 lat cat test co dinh, ta truot goc
du bao qua K thang gan nhat. Voi moi thang muc tieu t (cua so MO RONG):
  - train tren [0, t-V), early stopping tren val [t-V, t)
  - du bao 1 buoc cho thang t (phan vi), so voi thuc te y[t]
Gom sai so cua tat ca K du bao -> MAE/RMSE/MAPE/pinball va calibration tin cay
hon nhieu (K diem thay vi 12-24), ma moi fold van train tren gan het du lieu.

Dung cau hinh N-BEATS trien khai (tu results/tune_all.json).

Chay:  python -m src.backtest [K] [val_window]
Ket qua: results/backtest.json + report/figures/backtest.png
"""
import json
import sys

import numpy as np
import pandas as pd

from .data_loader import ROOT, WindowDataset, load_series, split_indices
from .evaluate import all_metrics, coverage, pinball_loss
from .train import predict_quantiles
from .tune_all import HORIZON, MEDIAN_IDX, QUANTILES, train_with_params

RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "report" / "figures"
MODEL = "NBEATS"


def main(K: int = 36, val_window: int = 12) -> None:
    with open(RESULTS_DIR / "tune_all.json") as f:
        params = json.load(f)["results"][MODEL]["best_params"]
    L = params["lookback"]

    y, months = load_series()
    n = len(y)
    scale = float(y[: n - K].mean())          # scale co dinh tu vung truoc backtest
    ys = y / scale
    data = {"ys": ys, "scale": scale, "Xs": None, "target_std": None, "n_features": 1}

    origins = list(range(n - K, n))           # K thang muc tieu gan nhat
    actuals, p50s, quantile_rows = [], [], []
    print(f"Backtest goc truot: {K} fold, val_window={val_window}, "
          f"du bao {months[origins[0]].date()} -> {months[origins[-1]].date()}")

    for i, t in enumerate(origins, 1):
        model, _, _ = train_with_params(
            MODEL, params, data, (0, t - val_window), (t - val_window, t))
        ds = WindowDataset(ys, L, HORIZON, (t, t + 1))     # dung 1 mau muc tieu = thang t
        yt, yq = predict_quantiles(model, ds)
        actuals.append(float(yt[0] * scale))
        quantile_rows.append((yq[0] * scale).tolist())
        p50s.append(float(yq[0, MEDIAN_IDX] * scale))
        if i % 6 == 0 or i == len(origins):
            print(f"  [{i}/{K}] {months[t].date()}  thuc te={actuals[-1]:,.0f}  P50={p50s[-1]:,.0f}")

    actuals = np.array(actuals)
    p50s = np.array(p50s)
    qarr = np.array(quantile_rows)

    m = all_metrics(actuals, p50s)
    m["PinballLoss"] = pinball_loss(actuals, qarr, QUANTILES)
    cov = coverage(actuals, qarr, QUANTILES)

    payload = {
        "K": K, "val_window": val_window,
        "origin_months": [months[t].strftime("%Y-%m") for t in origins],
        "actual": actuals.tolist(), "p50": p50s.tolist(),
        "quantiles": QUANTILES, "pred_quantiles": qarr.tolist(),
        "metrics": m, "coverage": cov,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "backtest.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n=== Backtest ({K} diem) ===")
    print(f"MAE={m['MAE']:,.0f}  RMSE={m['RMSE']:,.0f}  MAPE={m['MAPE']:.2f}%  Pinball={m['PinballLoss']:,.0f}")
    print("Coverage:", {k: round(v, 2) for k, v in cov.items()})
    print("Da luu results/backtest.json")
    _make_figure(payload)


def _make_figure(payload: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mth = pd.to_datetime(payload["origin_months"], format="%Y-%m")
    actual = np.array(payload["actual"])
    q = np.array(payload["quantiles"])
    qa = np.array(payload["pred_quantiles"])
    qi = {qq: i for i, qq in enumerate(payload["quantiles"])}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.fill_between(mth, qa[:, qi[0.1]], qa[:, qi[0.9]], color="crimson", alpha=0.15, label="P10-P90")
    ax.plot(mth, qa[:, qi[0.5]], color="crimson", lw=1.6, label="Dự báo P50")
    ax.plot(mth, actual, "k-o", lw=1.4, ms=3, label="Thực tế")
    ax.set_title(f"Backtest gốc trượt — dự báo 1 bước, {payload['K']} tháng")
    ax.set_ylabel("Quantity"); ax.legend(fontsize=9)

    ax = axes[1]
    cov = [payload["coverage"][f"{qq:.2f}"] for qq in payload["quantiles"]]
    ax.plot([0, 1], [0, 1], "b--", lw=1.2, label="Lý tưởng")
    ax.plot(q, cov, "r-o", lw=1.6, label=f"Backtest (n={payload['K']})")
    ax.set_xlabel("Mức phân vị danh nghĩa"); ax.set_ylabel("Tỷ lệ bao phủ thực tế")
    ax.set_title("Calibration trên backtest"); ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "backtest.png", dpi=130)
    plt.close(fig)
    print("Da luu report/figures/backtest.png")


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 36
    vw = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    main(K, vw)
