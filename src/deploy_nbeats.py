"""Trien khai mo hinh N-BEATS TOT NHAT tu tune_all (so sanh cong bang).

Doc bo sieu tham so N-BEATS tot nhat trong results/tune_all.json, huan luyen lai
tren train+val, danh gia test, du bao de quy 12 thang, va luu cac artifact trien
khai (model, params, metrics, forecast.json, forecast_plot.png) — dung cho web
app va cac hinh res1-5.

Chay:  python -m src.deploy_nbeats
"""
import json

import numpy as np
import pandas as pd
import torch

from .data_loader import ROOT, WindowDataset, load_series, split_indices
from .evaluate import (all_metrics, coverage, naive_forecast, pinball_loss,
                       seasonal_naive_forecast)
from .train import predict_quantiles
from .tune_all import HORIZON, MEDIAN_IDX, QUANTILES, train_with_params
from .tune_optuna import recursive_forecast

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    with open(RESULTS_DIR / "tune_all.json") as f:
        best = json.load(f)["results"]["NBEATS"]["best_params"]
    print(f"N-BEATS best params: {best}")

    y, months = load_series()
    n = len(y)
    train_end, val_end = split_indices(n)
    scale = float(y[:train_end].mean())
    ys = y / scale
    data = {"ys": ys, "scale": scale, "Xs": None, "target_std": None, "n_features": 1}
    lookback = best["lookback"]

    # So epoch toi uu (early stopping tren val) -> train lai tren train+val.
    _, _, best_epoch = train_with_params("NBEATS", best, data, (0, train_end), (train_end, val_end))
    model, _, _ = train_with_params("NBEATS", best, data, (0, val_end), None,
                                    fixed_epochs=max(best_epoch, 1), seed=0)

    # Danh gia test
    test_ds = WindowDataset(ys, lookback, HORIZON, (val_end, n))
    yt_s, yq_s = predict_quantiles(model, test_ds)
    y_true = yt_s * scale
    y_pred_q = yq_s * scale
    y_pred_med = y_pred_q[:, MEDIAN_IDX]

    metrics = {"RevIN+NBEATS": all_metrics(y_true, y_pred_med)}
    metrics["RevIN+NBEATS"]["PinballLoss"] = pinball_loss(y_true, y_pred_q, QUANTILES)
    metrics["Naive"] = all_metrics(y[val_end:], naive_forecast(y, (val_end, n)))
    metrics["SeasonalNaive"] = all_metrics(y[val_end:], seasonal_naive_forecast(y, (val_end, n)))
    cov = coverage(y_true, y_pred_q, QUANTILES)
    sigma_demand = float(np.std(y_true - y_pred_med, ddof=1))

    future_q = recursive_forecast(model, ys, lookback, steps=12) * scale
    future_med = future_q[:, MEDIAN_IDX]
    future_months = pd.date_range(months[-1], periods=13, freq="MS")[1:].strftime("%Y-%m").tolist()

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "best_model.pt")
    with open(MODELS_DIR / "best_params.json", "w") as f:
        json.dump({"params": best, "horizon": HORIZON, "quantiles": QUANTILES, "scale": scale,
                   "best_epoch": best_epoch, "sigma_demand_monthly": sigma_demand,
                   "source": "tune_all (so sanh cong bang)"}, f, indent=2)
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump({"metrics": metrics, "coverage": cov}, f, indent=2)
    with open(RESULTS_DIR / "forecast.json", "w") as f:
        json.dump({
            "quantiles": QUANTILES, "median_idx": MEDIAN_IDX,
            "history_months": months.strftime("%Y-%m").tolist(), "history": y.tolist(),
            "test_months": months[val_end:].strftime("%Y-%m").tolist(),
            "test_actual": y_true.tolist(), "test_pred": y_pred_med.tolist(),
            "test_pred_quantiles": y_pred_q.tolist(),
            "future_months": future_months, "future_forecast": future_med.tolist(),
            "future_quantiles": future_q.tolist(),
            "sigma_demand_monthly": sigma_demand, "coverage": cov, "metrics": metrics,
        }, f, indent=2)

    print(f"MAE={metrics['RevIN+NBEATS']['MAE']:,.0f}  MAPE={metrics['RevIN+NBEATS']['MAPE']:.2f}%  "
          f"Pinball={metrics['RevIN+NBEATS']['PinballLoss']:,.0f}  best_epoch={best_epoch}")
    print(f"sigma_demand={sigma_demand:,.0f}  n_params={sum(p.numel() for p in model.parameters()):,}")
    print("Da luu: models/best_model.pt, best_params.json, results/metrics.json, forecast.json")


if __name__ == "__main__":
    main()
