"""Tim sieu tham so bang Optuna (objective = pinball loss tren validation),
huan luyen lai mo hinh DA PHAN VI cuoi, danh gia test, luu checkpoint + ket qua
du bao phan vi cho web app.

Chay:  python -m src.tune_optuna [n_trials]
"""
import json
import sys

import numpy as np
import optuna
import pandas as pd
import torch

from .data_loader import ROOT, WindowDataset, load_series, split_indices
from .evaluate import all_metrics, coverage, naive_forecast, pinball_loss, seasonal_naive_forecast
from .nbeats import DEFAULT_QUANTILES
from .train import predict_quantiles, train_model

HORIZON = 1
QUANTILES = DEFAULT_QUANTILES
MEDIAN_IDX = QUANTILES.index(0.5)
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "lookback": trial.suggest_int("lookback", 6, 24),
        "n_stacks": trial.suggest_int("n_stacks", 1, 3),
        "n_blocks": trial.suggest_int("n_blocks", 1, 4),
        "hidden_size": trial.suggest_int("hidden_size", 64, 512, step=64),
        "n_layers": trial.suggest_int("n_layers", 2, 4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "revin_affine": trial.suggest_categorical("revin_affine", [True, False]),
    }


@torch.no_grad()
def recursive_forecast(model, series_scaled: np.ndarray, lookback: int, steps: int) -> np.ndarray:
    """Du bao de quy `steps` thang tiep theo, dua trung vi (P50) lam gia tri
    quay vong vao cua so input. Tra ve mang (steps, n_quantiles)."""
    model.eval()
    window = list(series_scaled[-lookback:])
    rows = []
    for _ in range(steps):
        x = torch.tensor(window[-lookback:], dtype=torch.float32).unsqueeze(0)
        q_vals = model(x).numpy()[0, 0, :]      # (n_quantiles,)
        rows.append(q_vals)
        window.append(float(q_vals[MEDIAN_IDX]))
    return np.array(rows)


def main(n_trials: int = 100) -> None:
    y, months = load_series()
    n = len(y)
    train_end, val_end = split_indices(n)
    scale = float(y[:train_end].mean())
    ys = y / scale

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        _, val_pinball, _ = train_model(
            ys, params, HORIZON,
            train_range=(0, train_end), val_range=(train_end, val_end),
            quantiles=QUANTILES, trial=trial,
        )
        return val_pinball * scale

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=30),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = dict(study.best_params)
    print(f"\nBest val pinball loss: {study.best_value:,.0f}")
    print(f"Best params: {best_params}")

    # Lay so epoch toi uu (early stopping tren val), roi train lai tren train+val.
    _, _, best_epoch = train_model(
        ys, best_params, HORIZON,
        train_range=(0, train_end), val_range=(train_end, val_end), quantiles=QUANTILES,
    )
    final_model, _, _ = train_model(
        ys, best_params, HORIZON,
        train_range=(0, val_end), val_range=None, quantiles=QUANTILES,
        fixed_epochs=max(best_epoch, 1),
    )

    # Danh gia tren test (mot lan duy nhat)
    lookback = best_params["lookback"]
    test_ds = WindowDataset(ys, lookback, HORIZON, (val_end, n))
    y_true_s, y_pred_q_s = predict_quantiles(final_model, test_ds)
    y_true = y_true_s * scale
    y_pred_q = y_pred_q_s * scale                 # (N, Q)
    y_pred_median = y_pred_q[:, MEDIAN_IDX]

    metrics = {"RevIN+NBEATS": all_metrics(y_true, y_pred_median)}
    metrics["RevIN+NBEATS"]["PinballLoss"] = pinball_loss(y_true, y_pred_q, QUANTILES)
    metrics["Naive"] = all_metrics(y[val_end:], naive_forecast(y, (val_end, n)))
    metrics["SeasonalNaive"] = all_metrics(y[val_end:], seasonal_naive_forecast(y, (val_end, n)))
    cov = coverage(y_true, y_pred_q, QUANTILES)
    print(json.dumps(metrics, indent=2))
    print("Coverage (calibration):", json.dumps(cov, indent=2))

    sigma_demand = float(np.std(y_true - y_pred_median, ddof=1))

    # Du bao 12 thang tiep theo (2025) bang de quy — tra ve phan vi
    future_q = recursive_forecast(final_model, ys, lookback, steps=12) * scale  # (12, Q)
    future_median = future_q[:, MEDIAN_IDX]
    future_months = pd.date_range(months[-1], periods=13, freq="MS")[1:].strftime("%Y-%m").tolist()

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    torch.save(final_model.state_dict(), MODELS_DIR / "best_model.pt")
    with open(MODELS_DIR / "best_params.json", "w") as f:
        json.dump({
            "params": best_params,
            "horizon": HORIZON,
            "quantiles": QUANTILES,
            "scale": scale,
            "best_epoch": best_epoch,
            "best_val_pinball": study.best_value,
            "sigma_demand_monthly": sigma_demand,
        }, f, indent=2)

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump({"metrics": metrics, "coverage": cov}, f, indent=2)

    with open(RESULTS_DIR / "forecast.json", "w") as f:
        json.dump({
            "quantiles": QUANTILES,
            "median_idx": MEDIAN_IDX,
            "history_months": months.strftime("%Y-%m").tolist(),
            "history": y.tolist(),
            "test_months": months[val_end:].strftime("%Y-%m").tolist(),
            "test_actual": y_true.tolist(),
            "test_pred": y_pred_median.tolist(),
            "test_pred_quantiles": y_pred_q.tolist(),
            "future_months": future_months,
            "future_forecast": future_median.tolist(),
            "future_quantiles": future_q.tolist(),
            "sigma_demand_monthly": sigma_demand,
            "coverage": cov,
            "metrics": metrics,
        }, f, indent=2)

    # Bieu do fan chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(months, y, label="Thuc te", color="steelblue", lw=1.2)
    fut_idx = pd.to_datetime(future_months, format="%Y-%m")
    # Dai phan vi P5-P95 va P25-P75
    qarr = np.array(QUANTILES)
    p05, p95 = future_q[:, 0], future_q[:, -1]
    p25 = future_q[:, np.argmin(np.abs(qarr - 0.25))]
    p75 = future_q[:, np.argmin(np.abs(qarr - 0.75))]
    ax.fill_between(fut_idx, p05, p95, color="crimson", alpha=0.15, label="P5–P95")
    ax.fill_between(fut_idx, p25, p75, color="crimson", alpha=0.25, label="P25–P75")
    ax.plot(fut_idx, future_median, label="Du bao trung vi (P50)", color="crimson", lw=1.8)
    ax.plot(months[val_end:], y_pred_median, label="Du bao (test)", color="darkorange", lw=1.8)
    ax.axvline(months[train_end], color="gray", ls=":", lw=1)
    ax.axvline(months[val_end], color="gray", ls=":", lw=1)
    ax.set_title("RevIN + N-BEATS: du bao nhu cau hang thang (phan vi)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "forecast_plot.png", dpi=130)

    print(f"\nSigma nhu cau (thang): {sigma_demand:,.0f}")
    print(f"Du bao thang toi ({future_months[0]}): "
          f"P10={future_q[0, np.argmin(np.abs(qarr-0.1))]:,.0f}  "
          f"P50={future_median[0]:,.0f}  "
          f"P90={future_q[0, np.argmin(np.abs(qarr-0.9))]:,.0f}")
    print("Da luu: models/best_model.pt, models/best_params.json, "
          "results/metrics.json, results/forecast.json, results/forecast_plot.png")


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(n_trials)
