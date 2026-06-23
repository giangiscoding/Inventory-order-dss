"""So sanh nhieu kien truc du bao DA PHAN VI tu xay dung tren cung mot giao thuc.

Tat ca cac mo hinh hoc sau (MLP, DLinear, TSMixer, NHITS, N-BEATS) duoc huan
luyen voi CUNG cau hinh (lookback, hidden, lr, batch, early stopping) va CUNG
ham mat mat (pinball loss) tren CUNG tap du lieu -> so sanh cong bang ve KIEN
TRUC loi. Moi mo hinh chay nhieu seed roi lay trung binh de giam phuong sai.

Ngoai ra co hai baseline diem: Naive va Seasonal Naive.

Chay:  python -m src.compare_models [n_seeds]
Ket qua: results/comparison.json + results/figures/cmp_*.png
"""
import copy
import json
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_loader import (ROOT, MultiWindowDataset, WindowDataset, load_multivariate,
                          load_series, split_indices)
from .evaluate import (all_metrics, coverage, mae, mape, naive_forecast,
                       pinball_loss, rmse, seasonal_naive_forecast, smape)
from .models import (DEFAULT_QUANTILES, MULTIVARIATE_MODELS, build_model,
                     build_model_multi)
from .train import pinball_loss_torch, predict_quantiles

HORIZON = 1
QUANTILES = DEFAULT_QUANTILES
MEDIAN_IDX = QUANTILES.index(0.5)
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "report" / "figures"

# Cau hinh chung cho TAT CA mo hinh hoc sau (de so sanh cong bang).
COMMON = dict(
    lookback=24,
    hidden_size=256,
    n_layers=3,
    dropout=0.1,
    learning_rate=1e-3,
    batch_size=8,
    revin_affine=False,
    max_epochs=300,
    patience=30,
)

DEEP_MODELS = ["MLP", "DLinear", "TSMixer", "NHITS", "NBEATS"]


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _make_dataset(name, data, t_range):
    """Tao dataset + thong tin descale tuy mo hinh don bien hay da bien."""
    if name in MULTIVARIATE_MODELS:
        ds = MultiWindowDataset(data["Xs"], COMMON["lookback"], HORIZON, t_range)
        return ds, data["target_std"]      # descale = x target_std
    ds = WindowDataset(data["ys"], COMMON["lookback"], HORIZON, t_range)
    return ds, data["scale"]               # descale = x scale (train mean)


def _build(name, n_features):
    if name in MULTIVARIATE_MODELS:
        return build_model_multi(name, COMMON["lookback"], n_features, HORIZON, len(QUANTILES),
                                 hidden_size=COMMON["hidden_size"], n_layers=COMMON["n_layers"],
                                 dropout=COMMON["dropout"], revin_affine=COMMON["revin_affine"])
    return build_model(name, COMMON["lookback"], HORIZON, len(QUANTILES),
                       hidden_size=COMMON["hidden_size"], n_layers=COMMON["n_layers"],
                       dropout=COMMON["dropout"], revin_affine=COMMON["revin_affine"])


def train_one(name, data, train_range, val_range, seed):
    """Huan luyen 1 mo hinh (don bien hoac da bien), early stopping theo pinball
    loss tren val. Tra ve (model, best_epoch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    q_tensor = torch.tensor(QUANTILES, dtype=torch.float32)

    train_ds, _ = _make_dataset(name, data, train_range)
    val_ds, _ = _make_dataset(name, data, val_range)
    loader = DataLoader(train_ds, batch_size=COMMON["batch_size"], shuffle=True)

    model = _build(name, data["n_features"])
    opt = torch.optim.Adam(model.parameters(), lr=COMMON["learning_rate"])

    best_val, best_state, best_epoch, no_improve = float("inf"), None, 0, 0
    for epoch in range(1, COMMON["max_epochs"] + 1):
        model.train()
        for x, y in loader:
            opt.zero_grad()
            loss = pinball_loss_torch(model(x), y, q_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        yt, yp = predict_quantiles(model, val_ds)
        val_pin = pinball_loss(yt, yp, QUANTILES)
        if val_pin < best_val - 1e-9:
            best_val, best_state, best_epoch, no_improve = val_pin, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            no_improve += 1
            if no_improve >= COMMON["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def evaluate_on_test(model, name, data, test_range):
    test_ds, descale = _make_dataset(name, data, test_range)
    yt_s, yp_q_s = predict_quantiles(model, test_ds)
    yt = yt_s * descale
    yp_q = yp_q_s * descale
    yp_med = yp_q[:, MEDIAN_IDX]
    m = all_metrics(yt, yp_med)
    m["PinballLoss"] = pinball_loss(yt, yp_q, QUANTILES)
    cov = coverage(yt, yp_q, QUANTILES)
    return m, cov, yt, yp_med, yp_q


def main(n_seeds: int = 3) -> None:
    seeds = list(range(n_seeds))
    y, months = load_series()
    n = len(y)
    train_end, val_end = split_indices(n)
    scale = float(y[:train_end].mean())
    ys = y / scale

    # Du lieu da bien cho TSMixer/DLinear: Quantity + 6 ngoai sinh |r| cao nhat.
    # Scale tung cot bang do lech chuan tap train (de O(1)); descale muc tieu = x target_std.
    Xmv, top_exog, _ = load_multivariate(k=6)
    col_std = Xmv[:train_end].std(axis=0)
    col_std[col_std == 0] = 1.0
    Xs = Xmv / col_std
    data = {
        "ys": ys, "scale": scale,
        "Xs": Xs, "target_std": float(col_std[0]), "n_features": Xmv.shape[1],
    }
    print(f"Bien ngoai sinh da bien (top-6 |r|): {top_exog}")

    results = {}          # name -> {metrics mean/std, params, ...}
    test_curves = {}      # name -> median forecast (last seed) for plotting
    cov_curves = {}       # name -> coverage (mean over seeds)

    # --- Baseline diem ---
    for bname, fn in [("Naive", naive_forecast),
                      ("SeasonalNaive", seasonal_naive_forecast)]:
        pred = fn(y, (val_end, n))
        yt = y[val_end:]
        results[bname] = {
            "type": "baseline",
            "MAE": {"mean": mae(yt, pred), "std": 0.0},
            "RMSE": {"mean": rmse(yt, pred), "std": 0.0},
            "MAPE": {"mean": mape(yt, pred), "std": 0.0},
            "sMAPE": {"mean": smape(yt, pred), "std": 0.0},
            "PinballLoss": {"mean": None, "std": None},
            "n_params": 0,
        }
        test_curves[bname] = pred.tolist()

    # --- Mo hinh hoc sau ---
    test_actual = None
    for name in DEEP_MODELS:
        runs = {"MAE": [], "RMSE": [], "MAPE": [], "sMAPE": [], "PinballLoss": []}
        covs = []
        last_med = None
        n_params = None
        t0 = time.time()
        for seed in seeds:
            model, best_epoch = train_one(name, data, (0, train_end), (train_end, val_end), seed)
            if n_params is None:
                n_params = count_params(model)
            m, cov, yt, yp_med, yp_q = evaluate_on_test(model, name, data, (val_end, n))
            for k in runs:
                runs[k].append(m[k])
            covs.append([cov[f"{q:.2f}"] for q in QUANTILES])
            last_med = yp_med
            test_actual = yt
        dt = time.time() - t0
        results[name] = {
            "type": "deep",
            "n_params": n_params,
            "train_time_s": round(dt, 1),
            **{k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in runs.items()},
        }
        test_curves[name] = last_med.tolist()
        cov_curves[name] = np.mean(covs, axis=0).tolist()
        print(f"{name:9s} | MAE={np.mean(runs['MAE']):8.1f}+-{np.std(runs['MAE']):6.1f} "
              f"| MAPE={np.mean(runs['MAPE']):5.2f}% | Pinball={np.mean(runs['PinballLoss']):7.1f} "
              f"| params={n_params:6d} | {dt:.1f}s")

    payload = {
        "protocol": COMMON,
        "multivariate_models": MULTIVARIATE_MODELS,
        "multivariate_features": ["Quantity"] + top_exog,
        "quantiles": QUANTILES,
        "seeds": seeds,
        "test_months": months[val_end:].strftime("%Y-%m").tolist(),
        "test_actual": test_actual.tolist() if test_actual is not None else None,
        "results": results,
        "test_curves": test_curves,
        "coverage_curves": cov_curves,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nDa luu results/comparison.json")

    _make_figures(payload)


def _make_figures(payload: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    res = payload["results"]
    order = ["Naive", "SeasonalNaive", "MLP", "DLinear", "TSMixer", "NHITS", "NBEATS"]
    order = [m for m in order if m in res]
    labels = ["N-BEATS" if m == "NBEATS" else m for m in order]
    colors = ["#9aa0a6", "#9aa0a6", "#4c72b0", "#55a868", "#c44e52", "#8172b3", "#dd8452"]

    # --- Fig A: MAE va MAPE (bar) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, metric, title in [(axes[0], "MAE", "MAE (don vi)"),
                              (axes[1], "MAPE", "MAPE (%)")]:
        means = [res[m][metric]["mean"] for m in order]
        stds = [res[m][metric]["std"] for m in order]
        bars = ax.bar(labels, means, yerr=stds, color=colors[:len(order)],
                      capsize=3, edgecolor="black", linewidth=0.4)
        ax.set_title(f"So sanh {title}")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=20)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}" if metric == "MAE" else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cmp1_mae_mape.png", dpi=130)
    plt.close(fig)

    # --- Fig B: chi cac mo hinh hoc sau, zoom MAE + Pinball + so tham so ---
    deep = [m for m in order if res[m]["type"] == "deep"]
    dlabels = ["N-BEATS" if m == "NBEATS" else m for m in deep]
    dcolors = {"MLP": "#4c72b0", "DLinear": "#55a868", "TSMixer": "#c44e52",
               "NHITS": "#8172b3", "NBEATS": "#dd8452"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, metric, title in [(axes[0], "MAE", "MAE"),
                              (axes[1], "PinballLoss", "Pinball Loss")]:
        means = [res[m][metric]["mean"] for m in deep]
        stds = [res[m][metric]["std"] for m in deep]
        ax.bar(dlabels, means, yerr=stds, color=[dcolors[m] for m in deep],
               capsize=3, edgecolor="black", linewidth=0.4)
        ax.set_title(f"{title} (chi mo hinh hoc sau)")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=20)
    params = [res[m]["n_params"] for m in deep]
    axes[2].bar(dlabels, params, color=[dcolors[m] for m in deep],
                edgecolor="black", linewidth=0.4)
    axes[2].set_title("So tham so mo hinh")
    axes[2].set_ylabel("So tham so")
    axes[2].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cmp2_deep_detail.png", dpi=130)
    plt.close(fig)

    # --- Fig C: du bao P50 cua tung mo hinh vs thuc te ---
    import pandas as pd
    months = pd.to_datetime(payload["test_months"], format="%Y-%m")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(months, payload["test_actual"], "k-o", lw=2.2, label="Thuc te", zorder=5)
    for m in deep:
        ax.plot(months, payload["test_curves"][m], marker=".", lw=1.3,
                color=dcolors[m], label=("N-BEATS" if m == "NBEATS" else m), alpha=0.9)
    ax.set_title("Du bao trung vi (P50) cac mo hinh vs thuc te (tap test 2024)")
    ax.set_ylabel("Quantity")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cmp3_test_forecast.png", dpi=130)
    plt.close(fig)

    # --- Fig D: calibration cac mo hinh ---
    fig, ax = plt.subplots(figsize=(7.2, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ly tuong")
    for m in deep:
        ax.plot(payload["quantiles"], payload["coverage_curves"][m], marker="o",
                color=dcolors[m], label=("N-BEATS" if m == "NBEATS" else m))
    ax.set_xlabel("Muc phan vi danh nghia")
    ax.set_ylabel("Ty le bao phu thuc te")
    ax.set_title("Calibration cac mo hinh (tap test, n=12)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cmp4_calibration.png", dpi=130)
    plt.close(fig)

    print("Da luu 4 bieu do so sanh vao report/figures/cmp*.png")


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    main(n_seeds)
