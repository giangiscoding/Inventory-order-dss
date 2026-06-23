"""Tinh chinh Optuna RIENG cho TUNG mo hinh -> so sanh CONG BANG.

Khac voi compare_models (cung mot cau hinh co dinh cho moi mo hinh), o day moi
kien truc duoc Optuna tim bo sieu tham so tot nhat cua RIENG no (cung so trial),
toi uu pinball loss tren validation. Sau do huan luyen lai bo tot nhat tren
train+val (so epoch tu early stopping) qua nhieu seed va danh gia tren test.

  - MLP, NHITS, N-BEATS : don bien (chi Quantity).
  - DLinear, TSMixer    : da bien (Quantity + 6 ngoai sinh |r| cao nhat).

Chay:  python -m src.tune_all [n_trials] [n_seeds]
Ket qua: results/tune_all.json + report/figures/tune_all.png
"""
import copy
import json
import sys
import time

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from .data_loader import (ROOT, MultiWindowDataset, WindowDataset, load_multivariate,
                          load_series, split_indices)
from .evaluate import all_metrics, coverage, pinball_loss
from .models import (DEFAULT_QUANTILES, MULTIVARIATE_MODELS, build_model,
                     build_model_multi)
from .train import pinball_loss_torch, predict_quantiles

HORIZON = 1
QUANTILES = DEFAULT_QUANTILES
MEDIAN_IDX = QUANTILES.index(0.5)
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "report" / "figures"

MODELS = ["MLP", "DLinear", "TSMixer", "NHITS", "NBEATS"]
MAX_EPOCHS = 300
PATIENCE = 30


def suggest(name: str, trial: optuna.Trial) -> dict:
    p = {
        "lookback": trial.suggest_int("lookback", 6, 24),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "revin_affine": trial.suggest_categorical("revin_affine", [True, False]),
    }
    if name == "DLinear":                       # DLinear: l015i tuyen tinh, rat it tham so
        p["kernel_size"] = trial.suggest_categorical("kernel_size", [7, 13, 25])
        return p
    p["hidden_size"] = trial.suggest_int("hidden_size", 64, 512, step=64)
    p["n_layers"] = trial.suggest_int("n_layers", 2, 4)
    p["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
    if name == "NBEATS":
        p["n_stacks"] = trial.suggest_int("n_stacks", 1, 3)
        p["n_blocks"] = trial.suggest_int("n_blocks", 1, 4)
    return p


def _build(name, params, n_features):
    common = dict(hidden_size=params.get("hidden_size", 256),
                  n_layers=params.get("n_layers", 3),
                  dropout=params.get("dropout", 0.0),
                  revin_affine=params["revin_affine"])
    if name in MULTIVARIATE_MODELS:
        return build_model_multi(name, params["lookback"], n_features, HORIZON,
                                 len(QUANTILES), kernel_size=params.get("kernel_size", 13),
                                 **common)
    return build_model(name, params["lookback"], HORIZON, len(QUANTILES),
                       n_stacks=params.get("n_stacks", 2), n_blocks=params.get("n_blocks", 2),
                       **common)


def _dataset(name, params, data, t_range):
    L = params["lookback"]
    if name in MULTIVARIATE_MODELS:
        return MultiWindowDataset(data["Xs"], L, HORIZON, t_range), data["target_std"]
    return WindowDataset(data["ys"], L, HORIZON, t_range), data["scale"]


def train_with_params(name, params, data, train_range, val_range,
                      fixed_epochs=None, seed=42, trial=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    q_tensor = torch.tensor(QUANTILES, dtype=torch.float32)

    train_ds, _ = _dataset(name, params, data, train_range)
    loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_ds = _dataset(name, params, data, val_range)[0] if val_range else None

    model = _build(name, params, data["n_features"])
    opt = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    best_val, best_state, best_epoch, no_improve = float("inf"), None, 0, 0
    n_epochs = fixed_epochs if fixed_epochs is not None else MAX_EPOCHS
    for epoch in range(1, n_epochs + 1):
        model.train()
        for x, y in loader:
            opt.zero_grad()
            loss = pinball_loss_torch(model(x), y, q_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if val_ds is None or fixed_epochs is not None:
            continue
        yt, yp = predict_quantiles(model, val_ds)
        val_pin = pinball_loss(yt, yp, QUANTILES)
        if trial is not None:
            trial.report(val_pin, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if val_pin < best_val - 1e-9:
            best_val, best_state, best_epoch, no_improve = val_pin, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = n_epochs
    return model, best_val, best_epoch


def eval_test(model, name, params, data, test_range):
    ds, descale = _dataset(name, params, data, test_range)
    yt, yq = predict_quantiles(model, ds)
    yt, yq = yt * descale, yq * descale
    m = all_metrics(yt, yq[:, MEDIAN_IDX])
    m["PinballLoss"] = pinball_loss(yt, yq, QUANTILES)
    cov = coverage(yt, yq, QUANTILES)
    return m, cov


def main(n_trials: int = 100, n_seeds: int = 3) -> None:
    y, months = load_series()
    n = len(y)
    train_end, val_end = split_indices(n)
    scale = float(y[:train_end].mean())
    ys = y / scale

    Xmv, top_exog, _ = load_multivariate(k=6)
    col_std = Xmv[:train_end].std(axis=0)
    col_std[col_std == 0] = 1.0
    Xs = Xmv / col_std
    data = {"ys": ys, "scale": scale, "Xs": Xs,
            "target_std": float(col_std[0]), "n_features": Xmv.shape[1]}
    print(f"Da bien (TSMixer/DLinear) dung: {['Quantity'] + top_exog}\n")

    results = {}
    for name in MODELS:
        t0 = time.time()

        def objective(trial):
            params = suggest(name, trial)
            _, val_pin, _ = train_with_params(
                name, params, data, (0, train_end), (train_end, val_end), trial=trial)
            return val_pin

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=30),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = dict(study.best_params)

        # So epoch toi uu (early stopping) roi danh gia test qua nhieu seed.
        _, _, best_epoch = train_with_params(name, best, data, (0, train_end), (train_end, val_end))
        runs = {"MAE": [], "RMSE": [], "MAPE": [], "PinballLoss": []}
        covs, n_params = [], None
        for seed in range(n_seeds):
            model, _, _ = train_with_params(name, best, data, (0, val_end), None,
                                            fixed_epochs=max(best_epoch, 1), seed=seed)
            if n_params is None:
                n_params = sum(p.numel() for p in model.parameters())
            m, cov = eval_test(model, name, best, data, (val_end, n))
            for k in runs:
                runs[k].append(m[k])
            covs.append([cov[f"{q:.2f}"] for q in QUANTILES])

        dt = time.time() - t0
        results[name] = {
            "multivariate": name in MULTIVARIATE_MODELS,
            "best_params": best,
            "best_val_pinball": float(study.best_value),
            "best_epoch": best_epoch,
            "n_params": n_params,
            "n_trials": n_trials,
            "coverage": np.mean(covs, axis=0).tolist(),
            **{k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in runs.items()},
        }
        print(f"{name:8s} | valPin={study.best_value:7.1f} | "
              f"MAE={np.mean(runs['MAE']):8,.0f}+-{np.std(runs['MAE']):5,.0f} | "
              f"MAPE={np.mean(runs['MAPE']):4.2f}% | Pin={np.mean(runs['PinballLoss']):6,.0f} | "
              f"par={n_params:6d} | {dt:.0f}s")

    payload = {"n_trials": n_trials, "n_seeds": n_seeds,
               "multivariate_features": ["Quantity"] + top_exog, "results": results}
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "tune_all.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("\nDa luu results/tune_all.json")
    _make_figure(results)


def _make_figure(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = sorted(results, key=lambda m: results[m]["PinballLoss"]["mean"], reverse=True)
    labels = [("N-BEATS" if m == "NBEATS" else m) +
              ("\n(đa biến)" if results[m]["multivariate"] else "") for m in order]
    colors = ["#dd8452" if m == "NBEATS" else
              ("#c44e52" if results[m]["multivariate"] else "#4c72b0") for m in order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, key, title in [(axes[0], "MAE", "MAE (test)"),
                           (axes[1], "PinballLoss", "Pinball loss (test)")]:
        means = [results[m][key]["mean"] for m in order]
        stds = [results[m][key]["std"] for m in order]
        bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=3,
                      edgecolor="black", linewidth=0.4)
        ax.set_title(f"{title} — mỗi mô hình tinh chỉnh Optuna riêng")
        ax.tick_params(axis="x", labelsize=8)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center",
                    va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tune_all.png", dpi=130)
    plt.close(fig)
    print("Da luu report/figures/tune_all.png")


if __name__ == "__main__":
    nt = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    main(nt, ns)
