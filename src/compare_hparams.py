"""So sanh TOP-5 bo sieu tham so cua RevIN+N-BEATS.

Chay Optuna (cung khong gian tim kiem va seed nhu tune_optuna -> tat dinh, cho
ket qua trung khop voi mo hinh trien khai), lay 5 trial COMPLETE tot nhat theo
pinball loss tren validation. Voi moi bo: huan luyen lai tren train+val (so
epoch xac dinh bang early stopping tren val) roi danh gia tren test.

Muc dich: kiem chung xem xep hang theo validation co khop voi test khong, va
cung cap mot bang so sanh nhieu bo sieu tham so (khong chi mot bo tot nhat).

Chay:  python -m src.compare_hparams [n_trials] [top_k]
Ket qua: results/hparam_top5.json + report/figures/hparam_top5.png
"""
import json
import sys

import numpy as np
import optuna
import torch

from .data_loader import ROOT, load_series, split_indices
from .tune_all import HORIZON, MEDIAN_IDX, QUANTILES, eval_test, suggest, train_with_params

RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "report" / "figures"


def evaluate_params(params, data, train_end, val_end, n):
    """Tra ve (val_pinball, best_epoch, test_metrics) cho mot bo sieu tham so
    N-BEATS, dung CUNG machinery voi tune_all de nhat quan."""
    _, val_pin, best_epoch = train_with_params(
        "NBEATS", params, data, (0, train_end), (train_end, val_end))
    model, _, _ = train_with_params(
        "NBEATS", params, data, (0, val_end), None, fixed_epochs=max(best_epoch, 1), seed=0)
    m, _ = eval_test(model, "NBEATS", params, data, (val_end, n))
    return val_pin * data["scale"], best_epoch, m


def main(n_trials: int = 100, top_k: int = 5) -> None:
    y, months = load_series()
    n = len(y)
    train_end, val_end = split_indices(n)
    scale = float(y[:train_end].mean())
    ys = y / scale
    data = {"ys": ys, "scale": scale, "Xs": None, "target_std": None, "n_features": 1}

    def objective(trial: optuna.Trial) -> float:
        params = suggest("NBEATS", trial)
        _, val_pin, _ = train_with_params(
            "NBEATS", params, data, (0, train_end), (train_end, val_end), trial=trial)
        return val_pin

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=30),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Chi xet cac trial chay xong (khong bi prune), xep theo val pinball.
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top = sorted(completed, key=lambda t: t.value)[:top_k]

    rows = []
    print(f"\n{'Hang':>4} {'Trial':>5} {'ValPin':>8} {'Epoch':>6} "
          f"{'TestMAE':>9} {'MAPE%':>6} {'TestPin':>8}")
    for rank, t in enumerate(top, 1):
        val_pin, best_epoch, m = evaluate_params(t.params, data, train_end, val_end, n)
        rows.append({
            "rank": rank, "trial": t.number, "val_pinball": val_pin,
            "best_epoch": best_epoch, "params": t.params, "test": m,
        })
        print(f"{rank:>4} {t.number:>5} {val_pin:>8.0f} {best_epoch:>6} "
              f"{m['MAE']:>9,.0f} {m['MAPE']:>6.2f} {m['PinballLoss']:>8.0f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "hparam_top5.json", "w") as f:
        json.dump({"n_trials": n_trials, "top_k": top_k,
                   "n_completed": len(completed), "rows": rows}, f, indent=2)
    print(f"\nDa luu results/hparam_top5.json")
    _make_figure(rows)


def _make_figure(rows: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = [f"#{r['rank']}\n(trial {r['trial']})" for r in rows]
    val = [r["val_pinball"] for r in rows]
    test_mae = [r["test"]["MAE"] for r in rows]
    test_pin = [r["test"]["PinballLoss"] for r in rows]
    # To dam bo tot nhat (rank 1 = mo hinh trien khai).
    colors = ["#dd8452"] + ["#4c72b0"] * (len(rows) - 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, vals, title in [
        (axes[0], val, "Pinball loss trên validation\n(tiêu chí xếp hạng)"),
        (axes[1], test_mae, "MAE trên test"),
        (axes[2], test_pin, "Pinball loss trên test"),
    ]:
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=8)
        ax.margins(y=0.15)
    fig.suptitle("Top-5 bộ siêu tham số RevIN+N-BEATS (cam = bộ triển khai)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hparam_top5.png", dpi=130)
    plt.close(fig)
    print("Da luu report/figures/hparam_top5.png")


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    main(n_trials, top_k)
