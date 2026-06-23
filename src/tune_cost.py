"""Tinh chinh sieu tham so theo TONG CHI PHI ton kho (decision-focused).

Lay y tuong tu bai bao RevIN-TSMixer (Nguyen et al., ESWA 2026): thay vi chi
toi uu sai so du bao (pinball/MAPE - "Scenario 1"), ta toi uu sieu tham so de
giam truc tiep TONG CHI PHI ton kho theo chinh sach (r, q) ("Scenario 2").

Luan diem: du bao chinh xac nhat chua chac cho quyet dinh ton kho re nhat. Mo
hinh tot ve chi phi can sai so NHO va it thien lech o vung anh huong chi phi.

Script tinh chinh N-BEATS (mo hinh trien khai) theo ca hai muc tieu va so sanh
tren test (MAPE va tong chi phi). Scenario 1 lay tu results/tune_all.json.

Chay:  python -m src.tune_cost [n_trials] [n_seeds]
Ket qua: results/cost_opt.json + report/figures/cost_opt.png
"""
import copy
import json
import sys

import numpy as np
import optuna

from .data_loader import ROOT, load_multivariate, load_series, split_indices
from .evaluate import all_metrics, pinball_loss
from .inventory import rq_total_cost
from .train import predict_quantiles
from .tune_all import (HORIZON, MEDIAN_IDX, QUANTILES, _dataset, suggest,
                       train_with_params)

RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "report" / "figures"

# Cau truc chi phi minh hoa (dong vi don vi), thong nhat voi vi du Newsvendor
# trong bao cao: thieu hang 30k/don vi, luu kho 4k/don vi/thang.
C_SHORT = 30_000.0     # chi phi thieu 1 don vi
H_HOLD = 4_000.0       # chi phi luu kho 1 don vi / thang
O_ORDER = 500_000.0    # chi phi co dinh moi lan dat
LEAD = 2.0             # lead time (thang)
MODEL = "NBEATS"

# Dai chi phi thieu hang de QUET (dong/don vi), lay TC_min nhu bai bao.
# Quet log tu nguong alpha>0 (~vai tram) den muc cao -> bao phu day muc phuc vu.
C_GRID = np.logspace(np.log10(300), np.log10(60_000), 60)

# sigma_D: "demand" = do lech chuan nhu cau (dung cong thuc bai bao);
#          "error"  = do lech chuan SAI SO du bao (decision-focused).
SIGMA_MODE = "demand"


def forecast_mu_sigma(model, params, data, t_range):
    """Tra ve (mu_D, sigma_D, MAPE) tu du bao P50 tren mot khoang."""
    ds, descale = _dataset(MODEL, params, data, t_range)
    yt, yq = predict_quantiles(model, ds)
    yt = yt * descale
    yp50 = yq[:, MEDIAN_IDX] * descale
    mu = float(np.mean(yp50))
    if SIGMA_MODE == "demand":
        sigma = float(np.std(yt, ddof=1))           # do lech chuan nhu cau thuc
    else:
        sigma = float(np.std(yt - yp50, ddof=1))    # do lech chuan sai so du bao
    mape = all_metrics(yt, yp50)["MAPE"]
    return mu, sigma, mape


def tc_curve(mu, sigma):
    """Tong chi phi theo tung muc c_s tren C_GRID."""
    return np.array([rq_total_cost(mu, sigma, cs, H_HOLD, O_ORDER, LEAD)[0] for cs in C_GRID])


def total_cost_min(model, params, data, t_range):
    """TC_min = min theo c_s (nhu bai bao). Tra ve (tc_min, c_star, curve)."""
    mu, sigma, _ = forecast_mu_sigma(model, params, data, t_range)
    curve = tc_curve(mu, sigma)
    j = int(np.argmin(curve))
    return float(curve[j]), float(C_GRID[j]), curve


def evaluate(params, data, train_end, val_end, n, n_seeds):
    """Huan luyen lai (n_seeds) tren train+val; tra ve test MAPE, TC_min, c_star
    trung binh va duong cong TC theo c_s (seed cuoi)."""
    _, _, best_epoch = train_with_params(MODEL, params, data, (0, train_end), (train_end, val_end))
    mapes, tcs, cstars, curve = [], [], [], None
    for seed in range(n_seeds):
        model, _, _ = train_with_params(MODEL, params, data, (0, val_end), None,
                                        fixed_epochs=max(best_epoch, 1), seed=seed)
        mu, sigma, mape = forecast_mu_sigma(model, params, data, (val_end, n))
        tc_min, c_star, curve = total_cost_min(model, params, data, (val_end, n))
        mapes.append(mape); tcs.append(tc_min); cstars.append(c_star)
    return {"MAPE_mean": float(np.mean(mapes)), "MAPE_std": float(np.std(mapes)),
            "TC_mean": float(np.mean(tcs)), "TC_std": float(np.std(tcs)),
            "c_star": float(np.mean(cstars)), "tc_curve": curve.tolist()}


def main(n_trials: int = 100, n_seeds: int = 3) -> None:
    y, months = load_series()
    n = len(y)
    train_end, val_end = split_indices(n)
    scale = float(y[:train_end].mean())
    ys = y / scale
    Xmv, _, _ = load_multivariate(k=6)
    col_std = Xmv[:train_end].std(axis=0)
    col_std[col_std == 0] = 1.0
    data = {"ys": ys, "scale": scale, "Xs": Xmv / col_std,
            "target_std": float(col_std[0]), "n_features": Xmv.shape[1]}

    # ----- Scenario 2: Optuna toi uu TONG CHI PHI tren validation -----
    def objective(trial):
        params = suggest(MODEL, trial)
        model, _, _ = train_with_params(MODEL, params, data, (0, train_end),
                                        (train_end, val_end), trial=trial)
        return total_cost_min(model, params, data, (train_end, val_end))[0]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=30),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_cost_params = dict(study.best_params)

    # ----- Scenario 1: lay bo tot nhat theo pinball tu tune_all -----
    with open(RESULTS_DIR / "tune_all.json") as f:
        best_acc_params = json.load(f)["results"][MODEL]["best_params"]

    s1 = evaluate(best_acc_params, data, train_end, val_end, n, n_seeds)
    s2 = evaluate(best_cost_params, data, train_end, val_end, n, n_seeds)

    payload = {
        "sigma_mode": SIGMA_MODE,
        "cost_params": {"holding": H_HOLD, "ordering": O_ORDER, "lead": LEAD},
        "c_grid": C_GRID.tolist(),
        "scenario1_accuracy": {"params": best_acc_params, **s1},
        "scenario2_cost": {"params": best_cost_params, **s2},
    }
    with open(RESULTS_DIR / "cost_opt.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'Kich ban':28s} {'MAPE%':>8s} {'TC_min':>14s} {'c_star':>9s}")
    print(f"{'S1: toi uu sai so (pinball)':28s} {s1['MAPE_mean']:>7.2f} {s1['TC_mean']:>14,.0f} {s1['c_star']:>9,.0f}")
    print(f"{'S2: toi uu tong chi phi':28s} {s2['MAPE_mean']:>7.2f} {s2['TC_mean']:>14,.0f} {s2['c_star']:>9,.0f}")
    print(f"\nGiam TC_min cua S2 so voi S1: {(1 - s2['TC_mean']/s1['TC_mean'])*100:.1f}%")
    print("Da luu results/cost_opt.json")
    _make_figure(payload)


def _make_figure(payload: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    s1, s2 = payload["scenario1_accuracy"], payload["scenario2_cost"]
    cg = np.array(payload["c_grid"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    # Trai: duong cong Tong chi phi theo c_s (hinh chu U)
    axes[0].plot(cg, s1["tc_curve"], color="#4c72b0", lw=1.8, label="S1: tối ưu sai số")
    axes[0].plot(cg, s2["tc_curve"], color="#dd8452", lw=1.8, label="S2: tối ưu tổng chi phí")
    for s, c in [(s1, "#4c72b0"), (s2, "#dd8452")]:
        axes[0].scatter([s["c_star"]], [s["TC_mean"]], color=c, zorder=5, s=30)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Chi phí thiếu hàng $c_s$ (đồng/đơn vị)")
    axes[0].set_ylabel("Tổng chi phí kỳ vọng")
    axes[0].set_title("Tổng chi phí theo $c_s$ (test); chấm = $TC_{\\min}$")
    axes[0].legend(fontsize=9)
    # Phai: MAPE
    labels = ["S1: tối ưu\nsai số", "S2: tối ưu\ntổng chi phí"]
    axes[1].bar(labels, [s1["MAPE_mean"], s2["MAPE_mean"]],
                yerr=[s1["MAPE_std"], s2["MAPE_std"]], color=["#4c72b0", "#dd8452"],
                capsize=3, edgecolor="black", linewidth=0.4)
    axes[1].set_title("MAPE trên test (%)")
    axes[1].tick_params(axis="x", labelsize=9)
    fig.suptitle("N-BEATS: tinh chỉnh theo sai số (S1) vs theo tổng chi phí (S2)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cost_opt.png", dpi=130)
    plt.close(fig)
    print("Da luu report/figures/cost_opt.png")


if __name__ == "__main__":
    nt = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    main(nt, ns)
