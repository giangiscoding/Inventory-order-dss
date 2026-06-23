"""Pipeline du lieu: doc CSV, chia 8/1/1 theo thoi gian, tao cua so truot."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Data" / "data_TSI_v2.csv"

TARGET_COL = "Quantity"


def load_dataframe(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
    df = df.sort_values("Month").reset_index(drop=True)
    return df


def load_series(path: Path = DATA_PATH) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Tra ve chuoi muc tieu Quantity va truc thoi gian."""
    df = load_dataframe(path)
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    months = pd.DatetimeIndex(df["Month"])
    return y, months


def split_indices(n: int, ratios: tuple[float, float, float] = (0.7, 0.1, 0.2)) -> tuple[int, int]:
    """Chia 7/1/2 theo thoi gian (test lon hon de danh gia on dinh hon).
    Tra ve (train_end, val_end): train [0, train_end), val [train_end, val_end),
    test [val_end, n)."""
    train_end = int(round(n * ratios[0]))
    val_end = train_end + int(round(n * ratios[1]))
    return train_end, val_end


EXOG_COLS = [
    "CompetitorQuantity", "PromotionAmount", "Construction", "CPI", "Exports",
    "Imports", "IPI", "RegisteredFDI", "DisbursedFDI", "RetailSales",
]


def select_top_exog(df: pd.DataFrame, train_end: int, k: int = 6) -> list[str]:
    """Chon k bien ngoai sinh co |tuong quan Pearson| voi Quantity cao nhat,
    tinh TREN TAP TRAIN (tranh ro ri du lieu)."""
    tr = df.iloc[:train_end]
    corrs = {c: abs(float(tr[c].corr(tr[TARGET_COL]))) for c in EXOG_COLS}
    return sorted(corrs, key=corrs.get, reverse=True)[:k]


def load_multivariate(path: Path = DATA_PATH, k: int = 6
                      ) -> tuple[np.ndarray, list[str], pd.DatetimeIndex]:
    """Tra ve ma tran dac trung (n, F) voi cot 0 = Quantity (muc tieu) va k bien
    ngoai sinh tuong quan cao nhat; kem danh sach ten bien ngoai sinh va truc
    thoi gian."""
    df = load_dataframe(path)
    train_end, _ = split_indices(len(df))
    top = select_top_exog(df, train_end, k)
    cols = [TARGET_COL] + top
    X = df[cols].to_numpy(dtype=np.float64)
    months = pd.DatetimeIndex(df["Month"])
    return X, top, months


class WindowDataset(Dataset):
    """Cua so truot: input `lookback` diem -> target `horizon` diem ke tiep.

    Mau co chi so muc tieu t su dung input y[t-lookback:t] va target y[t:t+horizon].
    `t_range` la khoang chi so muc tieu (start, end) de gioi han mau theo tap
    train/val/test ma khong ro ri du lieu tuong lai vao input.
    """

    def __init__(self, series: np.ndarray, lookback: int, horizon: int, t_range: tuple[int, int]):
        self.series = torch.as_tensor(series, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon
        start = max(t_range[0], lookback)
        end = min(t_range[1], len(series) - horizon + 1)
        self.targets = list(range(start, end))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        t = self.targets[idx]
        x = self.series[t - self.lookback : t]
        y = self.series[t : t + self.horizon]
        return x, y


class MultiWindowDataset(Dataset):
    """Cua so truot DA BIEN: input (lookback, F) gom F dac trung -> target
    (horizon,) la bien muc tieu (cot `target_idx`). Dung cho cac mo hinh da bien
    (TSMixer, DLinear). `features` co dang (n, F)."""

    def __init__(self, features: np.ndarray, lookback: int, horizon: int,
                 t_range: tuple[int, int], target_idx: int = 0):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.target_idx = target_idx
        start = max(t_range[0], lookback)
        end = min(t_range[1], len(features) - horizon + 1)
        self.targets = list(range(start, end))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        t = self.targets[idx]
        x = self.features[t - self.lookback : t]                 # (L, F)
        y = self.features[t : t + self.horizon, self.target_idx]  # (H,)
        return x, y
