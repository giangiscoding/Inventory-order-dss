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
