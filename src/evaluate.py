"""Metrics danh gia va cac baseline so sanh."""
import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, quantiles) -> float:
    """Pinball (quantile) loss trung binh tren toan bo mau va cac phan vi.

    y_true: (N,), y_pred_q: (N, Q), quantiles: list/array (Q,).
    Day la thuoc do "dung" cho du bao phan vi — phat sai so bat doi xung theo
    tung muc phan vi.
    """
    y_true = np.asarray(y_true).reshape(-1, 1)
    q = np.asarray(quantiles).reshape(1, -1)
    errors = y_true - y_pred_q
    loss = np.maximum(q * errors, (q - 1.0) * errors)
    return float(loss.mean())


def coverage(y_true: np.ndarray, y_pred_q: np.ndarray, quantiles) -> dict:
    """Ty le thuc te <= phan vi du bao, theo tung muc — dung de kiem tra hieu
    chinh (calibration). Ly tuong: coverage cua muc q ~= q."""
    y_true = np.asarray(y_true).reshape(-1, 1)
    cov = (y_true <= y_pred_q).mean(axis=0)
    return {f"{q:.2f}": float(c) for q, c in zip(quantiles, cov)}


def naive_forecast(series: np.ndarray, t_range: tuple[int, int]) -> np.ndarray:
    """Du bao = gia tri thang truoc."""
    return series[t_range[0] - 1 : t_range[1] - 1]


def seasonal_naive_forecast(series: np.ndarray, t_range: tuple[int, int], season: int = 12) -> np.ndarray:
    """Du bao = gia tri cung thang nam truoc."""
    return series[t_range[0] - season : t_range[1] - season]
