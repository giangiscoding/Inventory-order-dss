"""Vong lap huan luyen RevIN + N-BEATS da phan vi voi pinball loss va early
stopping theo pinball loss tren tap validation."""
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_loader import WindowDataset
from .nbeats import DEFAULT_QUANTILES, RevinNBeats


def make_model(params: dict, horizon: int, n_quantiles: int) -> RevinNBeats:
    return RevinNBeats(
        lookback=params["lookback"],
        horizon=horizon,
        n_quantiles=n_quantiles,
        n_stacks=params["n_stacks"],
        n_blocks=params["n_blocks"],
        hidden_size=params["hidden_size"],
        n_layers=params["n_layers"],
        dropout=params["dropout"],
        revin_affine=params["revin_affine"],
    )


def pinball_loss_torch(pred: torch.Tensor, target: torch.Tensor,
                       quantiles: torch.Tensor) -> torch.Tensor:
    """pred: (B, H, Q), target: (B, H), quantiles: (Q,)."""
    target = target.unsqueeze(-1)
    errors = target - pred
    q = quantiles.view(1, 1, -1)
    loss = torch.maximum(q * errors, (q - 1.0) * errors)
    return loss.mean()


@torch.no_grad()
def predict_quantiles(model: RevinNBeats, dataset: WindowDataset) -> tuple[np.ndarray, np.ndarray]:
    """Du bao toan bo dataset (horizon=1). Tra ve (y_true (N,), y_pred (N, Q))."""
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    trues, preds = [], []
    for x, y in loader:
        out = model(x)                       # (B, 1, Q)
        trues.append(y.numpy().reshape(-1))  # (B,)
        preds.append(out.numpy()[:, 0, :])   # (B, Q)
    return np.concatenate(trues), np.concatenate(preds)


def train_model(
    series_scaled: np.ndarray,
    params: dict,
    horizon: int,
    train_range: tuple[int, int],
    val_range: tuple[int, int] | None,
    quantiles: list[float] = DEFAULT_QUANTILES,
    max_epochs: int = 300,
    patience: int = 25,
    fixed_epochs: int | None = None,
    seed: int = 42,
    trial=None,
):
    """Huan luyen tren train_range; neu co val_range thi early stopping theo
    pinball loss tren validation. Tra ve (model, best_val_pinball, best_epoch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    q_tensor = torch.tensor(quantiles, dtype=torch.float32)
    lookback = params["lookback"]
    train_ds = WindowDataset(series_scaled, lookback, horizon, train_range)
    val_ds = WindowDataset(series_scaled, lookback, horizon, val_range) if val_range else None
    if len(train_ds) == 0:
        raise ValueError("Tap train rong — lookback qua lon so voi du lieu.")

    loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    model = make_model(params, horizon, len(quantiles))
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    n_epochs = fixed_epochs if fixed_epochs is not None else max_epochs

    for epoch in range(1, n_epochs + 1):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = pinball_loss_torch(model(x), y, q_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if val_ds is None or fixed_epochs is not None:
            continue

        from .evaluate import pinball_loss
        y_true, y_pred = predict_quantiles(model, val_ds)
        val_pinball = pinball_loss(y_true, y_pred, quantiles)
        if trial is not None:
            trial.report(val_pinball, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        if val_pinball < best_val - 1e-9:
            best_val = val_pinball
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = n_epochs
    return model, best_val, best_epoch
