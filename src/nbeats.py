"""N-BEATS (generic) tu xay dung + tich hop RevIN + du bao DA PHAN VI (quantile).

Thay vi chi du bao mot gia tri diem, moi block xuat ra `horizon * n_quantiles`
gia tri. Dau ra cuoi cung duoc rang buoc don dieu theo chieu phan vi (P5 <= P10
<= ... <= P95) bang tham so hoa "base + cong don softplus", tranh hien tuong
quantile crossing.

Tham khao: Oreshkin et al., "N-BEATS", ICLR 2020;
Wen et al., "A Multi-Horizon Quantile Recurrent Forecaster", 2017 (pinball loss).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .revin import RevIN

# Cac muc phan vi mac dinh (phai tang dan, chua 0.5 lam trung vi).
DEFAULT_QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


class NBeatsBlock(nn.Module):
    """Mot block generic: MLP chung -> backcast (lookback) va forecast
    (horizon * n_quantiles)."""

    def __init__(self, lookback: int, horizon: int, n_quantiles: int,
                 hidden_size: int, n_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = lookback
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_size), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        self.mlp = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(hidden_size, lookback)
        self.forecast_head = nn.Linear(hidden_size, horizon * n_quantiles)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(x)
        return self.backcast_head(h), self.forecast_head(h)


class NBeats(nn.Module):
    """Doubly-residual stacking: residual input qua tung block, forecast cong don;
    cuoi cung rang buoc don dieu theo chieu phan vi."""

    def __init__(self, lookback: int, horizon: int, n_quantiles: int, n_stacks: int,
                 n_blocks: int, hidden_size: int, n_layers: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            NBeatsBlock(lookback, horizon, n_quantiles, hidden_size, n_layers, dropout)
            for _ in range(n_stacks * n_blocks)
        )
        self.horizon = horizon
        self.n_quantiles = n_quantiles

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        forecast = torch.zeros(x.shape[0], self.horizon * self.n_quantiles,
                               device=x.device, dtype=x.dtype)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        forecast = forecast.view(x.shape[0], self.horizon, self.n_quantiles)
        # Rang buoc don dieu: phan vi thap nhat la base, cac phan vi sau cong don
        # cac so duong (softplus) -> dam bao khong cat nhau.
        base = forecast[..., :1]
        increments = F.softplus(forecast[..., 1:])
        return torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)


class RevinNBeats(nn.Module):
    """Pipeline: RevIN.normalize -> N-BEATS (da phan vi) -> RevIN.denormalize.

    Dau ra: tensor (batch, horizon, n_quantiles).
    """

    def __init__(self, lookback: int, horizon: int, n_quantiles: int, n_stacks: int = 2,
                 n_blocks: int = 2, hidden_size: int = 256, n_layers: int = 3,
                 dropout: float = 0.1, revin_affine: bool = True):
        super().__init__()
        self.revin = RevIN(affine=revin_affine)
        self.nbeats = NBeats(lookback, horizon, n_quantiles, n_stacks, n_blocks,
                             hidden_size, n_layers, dropout)
        self.n_quantiles = n_quantiles

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.revin.normalize(x)
        forecast = self.nbeats(x)
        return self.revin.denormalize(forecast)
