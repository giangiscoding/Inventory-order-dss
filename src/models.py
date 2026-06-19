"""Cac mo hinh du bao chuoi thoi gian DA PHAN VI tu xay dung de SO SANH.

Tat ca cac mo hinh deu:
  - Nhan input cua so (batch, lookback) — chuoi don bien.
  - Bao boc bang RevIN (chuan hoa theo tung instance, dao nguoc khi xuat).
  - Xuat ra tensor (batch, horizon, n_quantiles) DON DIEU theo chieu phan vi
    (P5 <= P10 <= ... <= P95) nho tham so hoa "base + cong don softplus" —
    tranh quantile crossing. Dung chung ham `monotone_quantiles`.

Nho vay viec so sanh la cong bang: cung giao thuc huan luyen (pinball loss),
cung tap du lieu, cung dau ra, chi khac kien truc loi (backbone).

Cac kien truc (tu cai dat, khong dung thu vien ngoai):
  - MLP        : MLP thuan don gian (moc so sanh hoc sau co ban).
  - DLinear    : phan ra xu huong/mua vu + tuyen tinh (Zeng et al., AAAI 2023).
  - TSMixer    : tron theo thoi gian va theo dac trung bang MLP (Chen et al., 2023).
  - NHITS      : lay mau da ty le + noi suy phan cap (Challu et al., AAAI 2023).
  - N-BEATS    : xem src/nbeats.py (RevinNBeats) — doubly-residual stacking.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nbeats import DEFAULT_QUANTILES, RevinNBeats  # noqa: F401  (tien re-export)
from .revin import RevIN


def monotone_quantiles(raw: torch.Tensor, horizon: int, n_quantiles: int) -> torch.Tensor:
    """Bien (batch, horizon*n_quantiles) -> (batch, horizon, n_quantiles) don dieu.

    Phan vi thap nhat la base; cac phan vi sau = base + cong don cac so duong
    (softplus) -> dam bao tang dan, khong cat nhau.
    """
    f = raw.view(raw.shape[0], horizon, n_quantiles)
    base = f[..., :1]
    increments = F.softplus(f[..., 1:])
    return torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)


def _mlp(in_dim: int, hidden: int, n_layers: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = hidden
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# 1. MLP thuan
# ---------------------------------------------------------------------------
class _MLPCore(nn.Module):
    def __init__(self, lookback, horizon, n_quantiles, hidden, n_layers, dropout):
        super().__init__()
        self.horizon, self.n_quantiles = horizon, n_quantiles
        self.net = _mlp(lookback, hidden, n_layers, dropout)
        self.head = nn.Linear(hidden, horizon * n_quantiles)

    def forward(self, x):
        return monotone_quantiles(self.head(self.net(x)), self.horizon, self.n_quantiles)


# ---------------------------------------------------------------------------
# 2. DLinear: phan ra trend (trung binh truot) + phan du, moi nhanh mot Linear
# ---------------------------------------------------------------------------
class _DLinearCore(nn.Module):
    def __init__(self, lookback, horizon, n_quantiles, kernel_size=13, **_):
        super().__init__()
        self.horizon, self.n_quantiles = horizon, n_quantiles
        self.kernel = min(kernel_size if kernel_size % 2 == 1 else kernel_size - 1,
                          lookback if lookback % 2 == 1 else lookback - 1)
        out = horizon * n_quantiles
        self.linear_trend = nn.Linear(lookback, out)
        self.linear_season = nn.Linear(lookback, out)

    def _moving_avg(self, x):  # x: (B, L) -> trend (B, L)
        pad = (self.kernel - 1) // 2
        xp = F.pad(x.unsqueeze(1), (pad, pad), mode="replicate")
        return F.avg_pool1d(xp, self.kernel, stride=1).squeeze(1)

    def forward(self, x):
        trend = self._moving_avg(x)
        season = x - trend
        out = self.linear_trend(trend) + self.linear_season(season)
        return monotone_quantiles(out, self.horizon, self.n_quantiles)


# ---------------------------------------------------------------------------
# 3. TSMixer: cac block tron theo thoi gian va theo dac trung (univariate C=1)
# ---------------------------------------------------------------------------
class _TSMixerBlock(nn.Module):
    def __init__(self, lookback, n_channels, hidden, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_channels)
        self.time = nn.Linear(lookback, lookback)
        self.norm2 = nn.LayerNorm(n_channels)
        self.feat1 = nn.Linear(n_channels, hidden)
        self.feat2 = nn.Linear(hidden, n_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, L, C)
        # Time-mixing: tron thong tin doc theo truc thoi gian.
        h = self.norm1(x).transpose(1, 2)           # (B, C, L)
        h = self.drop(F.relu(self.time(h))).transpose(1, 2)
        x = x + h
        # Feature-mixing: tron giua cac kenh dac trung.
        h = self.norm2(x)
        h = self.drop(self.feat2(F.relu(self.feat1(h))))
        return x + h


class _TSMixerCore(nn.Module):
    def __init__(self, lookback, horizon, n_quantiles, hidden, n_layers, dropout, n_channels=1):
        super().__init__()
        self.horizon, self.n_quantiles, self.C = horizon, n_quantiles, n_channels
        self.blocks = nn.ModuleList(
            _TSMixerBlock(lookback, n_channels, hidden, dropout) for _ in range(n_layers)
        )
        self.head = nn.Linear(lookback * n_channels, horizon * n_quantiles)

    def forward(self, x):  # x: (B, L)
        h = x.unsqueeze(-1)                          # (B, L, C=1)
        for blk in self.blocks:
            h = blk(h)
        h = h.reshape(h.shape[0], -1)               # (B, L*C)
        return monotone_quantiles(self.head(h), self.horizon, self.n_quantiles)


# ---------------------------------------------------------------------------
# 4. NHITS: lay mau da ty le (max-pool) + noi suy phan cap, doubly-residual
# ---------------------------------------------------------------------------
class _NHITSBlock(nn.Module):
    def __init__(self, lookback, horizon, n_quantiles, hidden, n_layers, dropout, pool_kernel):
        super().__init__()
        self.lookback, self.horizon, self.n_quantiles = lookback, horizon, n_quantiles
        self.pool = nn.MaxPool1d(pool_kernel, stride=pool_kernel, ceil_mode=True)
        pooled_len = -(-lookback // pool_kernel)     # ceil
        self.mlp = _mlp(pooled_len, hidden, n_layers, dropout)
        self.backcast_head = nn.Linear(hidden, pooled_len)  # noi suy ve lookback
        self.forecast_head = nn.Linear(hidden, horizon * n_quantiles)

    def forward(self, x):  # x: (B, L)
        pooled = self.pool(x.unsqueeze(1)).squeeze(1)        # (B, Lp)
        h = self.mlp(pooled)
        bc_low = self.backcast_head(h).unsqueeze(1)          # (B, 1, Lp)
        backcast = F.interpolate(bc_low, size=self.lookback, mode="linear",
                                 align_corners=False).squeeze(1)
        forecast = self.forecast_head(h)                     # (B, H*Q)
        return backcast, forecast


class _NHITSCore(nn.Module):
    def __init__(self, lookback, horizon, n_quantiles, hidden, n_layers, dropout,
                 pool_kernels=(1, 2, 4)):
        super().__init__()
        self.horizon, self.n_quantiles = horizon, n_quantiles
        self.blocks = nn.ModuleList(
            _NHITSBlock(lookback, horizon, n_quantiles, hidden, n_layers, dropout, k)
            for k in pool_kernels
        )

    def forward(self, x):  # x: (B, L)
        residual = x
        forecast = torch.zeros(x.shape[0], self.horizon * self.n_quantiles,
                               device=x.device, dtype=x.dtype)
        for blk in self.blocks:
            backcast, block_forecast = blk(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return monotone_quantiles(forecast, self.horizon, self.n_quantiles)


# ---------------------------------------------------------------------------
# Bao boc RevIN chung cho cac core o tren
# ---------------------------------------------------------------------------
class RevinForecaster(nn.Module):
    """RevIN.normalize -> core -> RevIN.denormalize. Dau ra (B, horizon, Q)."""

    def __init__(self, core: nn.Module, revin_affine: bool = True):
        super().__init__()
        self.revin = RevIN(affine=revin_affine)
        self.core = core

    def forward(self, x):
        x = self.revin.normalize(x)
        return self.revin.denormalize(self.core(x))


# ---------------------------------------------------------------------------
# Factory: tao mo hinh theo ten tu mot config chung
# ---------------------------------------------------------------------------
def build_model(name: str, lookback: int, horizon: int, n_quantiles: int,
                hidden_size: int = 256, n_layers: int = 3, dropout: float = 0.1,
                revin_affine: bool = True, **kw) -> nn.Module:
    name = name.lower()
    if name == "nbeats":
        return RevinNBeats(
            lookback=lookback, horizon=horizon, n_quantiles=n_quantiles,
            n_stacks=kw.get("n_stacks", 2), n_blocks=kw.get("n_blocks", 2),
            hidden_size=hidden_size, n_layers=n_layers, dropout=dropout,
            revin_affine=revin_affine,
        )
    if name == "mlp":
        core = _MLPCore(lookback, horizon, n_quantiles, hidden_size, n_layers, dropout)
    elif name == "dlinear":
        core = _DLinearCore(lookback, horizon, n_quantiles,
                            kernel_size=kw.get("kernel_size", 13))
    elif name == "tsmixer":
        core = _TSMixerCore(lookback, horizon, n_quantiles, hidden_size, n_layers, dropout)
    elif name == "nhits":
        core = _NHITSCore(lookback, horizon, n_quantiles, hidden_size, n_layers, dropout,
                          pool_kernels=kw.get("pool_kernels", (1, 2, 4)))
    else:
        raise ValueError(f"Mo hinh khong ho tro: {name}")
    return RevinForecaster(core, revin_affine=revin_affine)


MODEL_NAMES = ["MLP", "DLinear", "TSMixer", "NHITS", "NBEATS"]
