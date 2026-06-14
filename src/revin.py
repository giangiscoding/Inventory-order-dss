"""RevIN — Reversible Instance Normalization (tu xay dung).

Tham khao: Kim et al., "Reversible Instance Normalization for Accurate
Time-Series Forecasting against Distribution Shift", ICLR 2022.
"""
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Chuan hoa tung cua so input bang mean/std cua chinh no, co the kem
    bien doi affine hoc duoc; sau khi du bao thi dao nguoc ve thang do goc.

    Lam viec voi tensor (batch, length) — chuoi don bien.
    """

    def __init__(self, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.affine = affine
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.zeros(1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self._mean = x.mean(dim=-1, keepdim=True).detach()
        self._std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps).detach()
        out = (x - self._mean) / self._std
        if self.affine:
            out = out * self.gamma + self.beta
        return out

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        # x co the la (batch, length) hoac (batch, horizon, n_quantiles).
        # mean/std luu o dang (batch, 1) -> reshape de broadcast voi so chieu cua x.
        if self.affine:
            x = (x - self.beta) / (self.gamma + self.eps)
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        return x * self._std.view(shape) + self._mean.view(shape)
