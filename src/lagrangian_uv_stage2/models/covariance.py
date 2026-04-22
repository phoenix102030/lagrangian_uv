from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _raw_tril_to_spd(raw: torch.Tensor, min_diag: float) -> torch.Tensor:
    if raw.shape[-1] != 3:
        raise ValueError(f"Expected 3 lower-triangular parameters, got shape {raw.shape}.")

    l11 = F.softplus(raw[..., 0]) + min_diag
    l21 = raw[..., 1]
    l22 = F.softplus(raw[..., 2]) + min_diag

    zero = torch.zeros_like(l11)
    row1 = torch.stack([l11, zero], dim=-1)
    row2 = torch.stack([l21, l22], dim=-1)
    chol = torch.stack([row1, row2], dim=-2)
    return chol @ chol.transpose(-1, -2)


def _pairwise_squared_distance(coords: torch.Tensor) -> torch.Tensor:
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    return diff.pow(2).sum(dim=-1)


class SeparableCrossCovariance(nn.Module):
    def __init__(
        self,
        init_core_tril: list[float],
        init_log_spatial_scale: float,
        jitter: float,
        min_diag: float = 1.0e-4,
    ) -> None:
        super().__init__()
        self.core_tril_raw = nn.Parameter(torch.tensor(init_core_tril, dtype=torch.float32))
        self.log_spatial_scale = nn.Parameter(torch.tensor(init_log_spatial_scale, dtype=torch.float32))
        self.jitter = jitter
        self.min_diag = min_diag

    def forward(self, site_coords: torch.Tensor) -> torch.Tensor:
        core = _raw_tril_to_spd(self.core_tril_raw, self.min_diag)
        lengthscale = F.softplus(self.log_spatial_scale) + self.min_diag
        dist_sq = _pairwise_squared_distance(site_coords)
        spatial = torch.exp(-0.5 * dist_sq / (lengthscale**2))
        covariance = torch.kron(core, spatial)
        eye = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
        return covariance + self.jitter * eye


class PositiveDiagonal(nn.Module):
    def __init__(self, init_raw_diag: list[float], jitter: float) -> None:
        super().__init__()
        self.raw_diag = nn.Parameter(torch.tensor(init_raw_diag, dtype=torch.float32))
        self.jitter = jitter

    def forward(self) -> torch.Tensor:
        diag = F.softplus(self.raw_diag) + self.jitter
        return torch.diag(diag)
