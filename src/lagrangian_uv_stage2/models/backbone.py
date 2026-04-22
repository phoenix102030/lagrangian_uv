from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sanitize_tensor(tensor: torch.Tensor, finite_clip: float = 1.0e4) -> torch.Tensor:
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=finite_clip, neginf=-finite_clip)
    return torch.clamp(tensor, min=-finite_clip, max=finite_clip)


def _valid_group_count(channels: int, requested: int) -> int:
    for candidate in range(min(channels, requested), 0, -1):
        if channels % candidate == 0:
            return candidate
    return 1


def _raw_to_spd_2x2(raw: torch.Tensor, sigma_floor: float) -> torch.Tensor:
    if raw.shape[-1] != 3:
        raise ValueError(f"Expected the last dimension to have size 3, got shape {raw.shape}.")

    raw = _sanitize_tensor(raw)
    l11 = F.softplus(raw[..., 0]) + sigma_floor
    l21 = raw[..., 1]
    l22 = F.softplus(raw[..., 2]) + sigma_floor

    zero = torch.zeros_like(l11)
    row1 = torch.stack([l11, zero], dim=-1)
    row2 = torch.stack([l21, l22], dim=-1)
    chol = torch.stack([row1, row2], dim=-2)
    covariance = chol @ chol.transpose(-1, -2)
    covariance = _sanitize_tensor(covariance)
    eye = torch.eye(2, device=raw.device, dtype=raw.dtype)
    return covariance + sigma_floor * eye


class GridEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int, dropout: float, norm_groups: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.GroupNorm(_valid_group_count(32, norm_groups), 32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.GroupNorm(_valid_group_count(64, norm_groups), 64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.GroupNorm(_valid_group_count(128, norm_groups), 128),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _sanitize_tensor(x)
        features = self.features(x)
        features = _sanitize_tensor(features)
        projected = self.projection(features)
        return _sanitize_tensor(projected)


class ComponentAdvectionHead(nn.Module):
    def __init__(
        self,
        encoder: GridEncoder,
        hidden_dim: int,
        dropout: float,
        mean_scale: float,
        sigma_floor: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(hidden_dim, 2)
        self.cov_head = nn.Linear(hidden_dim, 3)
        self.mean_scale = mean_scale
        self.sigma_floor = sigma_floor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        hidden = _sanitize_tensor(self.hidden(encoded))
        mean = torch.tanh(_sanitize_tensor(self.mean_head(hidden))) * self.mean_scale
        covariance = _raw_to_spd_2x2(self.cov_head(hidden), self.sigma_floor)
        mean = _sanitize_tensor(mean)
        covariance = _sanitize_tensor(covariance)
        return mean, covariance, encoded


class DualBranchAdvectionNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        dropout: float,
        norm_groups: int,
        mean_scale: float,
        sigma_floor: float,
        share_encoder: bool,
        state_dim: int,
        forcing_scale: float,
    ) -> None:
        super().__init__()

        shared_encoder = GridEncoder(input_channels, hidden_dim, dropout, norm_groups)
        if share_encoder:
            u_encoder = shared_encoder
            v_encoder = shared_encoder
        else:
            u_encoder = shared_encoder
            v_encoder = GridEncoder(input_channels, hidden_dim, dropout, norm_groups)

        self.u_head = ComponentAdvectionHead(
            encoder=u_encoder,
            hidden_dim=hidden_dim,
            dropout=dropout,
            mean_scale=mean_scale,
            sigma_floor=sigma_floor,
        )
        self.v_head = ComponentAdvectionHead(
            encoder=v_encoder,
            hidden_dim=hidden_dim,
            dropout=dropout,
            mean_scale=mean_scale,
            sigma_floor=sigma_floor,
        )
        self.forcing_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )
        self.forcing_scale = forcing_scale

    def forward(self, nwp_u: torch.Tensor, nwp_v: torch.Tensor) -> dict[str, torch.Tensor]:
        mean_u, cov_u, feat_u = self.u_head(nwp_u)
        mean_v, cov_v, feat_v = self.v_head(nwp_v)
        means = torch.stack([mean_u, mean_v], dim=1)
        covariances = torch.stack([cov_u, cov_v], dim=1)
        features = torch.stack([feat_u, feat_v], dim=1)
        forcing_features = torch.cat([feat_u, feat_v], dim=-1)
        forcing = torch.tanh(_sanitize_tensor(self.forcing_head(forcing_features))) * self.forcing_scale
        forcing = _sanitize_tensor(forcing)
        return {"means": means, "covariances": covariances, "features": features, "forcing": forcing}
