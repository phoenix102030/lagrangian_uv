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


def _raw_to_spd(raw: torch.Tensor, matrix_dim: int, sigma_floor: float) -> torch.Tensor:
    expected_params = matrix_dim * (matrix_dim + 1) // 2
    if raw.shape[-1] != expected_params:
        raise ValueError(f"Expected the last dimension to have size {expected_params}, got shape {raw.shape}.")

    raw = _sanitize_tensor(raw).to(dtype=torch.float32)
    chol = raw.new_zeros(*raw.shape[:-1], matrix_dim, matrix_dim)
    tril_row, tril_col = torch.tril_indices(matrix_dim, matrix_dim, device=raw.device)
    chol[..., tril_row, tril_col] = raw

    diag_idx = torch.arange(matrix_dim, device=raw.device)
    diag_param_idx = torch.cumsum(torch.arange(1, matrix_dim + 1, device=raw.device), dim=0) - 1
    sigma_floor_tensor = raw.new_tensor(sigma_floor)
    chol[..., diag_idx, diag_idx] = F.softplus(raw[..., diag_param_idx]) + sigma_floor_tensor
    covariance = chol @ chol.transpose(-1, -2)
    covariance = _sanitize_tensor(covariance)
    eye = torch.eye(matrix_dim, device=raw.device, dtype=raw.dtype)
    return covariance + sigma_floor_tensor * eye


def _joint_covariance_blocks(
    joint_covariance: torch.Tensor,
    num_components: int,
    spatial_dim: int,
) -> torch.Tensor:
    blocks = []
    for component_idx in range(num_components):
        start = component_idx * spatial_dim
        end = start + spatial_dim
        blocks.append(joint_covariance[..., start:end, start:end])
    return torch.stack(blocks, dim=-3)


class TransformerSpatialExtractor(nn.Module):
    """CNN background encoder + station grid sampling + station transformer."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_sites: int,
        dropout: float = 0.1,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.num_sites = int(num_sites)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.GroupNorm(_valid_group_count(32, norm_groups), 32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.GroupNorm(_valid_group_count(64, norm_groups), 64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.GroupNorm(_valid_group_count(hidden_dim, norm_groups), hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.site_pos_embedding = nn.Parameter(torch.zeros(1, self.num_sites, hidden_dim))
        nn.init.normal_(self.site_pos_embedding, mean=0.0, std=0.02)

    @staticmethod
    def _normalize_site_coords(site_coords: torch.Tensor) -> torch.Tensor:
        coords_min = site_coords.min(dim=0, keepdim=True)[0]
        coords_max = site_coords.max(dim=0, keepdim=True)[0]
        coord_range = (coords_max - coords_min).clamp_min(1.0e-5)
        return 2.0 * (site_coords - coords_min) / coord_range - 1.0

    def forward(self, x: torch.Tensor, site_coords: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected NWP tensor with shape [time, channels, height, width], got {x.shape}.")

        x = _sanitize_tensor(x)
        num_steps = x.shape[0]
        feature_map = _sanitize_tensor(self.cnn(x))

        site_coords = site_coords.to(device=x.device, dtype=feature_map.dtype)
        site_coords_norm = self._normalize_site_coords(site_coords)
        grid = site_coords_norm.unsqueeze(0).unsqueeze(2).expand(num_steps, -1, -1, -1)
        point_features = F.grid_sample(feature_map, grid, align_corners=True)
        point_features = point_features.squeeze(-1).permute(0, 2, 1)

        tokens = point_features + self.site_pos_embedding.to(device=x.device, dtype=point_features.dtype)
        return _sanitize_tensor(self.transformer(tokens))


class JointAdvectionNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        dropout: float,
        norm_groups: int,
        mean_scale: float,
        sigma_floor: float,
        state_dim: int,
        forcing_scale: float,
        temporal_model: str = "gru",
        num_components: int = 2,
        spatial_dim: int = 2,
        num_sites: int = 3,
    ) -> None:
        super().__init__()
        self.num_components = int(num_components)
        self.spatial_dim = int(spatial_dim)
        self.num_sites = int(num_sites)
        self.joint_dim = self.num_components * self.spatial_dim
        self.state_dim = int(state_dim)
        if self.state_dim != self.num_components * self.num_sites:
            raise ValueError(
                "state_dim must equal num_components * num_sites for component-first state ordering."
            )

        self.spatial_extractor = TransformerSpatialExtractor(
            in_channels=2 * input_channels,
            hidden_dim=hidden_dim,
            num_sites=self.num_sites,
            dropout=dropout,
            norm_groups=norm_groups,
        )

        self.temporal_model = temporal_model.lower()
        if self.temporal_model == "gru":
            self.temporal = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        elif self.temporal_model in {"none", "identity"}:
            self.temporal = None
        else:
            raise ValueError("model.encoder.temporal_model must be 'gru' or 'none'.")

        self.hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(hidden_dim, self.joint_dim)
        self.cov_head = nn.Linear(hidden_dim, self.joint_dim * (self.joint_dim + 1) // 2)
        self.forcing_scale = float(forcing_scale)
        self.forcing_head = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_components),
            )
            if self.forcing_scale > 0.0
            else None
        )
        self.mean_scale = float(mean_scale)
        self.sigma_floor = float(sigma_floor)

    def _apply_temporal_model(self, encoded: torch.Tensor) -> torch.Tensor:
        if self.temporal is None:
            return encoded
        # Treat each site as an independent sequence while sharing the GRU.
        by_site = encoded.permute(1, 0, 2)
        smoothed, _ = self.temporal(by_site)
        return _sanitize_tensor(smoothed.permute(1, 0, 2))

    def forward(self, nwp_u: torch.Tensor, nwp_v: torch.Tensor, site_coords: torch.Tensor) -> dict[str, torch.Tensor]:
        if nwp_u.shape != nwp_v.shape:
            raise ValueError(f"nwp_u and nwp_v must have the same shape, got {nwp_u.shape} and {nwp_v.shape}.")
        if nwp_u.ndim != 4:
            raise ValueError(f"Expected NWP tensors with shape [time, channels, height, width], got {nwp_u.shape}.")

        num_steps = nwp_u.shape[0]
        joint_input = torch.cat([nwp_u, nwp_v], dim=1)
        encoded = self.spatial_extractor(joint_input, site_coords)
        encoded = self._apply_temporal_model(encoded)
        hidden = _sanitize_tensor(self.hidden(encoded))

        joint_mean = torch.tanh(_sanitize_tensor(self.mean_head(hidden))) * self.mean_scale
        means = joint_mean.view(num_steps, self.num_sites, self.num_components, self.spatial_dim)

        joint_covariance = _raw_to_spd(self.cov_head(hidden), self.joint_dim, self.sigma_floor)
        covariances = _joint_covariance_blocks(
            joint_covariance,
            num_components=self.num_components,
            spatial_dim=self.spatial_dim,
        )

        if self.forcing_head is None:
            forcing = torch.zeros((num_steps, self.state_dim), device=nwp_u.device, dtype=hidden.dtype)
        else:
            forcing_raw = torch.tanh(_sanitize_tensor(self.forcing_head(hidden))) * self.forcing_scale
            forcing = forcing_raw.permute(0, 2, 1).reshape(num_steps, self.state_dim)
            forcing = _sanitize_tensor(forcing)

        return {
            "means": _sanitize_tensor(means),
            "covariances": _sanitize_tensor(covariances),
            "joint_covariance": _sanitize_tensor(joint_covariance),
            "features": _sanitize_tensor(encoded),
            "forcing": forcing,
        }


class DualBranchAdvectionNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "DualBranchAdvectionNet is not part of the pure site-level IDE kernel. "
            "Set model.joint_component_advection=true."
        )
