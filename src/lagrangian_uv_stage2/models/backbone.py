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

    raw = _sanitize_tensor(raw)
    chol = raw.new_zeros(*raw.shape[:-1], matrix_dim, matrix_dim)
    tril_row, tril_col = torch.tril_indices(matrix_dim, matrix_dim, device=raw.device)
    chol[..., tril_row, tril_col] = raw

    diag_idx = torch.arange(matrix_dim, device=raw.device)
    diag_param_idx = torch.cumsum(torch.arange(1, matrix_dim + 1, device=raw.device), dim=0) - 1
    chol[..., diag_idx, diag_idx] = F.softplus(raw[..., diag_param_idx]) + sigma_floor
    covariance = chol @ chol.transpose(-1, -2)
    covariance = _sanitize_tensor(covariance)
    eye = torch.eye(matrix_dim, device=raw.device, dtype=raw.dtype)
    return covariance + sigma_floor * eye


def _raw_to_spd_2x2(raw: torch.Tensor, sigma_floor: float) -> torch.Tensor:
    raw = _sanitize_tensor(raw)
    return _raw_to_spd(raw, matrix_dim=2, sigma_floor=sigma_floor)


def _joint_covariance_blocks(joint_covariance: torch.Tensor, num_components: int, spatial_dim: int) -> torch.Tensor:
    blocks = []
    for component_idx in range(num_components):
        start = component_idx * spatial_dim
        end = start + spatial_dim
        blocks.append(joint_covariance[..., start:end, start:end])
    return torch.stack(blocks, dim=-3)


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
        temporal_model: str = "gru",
    ) -> None:
        super().__init__()
        self.encoder = encoder
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
        self.mean_head = nn.Linear(hidden_dim, 2)
        self.cov_head = nn.Linear(hidden_dim, 3)
        self.mean_scale = mean_scale
        self.sigma_floor = sigma_floor

    def _encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        if self.temporal is None:
            return encoded
        if encoded.ndim != 2:
            raise ValueError(f"Temporal advection head expects encoded shape [time, hidden], got {encoded.shape}.")
        smoothed, _ = self.temporal(encoded.unsqueeze(0))
        return _sanitize_tensor(smoothed.squeeze(0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self._encode_sequence(x)
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
        temporal_model: str = "gru",
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
            temporal_model=temporal_model,
        )
        self.v_head = ComponentAdvectionHead(
            encoder=v_encoder,
            hidden_dim=hidden_dim,
            dropout=dropout,
            mean_scale=mean_scale,
            sigma_floor=sigma_floor,
            temporal_model=temporal_model,
        )
        self.forcing_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )
        self.forcing_scale = float(forcing_scale)
        self.state_dim = int(state_dim)

    def forward(self, nwp_u: torch.Tensor, nwp_v: torch.Tensor) -> dict[str, torch.Tensor]:
        mean_u, cov_u, feat_u = self.u_head(nwp_u)
        mean_v, cov_v, feat_v = self.v_head(nwp_v)
        means = torch.stack([mean_u, mean_v], dim=1)
        covariances = torch.stack([cov_u, cov_v], dim=1)
        features = torch.stack([feat_u, feat_v], dim=1)

        if self.forcing_scale <= 0.0:
            forcing = torch.zeros(
                (nwp_u.shape[0], self.state_dim),
                device=nwp_u.device,
                dtype=feat_u.dtype,
            )
        else:
            forcing_features = torch.cat([feat_u, feat_v], dim=-1)
            forcing = torch.tanh(_sanitize_tensor(self.forcing_head(forcing_features))) * self.forcing_scale
            forcing = _sanitize_tensor(forcing)

        return {"means": means, "covariances": covariances, "features": features, "forcing": forcing}


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
    ) -> None:
        super().__init__()
        if num_components != 2 or spatial_dim != 2:
            raise ValueError("JointAdvectionNet currently expects two 2D components, e.g. u/v.")

        self.num_components = int(num_components)
        self.spatial_dim = int(spatial_dim)
        self.joint_dim = self.num_components * self.spatial_dim
        self.encoder = GridEncoder(2 * input_channels, hidden_dim, dropout, norm_groups)
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
        self.forcing_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )
        self.mean_scale = mean_scale
        self.sigma_floor = sigma_floor
        self.forcing_scale = float(forcing_scale)
        self.state_dim = int(state_dim)

    def _encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        if self.temporal is None:
            return encoded
        if encoded.ndim != 2:
            raise ValueError(f"Temporal advection head expects encoded shape [time, hidden], got {encoded.shape}.")
        smoothed, _ = self.temporal(encoded.unsqueeze(0))
        return _sanitize_tensor(smoothed.squeeze(0))

    def forward(self, nwp_u: torch.Tensor, nwp_v: torch.Tensor) -> dict[str, torch.Tensor]:
        joint_input = torch.cat([nwp_u, nwp_v], dim=1)
        encoded = self._encode_sequence(joint_input)
        hidden = _sanitize_tensor(self.hidden(encoded))

        joint_mean = torch.tanh(_sanitize_tensor(self.mean_head(hidden))) * self.mean_scale
        means = joint_mean.view(-1, self.num_components, self.spatial_dim)
        joint_covariance = _raw_to_spd(self.cov_head(hidden), self.joint_dim, self.sigma_floor)
        covariances = _joint_covariance_blocks(
            joint_covariance,
            num_components=self.num_components,
            spatial_dim=self.spatial_dim,
        )
        features = encoded.unsqueeze(1).expand(-1, self.num_components, -1)

        if self.forcing_scale <= 0.0:
            forcing = torch.zeros(
                (nwp_u.shape[0], self.state_dim),
                device=nwp_u.device,
                dtype=encoded.dtype,
            )
        else:
            forcing = torch.tanh(_sanitize_tensor(self.forcing_head(hidden))) * self.forcing_scale
            forcing = _sanitize_tensor(forcing)

        return {
            "means": _sanitize_tensor(means),
            "covariances": _sanitize_tensor(covariances),
            "joint_covariance": _sanitize_tensor(joint_covariance),
            "features": features,
            "forcing": forcing,
        }
