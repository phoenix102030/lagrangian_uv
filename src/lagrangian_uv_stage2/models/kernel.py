from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _sanitize_vector(vector: torch.Tensor, finite_clip: float = 1.0e4) -> torch.Tensor:
    vector = torch.nan_to_num(vector, nan=0.0, posinf=finite_clip, neginf=-finite_clip)
    return torch.clamp(vector, min=-finite_clip, max=finite_clip)


def _sanitize_matrix(matrix: torch.Tensor, finite_clip: float = 1.0e4) -> torch.Tensor:
    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=finite_clip, neginf=-finite_clip)
    matrix = torch.clamp(matrix, min=-finite_clip, max=finite_clip)
    return _symmetrize(matrix)


def _softplus_inverse(value: float) -> float:
    value = max(float(value), 1.0e-8)
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


def _analytical_2x2_inverse(matrix: torch.Tensor, jitter: float) -> torch.Tensor:
    a = matrix[..., 0, 0]
    b = matrix[..., 0, 1]
    c = matrix[..., 1, 0]
    d = matrix[..., 1, 1]

    det = (a * d - b * c).clamp_min(jitter)
    row1 = torch.stack([d / det, -b / det], dim=-1)
    row2 = torch.stack([-c / det, a / det], dim=-1)
    return _sanitize_matrix(torch.stack([row1, row2], dim=-2))


class StochasticAdvectionKernel(nn.Module):
    """Build a full 6x6 open-system IDE transition matrix.

    Rows are target states and columns are source states, ordered as
    [u(s1), u(s2), u(s3), v(s1), v(s2), v(s3)].
    """

    def __init__(
        self,
        num_sites: int,
        num_components: int,
        delta_t: float,
        kernel_jitter: float,
        identity_mix: float,
        kernel_decay: float,
        min_block_scale: float,
        allow_cross_component: bool = True,
        diagonal_block_scale_init: float = 1.0,
        cross_component_scale_init: float = 0.02,
        max_transition_value: float = 10.0,
    ) -> None:
        super().__init__()
        self.num_sites = int(num_sites)
        self.num_components = int(num_components)
        self.delta_t = float(delta_t)
        self.kernel_jitter = float(kernel_jitter)
        self.kernel_decay = float(kernel_decay)
        self.min_block_scale = float(min_block_scale)
        self.allow_cross_component = bool(allow_cross_component)
        self.max_transition_value = float(max_transition_value)

        scale_init = torch.full(
            (self.num_components, self.num_components),
            _softplus_inverse(cross_component_scale_init),
            dtype=torch.float32,
        )
        scale_init.fill_diagonal_(_softplus_inverse(diagonal_block_scale_init))
        self.block_scale_raw = nn.Parameter(scale_init)

        component_mask = torch.eye(self.num_components, dtype=torch.float32)
        if self.allow_cross_component:
            component_mask = torch.ones((self.num_components, self.num_components), dtype=torch.float32)
        self.register_buffer("component_mask", component_mask)
        self.register_buffer("identity_mix", torch.tensor(float(identity_mix), dtype=torch.float32))

    def _block_scales(self) -> torch.Tensor:
        return F.softplus(self.block_scale_raw) + self.min_block_scale

    def _block_drift_dispersion(
        self,
        mean_i: torch.Tensor,
        mean_j: torch.Tensor,
        cov_i: torch.Tensor,
        cov_j: torch.Tensor,
        same_component: bool,
        eye2: torch.Tensor,
        cross_cov_ij: torch.Tensor | None,
        cross_cov_ji: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if same_component:
            drift = self.delta_t * mean_j
            dispersion = eye2 + 2.0 * (self.delta_t**2) * cov_j
        else:
            drift = self.delta_t * (mean_i - mean_j)
            relative_covariance = cov_i + cov_j
            if cross_cov_ij is not None and cross_cov_ji is not None:
                relative_covariance = relative_covariance - cross_cov_ij - cross_cov_ji
            dispersion = eye2 + 2.0 * (self.delta_t**2) * relative_covariance

        drift = _sanitize_vector(drift)
        dispersion = _sanitize_matrix(dispersion + self.kernel_jitter * eye2)
        return drift, dispersion

    def forward(
        self,
        means: torch.Tensor,
        covariances: torch.Tensor,
        site_coords: torch.Tensor,
        joint_covariance: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if means.ndim != 4:
            raise ValueError(
                "means must have shape [time, num_sites, num_components, spatial_dim], "
                f"got {means.shape}."
            )
        if covariances.ndim != 5:
            raise ValueError(
                "covariances must have shape [time, num_sites, num_components, spatial_dim, spatial_dim], "
                f"got {covariances.shape}."
            )

        num_steps = means.shape[0]
        kernel_dtype = torch.float32
        means = _sanitize_vector(means.to(dtype=kernel_dtype))
        covariances = _sanitize_matrix(covariances.to(dtype=kernel_dtype))
        if joint_covariance is not None:
            joint_covariance = _sanitize_matrix(joint_covariance.to(dtype=kernel_dtype))
        site_coords = site_coords.to(device=means.device, dtype=kernel_dtype)

        block_scales = self._block_scales().to(means.device, kernel_dtype)
        component_mask = self.component_mask.to(means.device, kernel_dtype)
        identity_mix = torch.clamp(self.identity_mix.to(means.device, kernel_dtype), min=0.0, max=1.0)
        eye2 = torch.eye(2, device=means.device, dtype=kernel_dtype)
        state_dim = self.num_sites * self.num_components

        transitions = []
        drift_terms = []
        dispersion_terms = []

        target_coords = site_coords.unsqueeze(1)
        source_coords = site_coords.unsqueeze(0)

        for t in range(num_steps):
            component_rows = []
            drift_blocks = []
            dispersion_blocks = []

            for target_component in range(self.num_components):
                source_blocks = []
                target_drift_blocks = []
                target_dispersion_blocks = []

                for source_component in range(self.num_components):
                    mean_i = means[t, :, target_component, :]
                    mean_j = means[t, :, source_component, :]
                    cov_i = covariances[t, :, target_component, :, :]
                    cov_j = covariances[t, :, source_component, :, :]

                    cross_cov_ij = None
                    cross_cov_ji = None
                    if joint_covariance is not None and target_component != source_component:
                        i_start = target_component * 2
                        j_start = source_component * 2
                        cross_cov_ij = joint_covariance[t, :, i_start : i_start + 2, j_start : j_start + 2]
                        cross_cov_ji = joint_covariance[t, :, j_start : j_start + 2, i_start : i_start + 2]

                    drift, dispersion = self._block_drift_dispersion(
                        mean_i=mean_i,
                        mean_j=mean_j,
                        cov_i=cov_i,
                        cov_j=cov_j,
                        same_component=(target_component == source_component),
                        eye2=eye2,
                        cross_cov_ij=cross_cov_ij,
                        cross_cov_ji=cross_cov_ji,
                    )

                    centered = target_coords - source_coords - drift.unsqueeze(0)
                    dispersion_by_target = dispersion.unsqueeze(0).expand(self.num_sites, -1, -1, -1)
                    inv_dispersion = _analytical_2x2_inverse(dispersion_by_target, self.kernel_jitter)
                    quad_form = torch.matmul(
                        centered.unsqueeze(-2),
                        torch.matmul(inv_dispersion, centered.unsqueeze(-1)),
                    ).squeeze(-1).squeeze(-1)
                    quad_form = torch.nan_to_num(quad_form, nan=0.0, posinf=60.0, neginf=0.0)
                    quad_form = torch.clamp(quad_form, min=0.0, max=60.0)

                    block = torch.exp(torch.clamp(-0.5 * quad_form, min=-60.0, max=20.0))
                    block = component_mask[target_component, source_component] * block_scales[
                        target_component, source_component
                    ] * block
                    block = self.kernel_decay * torch.clamp(block, min=0.0, max=self.max_transition_value)
                    block = torch.nan_to_num(block, nan=0.0, posinf=self.max_transition_value, neginf=0.0)

                    source_blocks.append(block)
                    target_drift_blocks.append(drift)
                    target_dispersion_blocks.append(dispersion)

                component_rows.append(torch.cat(source_blocks, dim=1))
                drift_blocks.append(torch.stack(target_drift_blocks, dim=0))
                dispersion_blocks.append(torch.stack(target_dispersion_blocks, dim=0))

            transition = torch.cat(component_rows, dim=0)
            identity = torch.eye(state_dim, device=transition.device, dtype=transition.dtype)
            transition = (1.0 - identity_mix) * transition + identity_mix * identity
            transition = _sanitize_matrix(transition) if transition.shape[-1] == 2 else _sanitize_vector(transition)
            transition = transition.view(state_dim, state_dim)
            transitions.append(transition)
            drift_terms.append(torch.stack(drift_blocks, dim=0))
            dispersion_terms.append(torch.stack(dispersion_blocks, dim=0))

        transition_stack = torch.stack(transitions, dim=0)
        return (
            transition_stack,
            {
                "drift_terms": torch.stack(drift_terms, dim=0),
                "dispersion_terms": torch.stack(dispersion_terms, dim=0),
                "block_scales": block_scales,
                "component_mask": component_mask,
                "identity_mix": identity_mix,
                "kernel_decay": torch.as_tensor(
                    self.kernel_decay,
                    device=transition_stack.device,
                    dtype=transition_stack.dtype,
                ),
            },
        )
