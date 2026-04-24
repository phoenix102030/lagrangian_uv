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


def _stable_inverse_logdet(
    matrix: torch.Tensor,
    base_jitter: float,
    max_tries: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    candidate = _sanitize_matrix(matrix)
    diag_scale = candidate.diagonal().abs().mean().clamp_min(1.0)
    jitter = max(base_jitter, 1.0e-6)

    for _ in range(max_tries):
        repaired = candidate + (jitter * diag_scale) * identity
        repaired = _sanitize_matrix(repaired)
        chol, info = torch.linalg.cholesky_ex(repaired)
        if int(info.item()) == 0:
            inverse = torch.cholesky_inverse(chol)
            log_det = 2.0 * torch.log(torch.diagonal(chol)).sum()
            inverse = _sanitize_matrix(inverse)
            log_det = torch.nan_to_num(log_det, nan=0.0, posinf=20.0, neginf=-20.0)
            return repaired, inverse, log_det
        jitter *= 10.0

    safe_diag = torch.nan_to_num(candidate.diagonal(), nan=1.0, posinf=1.0e4, neginf=1.0)
    safe_diag = safe_diag.abs().clamp_min(jitter * diag_scale + 1.0e-4)
    repaired = torch.diag(safe_diag)
    repaired = repaired + (jitter * diag_scale) * identity
    repaired = _sanitize_matrix(repaired)
    inverse = torch.diag(1.0 / repaired.diagonal().clamp_min(1.0e-6))
    log_det = torch.log(repaired.diagonal().clamp_min(1.0e-6)).sum()
    inverse = _sanitize_matrix(inverse)
    log_det = torch.nan_to_num(log_det, nan=0.0, posinf=20.0, neginf=-20.0)
    return repaired, inverse, log_det


class StochasticAdvectionKernel(nn.Module):
    """Build the theorem-inspired redistribution matrix.

    The same-component block must not use ``mu_i - mu_i``. For a component
    propagated to itself, Theorem 2' has a shifted kernel with drift
    ``dt * mu_i`` and dispersion ``I + 2 * dt**2 * Sigma_i``. The older
    implementation made the dominant u->u and v->v paths zero-drift, so the
    advection net had almost no identifiable transport signal to learn.
    """

    def __init__(
        self,
        num_sites: int,
        num_components: int,
        delta_t: float,
        kernel_jitter: float,
        identity_mix: float,
        min_block_scale: float,
        allow_cross_component: bool = False,
        diagonal_block_scale_init: float = 1.0,
        cross_component_scale_init: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_sites = int(num_sites)
        self.num_components = int(num_components)
        self.delta_t = float(delta_t)
        self.kernel_jitter = float(kernel_jitter)
        self.min_block_scale = float(min_block_scale)
        self.allow_cross_component = bool(allow_cross_component)

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

        clipped_mix = min(max(identity_mix, 1.0e-6), 1.0 - 1.0e-6)
        self.identity_mix_logit = nn.Parameter(
            torch.tensor(math.log(clipped_mix / (1.0 - clipped_mix)), dtype=torch.float32)
        )

    def _block_scales(self) -> torch.Tensor:
        return F.softplus(self.block_scale_raw) + self.min_block_scale

    def _identity_mix(self) -> torch.Tensor:
        return torch.sigmoid(self.identity_mix_logit)

    def _block_drift_dispersion(
        self,
        mean_i: torch.Tensor,
        mean_j: torch.Tensor,
        cov_i: torch.Tensor,
        cov_j: torch.Tensor,
        same_component: bool,
        eye2: torch.Tensor,
        cross_cov_ij: torch.Tensor | None = None,
        cross_cov_ji: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if same_component:
            drift = self.delta_t * mean_i
            dispersion = eye2 + 2.0 * (self.delta_t**2) * cov_i
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
        # AMP autocast can make means/covariances float16. Keep the small
        # kernel linear algebra in float32: CUDA cholesky_ex is not implemented
        # for float16 on many PyTorch/CUDA builds, and these 2x2 inverses are
        # numerically sensitive anyway. The CNN/GRU can still run with AMP.
        num_steps = means.shape[0]
        kernel_dtype = torch.float32
        means = _sanitize_vector(means.to(dtype=kernel_dtype))
        covariances = _sanitize_matrix(covariances.to(dtype=kernel_dtype))
        if joint_covariance is not None:
            joint_covariance = _sanitize_matrix(joint_covariance.to(dtype=kernel_dtype))
        site_coords = site_coords.to(device=means.device, dtype=kernel_dtype)
        spatial_lags = _sanitize_vector(site_coords.unsqueeze(1) - site_coords.unsqueeze(0))
        block_scales = self._block_scales().to(means.device, kernel_dtype)
        component_mask = self.component_mask.to(means.device, kernel_dtype)
        identity_mix = self._identity_mix().to(means.device, kernel_dtype)
        eye2 = torch.eye(2, device=means.device, dtype=kernel_dtype)

        transitions = []
        drift_terms = []
        dispersion_terms = []
        state_dim = self.num_sites * self.num_components

        for t in range(num_steps):
            blocks = []
            drift_block_t = []
            dispersion_block_t = []

            for i in range(self.num_components):
                row_blocks = []
                row_drifts = []
                row_dispersion = []

                for j in range(self.num_components):
                    cross_cov_ij = None
                    cross_cov_ji = None
                    if joint_covariance is not None:
                        i_start = i * 2
                        j_start = j * 2
                        cross_cov_ij = joint_covariance[t, i_start : i_start + 2, j_start : j_start + 2]
                        cross_cov_ji = joint_covariance[t, j_start : j_start + 2, i_start : i_start + 2]
                    drift, dispersion = self._block_drift_dispersion(
                        mean_i=means[t, i],
                        mean_j=means[t, j],
                        cov_i=covariances[t, i],
                        cov_j=covariances[t, j],
                        same_component=(i == j),
                        eye2=eye2,
                        cross_cov_ij=cross_cov_ij,
                        cross_cov_ji=cross_cov_ji,
                    )
                    dispersion, inv_dispersion, log_det = _stable_inverse_logdet(
                        dispersion,
                        base_jitter=self.kernel_jitter,
                    )

                    centered = _sanitize_vector(spatial_lags - drift.view(1, 1, 2))
                    quad_form = torch.einsum("abk,kl,abl->ab", centered, inv_dispersion, centered)
                    quad_form = torch.nan_to_num(quad_form, nan=0.0, posinf=60.0, neginf=0.0)
                    quad_form = torch.clamp(quad_form, min=0.0, max=60.0)
                    exponent = torch.clamp(-0.5 * log_det - quad_form, min=-60.0, max=20.0)
                    block = component_mask[i, j] * block_scales[i, j] * torch.exp(exponent)
                    block = torch.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)

                    row_blocks.append(block)
                    row_drifts.append(drift)
                    row_dispersion.append(dispersion)

                blocks.append(torch.cat(row_blocks, dim=1))
                drift_block_t.append(torch.stack(row_drifts, dim=0))
                dispersion_block_t.append(torch.stack(row_dispersion, dim=0))

            kernel_matrix = torch.cat(blocks, dim=0)
            row_sum = kernel_matrix.sum(dim=1, keepdim=True).clamp_min(self.kernel_jitter)
            transition = kernel_matrix / row_sum
            transition = torch.nan_to_num(transition, nan=0.0, posinf=0.0, neginf=0.0)
            transition = transition / transition.sum(dim=1, keepdim=True).clamp_min(self.kernel_jitter)
            identity = torch.eye(state_dim, device=transition.device, dtype=transition.dtype)
            transition = (1.0 - identity_mix) * transition + identity_mix * identity
            transition = torch.nan_to_num(transition, nan=0.0, posinf=0.0, neginf=0.0)
            transition = transition / transition.sum(dim=1, keepdim=True).clamp_min(self.kernel_jitter)

            transitions.append(transition)
            drift_terms.append(torch.stack(drift_block_t, dim=0))
            dispersion_terms.append(torch.stack(dispersion_block_t, dim=0))

        return (
            torch.stack(transitions, dim=0),
            {
                "drift_terms": torch.stack(drift_terms, dim=0),
                "dispersion_terms": torch.stack(dispersion_terms, dim=0),
                "block_scales": block_scales,
                "component_mask": component_mask,
                "identity_mix": identity_mix,
            },
        )
