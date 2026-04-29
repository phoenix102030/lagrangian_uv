from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import JointAdvectionNet
from .covariance import PositiveDiagonal, SeparableCrossCovariance
from .kernel import StochasticAdvectionKernel


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported linear algebra dtype {name!r}. Choose from {sorted(mapping)}.")
    return mapping[name]


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _sanitize_symmetric_matrix(matrix: torch.Tensor, finite_clip: float = 1.0e6) -> torch.Tensor:
    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=finite_clip, neginf=-finite_clip)
    matrix = torch.clamp(matrix, min=-finite_clip, max=finite_clip)
    return _symmetrize(matrix)


def _sanitize_operator_matrix(matrix: torch.Tensor, finite_clip: float = 1.0e6) -> torch.Tensor:
    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=finite_clip, neginf=-finite_clip)
    return torch.clamp(matrix, min=-finite_clip, max=finite_clip)


def _sanitize_vector(vector: torch.Tensor, finite_clip: float = 1.0e6) -> torch.Tensor:
    vector = torch.nan_to_num(vector, nan=0.0, posinf=finite_clip, neginf=-finite_clip)
    return torch.clamp(vector, min=-finite_clip, max=finite_clip)


class Stage2LagrangianStateSpaceModel(nn.Module):
    """Kalman state-space model whose transition is the learned IDE kernel.

    The state is ordered component-first:
    [u(s1), u(s2), u(s3), v(s1), v(s2), v(s3)].
    """

    def __init__(self, config: dict[str, Any], site_coords: np.ndarray) -> None:
        super().__init__()

        model_cfg = config["model"]
        cov_cfg = config["covariance"]
        encoder_cfg = model_cfg["encoder"]

        self.num_sites = int(model_cfg["num_sites"])
        self.num_components = int(model_cfg["num_components"])
        self.spatial_dim = int(model_cfg.get("spatial_dim", 2))
        self.state_dim = self.num_sites * self.num_components
        self.transition_jitter = float(model_cfg["transition_jitter"])
        self.cholesky_jitter = float(model_cfg["cholesky_jitter"])
        self.linalg_dtype = _resolve_dtype(model_cfg.get("linear_algebra_dtype", "float32"))
        self.max_nll_per_timestep = float(config["training"].get("max_nll_per_timestep", 1.0e4))

        if not bool(model_cfg.get("joint_component_advection", True)):
            raise ValueError(
                "The pure IDE kernel model requires model.joint_component_advection=true "
                "so that the NWP encoder emits site-specific u/v advection parameters."
            )

        self.nll_weight = float(config["training"].get("nll_weight", 1.0))
        self.one_step_forecast_weight = float(config["training"].get("one_step_forecast_weight", 0.0))
        self.rollout_forecast_weight = float(config["training"].get("rollout_forecast_weight", 0.0))
        self.kernel_one_step_weight = float(config["training"].get("kernel_one_step_weight", 0.0))
        self.transition_row_sum_weight = float(config["training"].get("transition_row_sum_weight", 0.0))
        self.transition_row_sum_floor = float(config["training"].get("transition_row_sum_floor", 0.0))
        self.forecast_loss_horizon = int(config["training"].get("forecast_loss_horizon", 0))
        self.forecast_loss_min_context = int(config["training"].get("forecast_loss_min_context", 1))
        self.scheduled_sampling_ratio = float(config["training"].get("scheduled_sampling_start", 0.0))
        self.normalize_nll_by_state_dim = bool(config["training"].get("normalize_nll_by_state_dim", True))

        self.register_buffer("site_coords", torch.as_tensor(site_coords, dtype=torch.float32))

        input_channels = len(config["data"]["nwp"]["u_channel_indices"])
        self.advection_net = JointAdvectionNet(
            input_channels=input_channels,
            hidden_dim=int(encoder_cfg["hidden_dim"]),
            dropout=float(encoder_cfg["dropout"]),
            norm_groups=int(encoder_cfg["norm_groups"]),
            mean_scale=float(model_cfg["mean_scale"]),
            sigma_floor=float(model_cfg["sigma_floor"]),
            state_dim=self.state_dim,
            forcing_scale=float(model_cfg.get("forcing_scale", 0.0)),
            temporal_model=str(encoder_cfg.get("temporal_model", "gru")),
            num_components=self.num_components,
            spatial_dim=self.spatial_dim,
            num_sites=self.num_sites,
        )
        self.kernel = StochasticAdvectionKernel(
            num_sites=self.num_sites,
            num_components=self.num_components,
            delta_t=float(model_cfg["delta_t_hours"]),
            kernel_jitter=float(model_cfg["kernel_jitter"]),
            identity_mix=float(model_cfg.get("identity_mix", 0.0)),
            kernel_decay=float(model_cfg.get("kernel_decay", 1.0)),
            min_block_scale=float(model_cfg["min_block_scale"]),
            allow_cross_component=bool(model_cfg.get("allow_cross_component", True)),
            diagonal_block_scale_init=float(model_cfg.get("diagonal_block_scale_init", 1.0)),
            cross_component_scale_init=float(model_cfg.get("cross_component_scale_init", 0.02)),
            max_transition_value=float(model_cfg.get("max_transition_value", 10.0)),
        )
        self.process_covariance = SeparableCrossCovariance(
            init_core_tril=cov_cfg["process"]["init_core_tril"],
            init_log_spatial_scale=float(cov_cfg["process"]["init_log_spatial_scale"]),
            jitter=float(cov_cfg["process"]["jitter"]),
        )
        self.measurement_covariance = SeparableCrossCovariance(
            init_core_tril=cov_cfg["measurement"]["init_core_tril"],
            init_log_spatial_scale=float(cov_cfg["measurement"]["init_log_spatial_scale"]),
            jitter=float(cov_cfg["measurement"]["jitter"]),
        )
        self.initial_mean = nn.Parameter(
            torch.tensor(cov_cfg["initial_state"]["init_mean"], dtype=torch.float32)
        )
        self.initial_covariance = PositiveDiagonal(
            init_raw_diag=cov_cfg["initial_state"]["init_log_diag"],
            jitter=float(cov_cfg["initial_state"]["jitter"]),
        )

    def set_scheduled_sampling_ratio(self, ratio: float) -> None:
        self.scheduled_sampling_ratio = float(min(max(ratio, 0.0), 1.0))

    def _build_dynamics(
        self,
        kernel_transition: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        dynamics = _sanitize_operator_matrix(kernel_transition)
        row_sum = dynamics.sum(dim=-1)
        column_sum = dynamics.sum(dim=-2)
        return dynamics, {
            "kernel_transition": dynamics,
            "ide_kernel": dynamics,
            "transition_row_sum": row_sum,
            "transition_column_sum": column_sum,
            "transition_abs_mean": dynamics.abs().mean(),
        }

    def _compute_training_loss(
        self,
        outputs: dict[str, torch.Tensor],
        observations: torch.Tensor,
        nwp_u: torch.Tensor,
        nwp_v: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output_dtype = outputs["negative_log_likelihood"].dtype
        device = observations.device
        zero = torch.zeros((), device=device, dtype=output_dtype)
        nll = outputs["negative_log_likelihood"]
        normalized_nll = nll / float(self.state_dim) if self.normalize_nll_by_state_dim else nll

        if observations.shape[0] > 1:
            one_step_error = outputs["predicted_mean"][1:] - observations[1:]
            one_step_loss = F.smooth_l1_loss(outputs["predicted_mean"][1:], observations[1:])
            one_step_mae = one_step_error.abs().mean()
            one_step_rmse = torch.sqrt(torch.mean(one_step_error.square()).clamp_min(0.0))
        else:
            one_step_loss = zero
            one_step_mae = zero
            one_step_rmse = zero

        kernel_one_step_loss = zero
        kernel_one_step_mae = zero
        kernel_one_step_rmse = zero
        if self.kernel_one_step_weight > 0.0 and observations.shape[0] > 1:
            kernel_prediction = torch.einsum(
                "tij,tj->ti",
                outputs["transition"][1:].to(dtype=observations.dtype),
                observations[:-1],
            )
            kernel_prediction = kernel_prediction + outputs["forcing"][1:].to(dtype=observations.dtype)
            kernel_error = kernel_prediction - observations[1:]
            kernel_one_step_loss = F.smooth_l1_loss(kernel_prediction, observations[1:])
            kernel_one_step_mae = kernel_error.abs().mean()
            kernel_one_step_rmse = torch.sqrt(torch.mean(kernel_error.square()).clamp_min(0.0))

        rollout_loss = zero
        rollout_mae = zero
        rollout_rmse = zero
        rollout_horizon = min(
            self.forecast_loss_horizon,
            max(observations.shape[0] - self.forecast_loss_min_context, 0),
        )
        if self.rollout_forecast_weight > 0.0 and rollout_horizon > 0:
            context_end = observations.shape[0] - rollout_horizon
            if context_end >= self.forecast_loss_min_context:
                rollout = self._forecast_impl(
                    filtered_mean=outputs["filtered_mean"][context_end - 1],
                    filtered_cov=outputs["filtered_cov"][context_end - 1],
                    future_nwp_u=nwp_u[context_end:],
                    future_nwp_v=nwp_v[context_end:],
                    teacher_forcing_targets=observations[context_end:],
                    teacher_forcing_ratio=self.scheduled_sampling_ratio if self.training else 0.0,
                )
                rollout_target = observations[context_end:]
                rollout_error = rollout["forecast_mean"] - rollout_target
                rollout_loss = F.smooth_l1_loss(rollout["forecast_mean"], rollout_target)
                rollout_mae = rollout_error.abs().mean()
                rollout_rmse = torch.sqrt(torch.mean(rollout_error.square()).clamp_min(0.0))

        transition_row_sum_loss = zero
        if self.transition_row_sum_weight > 0.0 and self.transition_row_sum_floor > 0.0:
            row_sum = outputs["transition_row_sum"].to(dtype=output_dtype)
            transition_row_sum_loss = F.relu(self.transition_row_sum_floor - row_sum).square().mean()

        total_loss = (
            self.nll_weight * normalized_nll
            + self.one_step_forecast_weight * one_step_loss
            + self.rollout_forecast_weight * rollout_loss
            + self.kernel_one_step_weight * kernel_one_step_loss
            + self.transition_row_sum_weight * transition_row_sum_loss
        )
        return total_loss, {
            "negative_log_likelihood": nll,
            "normalized_negative_log_likelihood": normalized_nll,
            "one_step_forecast_loss": one_step_loss,
            "one_step_mae": one_step_mae,
            "one_step_rmse": one_step_rmse,
            "rollout_forecast_loss": rollout_loss,
            "rollout_mae": rollout_mae,
            "rollout_rmse": rollout_rmse,
            "kernel_one_step_loss": kernel_one_step_loss,
            "kernel_one_step_mae": kernel_one_step_mae,
            "kernel_one_step_rmse": kernel_one_step_rmse,
            "transition_row_sum_loss": transition_row_sum_loss,
        }

    def _forward_single(
        self,
        observations: torch.Tensor,
        nwp_u: torch.Tensor,
        nwp_v: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        advection = self.advection_net(nwp_u, nwp_v, self.site_coords)
        kernel_transition, kernel_aux = self.kernel(
            means=advection["means"],
            covariances=advection["covariances"],
            site_coords=self.site_coords,
            joint_covariance=advection.get("joint_covariance"),
        )
        dynamics, dynamics_aux = self._build_dynamics(kernel_transition)
        kalman = self.kalman_filter(
            observations=observations,
            transition=dynamics,
            forcing=advection["forcing"],
        )
        outputs = {
            **advection,
            **kernel_aux,
            **dynamics_aux,
            **kalman,
            "transition": dynamics,
            "dynamics_matrix": dynamics,
        }
        total_loss, loss_aux = self._compute_training_loss(
            outputs=outputs,
            observations=observations,
            nwp_u=nwp_u,
            nwp_v=nwp_v,
        )
        outputs.update(loss_aux)
        outputs["loss"] = total_loss
        return outputs

    def forward(
        self,
        observations: torch.Tensor,
        nwp_u: torch.Tensor,
        nwp_v: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if observations.ndim == 2:
            return self._forward_single(observations=observations, nwp_u=nwp_u, nwp_v=nwp_v)

        if observations.ndim == 3:
            aggregate_keys = (
                "loss",
                "negative_log_likelihood",
                "normalized_negative_log_likelihood",
                "one_step_forecast_loss",
                "one_step_mae",
                "one_step_rmse",
                "rollout_forecast_loss",
                "rollout_mae",
                "rollout_rmse",
                "kernel_one_step_loss",
                "kernel_one_step_mae",
                "kernel_one_step_rmse",
                "transition_row_sum_loss",
                "forcing_abs_mean",
                "transition_abs_mean",
                "transition_row_sum_mean",
                "transition_column_sum_mean",
            )
            values: dict[str, list[torch.Tensor]] = {key: [] for key in aggregate_keys}
            for batch_idx in range(observations.shape[0]):
                outputs = self._forward_single(
                    observations=observations[batch_idx],
                    nwp_u=nwp_u[batch_idx],
                    nwp_v=nwp_v[batch_idx],
                )
                for key in aggregate_keys:
                    if key == "forcing_abs_mean":
                        values[key].append(outputs["forcing"].abs().mean())
                    elif key == "transition_row_sum_mean":
                        values[key].append(outputs["transition_row_sum"].mean())
                    elif key == "transition_column_sum_mean":
                        values[key].append(outputs["transition_column_sum"].mean())
                    elif key in outputs:
                        value = outputs[key]
                        values[key].append(value.mean() if value.ndim > 0 else value)

            return {
                key: torch.stack(items, dim=0).mean()
                for key, items in values.items()
                if items
            }

        raise ValueError(
            "Expected observations to have shape [time, state_dim] or [batch, time, state_dim], "
            f"got {observations.shape}."
        )

    def kalman_filter(
        self,
        observations: torch.Tensor,
        transition: torch.Tensor,
        forcing: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if observations.ndim != 2:
            raise ValueError(
                f"Kalman filtering expects a [time, state_dim] tensor, got shape {observations.shape}."
            )

        output_dtype = observations.dtype
        linalg_dtype = self.linalg_dtype
        device = observations.device
        observations = observations.to(dtype=linalg_dtype)
        transition = transition.to(dtype=linalg_dtype)
        forcing = forcing.to(dtype=linalg_dtype)

        coords = self.site_coords.to(device=device, dtype=torch.float32)
        q_matrix = self.process_covariance(coords).to(device=device, dtype=linalg_dtype)
        r_matrix = self.measurement_covariance(coords).to(device=device, dtype=linalg_dtype)
        q_matrix = _sanitize_symmetric_matrix(q_matrix)
        r_matrix = _sanitize_symmetric_matrix(r_matrix)
        mean_prev = self.initial_mean.to(device=device, dtype=linalg_dtype)
        cov_prev = self.initial_covariance().to(device=device, dtype=linalg_dtype)
        cov_prev = _sanitize_symmetric_matrix(cov_prev)
        identity = torch.eye(self.state_dim, device=device, dtype=linalg_dtype)

        predicted_mean = []
        predicted_cov = []
        filtered_mean = []
        filtered_cov = []
        total_nll = observations.new_tensor(0.0)

        for t in range(observations.shape[0]):
            transition_t = transition[t]
            forcing_t = forcing[t]
            mean_pred = _sanitize_vector(transition_t @ mean_prev + forcing_t)
            cov_pred = _sanitize_symmetric_matrix(
                transition_t @ cov_prev @ transition_t.transpose(0, 1) + q_matrix
            )

            innovation = _sanitize_vector(observations[t] - mean_pred)
            innovation_cov = _sanitize_symmetric_matrix(
                cov_pred + r_matrix + (self.transition_jitter + self.cholesky_jitter) * identity
            )

            try:
                gain_t = torch.linalg.solve(innovation_cov, cov_pred)
                kalman_gain = _sanitize_operator_matrix(gain_t.transpose(0, 1))
                solved_innovation = torch.linalg.solve(innovation_cov, innovation.unsqueeze(-1)).squeeze(-1)
                solved_innovation = _sanitize_vector(solved_innovation)
                sign, log_abs_det = torch.linalg.slogdet(innovation_cov)
                log_det = torch.where(
                    sign > 0,
                    log_abs_det,
                    torch.full_like(log_abs_det, self.max_nll_per_timestep),
                )
                log_det = torch.nan_to_num(
                    log_det,
                    nan=self.max_nll_per_timestep,
                    posinf=self.max_nll_per_timestep,
                    neginf=0.0,
                )
                log_det = torch.clamp(log_det, min=0.0, max=self.max_nll_per_timestep)
                innovation_quad = innovation @ solved_innovation
                innovation_quad = torch.nan_to_num(
                    innovation_quad,
                    nan=self.max_nll_per_timestep,
                    posinf=self.max_nll_per_timestep,
                    neginf=0.0,
                )
                innovation_quad = torch.clamp(innovation_quad, min=0.0, max=self.max_nll_per_timestep)

                mean_filt = _sanitize_vector(mean_pred + kalman_gain @ innovation)
                posterior_projection = identity - kalman_gain
                cov_filt = posterior_projection @ cov_pred @ posterior_projection.transpose(0, 1)
                cov_filt = cov_filt + kalman_gain @ r_matrix @ kalman_gain.transpose(0, 1)
                cov_filt = _sanitize_symmetric_matrix(cov_filt)
            except RuntimeError:
                mean_filt = mean_pred
                cov_filt = cov_pred
                log_det = observations.new_tensor(0.0)
                innovation_quad = observations.new_tensor(0.0)

            contribution = 0.5 * (
                log_det + innovation_quad + self.state_dim * math.log(2.0 * math.pi)
            )
            contribution = torch.nan_to_num(
                contribution,
                nan=self.max_nll_per_timestep,
                posinf=self.max_nll_per_timestep,
                neginf=self.max_nll_per_timestep,
            )
            contribution = torch.clamp(contribution, min=0.0, max=self.max_nll_per_timestep)
            total_nll = total_nll + contribution

            predicted_mean.append(mean_pred)
            predicted_cov.append(cov_pred)
            filtered_mean.append(mean_filt)
            filtered_cov.append(cov_filt)

            mean_prev = mean_filt
            cov_prev = cov_filt

        return {
            "loss": (total_nll / observations.shape[0]).to(dtype=output_dtype),
            "negative_log_likelihood": (total_nll / observations.shape[0]).to(dtype=output_dtype),
            "predicted_mean": torch.stack(predicted_mean, dim=0).to(dtype=output_dtype),
            "predicted_cov": torch.stack(predicted_cov, dim=0).to(dtype=output_dtype),
            "filtered_mean": torch.stack(filtered_mean, dim=0).to(dtype=output_dtype),
            "filtered_cov": torch.stack(filtered_cov, dim=0).to(dtype=output_dtype),
            "process_covariance_matrix": q_matrix.to(dtype=output_dtype),
            "measurement_covariance_matrix": r_matrix.to(dtype=output_dtype),
        }

    def _forecast_impl(
        self,
        filtered_mean: torch.Tensor,
        filtered_cov: torch.Tensor,
        future_nwp_u: torch.Tensor,
        future_nwp_v: torch.Tensor,
        teacher_forcing_targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        advection = self.advection_net(future_nwp_u, future_nwp_v, self.site_coords)
        kernel_transition, kernel_aux = self.kernel(
            means=advection["means"],
            covariances=advection["covariances"],
            site_coords=self.site_coords,
            joint_covariance=advection.get("joint_covariance"),
        )
        dynamics, dynamics_aux = self._build_dynamics(kernel_transition)

        output_dtype = filtered_mean.dtype
        mean_prev = filtered_mean.to(dtype=self.linalg_dtype)
        cov_prev = filtered_cov.to(dtype=self.linalg_dtype)
        dynamics = dynamics.to(dtype=self.linalg_dtype)
        forcing = advection["forcing"].to(dtype=self.linalg_dtype)
        coords = self.site_coords.to(device=filtered_mean.device, dtype=torch.float32)
        q_matrix = self.process_covariance(coords).to(device=filtered_mean.device, dtype=self.linalg_dtype)
        q_matrix = _sanitize_symmetric_matrix(q_matrix)

        forecast_means = []
        forecast_covs = []
        for t in range(dynamics.shape[0]):
            transition_t = dynamics[t]
            mean_prev = _sanitize_vector(transition_t @ mean_prev + forcing[t])
            cov_prev = _sanitize_symmetric_matrix(
                transition_t @ cov_prev @ transition_t.transpose(0, 1) + q_matrix
            )
            forecast_means.append(mean_prev)
            forecast_covs.append(cov_prev)
            if teacher_forcing_targets is not None and t < dynamics.shape[0] - 1 and teacher_forcing_ratio > 0.0:
                teacher_state = teacher_forcing_targets[t].to(device=mean_prev.device, dtype=mean_prev.dtype)
                if teacher_forcing_ratio >= 1.0:
                    mean_prev = teacher_state
                else:
                    teacher_mask = torch.rand((), device=mean_prev.device) < teacher_forcing_ratio
                    teacher_mask = teacher_mask.to(dtype=mean_prev.dtype)
                    mean_prev = teacher_mask * teacher_state + (1.0 - teacher_mask) * mean_prev
                mean_prev = _sanitize_vector(mean_prev)

        return {
            "forecast_mean": torch.stack(forecast_means, dim=0).to(dtype=output_dtype),
            "forecast_cov": torch.stack(forecast_covs, dim=0).to(dtype=output_dtype),
            "transition": dynamics.to(dtype=output_dtype),
            **advection,
            **kernel_aux,
            **dynamics_aux,
        }

    @torch.no_grad()
    def forecast(
        self,
        filtered_mean: torch.Tensor,
        filtered_cov: torch.Tensor,
        future_nwp_u: torch.Tensor,
        future_nwp_v: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self._forecast_impl(
            filtered_mean=filtered_mean,
            filtered_cov=filtered_cov,
            future_nwp_u=future_nwp_u,
            future_nwp_v=future_nwp_v,
            teacher_forcing_targets=None,
            teacher_forcing_ratio=0.0,
        )
