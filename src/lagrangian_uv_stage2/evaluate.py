from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import load_config
from .data import DataBundle, WindowedSequenceDataset, build_data_bundle
from .models.state_space import Stage2LagrangianStateSpaceModel


def load_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> tuple[dict[str, Any], DataBundle, Stage2LagrangianStateSpaceModel, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = load_config(config_path) if config_path is not None else checkpoint["config"]
    bundle = build_data_bundle(config, scaler_state=checkpoint["scalers"])
    site_coords = np.asarray(checkpoint["site_coords"], dtype=np.float32)

    model = Stage2LagrangianStateSpaceModel(config, site_coords=site_coords)
    model.load_state_dict(checkpoint["model_state"])
    target_device = torch.device(device or config["training"]["device"])
    model.to(target_device)
    model.eval()
    return config, bundle, model, checkpoint


def _to_numpy(value: torch.Tensor | np.ndarray | float | int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _safe_name(name: str) -> str:
    return name.replace(".", "__").replace("/", "__")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable.")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _prepare_export_dir(export_dir: Path) -> None:
    if export_dir.exists():
        for child in export_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    export_dir.mkdir(parents=True, exist_ok=True)


def _compute_error_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    diff = prediction - target
    mae_per_feature = np.mean(np.abs(diff), axis=0)
    rmse_per_feature = np.sqrt(np.mean(diff**2, axis=0))
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae_per_feature": {name: float(value) for name, value in zip(feature_names, mae_per_feature)},
        "rmse_per_feature": {name: float(value) for name, value in zip(feature_names, rmse_per_feature)},
    }


def _compute_persistence_comparison(
    model_prediction: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    if target.shape[0] < 2:
        empty_metrics = _compute_error_metrics(model_prediction, target, feature_names)
        return {
            "persistence_metrics": empty_metrics,
            "model_metrics_on_persistence_horizon": empty_metrics,
            "improvement_vs_persistence_pct": {"mae": 0.0, "rmse": 0.0},
        }

    persistence_prediction = target[:-1]
    persistence_target = target[1:]
    model_target = model_prediction[1:]

    persistence_metrics = _compute_error_metrics(persistence_prediction, persistence_target, feature_names)
    model_metrics = _compute_error_metrics(model_target, persistence_target, feature_names)

    mae_base = max(persistence_metrics["mae"], 1.0e-8)
    rmse_base = max(persistence_metrics["rmse"], 1.0e-8)
    return {
        "persistence_metrics": persistence_metrics,
        "model_metrics_on_persistence_horizon": model_metrics,
        "improvement_vs_persistence_pct": {
            "mae": 100.0 * (persistence_metrics["mae"] - model_metrics["mae"]) / mae_base,
            "rmse": 100.0 * (persistence_metrics["rmse"] - model_metrics["rmse"]) / rmse_base,
        },
    }


def _stat_summary(values: np.ndarray, absolute: bool = False) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if absolute:
        array = np.abs(array)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _component_slices(feature_names: list[str], component_names: list[str]) -> dict[str, slice]:
    num_components = len(component_names)
    if num_components <= 0 or len(feature_names) % num_components != 0:
        return {}
    num_sites = len(feature_names) // num_components
    return {
        component_name: slice(component_idx * num_sites, (component_idx + 1) * num_sites)
        for component_idx, component_name in enumerate(component_names)
    }


def _matrix_structural_diagnostics(
    matrix: np.ndarray,
    feature_names: list[str],
    component_names: list[str],
    zero_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 2:
        matrix = matrix[None, ...]
    if matrix.ndim != 3:
        raise ValueError(f"Expected matrix diagnostics input with shape [time, dim, dim], got {matrix.shape}.")

    row_sums = matrix.sum(axis=-1)
    row_abs_sums = np.abs(matrix).sum(axis=-1)
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    spectral_radius = []
    for matrix_t in matrix:
        try:
            spectral_radius.append(float(np.max(np.abs(np.linalg.eigvals(matrix_t)))))
        except np.linalg.LinAlgError:
            spectral_radius.append(float("nan"))

    diagnostics: dict[str, Any] = {
        "shape": list(matrix.shape),
        "row_sum": _stat_summary(row_sums),
        "row_abs_sum": _stat_summary(row_abs_sums),
        "row_sum_deficit_from_one": _stat_summary(1.0 - row_sums),
        "diagonal": _stat_summary(diagonal),
        "off_diagonal_abs": _stat_summary(matrix - np.eye(matrix.shape[-1])[None, ...] * diagonal[..., None], absolute=True),
        "spectral_radius": _stat_summary(np.asarray(spectral_radius, dtype=np.float64)),
        "nonzero_fraction": float(np.mean(np.abs(matrix) > zero_tolerance)),
    }

    slices = _component_slices(feature_names, component_names)
    if slices:
        blocks: dict[str, Any] = {}
        for row_component, row_slice in slices.items():
            for col_component, col_slice in slices.items():
                block = matrix[:, row_slice, col_slice]
                key = f"{col_component}_to_{row_component}"
                blocks[key] = {
                    "value": _stat_summary(block),
                    "abs": _stat_summary(block, absolute=True),
                    "mean_abs": float(np.mean(np.abs(block))),
                    "max_abs": float(np.max(np.abs(block))),
                    "nonzero_fraction": float(np.mean(np.abs(block) > zero_tolerance)),
                }
        diagnostics["blocks"] = blocks
        diagnostics["row_sum_by_component"] = {
            component: _stat_summary(row_sums[:, component_slice])
            for component, component_slice in slices.items()
        }
    return diagnostics


def _inverse_transform_array(
    scaled: np.ndarray,
    bundle: DataBundle,
) -> np.ndarray:
    return _to_numpy(bundle.obs_scaler.inverse_transform(np.asarray(scaled, dtype=np.float32)))


def _metric_with_scaled_and_denorm(
    prediction_scaled: np.ndarray,
    target_scaled: np.ndarray,
    bundle: DataBundle,
) -> dict[str, Any]:
    prediction_denorm = _inverse_transform_array(prediction_scaled, bundle)
    target_denorm = _inverse_transform_array(target_scaled, bundle)
    return {
        "scaled": _compute_error_metrics(prediction_scaled, target_scaled, bundle.feature_names),
        "denormalized": _compute_error_metrics(prediction_denorm, target_denorm, bundle.feature_names),
    }


def _relative_improvement(model_mae: float, baseline_mae: float) -> float:
    return 100.0 * (baseline_mae - model_mae) / max(float(baseline_mae), 1.0e-8)


def _one_step_operator_diagnostics(
    observations_scaled: np.ndarray,
    transition: np.ndarray,
    kernel_transition: np.ndarray,
    forcing: np.ndarray,
    bundle: DataBundle,
) -> dict[str, Any]:
    observations_scaled = np.asarray(observations_scaled, dtype=np.float32)
    transition = np.asarray(transition, dtype=np.float32)
    kernel_transition = np.asarray(kernel_transition, dtype=np.float32)
    forcing = np.asarray(forcing, dtype=np.float32)

    if observations_scaled.shape[0] < 2:
        return {"available": False, "reason": "Need at least two timesteps for one-step diagnostics."}

    prev = observations_scaled[:-1]
    target = observations_scaled[1:]
    transition_t = transition[1:]
    kernel_t = kernel_transition[1:]
    forcing_t = forcing[1:]

    predictions = {
        "raw_persistence": prev,
        "ide_kernel": np.einsum("tij,tj->ti", kernel_t, prev),
        "ide_dynamics_without_forcing": np.einsum("tij,tj->ti", transition_t, prev),
        "ide_dynamics_with_forcing": np.einsum("tij,tj->ti", transition_t, prev) + forcing_t,
    }
    baseline = _metric_with_scaled_and_denorm(predictions["raw_persistence"], target, bundle)
    baseline_mae = baseline["denormalized"]["mae"]
    diagnostics: dict[str, Any] = {
        "available": True,
        "baseline": "raw_persistence",
        "metrics": {},
    }

    for name, prediction in predictions.items():
        metrics = _metric_with_scaled_and_denorm(prediction, target, bundle)
        metrics["mae_improvement_vs_raw_persistence_pct"] = _relative_improvement(
            metrics["denormalized"]["mae"],
            baseline_mae,
        )
        diagnostics["metrics"][name] = metrics

    target_delta = target - prev
    dynamics_delta = predictions["ide_dynamics_with_forcing"] - prev
    diagnostics["delta_stats_scaled"] = {
        "target_delta_abs": _stat_summary(target_delta, absolute=True),
        "dynamics_delta_abs": _stat_summary(dynamics_delta, absolute=True),
        "forcing_abs": _stat_summary(forcing_t, absolute=True),
    }
    return diagnostics


def _kalman_uncertainty_diagnostics(
    observations_scaled: np.ndarray,
    predicted_mean_scaled: np.ndarray,
    filtered_mean_scaled: np.ndarray,
    predicted_covariance: np.ndarray,
    filtered_covariance: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    observations_scaled = np.asarray(observations_scaled, dtype=np.float64)
    predicted_mean_scaled = np.asarray(predicted_mean_scaled, dtype=np.float64)
    filtered_mean_scaled = np.asarray(filtered_mean_scaled, dtype=np.float64)
    predicted_covariance = np.asarray(predicted_covariance, dtype=np.float64)
    filtered_covariance = np.asarray(filtered_covariance, dtype=np.float64)

    pred_error = predicted_mean_scaled - observations_scaled
    filt_error = filtered_mean_scaled - observations_scaled
    pred_var = np.clip(np.diagonal(predicted_covariance, axis1=-2, axis2=-1), 1.0e-12, None)
    filt_var = np.clip(np.diagonal(filtered_covariance, axis1=-2, axis2=-1), 1.0e-12, None)
    pred_std = np.sqrt(pred_var)
    filt_std = np.sqrt(filt_var)
    pred_abs_error = np.abs(pred_error)
    filt_abs_error = np.abs(filt_error)

    return {
        "predicted_std": _stat_summary(pred_std),
        "filtered_std": _stat_summary(filt_std),
        "filtered_to_predicted_variance_ratio": _stat_summary(filt_var / np.clip(pred_var, 1.0e-12, None)),
        "predicted_coverage": {
            "within_1sigma": float(np.mean(pred_abs_error <= pred_std)),
            "within_2sigma": float(np.mean(pred_abs_error <= 2.0 * pred_std)),
        },
        "filtered_coverage": {
            "within_1sigma": float(np.mean(filt_abs_error <= filt_std)),
            "within_2sigma": float(np.mean(filt_abs_error <= 2.0 * filt_std)),
        },
        "predicted_std_per_feature": {
            name: float(value) for name, value in zip(feature_names, np.mean(pred_std, axis=0))
        },
        "filtered_std_per_feature": {
            name: float(value) for name, value in zip(feature_names, np.mean(filt_std, axis=0))
        },
    }


def _advection_diagnostics(
    means: np.ndarray,
    covariances: np.ndarray,
    component_names: list[str],
    raw_means: np.ndarray | None = None,
    raw_covariances: np.ndarray | None = None,
    joint_covariance: np.ndarray | None = None,
) -> dict[str, Any]:
    means = np.asarray(means, dtype=np.float64)
    covariances = np.asarray(covariances, dtype=np.float64)
    diagnostics: dict[str, Any] = {
        "mean_norm": _stat_summary(np.linalg.norm(means, axis=-1)),
        "covariance_trace": _stat_summary(np.trace(covariances, axis1=-2, axis2=-1)),
        "components": {},
    }
    for component_idx, component_name in enumerate(component_names):
        if means.ndim == 4:
            component_means = means[:, :, component_idx]
            cov = covariances[:, :, component_idx]
        else:
            component_means = means[:, component_idx]
            cov = covariances[:, component_idx]
        eigvals = np.linalg.eigvalsh(cov)
        diagnostics["components"][component_name] = {
            "mu_x": _stat_summary(component_means[..., 0]),
            "mu_y": _stat_summary(component_means[..., 1]),
            "mu_norm": _stat_summary(np.linalg.norm(component_means, axis=-1)),
            "cov_trace": _stat_summary(np.trace(cov, axis1=-2, axis2=-1)),
            "cov_min_eigenvalue": _stat_summary(eigvals[..., 0]),
            "cov_max_eigenvalue": _stat_summary(eigvals[..., -1]),
            "cov_condition_number": _stat_summary(eigvals[..., -1] / np.clip(eigvals[..., 0], 1.0e-12, None)),
        }

    if raw_means is not None:
        raw_means = np.asarray(raw_means, dtype=np.float64)
        diagnostics["raw_component_mean_abs_difference"] = _stat_summary(
            raw_means[:, 0] - raw_means[:, 1],
            absolute=True,
        )
    if raw_covariances is not None:
        raw_covariances = np.asarray(raw_covariances, dtype=np.float64)
        diagnostics["raw_component_covariance_abs_difference"] = _stat_summary(
            raw_covariances[:, 0] - raw_covariances[:, 1],
            absolute=True,
        )
    if joint_covariance is not None and np.asarray(joint_covariance).size > 0:
        joint_covariance = np.asarray(joint_covariance, dtype=np.float64)
        eigvals = np.linalg.eigvalsh(joint_covariance)
        diagnostics["joint_covariance"] = {
            "shape": list(joint_covariance.shape),
            "trace": _stat_summary(np.trace(joint_covariance, axis1=-2, axis2=-1)),
            "min_eigenvalue": _stat_summary(eigvals[..., 0]),
            "max_eigenvalue": _stat_summary(eigvals[..., -1]),
            "condition_number": _stat_summary(eigvals[..., -1] / np.clip(eigvals[..., 0], 1.0e-12, None)),
        }
        if joint_covariance.shape[-2:] == (4, 4):
            cross_block = joint_covariance[..., 0:2, 2:4]
            diagnostics["joint_covariance"]["u_v_cross_block_abs"] = _stat_summary(cross_block, absolute=True)
            diagnostics["joint_covariance"]["u_v_cross_block_frobenius_norm"] = _stat_summary(
                np.linalg.norm(cross_block, axis=(-2, -1))
            )
    return diagnostics


def _forcing_diagnostics(forcing: np.ndarray) -> dict[str, Any]:
    forcing = np.asarray(forcing, dtype=np.float64)
    return {
        "abs": _stat_summary(forcing, absolute=True),
        "l2_norm": _stat_summary(np.linalg.norm(forcing, axis=-1)),
        "nonzero_fraction": float(np.mean(np.abs(forcing) > 1.0e-8)),
    }


def _build_findings(
    config: dict[str, Any],
    structural_diagnostics: dict[str, Any],
    one_step_diagnostics: dict[str, Any],
    predicted_metrics: dict[str, Any],
    filtered_metrics: dict[str, Any],
    persistence_comparison: dict[str, Any],
    forcing_diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    allow_cross_component = bool(config["model"].get("allow_cross_component", False))
    if not allow_cross_component:
        findings.append(
            {
                "severity": "info",
                "code": "cross_component_disabled",
                "message": "model.allow_cross_component is false, so u/v cross-component transition blocks are masked to zero by design.",
            }
        )

    transition_diag = structural_diagnostics.get("transition", {})
    blocks = transition_diag.get("blocks", {})
    cross_block_keys = [key for key in blocks if "_to_" in key and key.split("_to_")[0] != key.split("_to_")[1]]
    if cross_block_keys and all(blocks[key]["max_abs"] <= 1.0e-8 for key in cross_block_keys):
        findings.append(
            {
                "severity": "info",
                "code": "cross_component_blocks_zero",
                "message": "All cross-component transition blocks are numerically zero.",
                "max_abs_by_block": {key: blocks[key]["max_abs"] for key in cross_block_keys},
            }
        )

    row_sum_mean = transition_diag.get("row_sum", {}).get("mean")
    if row_sum_mean is not None and row_sum_mean < 0.99:
        horizon = int(config["evaluation"].get("forecast_horizon", 1))
        findings.append(
            {
                "severity": "warning",
                "code": "transition_row_sum_below_one",
                "message": "The final dynamics matrix has row sums below one, so open-loop forecasts can be damped toward zero.",
                "mean_row_sum": float(row_sum_mean),
                "approx_gain_after_configured_horizon": float(row_sum_mean**horizon),
            }
        )

    if one_step_diagnostics.get("available"):
        metrics = one_step_diagnostics["metrics"]
        persistence_mae = metrics["raw_persistence"]["denormalized"]["mae"]
        dynamics_mae = metrics["ide_dynamics_with_forcing"]["denormalized"]["mae"]
        kernel_mae = metrics["ide_kernel"]["denormalized"]["mae"]
        if dynamics_mae > persistence_mae:
            findings.append(
                {
                    "severity": "warning",
                    "code": "dynamics_one_step_underperforms_persistence",
                    "message": "Applying the learned dynamics to the previous true observation is worse than raw persistence on this window.",
                    "dynamics_mae": float(dynamics_mae),
                    "persistence_mae": float(persistence_mae),
                }
            )
        if kernel_mae > persistence_mae:
            findings.append(
                {
                    "severity": "warning",
                    "code": "kernel_one_step_underperforms_persistence",
                    "message": "The open-system IDE kernel transition alone is worse than raw persistence on this window.",
                    "kernel_mae": float(kernel_mae),
                    "persistence_mae": float(persistence_mae),
                }
            )

    persistence_model_mae = persistence_comparison["model_metrics_on_persistence_horizon"]["mae"]
    persistence_mae = persistence_comparison["persistence_metrics"]["mae"]
    if persistence_model_mae > persistence_mae and filtered_metrics["mae"] < predicted_metrics["mae"]:
        findings.append(
            {
                "severity": "warning",
                "code": "filter_update_hides_weak_prior",
                "message": "Predicted means lose to persistence, but filtered means improve after observing the target; the Kalman update is doing most of the correction.",
                "predicted_mae": float(predicted_metrics["mae"]),
                "filtered_mae": float(filtered_metrics["mae"]),
                "persistence_mae": float(persistence_mae),
            }
        )

    if forcing_diagnostics["nonzero_fraction"] == 0.0:
        findings.append(
            {
                "severity": "info",
                "code": "forcing_zero",
                "message": "Forcing is exactly zero for this run, so all forecast motion must come from the transition matrix.",
            }
        )
    return findings


def _select_sequence_window(
    bundle: DataBundle,
    config: dict[str, Any],
    split: str,
    window_index: int,
    online_start: int,
    online_length: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if split == "train":
        dataset = bundle.train_dataset
    elif split == "val":
        dataset = bundle.val_dataset
    elif split == "online":
        sequence = bundle.online_sequence
        if sequence is None:
            raise ValueError("Online data was not loaded into this DataBundle, so split='online' is unavailable.")
        sequence_length = int(sequence["obs"].shape[0])
        requested_length = int(online_length or config["evaluation"]["context_window"])
        start = int(online_start)
        end = min(start + requested_length, sequence_length)
        if start < 0 or start >= sequence_length or end <= start:
            raise ValueError(
                f"Invalid online slice start={start}, length={requested_length}, sequence_length={sequence_length}."
            )
        metadata = {
            "split": split,
            "window_index": 0,
            "start_index": start,
            "length": end - start,
            "sequence_length": sequence_length,
        }
        return sequence["obs"][start:end], sequence["nwp_u"][start:end], sequence["nwp_v"][start:end], metadata
    else:
        raise ValueError(f"Unsupported split {split!r}. Choose from train, val, online.")

    if len(dataset) == 0:
        raise ValueError(f"The {split!r} dataset has no available windows.")
    if window_index < 0 or window_index >= len(dataset):
        raise IndexError(f"Window index {window_index} is out of range for the {split!r} dataset of size {len(dataset)}.")

    item = dataset[window_index]
    metadata = {
        "split": split,
        "window_index": window_index,
        "start_index": int(item["start_index"]),
        "length": int(item["obs"].shape[0]),
        "num_windows_in_split": len(dataset),
    }
    return item["obs"], item["nwp_u"], item["nwp_v"], metadata


@torch.no_grad()
def rolling_forecast(
    model: Stage2LagrangianStateSpaceModel,
    bundle: DataBundle,
    config: dict[str, Any],
    device: str | torch.device,
    context_window_override: int | None = None,
    forecast_horizon_override: int | None = None,
    stride_override: int | None = None,
) -> dict[str, Any]:
    target_device = torch.device(device)
    sequence = bundle.online_sequence
    if sequence is None:
        raise ValueError("Online data was not loaded into this DataBundle, so rolling forecast is unavailable.")

    observations = sequence["obs"].to(target_device)
    nwp_u = sequence["nwp_u"].to(target_device)
    nwp_v = sequence["nwp_v"].to(target_device)

    context_window = int(context_window_override or config["evaluation"]["context_window"])
    forecast_horizon = int(forecast_horizon_override or config["evaluation"]["forecast_horizon"])
    stride = int(stride_override or config["evaluation"]["stride"])

    starts = range(0, len(observations) - context_window - forecast_horizon + 1, stride)

    forecast_collection = []
    target_collection = []
    persistence_collection = []
    transition_collection = []
    kernel_transition_collection = []
    forcing_collection = []
    per_roll_mae = []
    per_roll_persistence_mae = []

    for start in starts:
        context_end = start + context_window
        forecast_end = context_end + forecast_horizon

        context_obs = observations[start:context_end]
        context_u = nwp_u[start:context_end]
        context_v = nwp_v[start:context_end]
        future_u = nwp_u[context_end:forecast_end]
        future_v = nwp_v[context_end:forecast_end]
        target_obs = observations[context_end:forecast_end]

        filtered = model(context_obs, context_u, context_v)
        forecast = model.forecast(
            filtered_mean=filtered["filtered_mean"][-1],
            filtered_cov=filtered["filtered_cov"][-1],
            future_nwp_u=future_u,
            future_nwp_v=future_v,
        )

        forecast_denorm = bundle.obs_scaler.inverse_transform(forecast["forecast_mean"]).detach().cpu()
        target_denorm = bundle.obs_scaler.inverse_transform(target_obs).detach().cpu()
        last_observation_denorm = bundle.obs_scaler.inverse_transform(context_obs[-1]).detach().cpu()
        last_observation_denorm = last_observation_denorm.reshape(-1)
        persistence_denorm = last_observation_denorm.unsqueeze(0).repeat(forecast_horizon, 1)
        mae = (forecast_denorm - target_denorm).abs().mean(dim=0)
        persistence_mae = (persistence_denorm - target_denorm).abs().mean(dim=0)

        forecast_collection.append(forecast_denorm)
        target_collection.append(target_denorm)
        persistence_collection.append(persistence_denorm)
        transition_collection.append(forecast["transition"].detach().cpu())
        kernel_transition_collection.append(forecast["kernel_transition"].detach().cpu())
        forcing_collection.append(forecast["forcing"].detach().cpu())
        per_roll_mae.append(mae)
        per_roll_persistence_mae.append(persistence_mae)

    stacked_mae = torch.stack(per_roll_mae, dim=0)
    stacked_persistence_mae = torch.stack(per_roll_persistence_mae, dim=0)
    stacked_forecast = torch.stack(forecast_collection, dim=0)
    stacked_target = torch.stack(target_collection, dim=0)
    stacked_persistence = torch.stack(persistence_collection, dim=0)
    stacked_transition = torch.stack(transition_collection, dim=0)
    stacked_kernel_transition = torch.stack(kernel_transition_collection, dim=0)
    stacked_forcing = torch.stack(forcing_collection, dim=0)

    model_error_metrics = _compute_error_metrics(
        prediction=stacked_forecast.reshape(-1, stacked_forecast.shape[-1]).numpy(),
        target=stacked_target.reshape(-1, stacked_target.shape[-1]).numpy(),
        feature_names=bundle.feature_names,
    )
    persistence_error_metrics = _compute_error_metrics(
        prediction=stacked_persistence.reshape(-1, stacked_persistence.shape[-1]).numpy(),
        target=stacked_target.reshape(-1, stacked_target.shape[-1]).numpy(),
        feature_names=bundle.feature_names,
    )

    horizon_model_mae = (stacked_forecast - stacked_target).abs().mean(dim=(0, 2))
    horizon_model_rmse = torch.sqrt(((stacked_forecast - stacked_target) ** 2).mean(dim=(0, 2)))
    horizon_persistence_mae = (stacked_persistence - stacked_target).abs().mean(dim=(0, 2))
    horizon_persistence_rmse = torch.sqrt(((stacked_persistence - stacked_target) ** 2).mean(dim=(0, 2)))
    mae_base = max(persistence_error_metrics["mae"], 1.0e-8)
    rmse_base = max(persistence_error_metrics["rmse"], 1.0e-8)
    model_abs_error = (stacked_forecast - stacked_target).abs()
    persistence_abs_error = (stacked_persistence - stacked_target).abs()
    win_mask = model_abs_error < persistence_abs_error
    flat_forecast = stacked_forecast.reshape(-1, stacked_forecast.shape[-1])
    flat_target = stacked_target.reshape(-1, stacked_target.shape[-1])
    forecast_std = flat_forecast.std(dim=0, unbiased=False)
    target_std = flat_target.std(dim=0, unbiased=False)
    rolling_diagnostics = {
        "num_rolls": int(stacked_forecast.shape[0]),
        "model_beats_persistence_fraction": float(win_mask.float().mean()),
        "model_beats_persistence_fraction_by_feature": {
            name: float(value) for name, value in zip(bundle.feature_names, win_mask.float().mean(dim=(0, 1)))
        },
        "model_beats_persistence_fraction_by_horizon": [
            float(value) for value in win_mask.float().mean(dim=(0, 2))
        ],
        "model_bias_by_feature": {
            name: float(value) for name, value in zip(bundle.feature_names, (stacked_forecast - stacked_target).mean(dim=(0, 1)))
        },
        "model_error_mean_by_horizon": [
            float(value) for value in (stacked_forecast - stacked_target).mean(dim=(0, 2))
        ],
        "forecast_std_by_feature": {
            name: float(value) for name, value in zip(bundle.feature_names, forecast_std)
        },
        "target_std_by_feature": {
            name: float(value) for name, value in zip(bundle.feature_names, target_std)
        },
        "forecast_to_target_std_ratio_by_feature": {
            name: float(value)
            for name, value in zip(bundle.feature_names, forecast_std / target_std.clamp_min(1.0e-8))
        },
        "forecast_transition": _matrix_structural_diagnostics(
            stacked_transition.reshape(-1, stacked_transition.shape[-2], stacked_transition.shape[-1]).numpy(),
            feature_names=bundle.feature_names,
            component_names=config["data"]["component_names"],
        ),
        "forecast_kernel_transition": _matrix_structural_diagnostics(
            stacked_kernel_transition.reshape(
                -1,
                stacked_kernel_transition.shape[-2],
                stacked_kernel_transition.shape[-1],
            ).numpy(),
            feature_names=bundle.feature_names,
            component_names=config["data"]["component_names"],
        ),
        "forecast_forcing": _forcing_diagnostics(stacked_forcing.numpy()),
    }
    return {
        "feature_names": bundle.feature_names,
        "forecast_horizon": forecast_horizon,
        "forecast": stacked_forecast,
        "target": stacked_target,
        "persistence_forecast": stacked_persistence,
        "per_roll_mae": stacked_mae,
        "per_roll_persistence_mae": stacked_persistence_mae,
        "mean_mae": stacked_mae.mean(dim=0),
        "mean_persistence_mae": stacked_persistence_mae.mean(dim=0),
        "model_metrics": model_error_metrics,
        "persistence_metrics": persistence_error_metrics,
        "improvement_vs_persistence_pct": {
            "mae": 100.0 * (persistence_error_metrics["mae"] - model_error_metrics["mae"]) / mae_base,
            "rmse": 100.0 * (persistence_error_metrics["rmse"] - model_error_metrics["rmse"]) / rmse_base,
        },
        "horizon_metrics": {
            "model_mae": horizon_model_mae,
            "model_rmse": horizon_model_rmse,
            "persistence_mae": horizon_persistence_mae,
            "persistence_rmse": horizon_persistence_rmse,
        },
        "rolling_diagnostics": rolling_diagnostics,
    }


def rolling_forecast_from_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    device: str | torch.device | None = None,
    context_window_override: int | None = None,
    forecast_horizon_override: int | None = None,
    stride_override: int | None = None,
) -> dict[str, Any]:
    config, bundle, model, _ = load_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
    )
    return rolling_forecast(
        model=model,
        bundle=bundle,
        config=config,
        device=device or config["training"]["device"],
        context_window_override=context_window_override,
        forecast_horizon_override=forecast_horizon_override,
        stride_override=stride_override,
    )


def _collect_parameter_snapshot(model: Stage2LagrangianStateSpaceModel) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {"parameters": []}

    for name, value in model.state_dict().items():
        safe_key = _safe_name(name)
        array = _to_numpy(value).astype(np.float32, copy=False)
        arrays[safe_key] = array
        summary["parameters"].append(
            {
                "name": name,
                "export_key": safe_key,
                "shape": list(array.shape),
                "numel": int(array.size),
            }
        )

    return arrays, summary


def _export_advection_csv(
    export_dir: Path,
    means: np.ndarray,
    covariances: np.ndarray,
    component_names: list[str],
) -> None:
    mean_path = export_dir / "advection_means.csv"
    with mean_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_index", "site_index", "component", "mu_x", "mu_y"])
        for t in range(means.shape[0]):
            if means.ndim == 4:
                for site_idx in range(means.shape[1]):
                    for component_idx, component_name in enumerate(component_names):
                        writer.writerow(
                            [t, site_idx, component_name, means[t, site_idx, component_idx, 0], means[t, site_idx, component_idx, 1]]
                        )
            else:
                for component_idx, component_name in enumerate(component_names):
                    writer.writerow([t, "", component_name, means[t, component_idx, 0], means[t, component_idx, 1]])

    covariance_path = export_dir / "advection_covariances.csv"
    with covariance_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_index", "site_index", "component", "sigma_xx", "sigma_xy", "sigma_yx", "sigma_yy"])
        for t in range(covariances.shape[0]):
            if covariances.ndim == 6:
                for site_idx in range(covariances.shape[1]):
                    for component_idx, component_name in enumerate(component_names):
                        cov = covariances[t, site_idx, component_idx]
                        writer.writerow([t, site_idx, component_name, cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]])
            else:
                for component_idx, component_name in enumerate(component_names):
                    cov = covariances[t, component_idx]
                    writer.writerow([t, "", component_name, cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]])


def _export_joint_advection_covariance_csv(
    export_dir: Path,
    joint_covariance: np.ndarray,
    component_names: list[str],
) -> None:
    if joint_covariance.size == 0:
        return

    axes = ["x", "y"]
    labels = [f"{component}_{axis}" for component in component_names for axis in axes]
    path = export_dir / "joint_advection_covariances.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_index", "site_index", "row_index", "row_label", "col_index", "col_label", "value"])
        for t in range(joint_covariance.shape[0]):
            if joint_covariance.ndim == 4:
                for site_idx in range(joint_covariance.shape[1]):
                    for row_idx, row_label in enumerate(labels):
                        for col_idx, col_label in enumerate(labels):
                            writer.writerow(
                                [t, site_idx, row_idx, row_label, col_idx, col_label, joint_covariance[t, site_idx, row_idx, col_idx]]
                            )
            else:
                for row_idx, row_label in enumerate(labels):
                    for col_idx, col_label in enumerate(labels):
                        writer.writerow([t, "", row_idx, row_label, col_idx, col_label, joint_covariance[t, row_idx, col_idx]])


def _export_transition_csv(export_dir: Path, transition: np.ndarray, labels: list[str]) -> None:
    path = export_dir / "transition_matrix_long.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_index", "row_index", "row_label", "col_index", "col_label", "value"])
        for t in range(transition.shape[0]):
            for row_idx, row_label in enumerate(labels):
                for col_idx, col_label in enumerate(labels):
                    writer.writerow([t, row_idx, row_label, col_idx, col_label, transition[t, row_idx, col_idx]])


def _maybe_plot_diagnostics(
    export_dir: Path,
    arrays: dict[str, np.ndarray],
    feature_names: list[str],
    component_names: list[str],
    config: dict[str, Any],
) -> list[str]:
    try:
        import imageio.v2 as imageio
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    outputs: list[str] = []
    time_index = np.arange(arrays["advection_means"].shape[0])
    advection_means = arrays["advection_means"]
    advection_covariances = arrays["advection_covariances"]
    if advection_means.ndim == 4:
        mean_for_plot = advection_means.mean(axis=1)
    else:
        mean_for_plot = advection_means
    if advection_covariances.ndim == 6:
        covariance_for_plot = advection_covariances.mean(axis=1)
    else:
        covariance_for_plot = advection_covariances

    mean_path = export_dir / "advection_mean_timeseries.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for component_idx, component_name in enumerate(component_names):
        axes[component_idx].plot(time_index, mean_for_plot[:, component_idx, 0], label=f"{component_name}_x")
        axes[component_idx].plot(time_index, mean_for_plot[:, component_idx, 1], label=f"{component_name}_y")
        axes[component_idx].set_ylabel("Mean")
        axes[component_idx].set_title(f"Advection mean for {component_name}")
        axes[component_idx].legend(loc="upper right")
    axes[-1].set_xlabel("Time index")
    fig.tight_layout()
    fig.savefig(mean_path, dpi=140)
    plt.close(fig)
    outputs.append(str(mean_path))

    covariance_path = export_dir / "advection_covariance_timeseries.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for component_idx, component_name in enumerate(component_names):
        cov = covariance_for_plot[:, component_idx]
        axes[component_idx].plot(time_index, cov[:, 0, 0], label="sigma_xx")
        axes[component_idx].plot(time_index, cov[:, 0, 1], label="sigma_xy")
        axes[component_idx].plot(time_index, cov[:, 1, 1], label="sigma_yy")
        axes[component_idx].set_ylabel("Covariance")
        axes[component_idx].set_title(f"Advection covariance for {component_name}")
        axes[component_idx].legend(loc="upper right")
    axes[-1].set_xlabel("Time index")
    fig.tight_layout()
    fig.savefig(covariance_path, dpi=140)
    plt.close(fig)
    outputs.append(str(covariance_path))

    if "forcing" in arrays:
        forcing_path = export_dir / "forcing_timeseries.png"
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        half = arrays["forcing"].shape[-1] // 2
        u_labels = feature_names[:half]
        v_labels = feature_names[half:]
        for state_idx, label in enumerate(u_labels):
            axes[0].plot(time_index, arrays["forcing"][:, state_idx], label=label)
        for state_idx, label in enumerate(v_labels, start=half):
            axes[1].plot(time_index, arrays["forcing"][:, state_idx], label=label)
        axes[0].set_ylabel("Forcing")
        axes[0].set_title("NWP forcing for u states")
        axes[0].legend(loc="upper right", ncol=2)
        axes[1].set_ylabel("Forcing")
        axes[1].set_title("NWP forcing for v states")
        axes[1].legend(loc="upper right", ncol=2)
        axes[-1].set_xlabel("Time index")
        fig.tight_layout()
        fig.savefig(forcing_path, dpi=140)
        plt.close(fig)
        outputs.append(str(forcing_path))

    matrix_specs = [
        ("mean_dynamics_matrix.png", arrays["transition"].mean(axis=0), feature_names, feature_names, "Mean dynamics matrix"),
        ("mean_kernel_transition_matrix.png", arrays["kernel_transition"].mean(axis=0), feature_names, feature_names, "Mean kernel transition matrix"),
        ("process_covariance_matrix.png", arrays["process_covariance_matrix"], feature_names, feature_names, "Process covariance Q"),
        ("measurement_covariance_matrix.png", arrays["measurement_covariance_matrix"], feature_names, feature_names, "Measurement covariance R"),
        ("kernel_block_scales.png", arrays["block_scales"], component_names, component_names, "Kernel block scales"),
    ]

    for filename, matrix, x_labels, y_labels, title in matrix_specs:
        path = export_dir / filename
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        outputs.append(str(path))

    gif_path = export_dir / "transition_matrix_bubbles.gif"
    max_frames = int(config["evaluation"].get("max_gif_frames", 120))
    fps = int(config["evaluation"].get("gif_fps", 12))
    frame_indices = np.unique(np.linspace(0, arrays["transition"].shape[0] - 1, num=min(max_frames, arrays["transition"].shape[0]), dtype=int))
    vmax = float(np.max(arrays["transition"])) if arrays["transition"].size else 1.0
    vmax = max(vmax, 1.0e-6)
    frames = []

    for frame_idx in frame_indices:
        matrix = arrays["transition"][frame_idx]
        fig, ax = plt.subplots(figsize=(7, 7))
        rows, cols = np.indices(matrix.shape)
        colors = matrix.reshape(-1)
        sizes = 2400.0 * colors / vmax + 8.0
        scatter = ax.scatter(
            cols.reshape(-1),
            rows.reshape(-1),
            s=sizes,
            c=colors,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            edgecolors="black",
            linewidths=0.3,
        )
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()
        ax.set_xlim(-0.5, len(feature_names) - 0.5)
        ax.set_ylim(len(feature_names) - 0.5, -0.5)
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.set_title(f"Transition matrix at t={frame_idx}")
        fig.colorbar(scatter, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())
        frames.append(frame.copy())
        plt.close(fig)

    if frames:
        imageio.mimsave(gif_path, frames, fps=fps)
        outputs.append(str(gif_path))

    return outputs


@torch.no_grad()
def summarize_validation_windows(
    model: Stage2LagrangianStateSpaceModel,
    bundle: DataBundle,
    device: str | torch.device,
) -> dict[str, Any]:
    dataset: WindowedSequenceDataset = bundle.val_dataset
    target_device = torch.device(device)

    if len(dataset) == 0:
        return {"num_windows": 0, "windows": [], "aggregate": {}}

    window_records: list[dict[str, Any]] = []
    nll_values = []
    pred_mae = []
    pred_rmse = []
    filt_mae = []
    filt_rmse = []
    persistence_mae = []
    persistence_rmse = []
    persistence_model_mae = []
    persistence_model_rmse = []

    for window_index in range(len(dataset)):
        item = dataset[window_index]
        observations = item["obs"].to(target_device)
        nwp_u = item["nwp_u"].to(target_device)
        nwp_v = item["nwp_v"].to(target_device)
        outputs = model(observations, nwp_u, nwp_v)

        obs_denorm = _to_numpy(bundle.obs_scaler.inverse_transform(item["obs"]))
        predicted_denorm = _to_numpy(bundle.obs_scaler.inverse_transform(outputs["predicted_mean"]))
        filtered_denorm = _to_numpy(bundle.obs_scaler.inverse_transform(outputs["filtered_mean"]))

        predicted_metrics = _compute_error_metrics(predicted_denorm, obs_denorm, bundle.feature_names)
        filtered_metrics = _compute_error_metrics(filtered_denorm, obs_denorm, bundle.feature_names)
        persistence_comparison = _compute_persistence_comparison(
            model_prediction=predicted_denorm,
            target=obs_denorm,
            feature_names=bundle.feature_names,
        )
        nll = float(_to_numpy(outputs["negative_log_likelihood"]).reshape(-1)[0])

        window_records.append(
            {
                "window_index": window_index,
                "start_index": int(item["start_index"]),
                "negative_log_likelihood": nll,
                "predicted_metrics": predicted_metrics,
                "filtered_metrics": filtered_metrics,
                "persistence_comparison": persistence_comparison,
            }
        )
        nll_values.append(nll)
        pred_mae.append(predicted_metrics["mae"])
        pred_rmse.append(predicted_metrics["rmse"])
        filt_mae.append(filtered_metrics["mae"])
        filt_rmse.append(filtered_metrics["rmse"])
        persistence_mae.append(persistence_comparison["persistence_metrics"]["mae"])
        persistence_rmse.append(persistence_comparison["persistence_metrics"]["rmse"])
        persistence_model_mae.append(persistence_comparison["model_metrics_on_persistence_horizon"]["mae"])
        persistence_model_rmse.append(persistence_comparison["model_metrics_on_persistence_horizon"]["rmse"])

    mean_persistence_mae = float(np.mean(persistence_mae))
    mean_persistence_rmse = float(np.mean(persistence_rmse))
    mean_model_mae_on_persistence = float(np.mean(persistence_model_mae))
    mean_model_rmse_on_persistence = float(np.mean(persistence_model_rmse))
    return {
        "num_windows": len(dataset),
        "windows": window_records,
        "aggregate": {
            "mean_negative_log_likelihood": float(np.mean(nll_values)),
            "mean_predicted_mae": float(np.mean(pred_mae)),
            "mean_predicted_rmse": float(np.mean(pred_rmse)),
            "mean_filtered_mae": float(np.mean(filt_mae)),
            "mean_filtered_rmse": float(np.mean(filt_rmse)),
            "mean_persistence_mae": mean_persistence_mae,
            "mean_persistence_rmse": mean_persistence_rmse,
            "mean_model_mae_on_persistence_horizon": mean_model_mae_on_persistence,
            "mean_model_rmse_on_persistence_horizon": mean_model_rmse_on_persistence,
            "improvement_vs_persistence_pct": {
                "mae": 100.0 * (mean_persistence_mae - mean_model_mae_on_persistence) / max(mean_persistence_mae, 1.0e-8),
                "rmse": 100.0 * (mean_persistence_rmse - mean_model_rmse_on_persistence) / max(mean_persistence_rmse, 1.0e-8),
            },
        },
    }


@torch.no_grad()
def export_window_diagnostics(
    model: Stage2LagrangianStateSpaceModel,
    bundle: DataBundle,
    config: dict[str, Any],
    export_dir: str | Path,
    device: str | torch.device,
    split: str = "val",
    window_index: int = 0,
    online_start: int = 0,
    online_length: int | None = None,
) -> dict[str, Any]:
    export_dir = Path(export_dir).expanduser().resolve()
    _prepare_export_dir(export_dir)
    target_device = torch.device(device)

    observations, nwp_u, nwp_v, metadata = _select_sequence_window(
        bundle=bundle,
        config=config,
        split=split,
        window_index=window_index,
        online_start=online_start,
        online_length=online_length,
    )
    observations = observations.to(target_device)
    nwp_u = nwp_u.to(target_device)
    nwp_v = nwp_v.to(target_device)

    outputs = model(observations, nwp_u, nwp_v)
    obs_denorm = _to_numpy(bundle.obs_scaler.inverse_transform(observations))
    predicted_denorm = _to_numpy(bundle.obs_scaler.inverse_transform(outputs["predicted_mean"]))
    filtered_denorm = _to_numpy(bundle.obs_scaler.inverse_transform(outputs["filtered_mean"]))
    means = _to_numpy(outputs["means"])
    covariances = _to_numpy(outputs["covariances"])
    joint_covariance = _to_numpy(outputs["joint_covariance"]) if "joint_covariance" in outputs else None
    raw_means = _to_numpy(outputs["component_raw_means"]) if "component_raw_means" in outputs else None
    raw_covariances = (
        _to_numpy(outputs["component_raw_covariances"]) if "component_raw_covariances" in outputs else None
    )
    drift_terms = _to_numpy(outputs["drift_terms"])
    dispersion_terms = _to_numpy(outputs["dispersion_terms"])
    transition = _to_numpy(outputs["transition"])
    kernel_transition = _to_numpy(outputs["kernel_transition"])
    forcing = _to_numpy(outputs["forcing"])
    q_matrix = _to_numpy(outputs["process_covariance_matrix"])
    r_matrix = _to_numpy(outputs["measurement_covariance_matrix"])
    block_scales = _to_numpy(outputs["block_scales"])
    identity_mix = float(_to_numpy(outputs["identity_mix"]).reshape(-1)[0])
    kernel_decay = float(_to_numpy(outputs.get("kernel_decay", np.asarray([1.0], dtype=np.float32))).reshape(-1)[0])
    initial_mean = _to_numpy(model.initial_mean.detach().cpu())
    initial_covariance = _to_numpy(model.initial_covariance().detach().cpu())

    arrays = {
        "observations_scaled": _to_numpy(observations),
        "observations_denorm": obs_denorm,
        "predicted_mean_scaled": _to_numpy(outputs["predicted_mean"]),
        "predicted_mean_denorm": predicted_denorm,
        "filtered_mean_scaled": _to_numpy(outputs["filtered_mean"]),
        "filtered_mean_denorm": filtered_denorm,
        "predicted_covariance": _to_numpy(outputs["predicted_cov"]),
        "filtered_covariance": _to_numpy(outputs["filtered_cov"]),
        "advection_means": means,
        "advection_covariances": covariances,
        "forcing": forcing,
        "drift_terms": drift_terms,
        "dispersion_terms": dispersion_terms,
        "kernel_transition": kernel_transition,
        "transition": transition,
        "process_covariance_matrix": q_matrix,
        "measurement_covariance_matrix": r_matrix,
        "block_scales": block_scales,
        "component_mask": _to_numpy(outputs["component_mask"]),
        "identity_mix": np.asarray([identity_mix], dtype=np.float32),
        "kernel_decay": np.asarray([kernel_decay], dtype=np.float32),
        "initial_mean": initial_mean,
        "initial_covariance": initial_covariance,
        "site_coords": bundle.site_coords.astype(np.float32),
    }
    if raw_means is not None:
        arrays["component_raw_means"] = raw_means
    if raw_covariances is not None:
        arrays["component_raw_covariances"] = raw_covariances
    if joint_covariance is not None:
        arrays["joint_advection_covariance"] = joint_covariance

    np.savez_compressed(export_dir / "diagnostics_arrays.npz", **arrays)
    np.save(export_dir / "transition_matrices.npy", transition)
    np.save(export_dir / "kernel_transition_matrices.npy", kernel_transition)
    np.save(export_dir / "advection_means.npy", means)
    np.save(export_dir / "advection_covariances.npy", covariances)
    if joint_covariance is not None:
        np.save(export_dir / "joint_advection_covariances.npy", joint_covariance)
    np.save(export_dir / "forcing.npy", forcing)

    parameter_arrays, parameter_summary = _collect_parameter_snapshot(model)
    np.savez_compressed(export_dir / "learned_parameters.npz", **parameter_arrays)
    _write_json(export_dir / "parameter_summary.json", parameter_summary)

    _export_advection_csv(export_dir, means, covariances, config["data"]["component_names"])
    if joint_covariance is not None:
        _export_joint_advection_covariance_csv(export_dir, joint_covariance, config["data"]["component_names"])
    _export_transition_csv(export_dir, transition, bundle.feature_names)

    predicted_metrics = _compute_error_metrics(predicted_denorm, obs_denorm, bundle.feature_names)
    filtered_metrics = _compute_error_metrics(filtered_denorm, obs_denorm, bundle.feature_names)
    persistence_comparison = _compute_persistence_comparison(
        model_prediction=predicted_denorm,
        target=obs_denorm,
        feature_names=bundle.feature_names,
    )
    structural_diagnostics = {
        "transition": _matrix_structural_diagnostics(
            transition,
            feature_names=bundle.feature_names,
            component_names=config["data"]["component_names"],
        ),
        "kernel_transition": _matrix_structural_diagnostics(
            kernel_transition,
            feature_names=bundle.feature_names,
            component_names=config["data"]["component_names"],
        ),
        "process_covariance_matrix": _matrix_structural_diagnostics(
            q_matrix,
            feature_names=bundle.feature_names,
            component_names=config["data"]["component_names"],
        ),
        "measurement_covariance_matrix": _matrix_structural_diagnostics(
            r_matrix,
            feature_names=bundle.feature_names,
            component_names=config["data"]["component_names"],
        ),
    }
    one_step_diagnostics = _one_step_operator_diagnostics(
        observations_scaled=arrays["observations_scaled"],
        transition=transition,
        kernel_transition=kernel_transition,
        forcing=forcing,
        bundle=bundle,
    )
    uncertainty_diagnostics = _kalman_uncertainty_diagnostics(
        observations_scaled=arrays["observations_scaled"],
        predicted_mean_scaled=arrays["predicted_mean_scaled"],
        filtered_mean_scaled=arrays["filtered_mean_scaled"],
        predicted_covariance=arrays["predicted_covariance"],
        filtered_covariance=arrays["filtered_covariance"],
        feature_names=bundle.feature_names,
    )
    advection_parameter_diagnostics = _advection_diagnostics(
        means=means,
        covariances=covariances,
        component_names=config["data"]["component_names"],
        raw_means=raw_means,
        raw_covariances=raw_covariances,
        joint_covariance=joint_covariance,
    )
    forcing_parameter_diagnostics = _forcing_diagnostics(forcing)
    diagnostic_findings = _build_findings(
        config=config,
        structural_diagnostics=structural_diagnostics,
        one_step_diagnostics=one_step_diagnostics,
        predicted_metrics=predicted_metrics,
        filtered_metrics=filtered_metrics,
        persistence_comparison=persistence_comparison,
        forcing_diagnostics=forcing_parameter_diagnostics,
    )

    summary = {
        **metadata,
        "feature_names": bundle.feature_names,
        "component_names": config["data"]["component_names"],
        "training_objective": {
            "total_loss": float(_to_numpy(outputs["loss"]).reshape(-1)[0]),
            "negative_log_likelihood": float(_to_numpy(outputs["negative_log_likelihood"]).reshape(-1)[0]),
            "normalized_negative_log_likelihood": float(
                _to_numpy(outputs.get("normalized_negative_log_likelihood", outputs["negative_log_likelihood"])).reshape(-1)[0]
            ),
            "one_step_forecast_loss": float(_to_numpy(outputs["one_step_forecast_loss"]).reshape(-1)[0]),
            "rollout_forecast_loss": float(_to_numpy(outputs["rollout_forecast_loss"]).reshape(-1)[0]),
            "kernel_one_step_loss": float(_to_numpy(outputs["kernel_one_step_loss"]).reshape(-1)[0]),
        },
        "negative_log_likelihood": float(_to_numpy(outputs["negative_log_likelihood"]).reshape(-1)[0]),
        "advection_shapes": {
            "means": list(means.shape),
            "covariances": list(covariances.shape),
            "joint_covariance": list(joint_covariance.shape) if joint_covariance is not None else None,
            "forcing": list(forcing.shape),
            "drift_terms": list(drift_terms.shape),
            "dispersion_terms": list(dispersion_terms.shape),
            "kernel_transition": list(kernel_transition.shape),
            "transition": list(transition.shape),
        },
        "dynamics_parameters": {
            "identity_mix": identity_mix,
            "kernel_decay": kernel_decay,
            "component_mask": _to_numpy(outputs["component_mask"]).tolist(),
        },
        "predicted_metrics": predicted_metrics,
        "filtered_metrics": filtered_metrics,
        "persistence_comparison": persistence_comparison,
        "structural_diagnostics": structural_diagnostics,
        "one_step_operator_diagnostics": one_step_diagnostics,
        "kalman_uncertainty_diagnostics": uncertainty_diagnostics,
        "advection_parameter_diagnostics": advection_parameter_diagnostics,
        "forcing_diagnostics": forcing_parameter_diagnostics,
        "diagnostic_findings": diagnostic_findings,
        "files": {
            "arrays": str(export_dir / "diagnostics_arrays.npz"),
            "transitions_npy": str(export_dir / "transition_matrices.npy"),
            "kernel_transitions_npy": str(export_dir / "kernel_transition_matrices.npy"),
            "forcing_npy": str(export_dir / "forcing.npy"),
            "parameters_npz": str(export_dir / "learned_parameters.npz"),
            "parameter_summary": str(export_dir / "parameter_summary.json"),
            "advection_means_csv": str(export_dir / "advection_means.csv"),
            "advection_covariances_csv": str(export_dir / "advection_covariances.csv"),
            "joint_advection_covariances_npy": str(export_dir / "joint_advection_covariances.npy")
            if joint_covariance is not None
            else None,
            "joint_advection_covariances_csv": str(export_dir / "joint_advection_covariances.csv")
            if joint_covariance is not None
            else None,
            "transition_csv": str(export_dir / "transition_matrix_long.csv"),
            "diagnostic_summary": str(export_dir / "diagnostic_summary.json"),
            "validation_summary": str(export_dir / "validation_summary.json"),
        },
    }

    generated_plots = _maybe_plot_diagnostics(
        export_dir=export_dir,
        arrays=arrays,
        feature_names=bundle.feature_names,
        component_names=config["data"]["component_names"],
        config=config,
    )
    summary["generated_plots"] = generated_plots

    validation_summary = summarize_validation_windows(model=model, bundle=bundle, device=target_device)
    summary["validation_summary"] = validation_summary.get("aggregate", {})
    _write_json(export_dir / "validation_summary.json", validation_summary)
    _write_json(export_dir / "diagnostic_summary.json", summary)
    return summary


def export_window_diagnostics_from_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    split: str = "val",
    window_index: int = 0,
    online_start: int = 0,
    online_length: int | None = None,
) -> dict[str, Any]:
    config, bundle, model, _ = load_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
    )
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    base_output_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else (
        checkpoint_path.parent / "diagnostics" / f"{split}_window_{window_index:03d}"
    )
    return export_window_diagnostics(
        model=model,
        bundle=bundle,
        config=config,
        export_dir=base_output_dir,
        device=device or config["training"]["device"],
        split=split,
        window_index=window_index,
        online_start=online_start,
        online_length=online_length,
    )
