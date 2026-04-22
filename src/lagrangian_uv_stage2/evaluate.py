from __future__ import annotations

import csv
import json
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
        per_roll_mae.append(mae)
        per_roll_persistence_mae.append(persistence_mae)

    stacked_mae = torch.stack(per_roll_mae, dim=0)
    stacked_persistence_mae = torch.stack(per_roll_persistence_mae, dim=0)
    stacked_forecast = torch.stack(forecast_collection, dim=0)
    stacked_target = torch.stack(target_collection, dim=0)
    stacked_persistence = torch.stack(persistence_collection, dim=0)

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
        writer.writerow(["time_index", "component", "mu_x", "mu_y"])
        for t in range(means.shape[0]):
            for component_idx, component_name in enumerate(component_names):
                writer.writerow([t, component_name, means[t, component_idx, 0], means[t, component_idx, 1]])

    covariance_path = export_dir / "advection_covariances.csv"
    with covariance_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_index", "component", "sigma_xx", "sigma_xy", "sigma_yx", "sigma_yy"])
        for t in range(covariances.shape[0]):
            for component_idx, component_name in enumerate(component_names):
                cov = covariances[t, component_idx]
                writer.writerow([t, component_name, cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]])


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

    mean_path = export_dir / "advection_mean_timeseries.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for component_idx, component_name in enumerate(component_names):
        axes[component_idx].plot(time_index, arrays["advection_means"][:, component_idx, 0], label=f"{component_name}_x")
        axes[component_idx].plot(time_index, arrays["advection_means"][:, component_idx, 1], label=f"{component_name}_y")
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
        cov = arrays["advection_covariances"][:, component_idx]
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

    if "persistence_matrix" in arrays:
        matrix_specs.append(
            ("persistence_matrix.png", arrays["persistence_matrix"], feature_names, feature_names, "Persistence matrix")
        )
    if "residual_transition" in arrays:
        matrix_specs.append(
            (
                "mean_residual_transition_matrix.png",
                arrays["residual_transition"].mean(axis=0),
                feature_names,
                feature_names,
                "Mean residual transition matrix",
            )
        )

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
    export_dir.mkdir(parents=True, exist_ok=True)
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
    drift_terms = _to_numpy(outputs["drift_terms"])
    dispersion_terms = _to_numpy(outputs["dispersion_terms"])
    transition = _to_numpy(outputs["transition"])
    kernel_transition = _to_numpy(outputs["kernel_transition"])
    forcing = _to_numpy(outputs["forcing"])
    q_matrix = _to_numpy(outputs["process_covariance_matrix"])
    r_matrix = _to_numpy(outputs["measurement_covariance_matrix"])
    block_scales = _to_numpy(outputs["block_scales"])
    persistence_diagonal = _to_numpy(outputs["persistence_diagonal"])
    persistence_matrix = _to_numpy(outputs["persistence_matrix"])
    residual_gate = float(_to_numpy(outputs["residual_gate"]).reshape(-1)[0])
    residual_transition = _to_numpy(outputs["residual_transition"])
    identity_mix = float(_to_numpy(outputs["identity_mix"]).reshape(-1)[0])
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
        "persistence_diagonal": persistence_diagonal,
        "persistence_matrix": persistence_matrix,
        "residual_gate": np.asarray([residual_gate], dtype=np.float32),
        "residual_transition": residual_transition,
        "process_covariance_matrix": q_matrix,
        "measurement_covariance_matrix": r_matrix,
        "block_scales": block_scales,
        "identity_mix": np.asarray([identity_mix], dtype=np.float32),
        "initial_mean": initial_mean,
        "initial_covariance": initial_covariance,
        "site_coords": bundle.site_coords.astype(np.float32),
    }

    np.savez_compressed(export_dir / "diagnostics_arrays.npz", **arrays)
    np.save(export_dir / "transition_matrices.npy", transition)
    np.save(export_dir / "kernel_transition_matrices.npy", kernel_transition)
    np.save(export_dir / "advection_means.npy", means)
    np.save(export_dir / "advection_covariances.npy", covariances)
    np.save(export_dir / "forcing.npy", forcing)

    parameter_arrays, parameter_summary = _collect_parameter_snapshot(model)
    np.savez_compressed(export_dir / "learned_parameters.npz", **parameter_arrays)
    _write_json(export_dir / "parameter_summary.json", parameter_summary)

    _export_advection_csv(export_dir, means, covariances, config["data"]["component_names"])
    _export_transition_csv(export_dir, transition, bundle.feature_names)

    predicted_metrics = _compute_error_metrics(predicted_denorm, obs_denorm, bundle.feature_names)
    filtered_metrics = _compute_error_metrics(filtered_denorm, obs_denorm, bundle.feature_names)
    persistence_comparison = _compute_persistence_comparison(
        model_prediction=predicted_denorm,
        target=obs_denorm,
        feature_names=bundle.feature_names,
    )

    summary = {
        **metadata,
        "feature_names": bundle.feature_names,
        "component_names": config["data"]["component_names"],
        "training_objective": {
            "total_loss": float(_to_numpy(outputs["loss"]).reshape(-1)[0]),
            "negative_log_likelihood": float(_to_numpy(outputs["negative_log_likelihood"]).reshape(-1)[0]),
            "one_step_forecast_loss": float(_to_numpy(outputs["one_step_forecast_loss"]).reshape(-1)[0]),
            "rollout_forecast_loss": float(_to_numpy(outputs["rollout_forecast_loss"]).reshape(-1)[0]),
        },
        "negative_log_likelihood": float(_to_numpy(outputs["negative_log_likelihood"]).reshape(-1)[0]),
        "advection_shapes": {
            "means": list(means.shape),
            "covariances": list(covariances.shape),
            "forcing": list(forcing.shape),
            "drift_terms": list(drift_terms.shape),
            "dispersion_terms": list(dispersion_terms.shape),
            "kernel_transition": list(kernel_transition.shape),
            "transition": list(transition.shape),
        },
        "dynamics_parameters": {
            "persistence_diagonal": persistence_diagonal.tolist(),
            "residual_gate": residual_gate,
            "identity_mix": identity_mix,
        },
        "predicted_metrics": predicted_metrics,
        "filtered_metrics": filtered_metrics,
        "persistence_comparison": persistence_comparison,
        "files": {
            "arrays": str(export_dir / "diagnostics_arrays.npz"),
            "transitions_npy": str(export_dir / "transition_matrices.npy"),
            "kernel_transitions_npy": str(export_dir / "kernel_transition_matrices.npy"),
            "forcing_npy": str(export_dir / "forcing.npy"),
            "parameters_npz": str(export_dir / "learned_parameters.npz"),
            "parameter_summary": str(export_dir / "parameter_summary.json"),
            "advection_means_csv": str(export_dir / "advection_means.csv"),
            "advection_covariances_csv": str(export_dir / "advection_covariances.csv"),
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
