from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils.mat import load_mat_variable


def _sanitize_array(array: np.ndarray, finite_clip: float = 1.0e6) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=finite_clip, neginf=-finite_clip)
    return np.clip(array, -finite_clip, finite_clip).astype(np.float32)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, array: np.ndarray, axes: tuple[int, ...], eps: float) -> "Standardizer":
        array = _sanitize_array(array)
        mean = array.mean(axis=axes, keepdims=True)
        std = array.std(axis=axes, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    @classmethod
    def identity(cls, shape: tuple[int, ...]) -> "Standardizer":
        return cls(mean=np.zeros(shape, dtype=np.float32), std=np.ones(shape, dtype=np.float32))

    def transform(self, array: np.ndarray) -> np.ndarray:
        array = _sanitize_array(array)
        transformed = (array - self.mean) / self.std
        return _sanitize_array(transformed)

    def inverse_transform(self, value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(value, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=value.dtype, device=value.device)
            std = torch.as_tensor(self.std, dtype=value.dtype, device=value.device)
            return value * std + mean
        return value * self.std + self.mean

    def state_dict(self) -> dict[str, list]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict[str, list]) -> "Standardizer":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
        )


class WindowedSequenceDataset(Dataset):
    def __init__(
        self,
        observations: np.ndarray,
        nwp_u: np.ndarray,
        nwp_v: np.ndarray,
        window_size: int,
        stride: int,
    ) -> None:
        if not (len(observations) == len(nwp_u) == len(nwp_v)):
            raise ValueError("Observation and NWP sequences must have the same number of timesteps.")
        if len(observations) < window_size:
            raise ValueError(
                f"Sequence length {len(observations)} is shorter than the window size {window_size}."
            )

        self.observations = torch.from_numpy(observations).float()
        self.nwp_u = torch.from_numpy(nwp_u).float()
        self.nwp_v = torch.from_numpy(nwp_v).float()
        self.window_size = window_size
        self.start_indices = list(range(0, len(observations) - window_size + 1, stride))

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        start = self.start_indices[idx]
        end = start + self.window_size
        return {
            "obs": self.observations[start:end],
            "nwp_u": self.nwp_u[start:end],
            "nwp_v": self.nwp_v[start:end],
            "start_index": start,
        }


@dataclass
class DataBundle:
    train_dataset: WindowedSequenceDataset
    val_dataset: WindowedSequenceDataset
    online_sequence: dict[str, torch.Tensor]
    obs_scaler: Standardizer
    nwp_u_scaler: Standardizer
    nwp_v_scaler: Standardizer
    site_coords: np.ndarray
    feature_names: list[str]
    split_index: int


def _ensure_time_feature_matrix(array: np.ndarray, min_feature_count: int) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D observation array, got shape {array.shape}.")

    if array.shape[1] >= min_feature_count and array.shape[0] > array.shape[1]:
        return array
    if array.shape[0] >= min_feature_count and array.shape[1] > array.shape[0]:
        return array.T
    if array.shape[1] >= min_feature_count:
        return array
    if array.shape[0] >= min_feature_count:
        return array.T

    raise ValueError(
        "Could not infer the observation layout. "
        f"Observed shape: {array.shape}, expected at least {min_feature_count} features."
    )


def _ensure_hw_t_c(array: np.ndarray, grid_size: tuple[int, int], total_channels: int) -> np.ndarray:
    if array.ndim != 4:
        raise ValueError(f"Expected a 4D NWP array, got shape {array.shape}.")

    if tuple(array.shape[:2]) == grid_size and array.shape[-1] == total_channels:
        return array

    for perm in permutations(range(4)):
        permuted = np.transpose(array, perm)
        if tuple(permuted.shape[:2]) == grid_size and permuted.shape[-1] == total_channels:
            return permuted

    raise ValueError(
        "Could not infer the NWP layout. "
        f"Observed shape: {array.shape}, expected grid {grid_size} and >= {total_channels} channels."
    )


def _flatten_station_vector(array: np.ndarray, expected_sites: int) -> np.ndarray:
    vector = np.asarray(array, dtype=np.float32).reshape(-1)
    if vector.shape[0] != expected_sites:
        raise ValueError(
            f"Expected {expected_sites} station coordinates, but found {vector.shape[0]} values."
        )
    return vector


def _load_site_coords(config: dict[str, Any]) -> np.ndarray:
    coord_cfg = config["data"]["coordinates"]
    num_sites = int(config["model"]["num_sites"])

    if coord_cfg.get("manual") is not None:
        coords = np.asarray(coord_cfg["manual"], dtype=np.float32)
        if coords.shape != (num_sites, 2):
            raise ValueError(
                f"Manual station coordinates must have shape {(num_sites, 2)}, got {coords.shape}."
            )
        return coords

    lat = _flatten_station_vector(load_mat_variable(coord_cfg["source_path"], coord_cfg["lat_key"]), num_sites)
    lon = _flatten_station_vector(load_mat_variable(coord_cfg["source_path"], coord_cfg["lon_key"]), num_sites)
    return np.stack([lat, lon], axis=-1)


def _select_measurement_block(config: dict[str, Any], subset: str) -> np.ndarray:
    meas_cfg = config["data"]["measurement"]
    raw = load_mat_variable(meas_cfg[f"{subset}_path"], meas_cfg["variable_name"])
    raw = _ensure_time_feature_matrix(np.asarray(raw, dtype=np.float32), min_feature_count=max(meas_cfg["target_raw_indices"]) + 1)
    block = raw[:, meas_cfg["target_raw_indices"]]
    block = block[:, meas_cfg["component_first_order"]]
    return _sanitize_array(block)


def _select_nwp_blocks(config: dict[str, Any], subset: str) -> tuple[np.ndarray, np.ndarray]:
    nwp_cfg = config["data"]["nwp"]
    grid_size = tuple(nwp_cfg["grid_size"])
    raw = load_mat_variable(nwp_cfg[f"{subset}_path"], nwp_cfg["variable_name"])
    raw = _ensure_hw_t_c(np.asarray(raw, dtype=np.float32), grid_size=grid_size, total_channels=nwp_cfg["total_channels"])
    raw = np.transpose(raw, (2, 3, 0, 1))

    u_block = raw[:, nwp_cfg["u_channel_indices"], :, :]
    v_block = raw[:, nwp_cfg["v_channel_indices"], :, :]
    return _sanitize_array(u_block), _sanitize_array(v_block)


def _align_lengths(obs: np.ndarray, nwp_u: np.ndarray, nwp_v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = min(len(obs), len(nwp_u), len(nwp_v))
    return obs[:length], nwp_u[:length], nwp_v[:length]


def _build_feature_names(config: dict[str, Any]) -> list[str]:
    site_names = config["data"]["site_names"]
    component_names = config["data"]["component_names"]
    return [f"{component}_{site}_140m" for component in component_names for site in site_names]


def build_data_bundle(config: dict[str, Any], scaler_state: dict[str, dict[str, list]] | None = None) -> DataBundle:
    obs_offline = _select_measurement_block(config, "offline")
    nwp_u_offline, nwp_v_offline = _select_nwp_blocks(config, "offline")
    obs_offline, nwp_u_offline, nwp_v_offline = _align_lengths(obs_offline, nwp_u_offline, nwp_v_offline)

    obs_online = _select_measurement_block(config, "online")
    nwp_u_online, nwp_v_online = _select_nwp_blocks(config, "online")
    obs_online, nwp_u_online, nwp_v_online = _align_lengths(obs_online, nwp_u_online, nwp_v_online)

    split_fraction = float(config["data"]["split"]["train_fraction"])
    split_index = int(len(obs_offline) * split_fraction)
    window_size = int(config["data"]["windows"]["window_size"])
    split_index = max(split_index, window_size)
    split_index = min(split_index, len(obs_offline) - window_size)
    if split_index < window_size:
        raise ValueError(
            "The offline sequence is too short for the requested train/validation split and window size. "
            f"Sequence length={len(obs_offline)}, window_size={window_size}."
        )

    norm_cfg = config["data"]["normalization"]
    eps = float(norm_cfg["eps"])

    if scaler_state is None:
        if norm_cfg["normalize_observation"]:
            obs_scaler = Standardizer.fit(obs_offline[:split_index], axes=(0,), eps=eps)
        else:
            obs_scaler = Standardizer.identity((1, obs_offline.shape[-1]))

        if norm_cfg["normalize_nwp"]:
            nwp_u_scaler = Standardizer.fit(nwp_u_offline[:split_index], axes=(0, 2, 3), eps=eps)
            nwp_v_scaler = Standardizer.fit(nwp_v_offline[:split_index], axes=(0, 2, 3), eps=eps)
        else:
            nwp_u_scaler = Standardizer.identity((1, nwp_u_offline.shape[1], 1, 1))
            nwp_v_scaler = Standardizer.identity((1, nwp_v_offline.shape[1], 1, 1))
    else:
        obs_scaler = Standardizer.from_state_dict(scaler_state["obs"])
        nwp_u_scaler = Standardizer.from_state_dict(scaler_state["nwp_u"])
        nwp_v_scaler = Standardizer.from_state_dict(scaler_state["nwp_v"])

    obs_offline = obs_scaler.transform(obs_offline)
    nwp_u_offline = nwp_u_scaler.transform(nwp_u_offline)
    nwp_v_offline = nwp_v_scaler.transform(nwp_v_offline)

    obs_online = obs_scaler.transform(obs_online)
    nwp_u_online = nwp_u_scaler.transform(nwp_u_online)
    nwp_v_online = nwp_v_scaler.transform(nwp_v_online)

    window_cfg = config["data"]["windows"]
    train_dataset = WindowedSequenceDataset(
        observations=obs_offline[:split_index],
        nwp_u=nwp_u_offline[:split_index],
        nwp_v=nwp_v_offline[:split_index],
        window_size=int(window_cfg["window_size"]),
        stride=int(window_cfg["stride"]),
    )
    val_dataset = WindowedSequenceDataset(
        observations=obs_offline[split_index:],
        nwp_u=nwp_u_offline[split_index:],
        nwp_v=nwp_v_offline[split_index:],
        window_size=int(window_cfg["window_size"]),
        stride=int(window_cfg["stride"]),
    )

    online_sequence = {
        "obs": torch.from_numpy(obs_online).float(),
        "nwp_u": torch.from_numpy(nwp_u_online).float(),
        "nwp_v": torch.from_numpy(nwp_v_online).float(),
    }

    return DataBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        online_sequence=online_sequence,
        obs_scaler=obs_scaler,
        nwp_u_scaler=nwp_u_scaler,
        nwp_v_scaler=nwp_v_scaler,
        site_coords=_load_site_coords(config),
        feature_names=_build_feature_names(config),
        split_index=split_index,
    )


def serialize_scalers(bundle: DataBundle) -> dict[str, dict[str, list]]:
    return {
        "obs": bundle.obs_scaler.state_dict(),
        "nwp_u": bundle.nwp_u_scaler.state_dict(),
        "nwp_v": bundle.nwp_v_scaler.state_dict(),
    }
