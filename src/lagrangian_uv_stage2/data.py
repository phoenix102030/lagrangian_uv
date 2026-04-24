from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import mat as mat_utils
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


class InMemorySequenceSource:
    def __init__(self, array: np.ndarray) -> None:
        self.array = _sanitize_array(array)
        self.num_channels = int(self.array.shape[1])

    def __len__(self) -> int:
        return int(self.array.shape[0])

    def window(self, start: int, end: int) -> np.ndarray:
        return self.array[start:end]


class LazyNwpSequenceSource:
    def __init__(
        self,
        path: str,
        var_name: str,
        grid_size: tuple[int, int],
        total_channels: int,
        channel_indices: list[int],
        cache_chunk_size: int = 512,
        max_cache_chunks: int = 4,
    ) -> None:
        if mat_utils.h5py is None:
            raise RuntimeError("Lazy NWP loading requires h5py.")

        self.path = str(path)
        self.var_name = var_name
        self.grid_size = tuple(grid_size)
        self.total_channels = int(total_channels)
        self.channel_indices = np.asarray(channel_indices, dtype=np.int64)
        self.num_channels = int(len(channel_indices))
        self.cache_chunk_size = max(1, int(cache_chunk_size))
        self.max_cache_chunks = max(1, int(max_cache_chunks))

        self._file: Any | None = None
        self._dataset: Any | None = None
        self._chunk_cache: OrderedDict[int, np.ndarray] = OrderedDict()

        with mat_utils.h5py.File(self.path, "r") as handle:
            if self.var_name not in handle:
                raise KeyError(f"Variable {self.var_name!r} not found in {self.path}.")
            dataset = handle[self.var_name]
            self.source_shape = tuple(int(dim) for dim in dataset.shape)

        self.permutation = _infer_hw_t_c_permutation(self.source_shape, self.grid_size, self.total_channels)
        self.time_axis = int(self.permutation[2])
        self.length = int(self.source_shape[self.time_axis])

    def __len__(self) -> int:
        return self.length

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_dataset"] = None
        state["_chunk_cache"] = OrderedDict()
        return state

    def _ensure_open(self) -> Any:
        if self._dataset is None:
            self._file = mat_utils.h5py.File(self.path, "r")
            self._dataset = self._file[self.var_name]
        return self._dataset

    def _read_range(self, start: int, end: int) -> np.ndarray:
        dataset = self._ensure_open()
        source_slices: list[slice] = [slice(None)] * 4
        source_slices[self.time_axis] = slice(start, end)
        block = np.asarray(dataset[tuple(source_slices)], dtype=np.float32)
        block = np.transpose(block, self.permutation)
        block = block[:, :, :, self.channel_indices]
        block = np.transpose(block, (2, 3, 0, 1))
        return _sanitize_array(block)

    def _cached_chunk(self, chunk_index: int) -> np.ndarray:
        cached = self._chunk_cache.get(chunk_index)
        if cached is not None:
            self._chunk_cache.move_to_end(chunk_index)
            return cached

        start = chunk_index * self.cache_chunk_size
        end = min(start + self.cache_chunk_size, self.length)
        chunk = self._read_range(start, end)
        self._chunk_cache[chunk_index] = chunk
        self._chunk_cache.move_to_end(chunk_index)
        while len(self._chunk_cache) > self.max_cache_chunks:
            self._chunk_cache.popitem(last=False)
        return chunk

    def window(self, start: int, end: int) -> np.ndarray:
        if start < 0 or end < start or end > self.length:
            raise IndexError(f"Invalid window [{start}, {end}) for sequence length {self.length}.")

        pieces: list[np.ndarray] = []
        cursor = start
        while cursor < end:
            chunk_index = cursor // self.cache_chunk_size
            chunk = self._cached_chunk(chunk_index)
            chunk_start = chunk_index * self.cache_chunk_size
            local_start = cursor - chunk_start
            local_end = min(end - chunk_start, chunk.shape[0])
            pieces.append(chunk[local_start:local_end])
            cursor = chunk_start + local_end

        if not pieces:
            return np.empty((0, self.num_channels, *self.grid_size), dtype=np.float32)
        if len(pieces) == 1:
            return pieces[0]
        return np.concatenate(pieces, axis=0)


class StandardizedSequenceSource:
    def __init__(self, base_source: InMemorySequenceSource | LazyNwpSequenceSource, scaler: Standardizer) -> None:
        self.base_source = base_source
        self.scaler = scaler
        self.num_channels = int(base_source.num_channels)

    def __len__(self) -> int:
        return len(self.base_source)

    def window(self, start: int, end: int) -> np.ndarray:
        return self.scaler.transform(self.base_source.window(start, end))


class OffsetSequenceSource:
    def __init__(
        self,
        base_source: InMemorySequenceSource | LazyNwpSequenceSource | StandardizedSequenceSource,
        start_offset: int,
        end_offset: int,
    ) -> None:
        if start_offset < 0 or end_offset < start_offset or end_offset > len(base_source):
            raise ValueError(
                f"Invalid source slice [{start_offset}, {end_offset}) for source length {len(base_source)}."
            )
        self.base_source = base_source
        self.start_offset = int(start_offset)
        self.end_offset = int(end_offset)
        self.num_channels = int(base_source.num_channels)

    def __len__(self) -> int:
        return self.end_offset - self.start_offset

    def window(self, start: int, end: int) -> np.ndarray:
        if start < 0 or end < start or end > len(self):
            raise IndexError(f"Invalid local window [{start}, {end}) for sliced sequence length {len(self)}.")
        return self.base_source.window(self.start_offset + start, self.start_offset + end)


class WindowedSequenceDataset(Dataset):
    def __init__(
        self,
        observations: np.ndarray,
        nwp_u: InMemorySequenceSource | LazyNwpSequenceSource | StandardizedSequenceSource | OffsetSequenceSource,
        nwp_v: InMemorySequenceSource | LazyNwpSequenceSource | StandardizedSequenceSource | OffsetSequenceSource,
        window_size: int,
        stride: int,
    ) -> None:
        if not (len(observations) == len(nwp_u) == len(nwp_v)):
            raise ValueError("Observation and NWP sequences must have the same number of timesteps.")
        if len(observations) < window_size:
            raise ValueError(
                f"Sequence length {len(observations)} is shorter than the window size {window_size}."
            )

        self.observations = torch.from_numpy(_sanitize_array(observations)).float()
        self.nwp_u = nwp_u
        self.nwp_v = nwp_v
        self.window_size = int(window_size)
        self.start_indices = list(range(0, len(observations) - window_size + 1, stride))

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        start = self.start_indices[idx]
        end = start + self.window_size
        return {
            "obs": self.observations[start:end],
            "nwp_u": torch.from_numpy(self.nwp_u.window(start, end)).float(),
            "nwp_v": torch.from_numpy(self.nwp_v.window(start, end)).float(),
            "start_index": start,
        }


@dataclass
class DataBundle:
    train_dataset: WindowedSequenceDataset
    val_dataset: WindowedSequenceDataset
    online_sequence: dict[str, torch.Tensor] | None
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


def _infer_hw_t_c_permutation(
    shape: tuple[int, ...],
    grid_size: tuple[int, int],
    total_channels: int,
) -> tuple[int, int, int, int]:
    if len(shape) != 4:
        raise ValueError(f"Expected a 4D NWP array, got shape {shape}.")

    for perm in permutations(range(4)):
        permuted_shape = tuple(shape[axis] for axis in perm)
        if tuple(permuted_shape[:2]) == grid_size and permuted_shape[-1] == total_channels:
            return tuple(int(axis) for axis in perm)

    raise ValueError(
        "Could not infer the NWP layout. "
        f"Observed shape: {shape}, expected grid {grid_size} and {total_channels} channels."
    )


def _flatten_station_vector(array: np.ndarray, expected_sites: int) -> np.ndarray:
    vector = np.asarray(array, dtype=np.float32).reshape(-1)
    if vector.shape[0] != expected_sites:
        raise ValueError(
            f"Expected {expected_sites} station coordinates, but found {vector.shape[0]} values."
        )
    return vector


def _lat_lon_to_local_km(coords_lat_lon: np.ndarray) -> np.ndarray:
    coords_lat_lon = np.asarray(coords_lat_lon, dtype=np.float32)
    if coords_lat_lon.ndim != 2 or coords_lat_lon.shape[1] != 2:
        raise ValueError(f"Expected [num_sites, 2] latitude/longitude coordinates, got {coords_lat_lon.shape}.")

    lat = coords_lat_lon[:, 0]
    lon = coords_lat_lon[:, 1]
    lat0 = float(lat.mean())
    lon0 = float(lon.mean())
    km_per_degree_lat = 111.32
    km_per_degree_lon = km_per_degree_lat * np.cos(np.deg2rad(lat0))
    x = (lon - lon0) * km_per_degree_lon
    y = (lat - lat0) * km_per_degree_lat
    return np.stack([x, y], axis=-1).astype(np.float32)


def _maybe_convert_coords_to_local_km(coords: np.ndarray, coord_cfg: dict[str, Any]) -> np.ndarray:
    if not bool(coord_cfg.get("convert_to_local_km", True)):
        return np.asarray(coords, dtype=np.float32)

    order = str(coord_cfg.get("order", "lat_lon")).lower()
    coords = np.asarray(coords, dtype=np.float32)
    if order == "lat_lon":
        lat_lon = coords
    elif order == "lon_lat":
        lat_lon = coords[:, [1, 0]]
    else:
        raise ValueError("data.coordinates.order must be 'lat_lon' or 'lon_lat'.")

    local_km = _lat_lon_to_local_km(lat_lon)
    scale_km = float(coord_cfg.get("local_km_scale", 50.0))
    if scale_km <= 0.0:
        raise ValueError("data.coordinates.local_km_scale must be positive.")
    return (local_km / scale_km).astype(np.float32)


def _load_site_coords(config: dict[str, Any]) -> np.ndarray:
    coord_cfg = config["data"]["coordinates"]
    num_sites = int(config["model"]["num_sites"])

    if coord_cfg.get("manual") is not None:
        coords = np.asarray(coord_cfg["manual"], dtype=np.float32)
        if coords.shape != (num_sites, 2):
            raise ValueError(
                f"Manual station coordinates must have shape {(num_sites, 2)}, got {coords.shape}."
            )
        return _maybe_convert_coords_to_local_km(coords, coord_cfg)

    lat = _flatten_station_vector(load_mat_variable(coord_cfg["source_path"], coord_cfg["lat_key"]), num_sites)
    lon = _flatten_station_vector(load_mat_variable(coord_cfg["source_path"], coord_cfg["lon_key"]), num_sites)
    return _maybe_convert_coords_to_local_km(np.stack([lat, lon], axis=-1), coord_cfg)


def _select_measurement_block(config: dict[str, Any], subset: str) -> np.ndarray:
    meas_cfg = config["data"]["measurement"]
    raw = load_mat_variable(meas_cfg[f"{subset}_path"], meas_cfg["variable_name"])
    raw = _ensure_time_feature_matrix(
        np.asarray(raw, dtype=np.float32),
        min_feature_count=max(meas_cfg["target_raw_indices"]) + 1,
    )
    block = raw[:, meas_cfg["target_raw_indices"]]
    block = block[:, meas_cfg["component_first_order"]]
    return _sanitize_array(block)


def _select_nwp_source(config: dict[str, Any], subset: str, channel_indices: list[int]) -> InMemorySequenceSource | LazyNwpSequenceSource:
    nwp_cfg = config["data"]["nwp"]
    path = nwp_cfg[f"{subset}_path"]

    if mat_utils.h5py is not None:
        try:
            return LazyNwpSequenceSource(
                path=path,
                var_name=nwp_cfg["variable_name"],
                grid_size=tuple(nwp_cfg["grid_size"]),
                total_channels=int(nwp_cfg["total_channels"]),
                channel_indices=list(channel_indices),
            )
        except Exception:
            pass

    raw = load_mat_variable(path, nwp_cfg["variable_name"])
    permutation = _infer_hw_t_c_permutation(tuple(np.asarray(raw).shape), tuple(nwp_cfg["grid_size"]), int(nwp_cfg["total_channels"]))
    raw = np.transpose(np.asarray(raw, dtype=np.float32), permutation)
    raw = raw[:, :, :, channel_indices]
    raw = np.transpose(raw, (2, 3, 0, 1))
    return InMemorySequenceSource(raw)


def _fit_nwp_standardizer(
    source: InMemorySequenceSource | LazyNwpSequenceSource,
    train_length: int,
    eps: float,
    chunk_size: int = 256,
) -> Standardizer:
    if train_length <= 0:
        raise ValueError("train_length must be positive when fitting NWP standardization.")

    total_count = 0.0
    sum_array: np.ndarray | None = None
    sumsq_array: np.ndarray | None = None

    for start in range(0, train_length, chunk_size):
        end = min(start + chunk_size, train_length)
        chunk = source.window(start, end).astype(np.float64, copy=False)
        chunk_sum = chunk.sum(axis=(0, 2, 3), keepdims=True)
        chunk_sumsq = (chunk * chunk).sum(axis=(0, 2, 3), keepdims=True)
        chunk_count = float(chunk.shape[0] * chunk.shape[2] * chunk.shape[3])

        if sum_array is None:
            sum_array = chunk_sum
            sumsq_array = chunk_sumsq
        else:
            sum_array += chunk_sum
            sumsq_array += chunk_sumsq
        total_count += chunk_count

    if sum_array is None or sumsq_array is None or total_count <= 0.0:
        raise RuntimeError("Failed to accumulate NWP standardization statistics.")

    mean = sum_array / total_count
    variance = np.maximum((sumsq_array / total_count) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(std < eps, 1.0, std)
    return Standardizer(mean=mean.astype(np.float32), std=std.astype(np.float32))


def _materialize_sequence(
    source: InMemorySequenceSource | LazyNwpSequenceSource | StandardizedSequenceSource | OffsetSequenceSource,
    chunk_size: int = 256,
) -> np.ndarray:
    chunks = []
    for start in range(0, len(source), chunk_size):
        end = min(start + chunk_size, len(source))
        chunks.append(source.window(start, end))
    if not chunks:
        return np.empty((0, source.num_channels, 0, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _build_feature_names(config: dict[str, Any]) -> list[str]:
    site_names = config["data"]["site_names"]
    component_names = config["data"]["component_names"]
    return [f"{component}_{site}_140m" for component in component_names for site in site_names]


def build_data_bundle(
    config: dict[str, Any],
    scaler_state: dict[str, dict[str, list]] | None = None,
    include_online: bool = True,
) -> DataBundle:
    obs_offline = _select_measurement_block(config, "offline")
    nwp_cfg = config["data"]["nwp"]
    nwp_u_offline_source = _select_nwp_source(config, "offline", list(nwp_cfg["u_channel_indices"]))
    nwp_v_offline_source = _select_nwp_source(config, "offline", list(nwp_cfg["v_channel_indices"]))

    offline_length = min(len(obs_offline), len(nwp_u_offline_source), len(nwp_v_offline_source))
    obs_offline = obs_offline[:offline_length]

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
            nwp_u_scaler = _fit_nwp_standardizer(nwp_u_offline_source, train_length=split_index, eps=eps)
            nwp_v_scaler = _fit_nwp_standardizer(nwp_v_offline_source, train_length=split_index, eps=eps)
        else:
            nwp_u_scaler = Standardizer.identity((1, nwp_u_offline_source.num_channels, 1, 1))
            nwp_v_scaler = Standardizer.identity((1, nwp_v_offline_source.num_channels, 1, 1))
    else:
        obs_scaler = Standardizer.from_state_dict(scaler_state["obs"])
        nwp_u_scaler = Standardizer.from_state_dict(scaler_state["nwp_u"])
        nwp_v_scaler = Standardizer.from_state_dict(scaler_state["nwp_v"])

    obs_offline = obs_scaler.transform(obs_offline)
    nwp_u_offline_standardized = StandardizedSequenceSource(nwp_u_offline_source, nwp_u_scaler)
    nwp_v_offline_standardized = StandardizedSequenceSource(nwp_v_offline_source, nwp_v_scaler)

    online_sequence: dict[str, torch.Tensor] | None = None
    if include_online:
        obs_online = _select_measurement_block(config, "online")
        nwp_u_online_source = _select_nwp_source(config, "online", list(nwp_cfg["u_channel_indices"]))
        nwp_v_online_source = _select_nwp_source(config, "online", list(nwp_cfg["v_channel_indices"]))
        online_length = min(len(obs_online), len(nwp_u_online_source), len(nwp_v_online_source))
        obs_online = obs_scaler.transform(obs_online[:online_length])
        online_u = _materialize_sequence(
            OffsetSequenceSource(StandardizedSequenceSource(nwp_u_online_source, nwp_u_scaler), 0, online_length)
        )
        online_v = _materialize_sequence(
            OffsetSequenceSource(StandardizedSequenceSource(nwp_v_online_source, nwp_v_scaler), 0, online_length)
        )
        online_sequence = {
            "obs": torch.from_numpy(obs_online[:online_length]).float(),
            "nwp_u": torch.from_numpy(online_u[:online_length]).float(),
            "nwp_v": torch.from_numpy(online_v[:online_length]).float(),
        }

    window_cfg = config["data"]["windows"]
    train_dataset = WindowedSequenceDataset(
        observations=obs_offline[:split_index],
        nwp_u=OffsetSequenceSource(nwp_u_offline_standardized, 0, split_index),
        nwp_v=OffsetSequenceSource(nwp_v_offline_standardized, 0, split_index),
        window_size=int(window_cfg["window_size"]),
        stride=int(window_cfg["stride"]),
    )
    val_dataset = WindowedSequenceDataset(
        observations=obs_offline[split_index:],
        nwp_u=OffsetSequenceSource(nwp_u_offline_standardized, split_index, len(obs_offline)),
        nwp_v=OffsetSequenceSource(nwp_v_offline_standardized, split_index, len(obs_offline)),
        window_size=int(window_cfg["window_size"]),
        stride=int(window_cfg["stride"]),
    )

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
