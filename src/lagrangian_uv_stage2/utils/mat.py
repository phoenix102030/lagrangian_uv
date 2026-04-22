from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

try:
    from scipy import io as scipy_io
except ImportError:  # pragma: no cover
    scipy_io = None


_MATLAB_PRIVATE_KEYS = {"__header__", "__version__", "__globals__"}


def _pick_key(keys: list[str], requested: str | None) -> str:
    if requested is not None:
        if requested not in keys:
            raise KeyError(f"Variable {requested!r} not found. Available keys: {keys}")
        return requested

    public_keys = [key for key in keys if key not in _MATLAB_PRIVATE_KEYS and not key.startswith("#")]
    if len(public_keys) != 1:
        raise KeyError(
            "Could not infer a MATLAB variable automatically. "
            f"Available keys: {public_keys or keys}"
        )
    return public_keys[0]


def load_mat_variable(path: str | Path, var_name: str | None = None) -> np.ndarray:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    errors: list[str] = []

    if h5py is not None:
        try:
            with h5py.File(path, "r") as handle:
                key = _pick_key(list(handle.keys()), var_name)
                return np.asarray(handle[key])
        except Exception as exc:  # pragma: no cover
            errors.append(f"h5py failed: {exc}")

    if scipy_io is not None:
        try:
            content = scipy_io.loadmat(path)
            key = _pick_key(list(content.keys()), var_name)
            return np.asarray(content[key])
        except Exception as exc:  # pragma: no cover
            errors.append(f"scipy.io.loadmat failed: {exc}")

    dependency_hint = (
        "Install at least one MATLAB reader dependency: `pip install h5py scipy`."
    )
    details = " | ".join(errors) if errors else "No compatible MATLAB reader was available."
    raise RuntimeError(f"Unable to read {path}. {details} {dependency_hint}")
