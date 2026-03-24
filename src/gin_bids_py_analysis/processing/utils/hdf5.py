from __future__ import annotations

from typing import Any

import h5py
import numpy as np


def dataset_or_none(root: Any, key: str) -> h5py.Dataset | None:
    """Return the dataset located at slash-separated *key*, or None."""
    node: Any = root
    for part in key.split("/"):
        if part not in node:
            return None
        node = node[part]
    if isinstance(node, h5py.Dataset):
        return node
    return None


def decode_str_array(values: np.ndarray) -> list[str]:
    """Decode a numpy array containing bytes/strings into Python strings."""
    decoded: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def float_scalar(dataset: h5py.Dataset | None, default: float) -> float:
    if dataset is None:
        return default
    try:
        return float(dataset[()])
    except Exception:
        return default


def int_scalar(dataset: h5py.Dataset | None, default: int) -> int:
    if dataset is None:
        return default
    try:
        return int(dataset[()])
    except Exception:
        return default


def str_scalar(dataset: h5py.Dataset | None, default: str) -> str:
    if dataset is None:
        return default
    try:
        return str(dataset.asstr()[()])
    except Exception:
        return default


def coerce_feature_time(
    values: np.ndarray,
    *,
    n_features: int,
    n_times: int,
) -> np.ndarray:
    """
    Coerce 1D/2D inputs into a (n_features, n_times) float array.

    Extra values are truncated and missing values are filled with NaN.
    """
    arr = np.asarray(values, dtype=np.float64)
    out = np.full((n_features, n_times), np.nan, dtype=np.float64)

    if arr.ndim == 2:
        if arr.shape == (n_features, n_times):
            return arr
        if arr.shape == (n_times, n_features):
            return arr.T
        rows = min(n_features, arr.shape[0])
        cols = min(n_times, arr.shape[1])
        out[:rows, :cols] = arr[:rows, :cols]
        return out

    if arr.ndim == 1:
        if n_features == 1 and arr.size == n_times:
            return arr.reshape(1, n_times)
        if n_times == 1 and arr.size == n_features:
            return arr.reshape(n_features, 1)
        if arr.size == (n_features * n_times):
            return arr.reshape(n_features, n_times)
        flat = out.reshape(-1)
        limit = min(flat.size, arr.size)
        flat[:limit] = arr[:limit]
        return out

    raise ValueError(
        f"Metric dataset has unsupported shape {arr.shape!r}; expected 1D or 2D array."
    )
