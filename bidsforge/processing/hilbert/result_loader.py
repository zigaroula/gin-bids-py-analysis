"""Load pre-computed Hilbert processing results from disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.processing.utils.hdf5 import (
    dataset_or_none,
    decode_str_array,
    float_scalar,
    str_scalar,
)
from bidsforge.processing.utils.matlab import (
    mat_float,
    mat_root,
    mat_str,
    mat_str_list,
)

from .result import HilbertProcessingResult


def load_hilbert_result(path: Path | str) -> HilbertProcessingResult:
    """Load a pre-computed ``HilbertProcessingResult`` from *path*."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Hilbert result file not found: {path}")
    if path.suffix.lower() == ".mat":
        return _load_from_matlab(path)
    return _load_from_hdf5(path)


def _load_from_hdf5(path: Path) -> HilbertProcessingResult:
    with h5py.File(path, "r") as fh:
        _require_hdf5_schema(fh, path.name)
        envelope = np.asarray(fh["data/envelope"][:], dtype=np.float32)
        windows = np.asarray(fh["axes/smoothing_window_ms"][:], dtype=np.int64).ravel()
        smoothed = {
            int(window): envelope[idx]
            for idx, window in enumerate(windows)
        }
        channel_names = decode_str_array(np.asarray(fh["axes/channel"][:], dtype=object))
        bins = np.asarray(fh["meta/band_limits_hz"][:], dtype=np.float64).ravel().tolist()

        metadata = _load_hdf5_metadata(fh)
        raw_bids_path = str_scalar(
            dataset_or_none(fh, "provenance/raw_bids_path"),
            default=str(path),
        )
        events = _load_hdf5_events(fh)

        return HilbertProcessingResult(
            source_group=BIDSFileGroup(primary=BIDSFile.from_path(raw_bids_path)),
            metadata=metadata,
            smoothed=smoothed,
            channel_names=channel_names,
            bins=bins,
            downsampled_fs=float_scalar(
                dataset_or_none(fh, "meta/downsampled_frequency_hz"),
                default=0.0,
            ),
            original_fs=float_scalar(
                dataset_or_none(fh, "meta/sampling_frequency_hz"),
                default=0.0,
            ),
            events=events,
        )


def _load_from_matlab(path: Path) -> HilbertProcessingResult:
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    data = mat_root(mat, "hilbert")
    meta = data.meta
    _require_matlab_schema(meta, path.name)

    windows = np.asarray(data.axes.smoothing_window_ms, dtype=np.int64).ravel()
    channel_names = mat_str_list(getattr(data.axes, "channel", None))
    envelope = _normalize_matlab_envelope(
        np.asarray(data.data.envelope, dtype=np.float32),
        n_windows=len(windows),
        n_channels=len(channel_names),
        path_name=path.name,
    )
    smoothed = {
        int(window): envelope[idx]
        for idx, window in enumerate(windows)
    }
    bins = np.asarray(meta.band_limits_hz, dtype=np.float64).ravel().tolist()
    metadata = _load_matlab_metadata(meta)
    raw_bids_path = mat_str(
        getattr(data.provenance, "raw_bids_path", None),
        default=str(path),
    )
    events = _load_matlab_events(getattr(data, "events", None))

    return HilbertProcessingResult(
        source_group=BIDSFileGroup(primary=BIDSFile.from_path(raw_bids_path)),
        metadata=metadata,
        smoothed=smoothed,
        channel_names=channel_names,
        bins=bins,
        downsampled_fs=mat_float(
            getattr(meta, "downsampled_frequency_hz", None),
            default=0.0,
        ),
        original_fs=mat_float(
            getattr(meta, "sampling_frequency_hz", None),
            default=0.0,
        ),
        events=events,
    )


def _normalize_matlab_envelope(
    envelope: np.ndarray,
    *,
    n_windows: int,
    n_channels: int,
    path_name: str,
) -> np.ndarray:
    """Return MATLAB-loaded envelope data as [window, channel, time]."""
    arr = np.asarray(envelope, dtype=np.float32)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2:
        if n_windows == 1 and arr.shape[0] == n_channels:
            return arr.reshape(1, arr.shape[0], arr.shape[1])
        if n_channels == 1 and arr.shape[0] == n_windows:
            return arr.reshape(arr.shape[0], 1, arr.shape[1])
        if n_windows == 1 and n_channels == 1:
            return arr.reshape(1, 1, -1)
    if arr.ndim == 1:
        if n_windows == 1 and n_channels == 1:
            return arr.reshape(1, 1, arr.shape[0])
        if n_windows == 1 and n_channels > 0 and arr.size % n_channels == 0:
            return arr.reshape(1, n_channels, arr.size // n_channels)
        if n_channels == 1 and n_windows > 0 and arr.size % n_windows == 0:
            return arr.reshape(n_windows, 1, arr.size // n_windows)

    raise ValueError(
        f"{path_name}: unsupported MATLAB Hilbert envelope shape {arr.shape}; "
        "expected data shaped as smoothing_window x channel x time."
    )


def _load_hdf5_metadata(fh: h5py.File) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "montage_mode": str_scalar(dataset_or_none(fh, "meta/montage_mode"), default=""),
        "unit": str_scalar(dataset_or_none(fh, "meta/unit"), default="amplitude"),
        "centered": bool(dataset_or_none(fh, "meta/centered")[()])
        if dataset_or_none(fh, "meta/centered") is not None
        else False,
    }
    for key in (
        "events_source_requested",
        "events_source_resolved",
        "events_onset_precision",
        "events_file",
        "event_sample_shift_samples",
    ):
        value = str_scalar(dataset_or_none(fh, f"meta/{key}"), default="")
        if value:
            metadata[key] = value
    return metadata


def _load_matlab_metadata(meta: object) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "montage_mode": mat_str(getattr(meta, "montage_mode", None), default=""),
        "unit": mat_str(getattr(meta, "unit", None), default="amplitude"),
        "centered": bool(np.asarray(getattr(meta, "centered", False)).ravel()[0]),
    }
    for key in (
        "events_source_requested",
        "events_source_resolved",
        "events_onset_precision",
        "events_file",
        "event_sample_shift_samples",
    ):
        value = mat_str(getattr(meta, key, None), default="")
        if value:
            metadata[key] = value
    return metadata


def _load_hdf5_events(fh: h5py.File) -> list[dict[str, Any]] | None:
    if "events" not in fh:
        return None
    events_group = fh["events"]
    onsets = np.asarray(events_group["onset"][:], dtype=np.int64)
    durations = np.asarray(events_group["duration"][:], dtype=np.int64)
    types = decode_str_array(np.asarray(events_group["type"][:], dtype=object))
    descriptions = decode_str_array(
        np.asarray(events_group["description"][:], dtype=object)
    )
    return [
        {
            "onset": int(onset),
            "duration": int(duration),
            "type": event_type,
            "description": description,
        }
        for onset, duration, event_type, description in zip(
            onsets,
            durations,
            types,
            descriptions,
        )
    ]


def _load_matlab_events(events: object | None) -> list[dict[str, Any]] | None:
    if events is None:
        return None
    onsets = np.asarray(getattr(events, "onset", []), dtype=np.int64).ravel()
    durations = np.asarray(getattr(events, "duration", []), dtype=np.int64).ravel()
    types = mat_str_list(getattr(events, "type", None))
    descriptions = mat_str_list(getattr(events, "description", None))
    return [
        {
            "onset": int(onset),
            "duration": int(duration),
            "type": event_type,
            "description": description,
        }
        for onset, duration, event_type, description in zip(
            onsets,
            durations,
            types,
            descriptions,
        )
    ]


def _require_hdf5_schema(fh: h5py.File, path_name: str) -> None:
    schema_name = str_scalar(dataset_or_none(fh, "meta/schema_name"), default="")
    schema_version = str_scalar(dataset_or_none(fh, "meta/schema_version"), default="")
    if schema_name != "hilbert" or schema_version != "1.0":
        raise ValueError(
            f"{path_name}: unsupported Hilbert schema "
            f"(schema_name={schema_name!r}, schema_version={schema_version!r})."
        )


def _require_matlab_schema(meta: object, path_name: str) -> None:
    schema_name = mat_str(getattr(meta, "schema_name", None), default="")
    schema_version = mat_str(getattr(meta, "schema_version", None), default="")
    if schema_name != "hilbert" or schema_version != "1.0":
        raise ValueError(
            f"{path_name}: unsupported Hilbert schema "
            f"(schema_name={schema_name!r}, schema_version={schema_version!r})."
        )
