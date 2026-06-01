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
from bidsforge.processing.utils.matlab import mat_float, mat_root, mat_str, mat_str_list

from .result import TimeFrequencyProcessingResult


def load_time_frequency_result(path: Path | str) -> TimeFrequencyProcessingResult:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Time-frequency result file not found: {path}")
    if path.suffix.lower() == ".mat":
        return _load_from_matlab(path)
    return _load_from_hdf5(path)


def _load_from_hdf5(path: Path) -> TimeFrequencyProcessingResult:
    with h5py.File(path, "r") as fh:
        _require_hdf5_schema(fh, path.name)
        metadata = _load_hdf5_metadata(fh)
        raw_bids_path = str_scalar(dataset_or_none(fh, "provenance/raw_bids_path"), default=str(path))
        return TimeFrequencyProcessingResult(
            source_group=BIDSFileGroup(primary=BIDSFile.from_path(raw_bids_path)),
            metadata=metadata,
            power_db=np.asarray(fh["data/power_db"][:], dtype=np.float32),
            baseline_db=np.asarray(fh["data/baseline_db"][:], dtype=np.float32),
            trial_ids=decode_str_array(np.asarray(fh["axes/trial_id"][:], dtype=object)),
            channel_names=decode_str_array(np.asarray(fh["axes/channel"][:], dtype=object)),
            frequency_hz=np.asarray(fh["axes/frequency_hz"][:], dtype=np.float64),
            time_s=np.asarray(fh["axes/time_s"][:], dtype=np.float64),
            original_fs=float_scalar(dataset_or_none(fh, "meta/sampling_frequency_hz"), default=0.0),
            events=_load_hdf5_events(fh),
        )


def _load_from_matlab(path: Path) -> TimeFrequencyProcessingResult:
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    root = mat_root(mat, "time_frequency")
    meta = root.meta
    _require_matlab_schema(meta, path.name)
    raw_bids_path = mat_str(getattr(root.provenance, "raw_bids_path", None), default=str(path))
    return TimeFrequencyProcessingResult(
        source_group=BIDSFileGroup(primary=BIDSFile.from_path(raw_bids_path)),
        metadata=_load_matlab_metadata(meta),
        power_db=np.asarray(root.data.power_db, dtype=np.float32),
        baseline_db=np.asarray(root.data.baseline_db, dtype=np.float32),
        trial_ids=mat_str_list(getattr(root.axes, "trial_id", None)),
        channel_names=mat_str_list(getattr(root.axes, "channel", None)),
        frequency_hz=np.asarray(root.axes.frequency_hz, dtype=np.float64).ravel(),
        time_s=np.asarray(root.axes.time_s, dtype=np.float64).ravel(),
        original_fs=mat_float(getattr(meta, "sampling_frequency_hz", None), default=0.0),
        events=_load_matlab_events(getattr(root, "events", None)),
    )


def _load_hdf5_metadata(fh: h5py.File) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if "meta" not in fh:
        return metadata
    for key, node in fh["meta"].items():
        if isinstance(node, h5py.Dataset):
            if node.shape == ():
                value = node[()]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                metadata[key] = value
    return metadata


def _load_matlab_metadata(meta: object) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "apply_baseline",
        "dimension_order",
        "baseline_dimension_order",
        "time_frequency_method",
        "fieldtrip_polyorder",
        "fieldtrip_pad_s",
        "fieldtrip_padtype",
        "fieldtrip_scaling",
    ):
        value = getattr(meta, key, None)
        if value is not None:
            if key in {"fieldtrip_polyorder", "fieldtrip_pad_s"}:
                out[key] = mat_float(value, default=float("nan"))
            elif key in {"apply_baseline", "fieldtrip_scaling"}:
                out[key] = bool(np.asarray(value).ravel()[0])
            else:
                out[key] = mat_str(value, default="")
    return out


def _load_hdf5_events(fh: h5py.File) -> list[dict[str, Any]] | None:
    if "events" not in fh:
        return None
    events_group = fh["events"]
    onsets = np.asarray(events_group["onset"][:], dtype=np.float64)
    durations = np.asarray(events_group["duration"][:], dtype=np.float64)
    types = decode_str_array(np.asarray(events_group["type"][:], dtype=object))
    descriptions = decode_str_array(np.asarray(events_group["description"][:], dtype=object))
    return [
        {"onset": float(o), "duration": float(d), "type": t, "description": desc}
        for o, d, t, desc in zip(onsets, durations, types, descriptions)
    ]


def _load_matlab_events(events: object | None) -> list[dict[str, Any]] | None:
    if events is None:
        return None
    onsets = np.asarray(getattr(events, "onset", []), dtype=np.float64).ravel()
    durations = np.asarray(getattr(events, "duration", []), dtype=np.float64).ravel()
    types = mat_str_list(getattr(events, "type", None))
    descriptions = mat_str_list(getattr(events, "description", None))
    return [
        {"onset": float(o), "duration": float(d), "type": t, "description": desc}
        for o, d, t, desc in zip(onsets, durations, types, descriptions)
    ]


def _require_hdf5_schema(fh: h5py.File, path_name: str) -> None:
    schema_name = str_scalar(dataset_or_none(fh, "meta/schema_name"), default="")
    schema_version = str_scalar(dataset_or_none(fh, "meta/schema_version"), default="")
    if schema_name != "time_frequency" or schema_version != "1.0":
        raise ValueError(
            f"{path_name}: unsupported time-frequency schema "
            f"(schema_name={schema_name!r}, schema_version={schema_version!r})."
        )


def _require_matlab_schema(meta: object, path_name: str) -> None:
    schema_name = mat_str(getattr(meta, "schema_name", None), default="")
    schema_version = mat_str(getattr(meta, "schema_version", None), default="")
    if schema_name != "time_frequency" or schema_version != "1.0":
        raise ValueError(
            f"{path_name}: unsupported time-frequency schema "
            f"(schema_name={schema_name!r}, schema_version={schema_version!r})."
        )
