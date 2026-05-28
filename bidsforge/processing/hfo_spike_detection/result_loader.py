"""Load pre-computed HFO/spike detection results from disk."""

from __future__ import annotations

import csv
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

from .result import HfoSpikeDetectorProcessingResult


def load_hfo_spike_detection_result(path: Path | str) -> HfoSpikeDetectorProcessingResult:
    """Load a pre-computed ``HfoSpikeDetectorProcessingResult`` from *path*."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HFO/spike detection result file not found: {path}")
    ext = path.suffix.lower()
    if ext == ".mat":
        return _load_from_matlab(path)
    if ext == ".tsv":
        return _load_from_tsv(path)
    return _load_from_hdf5(path)


def _load_from_hdf5(path: Path) -> HfoSpikeDetectorProcessingResult:
    with h5py.File(path, "r") as fh:
        _require_hdf5_schema(fh, path.name)
        event_group = fh["events"]
        markers = _markers_from_arrays(
            onsets=np.asarray(event_group["onset"][:], dtype=np.float64),
            durations=np.asarray(event_group["duration"][:], dtype=np.float64),
            channels=decode_str_array(np.asarray(event_group["channel"][:], dtype=object)),
            event_types=decode_str_array(np.asarray(event_group["event_type"][:], dtype=object)),
            peak_freqs=np.asarray(event_group["peak_frequency"][:], dtype=np.float64),
            sample_indices=np.asarray(event_group["sample_index"][:], dtype=np.int64),
            freq_indices=np.asarray(event_group["frequency_index"][:], dtype=np.int64),
            strengths=np.asarray(event_group["detection_strength"][:], dtype=np.float64),
            colors=decode_str_array(np.asarray(event_group["color"][:], dtype=object))
            if "color" in event_group
            else None,
        )
        channel_names = decode_str_array(
            np.asarray(fh["axes/channel_names"][:], dtype=object)
        )
        freq_band_ds = dataset_or_none(fh, "counts/freq_band")
        n_spk_ds = dataset_or_none(fh, "counts/n_spk")
        n_osc_ds = dataset_or_none(fh, "counts/n_osc")
        detection_charac_ds = dataset_or_none(fh, "features/detection_charac")
        raw_bids_path = str_scalar(
            dataset_or_none(fh, "provenance/raw_bids_path"),
            default=str(path),
        )
        original_fs = float_scalar(dataset_or_none(fh, "meta/original_fs"), default=0.0)
        duration_seconds = float_scalar(
            dataset_or_none(fh, "meta/duration_seconds"),
            default=0.0,
        )
        n_samples = int(round(duration_seconds * original_fs)) if original_fs > 0 else 0
        freq_band = (
            np.asarray(freq_band_ds[:], dtype=np.float32)
            if freq_band_ds is not None
            else np.array([])
        )
        if freq_band.ndim == 1 and freq_band.size == 2:
            freq_band = freq_band.reshape(1, 2)

        return HfoSpikeDetectorProcessingResult(
            source_group=BIDSFileGroup(primary=BIDSFile.from_path(raw_bids_path)),
            metadata={
                "montage_mode": str_scalar(
                    dataset_or_none(fh, "meta/montage_mode"),
                    default="",
                ),
                "n_samples": n_samples,
            },
            markers=markers,
            channel_names=channel_names,
            freq_band=freq_band,
            n_spk=(
                np.asarray(n_spk_ds[:], dtype=np.int64).ravel()
                if n_spk_ds is not None
                else np.array([])
            ),
            n_osc=_mat_array_or_empty(
                np.asarray(n_osc_ds[:], dtype=np.int64)
                if n_osc_ds is not None
                else None,
                dtype=np.int64,
            ),
            detection_charac=(
                np.asarray(detection_charac_ds[:], dtype=np.float64)
                if detection_charac_ds is not None
                else np.array([])
            ),
            original_fs=original_fs,
        )


def _load_from_matlab(path: Path) -> HfoSpikeDetectorProcessingResult:
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    data = mat_root(mat, "hfo_spike_detection")
    meta = data.meta
    _require_matlab_schema(meta, path.name)
    events = data.events
    markers = _markers_from_arrays(
        onsets=np.asarray(getattr(events, "onset", []), dtype=np.float64).ravel(),
        durations=np.asarray(getattr(events, "duration", []), dtype=np.float64).ravel(),
        channels=mat_str_list(getattr(events, "channel", None)),
        event_types=mat_str_list(getattr(events, "event_type", None)),
        peak_freqs=np.asarray(getattr(events, "peak_frequency", []), dtype=np.float64).ravel(),
        sample_indices=np.asarray(getattr(events, "sample_index", []), dtype=np.int64).ravel(),
        freq_indices=np.asarray(getattr(events, "frequency_index", []), dtype=np.int64).ravel(),
        strengths=np.asarray(getattr(events, "detection_strength", []), dtype=np.float64).ravel(),
        colors=mat_str_list(getattr(events, "color", None)),
    )
    original_fs = mat_float(getattr(meta, "original_fs", None), default=0.0)
    duration_seconds = mat_float(getattr(meta, "duration_seconds", None), default=0.0)
    n_samples = int(round(duration_seconds * original_fs)) if original_fs > 0 else 0
    raw_bids_path = mat_str(
        getattr(data.provenance, "raw_bids_path", None),
        default=str(path),
    )
    freq_band = _mat_array_or_empty(
        getattr(data.counts, "freq_band", None),
        dtype=np.float32,
    )
    if freq_band.ndim == 1 and freq_band.size == 2:
        freq_band = freq_band.reshape(1, 2)

    return HfoSpikeDetectorProcessingResult(
        source_group=BIDSFileGroup(primary=BIDSFile.from_path(raw_bids_path)),
        metadata={
            "montage_mode": mat_str(getattr(meta, "montage_mode", None), default=""),
            "n_samples": n_samples,
        },
        markers=markers,
        channel_names=mat_str_list(getattr(data.axes, "channel_names", None)),
        freq_band=freq_band,
        n_spk=_mat_array_or_empty(getattr(data.counts, "n_spk", None), dtype=np.int64).ravel(),
        n_osc=_mat_array_or_empty(getattr(data.counts, "n_osc", None), dtype=np.int64),
        detection_charac=_mat_array_or_empty(
            getattr(data.features, "detection_charac", None),
            dtype=np.float64,
        ),
        original_fs=original_fs,
    )


def _load_from_tsv(path: Path) -> HfoSpikeDetectorProcessingResult:
    markers: list[dict[str, Any]] = []
    channels: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            marker = {
                "onset_time_seconds": _float_from_row(row, "onset"),
                "duration": _float_from_row(row, "duration"),
                "channel_label": row.get("channel", ""),
                "event_type": row.get("event_type", ""),
                "peak_frequency_hz": _float_from_row(row, "peak_frequency"),
                "sample_index": _int_from_row(row, "sample_index"),
                "frequency_index": _int_from_row(row, "frequency_index"),
                "detection_strength": _float_from_row(row, "detection_strength"),
                "visualization_color": row.get("color", "#808080") or "#808080",
            }
            markers.append(marker)
            if marker["channel_label"] and marker["channel_label"] not in channels:
                channels.append(marker["channel_label"])

    return HfoSpikeDetectorProcessingResult(
        source_group=BIDSFileGroup(primary=BIDSFile.from_path(path)),
        markers=markers,
        channel_names=channels,
    )


def _markers_from_arrays(
    *,
    onsets: np.ndarray,
    durations: np.ndarray,
    channels: list[str],
    event_types: list[str],
    peak_freqs: np.ndarray,
    sample_indices: np.ndarray,
    freq_indices: np.ndarray,
    strengths: np.ndarray,
    colors: list[str] | None,
) -> list[dict[str, Any]]:
    if colors is None:
        colors = ["#808080"] * len(onsets)
    return [
        {
            "onset_time_seconds": float(onset),
            "duration": float(duration),
            "channel_label": channel,
            "event_type": event_type,
            "peak_frequency_hz": float(peak_freq),
            "sample_index": int(sample_index),
            "frequency_index": int(freq_index),
            "detection_strength": float(strength),
            "visualization_color": color,
        }
        for (
            onset,
            duration,
            channel,
            event_type,
            peak_freq,
            sample_index,
            freq_index,
            strength,
            color,
        ) in zip(
            onsets,
            durations,
            channels,
            event_types,
            peak_freqs,
            sample_indices,
            freq_indices,
            strengths,
            colors,
        )
    ]


def _mat_array_or_empty(value: object, *, dtype: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    arr = np.asarray(value, dtype=dtype)
    if arr.size == 0:
        return np.array([])
    return arr


def _float_from_row(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def _int_from_row(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "") or 0))
    except ValueError:
        return 0


def _require_hdf5_schema(fh: h5py.File, path_name: str) -> None:
    schema_name = str_scalar(dataset_or_none(fh, "meta/schema_name"), default="")
    schema_version = str_scalar(dataset_or_none(fh, "meta/schema_version"), default="")
    if schema_name != "hfo_spike_detection" or schema_version != "2.1":
        raise ValueError(
            f"{path_name}: unsupported HFO/spike detection schema "
            f"(schema_name={schema_name!r}, schema_version={schema_version!r})."
        )


def _require_matlab_schema(meta: object, path_name: str) -> None:
    schema_name = mat_str(getattr(meta, "schema_name", None), default="")
    schema_version = mat_str(getattr(meta, "schema_version", None), default="")
    if schema_name != "hfo_spike_detection" or schema_version != "2.1":
        raise ValueError(
            f"{path_name}: unsupported HFO/spike detection schema "
            f"(schema_name={schema_name!r}, schema_version={schema_version!r})."
        )
