#!/usr/bin/env python3
"""
Compare MATLAB b3 regression outputs against Python regression outputs.

Default configuration targets subject GRE_2021_AICb:
- MATLAB b3 folders under C:\\GRE\\dev\\clarissa\\seeg\\b3_BPF_indiv_analyses
- Python regression output under bids/derivatives/regression

The script compares, by channel and time point:
- MATLAB log.<reg_name>.<realign>.<f_range>.<smoothing>.dots
- Python regression.condition_[a|b].slope

Usage:
    .venv\\Scripts\\python scripts\\compare_b3_vs_python_regression.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import scipy.io

from gin_bids_py_analysis.bids import BIDSDataset
from gin_bids_py_analysis.processing.trial_stats.regression.stats import (
    compute_linear_regression_maps,
)
from gin_bids_py_analysis.processing.trial_stats.regression import (
    RegressionProcessing,
    load_regression_result,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from trial_slope_shared import (  # noqa: E402
    BIDS_ROOT as PYTHON_SOURCE_BIDS_ROOT,
    PARAMS as PYTHON_REGRESSION_PARAMS,
    RESOLVER as PYTHON_REGRESSION_RESOLVER,
    build_trial_annotators,
    build_trial_slope_groups,
    load_roi_channels_from_csv,
)
from compare_subject_config import (  # noqa: E402
    BEHAVIOR_TSV_PATH,
    MATLAB_B2_PATH,
    MATLAB_B3_ROOT,
    PYTHON_REGRESSION_PATH,
)


# ---------------------------------------------------------------------------
# File paths and MATLAB path selection
# ---------------------------------------------------------------------------

MATLAB_LOG_DATA_FILENAME = "log_data.mat"
MATLAB_REALIGN = "onset"
MATLAB_F_RANGE_NAME = "f50f150"
MATLAB_SMOOTHING_NAME = "sm250"
MATLAB_TERM_INDEX = 0

# MATLAB regression name -> Python condition field mapping.
MATLAB_PYTHON_REGRESSION_MAP = [
    ("P_Rating", "condition_a"),
    ("UP_Rating", "condition_b"),
]

TOP_N = 12

# ---------------------------------------------------------------------------
# Bundle selection
# ---------------------------------------------------------------------------
# Set to True/False to include or skip each comparison bundle in the viewer.

# Replay OLS on Python epochs (re-runs regression on the Python-extracted epochs).
INCLUDE_REPLAY_PYTHON_EPOCHS: bool = False

# Replay OLS on MATLAB b2 epochs (re-runs regression on MATLAB-extracted epochs).
INCLUDE_REPLAY_MATLAB_B2: bool = False

# Direct comparison: saved Python HDF5 slopes vs MATLAB b3 slopes.
INCLUDE_HDF5_VS_MATLAB: bool = True
FLIP_UNPLEASANT_MATLAB_IN_HDF5_VS_MATLAB: bool = True


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

_FIRST_CONTACT_RE = re.compile(r"^([A-Za-z]+\d+)")


def _squeeze(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        while obj.ndim > 0 and obj.size == 1:
            obj = obj.flat[0]
    return obj


def _mat_attr(struct: object, *names: str) -> object | None:
    for name in names:
        val = getattr(struct, name, None)
        if val is not None:
            return _squeeze(val)
    return None


def _first_contact(name: str) -> str:
    match = _FIRST_CONTACT_RE.match(str(name).strip())
    return match.group(1) if match is not None else str(name).strip()


def _norm(name: str) -> str:
    return re.sub(r"[\s\-_\.]", "", str(name).strip()).casefold()


def match_channels(
    matlab_channels: list[str],
    python_channels: list[str],
) -> dict[int, int]:
    """Match MATLAB bipoles to Python channels using the first contact."""
    python_by_norm = {_norm(ch): i for i, ch in enumerate(python_channels)}
    mapping: dict[int, int] = {}
    for mat_idx, mat_name in enumerate(matlab_channels):
        key = _norm(_first_contact(mat_name))
        if key in python_by_norm:
            mapping[mat_idx] = python_by_norm[key]
    return mapping


def match_exact_channels(
    source_channels: list[str],
    target_channels: list[str],
) -> dict[int, int]:
    """Match channels by exact normalized name."""
    target_by_norm = {_norm(ch): i for i, ch in enumerate(target_channels)}
    mapping: dict[int, int] = {}
    for src_idx, src_name in enumerate(source_channels):
        key = _norm(src_name)
        if key in target_by_norm:
            mapping[src_idx] = target_by_norm[key]
    return mapping


def _extract_channels_scipy(hdr: object) -> list[str]:
    raw = getattr(hdr, "chan_info", None)
    if raw is not None:
        arr = np.asarray(raw)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return [str(arr[i, 1]).strip() for i in range(arr.shape[0])]
        if arr.ndim == 1:
            names: list[str] = []
            for row in arr:
                row_arr = np.asarray(row).flatten()
                if len(row_arr) >= 2:
                    names.append(str(row_arr[1]).strip())
            if names:
                return names
    for fallback_name in ("chanlabels", "label"):
        fallback = getattr(hdr, fallback_name, None)
        if fallback is not None:
            return [str(el).strip() for el in np.asarray(fallback).flatten()]
    return []


def _read_h5_char_dataset(dataset: object) -> str:
    import h5py  # noqa: PLC0415

    ds = dataset
    if h5py.check_string_dtype(ds.dtype):
        raw = ds.asstr()[()]
        arr = np.asarray(raw).flatten()
        return str(arr[0]).strip() if arr.size else ""
    data = np.asarray(ds[()])
    return "".join(chr(int(c)) for c in data.flatten()).strip()


def _read_h5_cell_strings(f: object, dataset: object) -> list[str]:
    import h5py  # noqa: PLC0415

    ds = dataset
    if h5py.check_string_dtype(ds.dtype):
        raw = ds.asstr()[()]
        return [str(s).strip() for s in np.asarray(raw).flatten()]
    refs = np.asarray(ds[()])
    refs = np.transpose(refs)
    out: list[str] = []
    for ref in refs.flatten():
        out.append(_read_h5_char_dataset(f[ref]))
    return out


def _h5_deref_scalar(f: object, obj: object) -> object:
    import h5py  # noqa: PLC0415

    if isinstance(obj, h5py.Dataset) and obj.dtype.kind == "O":
        refs = np.asarray(obj[()])
        if refs.size == 1:
            return f[refs.flat[0]]
    return obj


def _h5_struct_field(f: object, struct_obj: object, field: str) -> object:
    import h5py  # noqa: PLC0415

    obj = _h5_deref_scalar(f, struct_obj)
    if not isinstance(obj, h5py.Group):
        raise KeyError(f"Expected an HDF5 group while resolving field {field!r}.")
    if field not in obj:
        raise KeyError(f"Field {field!r} not found. Available: {list(obj.keys())!r}")
    return _h5_deref_scalar(f, obj[field])


def _h5_numeric_array(dataset: object, *, dtype: type = np.float64) -> np.ndarray:
    arr = np.asarray(dataset[()], dtype=dtype)
    arr = np.transpose(arr)
    return np.asarray(arr, dtype=dtype)


def _extract_channels_hdf5(f: object, hdr_obj: object) -> list[str]:
    import h5py  # noqa: PLC0415

    hdr = _h5_deref_scalar(f, hdr_obj)
    if not isinstance(hdr, h5py.Group):
        return []
    if "chan_info" in hdr:
        chan_info = hdr["chan_info"]
        refs = np.asarray(chan_info[()])
        if refs.ndim == 2:
            refs = np.transpose(refs)
            if refs.shape[1] >= 2:
                return [_read_h5_char_dataset(f[ref]) for ref in refs[:, 1]]
    for field in ("chanlabels", "label"):
        if field in hdr:
            return _read_h5_cell_strings(f, hdr[field])
    return []


# ---------------------------------------------------------------------------
# MATLAB b3 loader
# ---------------------------------------------------------------------------

def _select_term_matrix(dots: np.ndarray, *, term_index: int) -> np.ndarray:
    arr = np.asarray(dots, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if term_index < 0 or term_index >= arr.shape[2]:
            raise IndexError(
                f"term_index={term_index} is outside dots.shape[2]={arr.shape[2]}."
            )
        return np.asarray(arr[:, :, term_index], dtype=np.float64)
    raise ValueError(f"Unexpected dots shape: {arr.shape!r}")


def _load_matlab_b3_scipy(
    path: Path,
    *,
    regression_name: str,
    realign: str,
    f_range_name: str,
    smoothing_name: str,
    term_index: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    log = mat.get("log")
    hdr = mat.get("hdr")
    if log is None:
        raise KeyError(f"{path.name}: missing 'log' variable.")
    if hdr is None:
        raise KeyError(f"{path.name}: missing 'hdr' variable.")

    reg = getattr(log, regression_name, None)
    if reg is None:
        raise KeyError(f"{path.name}: regression {regression_name!r} not found in log.")
    realign_node = getattr(reg, realign, None)
    if realign_node is None:
        raise KeyError(f"{path.name}: realign {realign!r} not found under {regression_name!r}.")
    freq_node = getattr(realign_node, f_range_name, None)
    if freq_node is None:
        raise KeyError(f"{path.name}: band {f_range_name!r} not found under {regression_name!r}/{realign!r}.")
    smooth_node = getattr(freq_node, smoothing_name, None)
    if smooth_node is None:
        raise KeyError(
            f"{path.name}: smoothing {smoothing_name!r} not found under "
            f"{regression_name!r}/{realign!r}/{f_range_name!r}."
        )

    dots = _select_term_matrix(
        np.asarray(getattr(smooth_node, "dots"), dtype=np.float64),
        term_index=term_index,
    )
    time_node = getattr(realign_node, "time", None)
    if time_node is None:
        raise KeyError(f"{path.name}: missing log.{regression_name}.{realign}.time.timelist.")
    timelist = np.asarray(getattr(time_node, "timelist"), dtype=np.float64).ravel()
    channels = _extract_channels_scipy(hdr)
    return dots, timelist, channels


def _load_matlab_b3_hdf5(
    path: Path,
    *,
    regression_name: str,
    realign: str,
    f_range_name: str,
    smoothing_name: str,
    term_index: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import h5py  # noqa: PLC0415

    with h5py.File(str(path), "r") as f:
        log = _h5_deref_scalar(f, f["log"])
        hdr = _h5_deref_scalar(f, f["hdr"])

        reg = _h5_struct_field(f, log, regression_name)
        realign_node = _h5_struct_field(f, reg, realign)
        freq_node = _h5_struct_field(f, realign_node, f_range_name)
        smooth_node = _h5_struct_field(f, freq_node, smoothing_name)
        dots_ds = _h5_struct_field(f, smooth_node, "dots")
        dots = _select_term_matrix(
            _h5_numeric_array(dots_ds, dtype=np.float64),
            term_index=term_index,
        )

        time_node = _h5_struct_field(f, realign_node, "time")
        timelist_ds = _h5_struct_field(f, time_node, "timelist")
        timelist = _h5_numeric_array(timelist_ds, dtype=np.float64).ravel()

        channels = _extract_channels_hdf5(f, hdr)
        return dots, timelist, channels


def load_matlab_b3_regression(
    path: Path,
    *,
    regression_name: str,
    realign: str,
    f_range_name: str,
    smoothing_name: str,
    term_index: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    try:
        return _load_matlab_b3_scipy(
            path,
            regression_name=regression_name,
            realign=realign,
            f_range_name=f_range_name,
            smoothing_name=smoothing_name,
            term_index=term_index,
        )
    except NotImplementedError:
        return _load_matlab_b3_hdf5(
            path,
            regression_name=regression_name,
            realign=realign,
            f_range_name=f_range_name,
            smoothing_name=smoothing_name,
            term_index=term_index,
        )


def _load_matlab_b3_regressor_scipy(
    path: Path,
    *,
    regression_name: str,
) -> np.ndarray:
    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    log = mat.get("log")
    if log is None:
        raise KeyError(f"{path.name}: missing 'log' variable.")
    reg = getattr(log, regression_name, None)
    if reg is None:
        raise KeyError(f"{path.name}: regression {regression_name!r} not found in log.")
    regressor = np.asarray(getattr(reg, "regressor"), dtype=np.float64).ravel()
    return regressor


def _load_matlab_b3_regressor_hdf5(
    path: Path,
    *,
    regression_name: str,
) -> np.ndarray:
    import h5py  # noqa: PLC0415

    with h5py.File(str(path), "r") as f:
        log = _h5_deref_scalar(f, f["log"])
        reg = _h5_struct_field(f, log, regression_name)
        regressor_ds = _h5_struct_field(f, reg, "regressor")
        return _h5_numeric_array(regressor_ds, dtype=np.float64).ravel()


def load_matlab_b3_regressor(
    path: Path,
    *,
    regression_name: str,
) -> np.ndarray:
    try:
        return _load_matlab_b3_regressor_scipy(path, regression_name=regression_name)
    except NotImplementedError:
        return _load_matlab_b3_regressor_hdf5(path, regression_name=regression_name)


def _load_matlab_b2_hdf5(
    path: Path,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    import h5py  # noqa: PLC0415

    with h5py.File(str(path), "r") as f:
        raw = np.asarray(f["alldata"][()])
        alldata = np.asarray(raw.T, dtype=np.float64)
        channels: list[str] = []
        timelist = np.array([], dtype=np.float64)
        if "hdr" in f:
            hdr = _h5_deref_scalar(f, f["hdr"])
            if isinstance(hdr, h5py.Group):
                if "chan_info" in hdr:
                    channels = _extract_channels_hdf5(f, hdr)
                elif "label" in hdr:
                    channels = _read_h5_cell_strings(f, hdr["label"])
                if "timelist" in hdr:
                    timelist = np.asarray(hdr["timelist"][()], dtype=np.float64).ravel()
        return alldata, channels, timelist


def load_matlab_b2_data(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    try:
        mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        return _load_matlab_b2_hdf5(path)

    alldata = np.asarray(mat["alldata"], dtype=np.float64)
    if alldata.ndim == 2:
        alldata = alldata[np.newaxis, ...]
    if alldata.ndim != 3:
        raise ValueError(f"Unexpected b2 alldata shape: {alldata.shape!r}")

    hdr = mat.get("hdr")
    channels: list[str] = []
    timelist = np.array([], dtype=np.float64)
    if hdr is not None:
        channels = _extract_channels_scipy(hdr)
        timelist_raw = getattr(hdr, "timelist", None)
        if timelist_raw is not None:
            timelist = np.asarray(timelist_raw, dtype=np.float64).ravel()
    return alldata, channels, timelist


def _load_behavior_pleasantness(path: Path) -> np.ndarray:
    pleasantness: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")
        for row in reader:
            pleasantness.append(int(float(row["pleasant"])))
    return np.asarray(pleasantness, dtype=int)


def _condition_trial_value(regression_name: str) -> int:
    if regression_name == "P_Rating":
        return 1
    if regression_name == "UP_Rating":
        return 2
    raise ValueError(f"Unsupported regression name: {regression_name!r}")


def replay_regression_from_matlab_b2(
    *,
    b2_data: np.ndarray,
    b2_time: np.ndarray,
    b2_channels: list[str],
    matlab_regression_name: str,
    matlab_values: np.ndarray,
    matlab_regressor: np.ndarray,
    matlab_time: np.ndarray,
    matlab_channels: list[str],
    pleasantness: np.ndarray,
) -> ComparisonBundle:
    trial_value = _condition_trial_value(matlab_regression_name)
    trial_mask = np.asarray(pleasantness == trial_value, dtype=bool)
    if b2_data.shape[0] != pleasantness.size:
        raise ValueError(
            f"Behavior/b2 trial mismatch: pleasantness has {pleasantness.size} rows, "
            f"b2 has {b2_data.shape[0]} trials."
        )
    if int(np.sum(trial_mask)) != matlab_regressor.size:
        raise ValueError(
            f"Condition trial mismatch for {matlab_regression_name}: "
            f"{int(np.sum(trial_mask))} behavior trials vs {matlab_regressor.size} regressor rows."
        )

    target_time = np.asarray(matlab_time, dtype=np.float64).ravel()
    source_time = np.asarray(b2_time, dtype=np.float64).ravel()
    time_idx = np.array(
        [int(np.argmin(np.abs(source_time - t))) for t in target_time],
        dtype=int,
    )
    max_time_err = (
        float(np.max(np.abs(source_time[time_idx] - target_time)))
        if target_time.size
        else 0.0
    )

    epochs = np.asarray(b2_data[trial_mask][:, time_idx, :], dtype=np.float64)
    slopes, _, _, _, _stats_valid = compute_linear_regression_maps(
        np.asarray(matlab_regressor, dtype=np.float64).ravel(),
        np.transpose(epochs, (0, 2, 1)),
        n_features=epochs.shape[2],
        n_times=epochs.shape[1],
    )

    channel_mapping = match_exact_channels(matlab_channels, b2_channels)
    matched_mat_indices, matched_b2_indices, mat_stack, py_stack = _matlab_driven_channel_stacks(
        matlab_values=matlab_values,
        python_values=np.asarray(slopes, dtype=np.float64),
        channel_mapping=channel_mapping,
    )
    return ComparisonBundle(
        label=f"{matlab_regression_name} vs Python OLS replay on MATLAB b2",
        matlab_regression_name=matlab_regression_name,
        python_condition_name="matlab_b2_replay",
        matlab_time=target_time,
        python_time_aligned=target_time,
        matched_mat_indices=matched_mat_indices,
        matched_py_indices=matched_b2_indices,
        matlab_channels=list(matlab_channels),
        python_channels=list(b2_channels),
        matlab_values=mat_stack,
        python_values=py_stack,
        time_alignment_error_s=max_time_err,
    )


def replay_regression_from_python_epochs(
    *,
    python_result: object,
    python_condition_field: str,
    matlab_regression_name: str,
    matlab_values: np.ndarray,
    matlab_time: np.ndarray,
    matlab_channels: list[str],
    predictor_mode: str,
) -> ComparisonBundle:
    if python_condition_field == "condition_a":
        condition_name = str(python_result.condition_a)
        epochs = np.asarray(python_result.condition_a_epochs, dtype=np.float64)
        predictor_raw = np.asarray(
            python_result.condition_a_predictor_raw_values,
            dtype=np.float64,
        ).ravel()
        predictor_transformed = np.asarray(
            python_result.condition_a_predictor_transformed_values,
            dtype=np.float64,
        ).ravel()
        predictor_effective = np.asarray(
            python_result.condition_a_predictor_values,
            dtype=np.float64,
        ).ravel()
    elif python_condition_field == "condition_b":
        condition_name = str(python_result.condition_b)
        epochs = np.asarray(python_result.condition_b_epochs, dtype=np.float64)
        predictor_raw = np.asarray(
            python_result.condition_b_predictor_raw_values,
            dtype=np.float64,
        ).ravel()
        predictor_transformed = np.asarray(
            python_result.condition_b_predictor_transformed_values,
            dtype=np.float64,
        ).ravel()
        predictor_effective = np.asarray(
            python_result.condition_b_predictor_values,
            dtype=np.float64,
        ).ravel()
    else:
        raise ValueError(f"Unsupported python condition field: {python_condition_field!r}")

    if predictor_mode == "raw":
        predictor_values = predictor_raw
    elif predictor_mode == "transformed":
        predictor_values = predictor_transformed
    elif predictor_mode == "effective":
        predictor_values = predictor_effective
    else:
        raise ValueError(
            f"Unsupported predictor_mode={predictor_mode!r}. "
            "Use 'raw', 'transformed', or 'effective'."
        )

    target_time = np.asarray(matlab_time, dtype=np.float64).ravel()
    source_time = np.asarray(python_result.time_axis_s, dtype=np.float64).ravel()
    time_idx = np.array(
        [int(np.argmin(np.abs(source_time - t))) for t in target_time],
        dtype=int,
    )
    max_time_err = (
        float(np.max(np.abs(source_time[time_idx] - target_time)))
        if target_time.size
        else 0.0
    )

    epochs_aligned = np.asarray(epochs[:, :, time_idx], dtype=np.float64)
    slopes, _, _, _, _stats_valid = compute_linear_regression_maps(
        predictor_values,
        epochs_aligned,
        n_features=epochs_aligned.shape[1],
        n_times=epochs_aligned.shape[2],
    )

    channel_mapping = match_channels(matlab_channels, list(python_result.channel_names))
    matched_mat_indices, matched_py_indices, mat_stack, py_stack = _matlab_driven_channel_stacks(
        matlab_values=matlab_values,
        python_values=slopes,
        channel_mapping=channel_mapping,
    )
    return ComparisonBundle(
        label=(
            f"{matlab_regression_name} vs Python epoch replay "
            f"({condition_name}, predictor={predictor_mode})"
        ),
        matlab_regression_name=matlab_regression_name,
        python_condition_name=f"{condition_name}_epoch_replay_{predictor_mode}",
        matlab_time=target_time,
        python_time_aligned=target_time,
        matched_mat_indices=matched_mat_indices,
        matched_py_indices=matched_py_indices,
        matlab_channels=list(matlab_channels),
        python_channels=list(python_result.channel_names),
        matlab_values=mat_stack,
        python_values=py_stack,
        time_alignment_error_s=max_time_err,
    )


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _align_python_to_matlab_time(
    matlab_time: np.ndarray,
    python_time: np.ndarray,
    python_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    mat_time = np.asarray(matlab_time, dtype=np.float64).ravel()
    py_time = np.asarray(python_time, dtype=np.float64).ravel()
    py_idx = np.array([int(np.argmin(np.abs(py_time - t))) for t in mat_time], dtype=int)
    max_err = float(np.max(np.abs(py_time[py_idx] - mat_time))) if mat_time.size else 0.0
    aligned = np.asarray(python_values, dtype=np.float64)[:, py_idx]
    return aligned, py_idx, max_err


def _matlab_driven_channel_stacks(
    *,
    matlab_values: np.ndarray,
    python_values: np.ndarray,
    channel_mapping: dict[int, int],
) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
    """Return stacks ordered by MATLAB channel, filling missing Python rows with NaN."""
    mat_stack = np.asarray(matlab_values, dtype=np.float64)
    py_source = np.asarray(python_values, dtype=np.float64)
    matched_mat_indices = list(range(mat_stack.shape[0]))
    matched_py_indices = [int(channel_mapping.get(mat_idx, -1)) for mat_idx in matched_mat_indices]
    py_stack = np.full_like(mat_stack, np.nan, dtype=np.float64)
    for row_idx, py_idx in enumerate(matched_py_indices):
        if 0 <= py_idx < py_source.shape[0]:
            py_stack[row_idx, :] = py_source[py_idx, :]
    return matched_mat_indices, matched_py_indices, mat_stack, py_stack


def _python_channel_name(channels: list[str], py_idx: int) -> str:
    if 0 <= int(py_idx) < len(channels):
        return channels[int(py_idx)]
    return "(no HDF5)"


def filter_bundle_to_finite_matlab(bundle: ComparisonBundle) -> ComparisonBundle:
    """Keep MATLAB channels whose MATLAB b3 trace contains at least one finite value."""
    keep_rows = [
        row_idx
        for row_idx in range(bundle.matlab_values.shape[0])
        if np.any(np.isfinite(bundle.matlab_values[row_idx]))
    ]
    if len(keep_rows) == bundle.matlab_values.shape[0]:
        return bundle

    keep = np.asarray(keep_rows, dtype=np.int64)
    return ComparisonBundle(
        label=bundle.label,
        matlab_regression_name=bundle.matlab_regression_name,
        python_condition_name=bundle.python_condition_name,
        matlab_time=bundle.matlab_time,
        python_time_aligned=bundle.python_time_aligned,
        matched_mat_indices=[bundle.matched_mat_indices[i] for i in keep_rows],
        matched_py_indices=[bundle.matched_py_indices[i] for i in keep_rows],
        matlab_channels=bundle.matlab_channels,
        python_channels=bundle.python_channels,
        matlab_values=bundle.matlab_values[keep, :] if keep.size else bundle.matlab_values[:0, :],
        python_values=bundle.python_values[keep, :] if keep.size else bundle.python_values[:0, :],
        time_alignment_error_s=bundle.time_alignment_error_s,
    )


def _channel_metrics(
    matlab_trace: np.ndarray,
    python_trace: np.ndarray,
) -> tuple[float, float, float, float]:
    a = np.asarray(matlab_trace, dtype=np.float64).ravel()
    b = np.asarray(python_trace, dtype=np.float64).ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(finite)) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    a = a[finite]
    b = b[finite]
    mean_abs = float(np.mean(np.abs(b - a)))
    rms_diff = float(np.sqrt(np.mean((b - a) ** 2)))
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    denom_corr = float(np.sqrt(np.sum(a_centered * a_centered) * np.sum(b_centered * b_centered)))
    corr = float(np.sum(a_centered * b_centered) / denom_corr) if denom_corr > 0.0 else float("nan")
    denom_slope = float(np.sum(a_centered * a_centered))
    slope = float(np.sum(a_centered * b_centered) / denom_slope) if denom_slope > 0.0 else float("nan")
    return mean_abs, rms_diff, corr, slope


@dataclass
class ComparisonBundle:
    label: str
    matlab_regression_name: str
    python_condition_name: str
    matlab_time: np.ndarray
    python_time_aligned: np.ndarray
    matched_mat_indices: list[int]
    matched_py_indices: list[int]
    matlab_channels: list[str]
    python_channels: list[str]
    matlab_values: np.ndarray  # (n_matched_channels, n_times)
    python_values: np.ndarray  # (n_matched_channels, n_times)
    time_alignment_error_s: float


def _bundle_metric_summary(bundle: ComparisonBundle) -> dict[str, float]:
    rows: list[tuple[float, float, float, float]] = []
    for row_idx in range(bundle.matlab_values.shape[0]):
        rows.append(
            _channel_metrics(
                bundle.matlab_values[row_idx],
                bundle.python_values[row_idx],
            )
        )
    if not rows:
        return {
            "median_abs_slope_dev": float("nan"),
            "max_abs_slope_dev": float("nan"),
            "median_corr": float("nan"),
            "min_corr": float("nan"),
            "mean_mean_abs": float("nan"),
        }
    mean_abs_all = np.array([row[0] for row in rows if np.isfinite(row[0])], dtype=np.float64)
    corrs = np.array([row[2] for row in rows if np.isfinite(row[2])], dtype=np.float64)
    abs_slope_dev = np.array(
        [abs(row[3] - 1.0) for row in rows if np.isfinite(row[3])],
        dtype=np.float64,
    )
    return {
        "median_abs_slope_dev": float(np.median(abs_slope_dev)) if abs_slope_dev.size else float("nan"),
        "max_abs_slope_dev": float(np.max(abs_slope_dev)) if abs_slope_dev.size else float("nan"),
        "median_corr": float(np.median(corrs)) if corrs.size else float("nan"),
        "min_corr": float(np.min(corrs)) if corrs.size else float("nan"),
        "mean_mean_abs": float(np.mean(mean_abs_all)) if mean_abs_all.size else float("nan"),
    }


def print_bundle_summary(bundle: ComparisonBundle, *, top_n: int = TOP_N) -> None:
    rows: list[dict[str, object]] = []
    for row_idx, (mat_idx, py_idx) in enumerate(zip(bundle.matched_mat_indices, bundle.matched_py_indices)):
        mean_abs, rms_diff, corr, slope = _channel_metrics(
            bundle.matlab_values[row_idx],
            bundle.python_values[row_idx],
        )
        rows.append(
            {
                "row_idx": row_idx,
                "mat_idx": mat_idx,
                "py_idx": py_idx,
                "mat_name": bundle.matlab_channels[mat_idx],
                "py_name": _python_channel_name(bundle.python_channels, py_idx),
                "mean_abs": mean_abs,
                "rms_diff": rms_diff,
                "corr": corr,
                "slope": slope,
            }
        )

    if not rows:
        print(f"\n{bundle.label}: no finite matched channels.")
        return

    valid_rows = [
        row
        for row in rows
        if all(
            np.isfinite(float(row[key]))
            for key in ("mean_abs", "rms_diff", "corr", "slope")
        )
    ]
    skipped_rows = len(rows) - len(valid_rows)
    if not valid_rows:
        print(
            f"\n{bundle.label}: matched {len(rows)} channel(s), but none had enough "
            "overlapping finite samples for metrics."
        )
        return

    abs_slope_dev = np.array(
        [abs(float(row["slope"]) - 1.0) for row in valid_rows],
        dtype=np.float64,
    )
    corrs = np.array(
        [float(row["corr"]) for row in valid_rows],
        dtype=np.float64,
    )
    mean_abs_all = np.array(
        [float(row["mean_abs"]) for row in valid_rows],
        dtype=np.float64,
    )

    print(f"\n=== {bundle.label} ===")
    print(
        f"  MATLAB channels shown: {len(rows)}"
        f"  |  finite MATLAB/Python overlaps: {len(valid_rows)}"
        f"  |  skipped (NaN overlap): {skipped_rows}"
        f"  |  time alignment max error: {bundle.time_alignment_error_s:.6f} s"
    )
    if abs_slope_dev.size and corrs.size and mean_abs_all.size:
        print(
            f"  summary: median |slope-1|={float(np.median(abs_slope_dev)):.4f},"
            f" max |slope-1|={float(np.max(abs_slope_dev)):.4f},"
            f" median corr={float(np.median(corrs)):.4f},"
            f" min corr={float(np.min(corrs)):.4f},"
            f" mean mean|d|={float(np.mean(mean_abs_all)):.5f}"
        )

    def _print_table(title: str, *, sort_key: object, reverse: bool) -> None:
        print(f"\n  {title}")
        print(
            f"    {'MAT':>5}  {'PY':>5}  {'MATLAB bipole':<16}  {'Python':<10}"
            f"  {'mean|d|':>9}  {'RMSd':>9}  {'corr':>8}  {'slope':>8}"
        )
        ordered = sorted(valid_rows, key=sort_key, reverse=reverse)
        for row in ordered[:top_n]:
            py_idx = int(row["py_idx"])
            py_idx_text = f"[{py_idx:4d}]" if py_idx >= 0 else "   n/a"
            print(
                f"    [{int(row['mat_idx']):4d}]  {py_idx_text}  "
                f"{str(row['mat_name'])[:16]:<16}  {str(row['py_name'])[:10]:<10}  "
                f"{float(row['mean_abs']):9.5f}  {float(row['rms_diff']):9.5f}  "
                f"{float(row['corr']):8.4f}  {float(row['slope']):8.4f}"
            )

    _print_table(
        "Largest amplitude mismatches (|slope-1|)",
        sort_key=lambda row: abs(float(row["slope"]) - 1.0) if np.isfinite(float(row["slope"])) else -np.inf,
        reverse=True,
    )
    _print_table(
        "Lowest waveform similarity (corr)",
        sort_key=lambda row: float(row["corr"]) if np.isfinite(float(row["corr"])) else np.inf,
        reverse=False,
    )
    _print_table(
        "Largest absolute differences (mean|d|)",
        sort_key=lambda row: float(row["mean_abs"]) if np.isfinite(float(row["mean_abs"])) else -np.inf,
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------

class RegressionComparisonViewer:
    def __init__(self, bundles: list[ComparisonBundle]) -> None:
        if not bundles:
            raise ValueError("At least one comparison bundle is required.")
        self.bundles = bundles
        self._bundle_idx = 0
        self._ch_idx = 0
        self._build_figure()

    def _current_bundle(self) -> ComparisonBundle:
        return self.bundles[self._bundle_idx]

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(15, 7))
        self.fig.suptitle("MATLAB b3 vs Python regression comparison", fontsize=13, fontweight="bold")
        self.ax = self.fig.add_axes([0.07, 0.22, 0.72, 0.68])

        ax_prev_reg = self.fig.add_axes([0.07, 0.09, 0.09, 0.05])
        ax_next_reg = self.fig.add_axes([0.18, 0.09, 0.09, 0.05])
        ax_prev_ch = self.fig.add_axes([0.33, 0.09, 0.09, 0.05])
        ax_next_ch = self.fig.add_axes([0.44, 0.09, 0.09, 0.05])
        self.btn_prev_reg = Button(ax_prev_reg, "< Reg", color="lightblue")
        self.btn_next_reg = Button(ax_next_reg, "Reg >", color="lightblue")
        self.btn_prev_ch = Button(ax_prev_ch, "< Ch", color="lightgoldenrodyellow")
        self.btn_next_ch = Button(ax_next_ch, "Ch >", color="lightgoldenrodyellow")
        self.btn_prev_reg.on_clicked(lambda _e: self._step(reg_delta=-1))
        self.btn_next_reg.on_clicked(lambda _e: self._step(reg_delta=+1))
        self.btn_prev_ch.on_clicked(lambda _e: self._step(ch_delta=-1))
        self.btn_next_ch.on_clicked(lambda _e: self._step(ch_delta=+1))

        self.info_text = self.fig.text(
            0.82,
            0.55,
            "",
            ha="left",
            va="center",
            fontsize=10,
            bbox={"facecolor": "#fff8dc", "alpha": 0.95, "boxstyle": "round"},
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._redraw()

    def _on_key(self, event: object) -> None:
        key = getattr(event, "key", None)
        if key == "left":
            self._step(ch_delta=-1)
        elif key == "right":
            self._step(ch_delta=+1)
        elif key == "up":
            self._step(reg_delta=-1)
        elif key == "down":
            self._step(reg_delta=+1)

    def _step(self, *, ch_delta: int = 0, reg_delta: int = 0) -> None:
        if reg_delta:
            self._bundle_idx = max(0, min(self._bundle_idx + reg_delta, len(self.bundles) - 1))
            self._ch_idx = min(self._ch_idx, self._current_bundle().matlab_values.shape[0] - 1)
        if ch_delta:
            n_ch = self._current_bundle().matlab_values.shape[0]
            self._ch_idx = max(0, min(self._ch_idx + ch_delta, n_ch - 1))
        self._redraw()

    def _redraw(self) -> None:
        bundle = self._current_bundle()
        n_ch = bundle.matlab_values.shape[0]
        self._ch_idx = max(0, min(self._ch_idx, n_ch - 1))
        mat_idx = bundle.matched_mat_indices[self._ch_idx]
        py_idx = bundle.matched_py_indices[self._ch_idx]
        mat_trace = bundle.matlab_values[self._ch_idx]
        py_trace = bundle.python_values[self._ch_idx]
        mean_abs, rms_diff, corr, slope = _channel_metrics(mat_trace, py_trace)
        py_name = _python_channel_name(bundle.python_channels, py_idx)

        self.ax.clear()
        self.ax.plot(bundle.matlab_time, mat_trace, color="steelblue", linewidth=4.6, label="MATLAB b3")
        if np.any(np.isfinite(py_trace)):
            self.ax.plot(bundle.matlab_time, py_trace, color="darkorange", linewidth=1.6, label="Python regression")
        self.ax.axvline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("Time relative to anchor onset (s)")
        self.ax.set_ylabel("Regression slope")
        self.ax.legend(loc="upper right")
        self.ax.set_title(
            f"{bundle.label}  |  channel {self._ch_idx + 1}/{n_ch}"
            f"  |  MATLAB: {bundle.matlab_channels[mat_idx]} -> Python: {py_name}",
            fontsize=11,
        )

        info = (
            f"Comparison: {bundle.label}\n"
            f"MATLAB reg: {bundle.matlab_regression_name}\n"
            f"Python cond: {bundle.python_condition_name}\n"
            f"MATLAB idx: {mat_idx}\n"
            f"Python idx: {py_idx}\n"
            f"MATLAB bipole: {bundle.matlab_channels[mat_idx]}\n"
            f"Python channel: {py_name}\n"
            f"Time align max err: {bundle.time_alignment_error_s:.6f}s\n"
            f"mean|d|: {mean_abs:.5f}\n"
            f"RMSd: {rms_diff:.5f}\n"
            f"corr: {corr:.5f}\n"
            f"slope: {slope:.5f}"
        )
        self.info_text.set_text(info)
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _python_condition_values(result: object, condition_field: str) -> tuple[str, np.ndarray]:
    if condition_field == "condition_a":
        return str(result.condition_a), np.asarray(result.condition_a_slope, dtype=np.float64)
    if condition_field == "condition_b":
        return str(result.condition_b), np.asarray(result.condition_b_slope, dtype=np.float64)
    raise ValueError(f"Unsupported python condition field: {condition_field!r}")


def _python_condition_predictors(
    result: object,
    condition_field: str,
) -> tuple[str, np.ndarray, np.ndarray]:
    if condition_field == "condition_a":
        return (
            str(result.condition_a),
            np.asarray(result.condition_a_predictor_raw_values, dtype=np.float64).ravel(),
            np.asarray(result.condition_a_predictor_transformed_values, dtype=np.float64).ravel(),
        )
    if condition_field == "condition_b":
        return (
            str(result.condition_b),
            np.asarray(result.condition_b_predictor_raw_values, dtype=np.float64).ravel(),
            np.asarray(result.condition_b_predictor_transformed_values, dtype=np.float64).ravel(),
        )
    raise ValueError(f"Unsupported python condition field: {condition_field!r}")


def _describe_vector(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "n=0 finite=0"
    return (
        f"n={arr.size}, finite={finite.size}, "
        f"mean={float(np.mean(finite)):.6f}, std={float(np.std(finite)):.6f}, "
        f"min={float(np.min(finite)):.6f}, max={float(np.max(finite)):.6f}"
    )


def _target_subject_from_regression_path(path: Path) -> str:
    match = re.search(r"sub-([^_]+)", path.stem)
    if match is None:
        raise ValueError(f"Could not extract subject from regression path: {path}")
    return str(match.group(1))


def compute_python_regression_from_source(
    *,
    target_subject: str,
) -> object:
    manual_region_channels = load_roi_channels_from_csv()
    annotators = build_trial_annotators(manual_region_channels)
    ds = BIDSDataset(PYTHON_SOURCE_BIDS_ROOT)
    groups = build_trial_slope_groups(ds)
    for group in groups:
        if str(group.primary.get("subject", "")).strip() == str(target_subject).strip():
            processor = RegressionProcessing(
                PYTHON_REGRESSION_PARAMS,
                resolver=PYTHON_REGRESSION_RESOLVER,
                annotators=annotators,
            )
            return processor.process_group(group)
    raise ValueError(
        f"Subject {target_subject!r} not found in trial slope groups built from "
        f"{PYTHON_SOURCE_BIDS_ROOT}."
    )


def print_saved_vs_fresh_regression_diff(saved_result: object, fresh_result: object) -> None:
    print("\n=== Saved regression vs fresh in-memory recompute ===")
    for condition_field in ("condition_a", "condition_b"):
        saved_slope = np.asarray(
            getattr(saved_result, f"{condition_field}_slope"),
            dtype=np.float64,
        )
        fresh_slope = np.asarray(
            getattr(fresh_result, f"{condition_field}_slope"),
            dtype=np.float64,
        )
        finite = np.isfinite(saved_slope) & np.isfinite(fresh_slope)
        if not finite.any():
            print(f"  {condition_field}: no overlapping finite slope values.")
            continue
        diff = fresh_slope[finite] - saved_slope[finite]
        print(
            f"  {condition_field}: max|diff|={float(np.max(np.abs(diff))):.6g}, "
            f"mean|diff|={float(np.mean(np.abs(diff))):.6g}, "
            f"corr={float(np.corrcoef(saved_slope[finite], fresh_slope[finite])[0, 1]):.6g}"
        )


def _default_matlab_path(regression_name: str) -> Path:
    return MATLAB_B3_ROOT / regression_name / MATLAB_LOG_DATA_FILENAME


def main() -> None:
    if not (
        INCLUDE_REPLAY_PYTHON_EPOCHS
        or INCLUDE_REPLAY_MATLAB_B2
        or INCLUDE_HDF5_VS_MATLAB
    ):
        sys.exit("ERROR: all comparison panels are disabled by INCLUDE_* flags.")

    py_result: object | None = None
    if INCLUDE_HDF5_VS_MATLAB or INCLUDE_REPLAY_PYTHON_EPOCHS:
        print(f"Loading Python regression file: {PYTHON_REGRESSION_PATH}")
        if not PYTHON_REGRESSION_PATH.exists():
            sys.exit(f"ERROR: Python regression file not found: {PYTHON_REGRESSION_PATH}")
        py_result = load_regression_result(PYTHON_REGRESSION_PATH)
        print(
            f"  channels={len(py_result.channel_names)}, times={len(py_result.time_axis_s)}, "
            f"conditions={py_result.condition_a}/{py_result.condition_b}, predictor={py_result.predictor}"
        )
    else:
        print("Skipping Python regression HDF5 load because dependent panels are disabled.")

    fresh_py_result: object | None = None
    if INCLUDE_REPLAY_PYTHON_EPOCHS:
        target_subject = _target_subject_from_regression_path(PYTHON_REGRESSION_PATH)
        try:
            print(
                "\nRecomputing Python regression in memory from source Hilbert file "
                f"for subject {target_subject}..."
            )
            fresh_py_result = compute_python_regression_from_source(
                target_subject=target_subject,
            )
            if py_result is not None:
                print_saved_vs_fresh_regression_diff(py_result, fresh_py_result)
        except Exception as exc:
            print(
                "\nWARNING: fresh in-memory Python regression replay could not be computed:\n"
                f"  {type(exc).__name__}: {exc}"
            )
    else:
        print("\nSkipping Python epoch replay because INCLUDE_REPLAY_PYTHON_EPOCHS=False.")

    b2_data: np.ndarray | None = None
    b2_channels: list[str] = []
    b2_time = np.array([], dtype=np.float64)
    pleasantness = np.array([], dtype=int)
    if not INCLUDE_REPLAY_MATLAB_B2:
        print("Skipping MATLAB b2 replay because INCLUDE_REPLAY_MATLAB_B2=False.")
    elif MATLAB_B2_PATH.exists() and BEHAVIOR_TSV_PATH.exists():
        print(f"\nLoading MATLAB b2 file: {MATLAB_B2_PATH}")
        b2_data, b2_channels, b2_time = load_matlab_b2_data(MATLAB_B2_PATH)
        pleasantness = _load_behavior_pleasantness(BEHAVIOR_TSV_PATH)
        print(
            f"  b2 alldata shape={b2_data.shape}, channels={len(b2_channels)}, "
            f"time range={float(b2_time[0]):.3f}->{float(b2_time[-1]):.3f}"
        )
        print(
            f"  behavior rows={pleasantness.size}, "
            f"pleasant={int(np.sum(pleasantness == 1))}, "
            f"unpleasant={int(np.sum(pleasantness == 2))}"
        )
    else:
        print(
            "\nWARNING: MATLAB b2 replay disabled because at least one input is missing:\n"
            f"  b2 file: {MATLAB_B2_PATH}\n"
            f"  behavior TSV: {BEHAVIOR_TSV_PATH}"
        )

    bundles: list[ComparisonBundle] = []
    for matlab_regression_name, python_condition_field in MATLAB_PYTHON_REGRESSION_MAP:
        matlab_path = _default_matlab_path(matlab_regression_name)
        print(f"\nLoading MATLAB b3 file for {matlab_regression_name}: {matlab_path}")
        if not matlab_path.exists():
            print("  WARNING: file not found, skipping.")
            continue

        matlab_values, matlab_time, matlab_channels = load_matlab_b3_regression(
            matlab_path,
            regression_name=matlab_regression_name,
            realign=MATLAB_REALIGN,
            f_range_name=MATLAB_F_RANGE_NAME,
            smoothing_name=MATLAB_SMOOTHING_NAME,
            term_index=MATLAB_TERM_INDEX,
        )
        print(
            f"  MATLAB dots shape={matlab_values.shape}, MATLAB channels={len(matlab_channels)}"
        )

        if INCLUDE_HDF5_VS_MATLAB:
            if py_result is None:
                raise RuntimeError("Internal error: py_result is required for HDF5 comparison.")
            matlab_regressor = load_matlab_b3_regressor(
                matlab_path,
                regression_name=matlab_regression_name,
            )
            python_condition_name, python_values = _python_condition_values(
                py_result,
                python_condition_field,
            )
            _, python_predictor_raw, python_predictor_transformed = _python_condition_predictors(
                py_result,
                python_condition_field,
            )
            print(f"  Python slope shape={python_values.shape}")
            print(f"  MATLAB regressor: {_describe_vector(matlab_regressor)}")
            print(f"  Python predictor raw: {_describe_vector(python_predictor_raw)}")
            print(f"  Python predictor transformed: {_describe_vector(python_predictor_transformed)}")
            if matlab_regressor.size != python_predictor_raw.size:
                print(
                    "  WARNING: predictor trial-count mismatch "
                    f"(MATLAB={matlab_regressor.size}, Python={python_predictor_raw.size})."
                )
            if python_condition_field == "condition_b":
                print(
                    "  NOTE: Python uses the transformed unpleasant predictor for this output. "
                    "If MATLAB keeps the raw regressor sign, expect a sign inversion here."
                )
            channel_mapping = match_channels(matlab_channels, list(py_result.channel_names))

            python_values_aligned, _py_idx, max_time_err = _align_python_to_matlab_time(
                matlab_time,
                np.asarray(py_result.time_axis_s, dtype=np.float64),
                python_values,
            )
            matched_mat_indices, matched_py_indices, mat_stack, py_stack = (
                _matlab_driven_channel_stacks(
                    matlab_values=matlab_values,
                    python_values=python_values_aligned,
                    channel_mapping=channel_mapping,
                )
            )
            flip_matlab_trace = (
                FLIP_UNPLEASANT_MATLAB_IN_HDF5_VS_MATLAB
                and python_condition_field == "condition_b"
            )

            bundle = ComparisonBundle(
                label=(
                    f"{matlab_regression_name}"
                    f"{' (MATLAB flipped)' if flip_matlab_trace else ''}"
                    f" vs {python_condition_name}"
                    f"{' (pipeline output)' if python_condition_field == 'condition_b' else ''}"
                ),
                matlab_regression_name=matlab_regression_name,
                python_condition_name=python_condition_name,
                matlab_time=np.asarray(matlab_time, dtype=np.float64),
                python_time_aligned=np.asarray(matlab_time, dtype=np.float64),
                matched_mat_indices=matched_mat_indices,
                matched_py_indices=matched_py_indices,
                matlab_channels=list(matlab_channels),
                python_channels=list(py_result.channel_names),
                matlab_values=-mat_stack if flip_matlab_trace else mat_stack,
                python_values=py_stack,
                time_alignment_error_s=max_time_err,
            )
            n_matlab_before_filter = bundle.matlab_values.shape[0]
            bundle = filter_bundle_to_finite_matlab(bundle)
            print(
                "  MATLAB non-NaN channel filter: "
                f"{bundle.matlab_values.shape[0]}/{n_matlab_before_filter} kept "
                f"(Python HDF5 match for {len(channel_mapping)})"
            )
            print_bundle_summary(bundle)
            if python_condition_field == "condition_b":
                neg_bundle = ComparisonBundle(
                    label=f"{matlab_regression_name} vs NEG_{python_condition_name}",
                    matlab_regression_name=matlab_regression_name,
                    python_condition_name=f"NEG_{python_condition_name}",
                    matlab_time=bundle.matlab_time,
                    python_time_aligned=bundle.python_time_aligned,
                    matched_mat_indices=bundle.matched_mat_indices,
                    matched_py_indices=bundle.matched_py_indices,
                    matlab_channels=bundle.matlab_channels,
                    python_channels=bundle.python_channels,
                    matlab_values=bundle.matlab_values,
                    python_values=-bundle.python_values,
                    time_alignment_error_s=max_time_err,
                )
                neg_summary = _bundle_metric_summary(neg_bundle)
                print(
                    "\n  Oracle check with sign-flipped Python slope:"
                    f" median |slope-1|={neg_summary['median_abs_slope_dev']:.4f},"
                    f" median corr={neg_summary['median_corr']:.4f},"
                    f" mean mean|d|={neg_summary['mean_mean_abs']:.5f}"
                )

            bundles.append(bundle)

        if fresh_py_result is not None:
            predictor_mode = "raw" if python_condition_field == "condition_b" else "effective"
            source_replay_bundle = replay_regression_from_python_epochs(
                python_result=fresh_py_result,
                python_condition_field=python_condition_field,
                matlab_regression_name=matlab_regression_name,
                matlab_values=matlab_values,
                matlab_time=matlab_time,
                matlab_channels=matlab_channels,
                predictor_mode=predictor_mode,
            )
            source_replay_bundle = filter_bundle_to_finite_matlab(source_replay_bundle)
            print_bundle_summary(source_replay_bundle)
            if python_condition_field == "condition_b":
                print(
                    "\n  Note: this replay uses the RAW unpleasant predictor to stay "
                    "comparable to MATLAB b3. The saved Python slope output below still "
                    "uses the pipeline's transformed predictor (sign-flipped)."
                )
            bundles.append(source_replay_bundle)

        if b2_data is not None and b2_time.size and pleasantness.size:
            matlab_regressor = load_matlab_b3_regressor(
                matlab_path,
                regression_name=matlab_regression_name,
            )
            print(f"  MATLAB regressor: {_describe_vector(matlab_regressor)}")
            replay_bundle = replay_regression_from_matlab_b2(
                b2_data=b2_data,
                b2_time=b2_time,
                b2_channels=b2_channels,
                matlab_regression_name=matlab_regression_name,
                matlab_values=matlab_values,
                matlab_regressor=matlab_regressor,
                matlab_time=matlab_time,
                matlab_channels=matlab_channels,
                pleasantness=pleasantness,
            )
            replay_bundle = filter_bundle_to_finite_matlab(replay_bundle)
            print_bundle_summary(replay_bundle)
            replay_summary = _bundle_metric_summary(replay_bundle)
            print(
                "\n  Replay verdict from MATLAB b2:"
                f" median |slope-1|={replay_summary['median_abs_slope_dev']:.6g},"
                f" median corr={replay_summary['median_corr']:.6g},"
                f" mean mean|d|={replay_summary['mean_mean_abs']:.6g}"
            )
            bundles.append(replay_bundle)

    if not bundles:
        sys.exit(
            "ERROR: no MATLAB b3 regression file could be loaded. "
            "Check MATLAB_B3_ROOT and the generated log_data.mat files."
        )

    print("\nLaunching regression comparison viewer...")
    print("  Keyboard: left/right for channels, up/down for regression.")
    viewer = RegressionComparisonViewer(bundles)
    viewer.show()


if __name__ == "__main__":
    main()
