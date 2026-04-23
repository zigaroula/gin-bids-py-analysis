#!/usr/bin/env python3
"""
Quality analysis: compare MATLAB-epoched gamma vs BrainVision-derived gamma.

MATLAB file  : alldata (N_trials × N_samples × N_bipoles), hdr, opts_log.
               Epochs are around events 11/12, starting from event 5, tmin=-1 s, tmax=10 s.
BrainVision  : continuous Hilbert-processed signal with embedded events.
               Re-epoched here with the same parameters.

Usage:
    .venv\\Scripts\\python scripts\\compare_matlab_vs_bids.py
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.io
from matplotlib.widgets import Button, Slider

from gin_bids_py_analysis.processing.utils.epoch_quality import (
    apply_trial_nan_mask,
    detect_outlier_trial_channel_pairs_by_max,
    detect_outlier_trial_channel_pairs_by_mean,
    reject_channels_by_nan_trial_ratio,
    reject_channels_by_trial_max_spread,
    reject_channels_by_trial_mean_spread,
)
from gin_bids_py_analysis.processing.utils.statistics import (
    compute_baseline_outlier_mask,
    zscore_activity_by_baseline,
)

# ---------------------------------------------------------------------------
# File paths  (edit if needed)
# ---------------------------------------------------------------------------

MATLAB_PATH = Path(
    r"C:\GRE\dev\clarissa\seeg\b2_BPF_apply_options\edGRE_2021_AICb_MD_CHOICE_Partie1_BPF_f50f150_sf100_sm250_onset.mat"
)

# MNE needs the .vhdr header — derive it from the .eeg path.
BV_EEG_PATH = Path(
    r"D:/Boulot/clarissa_bids/derivatives/hilbert"
    r"/sub-GRE2021AICb/ieeg/sub-GRE2021AICb_task-MDCHOICE_desc-bgasm250_ieeg.eeg"
)
BV_VHDR_PATH = BV_EEG_PATH.with_suffix(".vhdr")

# Epoching window to apply to the BrainVision file.
# BV is epoched with the SAME window as the MATLAB alldata so that the
# PRECLEAN mean/max are computed over the identical sample set.
MAT_TMIN_S: float = -0.51   # MATLAB epoch start relative to anchor
MAT_TMAX_S: float = 5.49  # MATLAB epoch end relative to anchor
TMIN_S: float = -0.51  # BV epoching start — matches MATLAB to avoid 1-sample preclean discrepancy
TMAX_S: float = 5.49   # BV epoching end  — matches MATLAB
BV_DISPLAY_OFFSET_S: float = 0.0  # shift BV time axis by this amount for display
ANCHOR_CODES: set[str] = {"11", "12"}
EXPERIMENT_START_CODE: str = "5"
# SPM uses event samples as 1-based indices during epoch extraction; when
# epoching the BrainVision derivative with MNE's zero-based samples, subtract
# one output-rate sample to reproduce the extracted MATLAB epochs.
BV_EVENT_SAMPLE_SHIFT_SAMPLES: int = -1

# Global z-score baseline applied to BrainVision epochs.
ZSCORE_BV: bool = True
BASELINE_TMIN_S: float = -0.25
BASELINE_TMAX_S: float = -0.05
# MATLAB f_baseline_normalization uses rmoutliers() with default = median/MAD criterion,
# applied on per-trial baseline means.  Set True to replicate this step exactly.
BASELINE_REMOVE_OUTLIERS: bool = True
# Outlier-detection criterion for per-trial baseline means.
# 'median_mad' : median ± 3 × 1.4826 × MAD  (MATLAB rmoutliers default → f_baseline_normalization)
# 'mean'       : mean ± 3σ  (MATLAB rmoutliers(..., 'mean', 'ThresholdFactor', 3) → b2 HGA cleaning)
BASELINE_OUTLIER_METHOD: str = "median_mad"
# Replicate MATLAB b2: NaN-ise entire channels where ≥ this fraction of trials are NaN after
# PRECLEAN and removebadchannelsSd (MATLAB threshold = 0.25, i.e. 25 %).
REMOVE_HIGH_NAN_CHANNELS: bool = True
HIGH_NAN_CHANNEL_THRESHOLD: float = 0.25
# Pre-z-score NaN masking using mean±3σ and max±3σ over the full epoch.
# NOT present in f_baseline_normalization — only enable if MATLAB b2 does a separate
# pre-cleaning step before calling f_baseline_normalization.
# Current evidence: keeping this False (no PRECLEAN) + BASELINE_REMOVE_OUTLIERS=True
# is the closest match to MATLAB's f_baseline_normalization alone.
PRECLEAN_BV: bool = True
PRECLEAN_THRESHOLD: float = 3.0
# When MATLAB bsl_info exposes the exact baseline keep/outlier masks, use that
# mask directly for the BV z-score reference instead of recomputing outliers in
# Python.  Keep this False for the default "Python-only" comparison mode.
USE_MATLAB_BASELINE_KEEP_MASK: bool = False
# Replicate MATLAB's removebadchannelsSd step: NaN-ise entire channels whose
# across-trial spread of mean HGA is an outlier (threshold=1σ, 'mean' method).
# Applied after PRECLEAN, before z-scoring, matching b2_BPF_apply_options.m.
REJECT_BAD_CHANNELS_SD: bool = True
REJECT_BAD_CHANNELS_SD_THRESHOLD: float = 1.0
# Replicate MATLAB b2 opts.removenegratings: NaN-ise trials where the behavioral
# rating is negative (rating < MIN_RATING) across all channels.
# opts.removenegratings = 0 in b2_BPF_apply_options.m — set False to match exactly.
# Set True to replicate b2_BPF_apply_options_R1.m or other variants that enable this.
REMOVE_NEGATIVE_RATINGS: bool = False
MIN_RATING: float = 0.0
# Replicate MATLAB b2 removeoutlierRTs: NaN-ise trials where RT > maxRT before z-scoring.
# param.maxRT = 20 in init_2021_CB_seeg_R1.m.
REMOVE_OUTLIER_RTS: bool = True
MAX_RT_S: float = 20.0
RT_COLUMN: str = "RT"  # column name in BEH_TSV_PATH for reaction time
BEH_TSV_PATH = Path(
    r"D:/Boulot/clarissa_bids"
    r"/sub-GRE2021AICb/beh/sub-GRE2021AICb_task-MDCHOICE_beh.tsv"
)


# Optional: path to the MATLAB bsl_info .mat file saved by b2 alongside the alldata file.
# When set, MATLAB's per-channel z-score mean/σ are loaded and displayed in the infobox
# so you can directly compare them with Python's reference values.
# Typical path: <b2_BPF_backup>/<bpf_mat_name_norealign>_bsl_info.mat
#BSL_INFO_PATH: Path | None = None
BSL_INFO_PATH = Path(
    r"C:\GRE\dev\clarissa\seeg\b2_BPF_apply_options\edGRE_2021_AICb_MD_CHOICE_Partie1_BPF_f50f150_sf100_sm250_bsl_info.mat"
)

# When True, apply MATLAB's opts_log trial-rejection and bad-channel masks
# directly to the BV data instead of recomputing Python's own PRECLEAN and
# removebadchannelsSd.  Keep this False for the default "Python-only"
# comparison mode; set it True only for oracle diagnostics.
USE_MATLAB_REJECTION_MASKS: bool = False
# Report how much the BV window needs to slide to best match the MATLAB output.
SHIFT_DIAGNOSTIC_MAX_OFFSET: int = 2
RUN_BASELINE_STRATEGY_EXPERIMENTS: bool = False
BASELINE_EXPERIMENT_TOP_N: int = 10


# ===========================================================================
# MATLAB loading
# ===========================================================================

def _squeeze(obj: object) -> object:
    """Recursively squeeze 1-element numpy arrays to their scalar contents."""
    if isinstance(obj, np.ndarray):
        while obj.ndim > 0 and obj.size == 1:
            obj = obj.flat[0]
    return obj


def _mat_attr(struct: object, *names: str) -> object | None:
    """Access a MATLAB struct attribute, trying several candidate names."""
    for name in names:
        val = getattr(struct, name, None)
        if val is not None:
            return _squeeze(val)
    return None


def _extract_channels_scipy(hdr: object) -> list[str]:
    """Extract full bipole names from hdr.chan_info (column index 1).

    chan_info is a 3-column cell array:  [id, bipole_name, roi]
    e.g.  ['1', 'Bp02Bp01', 'vmPFC']
    """
    raw = getattr(hdr, "chan_info", None)
    if raw is not None:
        arr = np.asarray(raw)
        # scipy squeeze_me may give (N,) array of objects or (N,3) array.
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return [str(arr[i, 1]).strip() for i in range(arr.shape[0])]
        if arr.ndim == 1:
            # Each element is itself a row array/list.
            names: list[str] = []
            for row in arr:
                row_arr = np.asarray(row).flatten()
                if len(row_arr) >= 2:
                    names.append(str(row_arr[1]).strip())
            if names:
                return names
    # Fallback: old hdr.label path
    fallback = getattr(hdr, "label", None)
    if fallback is not None:
        return [str(el).strip() for el in np.asarray(fallback).flatten()]
    return []


def _extract_sfreq_scipy(hdr: object, fallback: float) -> float:
    val = _mat_attr(hdr, "Fs", "fsample", "fs", "srate", "sfreq", "freq")
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    return fallback


def _sfreq_from_filename(path: Path, fallback: float = 100.0) -> float:
    """Try to parse sampling frequency from a filename like …_sf100_…"""
    m = re.search(r"_sf(\d+)", path.stem, re.IGNORECASE)
    return float(m.group(1)) if m else fallback


# --- h5py helpers (MATLAB v7.3) -------------------------------------------

def _hdf5_cell_strings(f: object, ds: object) -> list[str]:
    """Read a MATLAB cell array of strings stored as HDF5 object references."""
    import h5py  # noqa: PLC0415
    data = ds[()]
    # Some files store strings directly as variable-length dtype.
    if h5py.check_string_dtype(ds.dtype):
        raw = ds.asstr()[()]
        return [str(s).strip() for s in np.asarray(raw).flatten()]
    # Object reference array — each ref points to a uint16 character dataset.
    strings: list[str] = []
    for ref in np.asarray(data).flatten():
        chars = f[ref][()]
        strings.append("".join(chr(int(c)) for c in np.asarray(chars).flatten()))
    return strings


def _hdf5_scalar(f: object, ds: object) -> float | None:
    try:
        val = ds[()]
        return float(np.asarray(val).flat[0])
    except Exception:  # noqa: BLE001
        return None


def _hdf5_chan_info_column(f: object, ds: object, col: int) -> list[str]:
    """Extract one column (0-indexed) from a MATLAB chan_info cell array in HDF5.

    chan_info is stored as an (N_rows, 3) array of object references.
    Each cell is a uint16 character array (MATLAB string encoding).
    """
    import h5py  # noqa: PLC0415
    data = np.asarray(ds[()])
    if data.ndim == 1:
        # Single-column squeeze — treat as if col==0
        refs = data
    elif data.ndim == 2:
        # HDF5 transposes MATLAB arrays: shape is (3, N) not (N, 3)
        # so column `col` in MATLAB == row `col` in HDF5
        if col < data.shape[0]:
            refs = data[col, :]
        else:
            refs = data[0, :]
    else:
        return []

    strings: list[str] = []
    for ref in refs.flatten():
        try:
            chars = f[ref][()]
            strings.append("".join(chr(int(c)) for c in np.asarray(chars).flatten()))
        except Exception:  # noqa: BLE001
            strings.append("")
    return strings


def _hdf5_is_matlab_empty(ds: object) -> bool:
    """Return True when a v7.3 dataset represents an empty MATLAB array."""
    try:
        attr = ds.attrs.get("MATLAB_empty")
    except Exception:  # noqa: BLE001
        return False
    if attr is None:
        return False
    try:
        return bool(np.asarray(attr).flat[0])
    except Exception:  # noqa: BLE001
        return False


def _hdf5_numeric_vector(ds: object) -> np.ndarray:
    """Read a numeric vector from HDF5, preserving MATLAB empty arrays."""
    if _hdf5_is_matlab_empty(ds):
        return np.empty((0,), dtype=np.float64)
    return np.asarray(ds[()], dtype=np.float64).ravel()


# --- opts_log extraction helpers (scipy + h5py) ---------------------------

def _extract_opts_log_scipy(mat: dict) -> dict:
    """Extract trial/channel rejection info from opts_log in a scipy-loaded MATLAB file.

    Returns a dict with the following optional keys:
        ``outlier_trials_mean`` / ``outlier_trials_max``: bool (n_trials, n_chan)
        ``bad_channels_mean``   / ``bad_channels_max``:  bool (n_chan,)
        ``high_nan_channel_indices``: int64 1-D, 0-based
        ``neg_rating_indices``  / ``outlier_rt_indices``: int64 1-D, 0-based
    """
    opts_log_raw = mat.get("opts_log")
    if opts_log_raw is None:
        return {}
    result: dict = {}
    try:
        hga = getattr(opts_log_raw, "HGA", None)
        if hga is not None:
            for key, attr in [
                ("outlier_trials_mean", "OutlierTrialsMean"),
                ("outlier_trials_max",  "OutlierTrialsMax"),
            ]:
                val = getattr(hga, attr, None)
                if val is not None:
                    arr = np.asarray(val, dtype=bool)
                    if arr.ndim == 2:
                        result[key] = arr  # (n_trials, n_chan)
            for key, attr in [
                ("bad_channels_mean", "channelsToRemoveSD"),
                ("bad_channels_max",  "channelsToRemoveSD_max"),
            ]:
                val = getattr(hga, attr, None)
                if val is not None:
                    result[key] = np.asarray(val, dtype=bool).flatten()
            for key, attr in [
                ("mean_trial_vals", "MeanTrialVals"),
                ("max_trial_vals",  "MaxTrialVals"),
            ]:
                val = getattr(hga, attr, None)
                if val is not None:
                    arr = np.asarray(val, dtype=float)
                    if arr.ndim == 2:
                        result[key] = arr  # (n_trials, n_chan)
            for key, attr in [
                ("allchannels_sd",     "allchannelsSd"),
                ("allchannels_sd_max", "allchannelsSd_max"),
            ]:
                val = getattr(hga, attr, None)
                if val is not None:
                    result[key] = np.asarray(val, dtype=float).flatten()
            val = getattr(hga, "channelsToRemove_25pcTrialsRemoved", None)
            if val is not None:
                arr = np.asarray(val, dtype=float).flatten()
                arr = arr[np.isfinite(arr)]
                result["high_nan_channel_indices"] = arr.astype(int) - 1
        for key, attr in [
            ("neg_rating_indices", "negratings"),
            ("outlier_rt_indices", "outlierRTs"),
        ]:
            val = getattr(opts_log_raw, attr, None)
            if val is not None:
                arr = np.asarray(val, dtype=float).flatten()
                result[key] = arr[np.isfinite(arr)].astype(int) - 1  # 0-based
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: could not parse opts_log: {exc}")
    return result


def _extract_opts_log_hdf5(f: object) -> dict:
    """Extract trial/channel rejection info from opts_log in an HDF5 MATLAB file.

    Same return format as :func:`_extract_opts_log_scipy`.
    """
    result: dict = {}
    if "opts_log" not in f:
        return result
    og = f["opts_log"]
    try:
        if "HGA" in og:
            hga = og["HGA"]
            for key, ds_name in [
                ("outlier_trials_mean", "OutlierTrialsMean"),
                ("outlier_trials_max",  "OutlierTrialsMax"),
            ]:
                if ds_name in hga:
                    # HDF5 transposes MATLAB (n_trials, n_chan) → (n_chan, n_trials)
                    arr = np.asarray(hga[ds_name][()], dtype=bool)
                    if arr.ndim == 2:
                        arr = arr.T  # back to (n_trials, n_chan)
                    result[key] = arr
            for key, ds_name in [
                ("bad_channels_mean", "channelsToRemoveSD"),
                ("bad_channels_max",  "channelsToRemoveSD_max"),
            ]:
                if ds_name in hga:
                    result[key] = np.asarray(hga[ds_name][()], dtype=bool).flatten()
            for key, ds_name in [
                ("mean_trial_vals", "MeanTrialVals"),
                ("max_trial_vals",  "MaxTrialVals"),
            ]:
                if ds_name in hga:
                    arr = np.asarray(hga[ds_name][()], dtype=float)
                    if arr.ndim == 2:
                        arr = arr.T  # HDF5 transposes MATLAB (n_trials, n_chan) → (n_chan, n_trials)
                    result[key] = arr
            for key, ds_name in [
                ("allchannels_sd",     "allchannelsSd"),
                ("allchannels_sd_max", "allchannelsSd_max"),
            ]:
                if ds_name in hga:
                    result[key] = np.asarray(hga[ds_name][()], dtype=float).flatten()
            if "channelsToRemove_25pcTrialsRemoved" in hga:
                arr = _hdf5_numeric_vector(hga["channelsToRemove_25pcTrialsRemoved"])
                result["high_nan_channel_indices"] = arr[np.isfinite(arr)].astype(int) - 1
        for key, ds_name in [
            ("neg_rating_indices", "negratings"),
            ("outlier_rt_indices", "outlierRTs"),
        ]:
            if ds_name in og:
                arr = _hdf5_numeric_vector(og[ds_name])
                result[key] = arr[np.isfinite(arr)].astype(int) - 1  # 0-based
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: could not parse opts_log from HDF5: {exc}")
    return result


def _load_hdf5_matlab(path: Path) -> tuple[np.ndarray, list[str], float, dict, np.ndarray | None]:
    """Load alldata, channel names, sfreq, opts_log info, and hdr.timelist from MATLAB v7.3."""
    import h5py  # noqa: PLC0415

    with h5py.File(str(path), "r") as f:
        # alldata: MATLAB stores (N_trials, N_samples, N_bipoles) as Fortran-order.
        # h5py reads it transposed → (N_bipoles, N_samples, N_trials).
        # Reversing all axes gives back (N_trials, N_samples, N_bipoles).
        raw_data = f["alldata"][()]
        alldata: np.ndarray = raw_data.T  # (N_trials, N_samples, N_bipoles)

        # Channel names from hdr.chan_info column 1 (bipole name), fallback hdr.label
        channels: list[str] = []
        sfreq: float = _sfreq_from_filename(path)
        timelist: np.ndarray | None = None

        if "hdr" in f:
            hdr_group = f["hdr"]
            if "chan_info" in hdr_group:
                channels = _hdf5_chan_info_column(f, hdr_group["chan_info"], col=1)
            elif "label" in hdr_group:
                channels = _hdf5_cell_strings(f, hdr_group["label"])
            if "timelist" in hdr_group:
                timelist = np.asarray(hdr_group["timelist"][()], dtype=np.float64).ravel()
            for fname in ("Fs", "fsample", "fs", "srate", "sfreq"):
                if fname in hdr_group:
                    val = _hdf5_scalar(f, hdr_group[fname])
                    if val is not None:
                        sfreq = val
                        break

        opts_log_info = _extract_opts_log_hdf5(f)

    return alldata, channels, sfreq, opts_log_info, timelist


def load_matlab(path: Path) -> tuple[np.ndarray, list[str], float, dict, np.ndarray | None]:
    """Load alldata, channel names, sfreq, opts_log info, and hdr.timelist when available.

    alldata shape: (N_trials, N_samples, N_bipoles)
    opts_log_info: dict with trial/channel rejection info (may be empty if not in file)
    """
    try:
        mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        # MATLAB v7.3 — HDF5-based
        print("  (MATLAB v7.3 detected, using h5py loader)")
        return _load_hdf5_matlab(path)

    alldata = np.asarray(mat["alldata"])
    # With squeeze_me, a (1,N,M) would collapse; make sure we have 3D.
    if alldata.ndim == 2:
        # Single trial: (N_samples, N_bipoles) → add trial dimension
        alldata = alldata[np.newaxis, ...]
    if alldata.ndim != 3:
        raise ValueError(f"Unexpected alldata shape: {alldata.shape}")

    hdr = mat.get("hdr")
    channels: list[str] = []
    sfreq: float = _sfreq_from_filename(path)
    timelist: np.ndarray | None = None

    if hdr is not None:
        channels = _extract_channels_scipy(hdr)
        sfreq = _extract_sfreq_scipy(hdr, fallback=sfreq)
        timelist_raw = getattr(hdr, "timelist", None)
        if timelist_raw is not None:
            timelist = np.asarray(timelist_raw, dtype=np.float64).ravel()

    opts_log_info = _extract_opts_log_scipy(mat)
    return alldata, channels, sfreq, opts_log_info, timelist


def _matlab_nearest_index(time_axis_s: np.ndarray, target_s: float) -> int:
    """Return the MATLAB-style nearest-sample index for a target time."""
    time_axis = np.asarray(time_axis_s, dtype=np.float64).ravel()
    return int(np.argmin(np.abs(time_axis - float(target_s))))


def _matlab_baseline_mask(
    time_axis_s: np.ndarray,
    baseline_tmin_s: float,
    baseline_tmax_s: float,
) -> tuple[np.ndarray, int, int]:
    """Return the exact baseline samples used by MATLAB's nearest-index logic."""
    time_axis = np.asarray(time_axis_s, dtype=np.float64).ravel()
    idx_start = _matlab_nearest_index(time_axis, baseline_tmin_s)
    idx_stop = _matlab_nearest_index(time_axis, baseline_tmax_s) - 1
    if idx_stop < idx_start:
        raise ValueError(
            "MATLAB baseline window is empty after applying the exclusive upper bound: "
            f"{baseline_tmin_s=} {baseline_tmax_s=}."
        )
    mask = np.zeros(time_axis.shape, dtype=bool)
    mask[idx_start:idx_stop + 1] = True
    return mask, idx_start, idx_stop


# ===========================================================================
# BrainVision epoching
# ===========================================================================

def _annotation_code(description: str) -> str | None:
    """Extract numeric event code from a BrainVision annotation description."""
    m = re.search(r"S\s*(\d+)", str(description))
    return m.group(1) if m else None


def epoch_brainvision(
    raw: mne.io.BaseRaw,
    tmin_s: float = TMIN_S,
    tmax_s: float = TMAX_S,
    anchor_codes: set[str] = ANCHOR_CODES,
    experiment_start_code: str = EXPERIMENT_START_CODE,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Epoch the continuous BrainVision recording.

    Returns:
        epochs     : (N_trials, N_channels, N_samples)  float64
        time_axis  : (N_samples,)  seconds relative to anchor
        ch_names   : list of channel names in BrainVision order
    """
    sfreq = float(raw.info["sfreq"])

    # --- Find experiment start ---
    t_start = -np.inf
    for ann in raw.annotations:
        if _annotation_code(str(ann["description"])) == experiment_start_code:
            t_start = float(ann["onset"])
            break  # first occurrence

    if np.isinf(t_start):
        print(f"  WARNING: experiment start event '{experiment_start_code}' not found."
              " Epoching over the entire recording.")

    # --- Collect anchor event onsets (sorted, after experiment start) ---
    anchor_onsets: list[float] = []
    for ann in raw.annotations:
        code = _annotation_code(str(ann["description"]))
        if code in anchor_codes:
            onset = float(ann["onset"])
            if onset > t_start:
                anchor_onsets.append(onset)
    anchor_onsets.sort()

    print(f"  Found {len(anchor_onsets)} anchor events [codes {sorted(anchor_codes)}]"
          f" after t_start={t_start:.3f}s")
    if not anchor_onsets:
        raise RuntimeError("No anchor events found in BrainVision file.")

    # --- Build MNE events array from onsets ---
    n_trials = len(anchor_onsets)
    events = np.column_stack([
        np.round(np.array(anchor_onsets) * sfreq).astype(np.int64) + BV_EVENT_SAMPLE_SHIFT_SAMPLES,
        np.zeros(n_trials, dtype=np.int64),
        np.ones(n_trials, dtype=np.int64),
    ])

    # --- Extract epochs via MNE ---
    epochs_mne = mne.Epochs(
        raw,
        events=events,
        event_id={"anchor": 1},
        tmin=tmin_s,
        tmax=tmax_s,
        baseline=None,
        preload=True,
        verbose=False,
        reject_by_annotation=False,
    )
    data = epochs_mne.get_data()                          # (N_kept, N_ch, N_samp)
    time_axis = epochs_mne.times                          # seconds

    dropped = n_trials - data.shape[0]
    if dropped:
        print(f"  WARNING: {dropped} trials were dropped by MNE (edge effects).")

    return data, time_axis, list(raw.ch_names)


# ===========================================================================
# Channel matching
# ===========================================================================

# Regex to extract the *first contact* from a concatenated bipole name.
# e.g.  "Bp02Bp01" → "Bp02",  "AIC3AIC2" → "AIC3",  "A01B02" → "A01"
# Pattern: one or more letters followed by one or more digits.
_FIRST_CONTACT_RE = re.compile(r"^([A-Za-z]+\d+)")


def _first_contact(bipole_name: str) -> str:
    """Return the first electrode contact from a concatenated bipole string.

    - ``"Bp02Bp01"``  → ``"Bp02"``
    - ``"AIC3AIC2"``  → ``"AIC3"``
    - Already a bare contact (``"Bp02"``) → returned as-is.
    """
    m = _FIRST_CONTACT_RE.match(bipole_name)
    return m.group(1) if m else bipole_name


def _norm(name: str) -> str:
    """Normalize a channel name for fuzzy comparison: lowercase, no separators."""
    return re.sub(r"[\s\-_\.]", "", name).casefold()


def match_channels(
    matlab_channels: list[str],
    bv_channels: list[str],
) -> dict[int, int]:
    """Return {matlab_idx: bv_idx} for channels matched by first-contact name.

    MATLAB stores full bipole names like ``"Bp02Bp01"``; BrainVision stores
    only the first contact ``"Bp02"``.  The first contact is extracted from
    the MATLAB name before normalised comparison.
    """
    # Index BV channels by their normalised name
    bv_by_norm: dict[str, int] = {}
    for i, ch in enumerate(bv_channels):
        bv_by_norm[_norm(ch)] = i

    mapping: dict[int, int] = {}
    for mi, mc in enumerate(matlab_channels):
        first = _first_contact(mc)
        norm_mc = _norm(first)
        if norm_mc in bv_by_norm:
            mapping[mi] = bv_by_norm[norm_mc]

    return mapping


# ===========================================================================
# Rejection comparison
# ===========================================================================

def print_rejection_comparison(
    mat_rej: dict,
    py_rej: dict,
    mat_channels: list[str],
    bv_channels: list[str],
    channel_mapping: dict[int, int],
    n_trials: int,
) -> None:
    """Print a side-by-side comparison of MATLAB opts_log vs Python pipeline rejections."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("REJECTION COMPARISON  —  MATLAB opts_log  vs  Python pipeline")
    print(sep)

    if not mat_rej:
        print("  (opts_log not found in MATLAB file — skipping comparison)")
        print(sep + "\n")
        return

    mat_mi = sorted(channel_mapping.keys())
    bv_mi  = [channel_mapping[m] for m in mat_mi]

    def _b(mask: np.ndarray | None, idx: int) -> str:
        """Single-channel bool flag from a 1-D mask."""
        if mask is None or idx >= len(mask):
            return "n/a"
        return "YES" if bool(mask[idx]) else "no"

    # ── Channel-level ─────────────────────────────────────────────────────────
    print("\n▸ CHANNEL-LEVEL REJECTIONS  (matched channels only)")
    print(f"  {'MATLAB channel':<20} {'BV channel':<16}"
          f"  {'MAT mean':>9} {'PY mean':>9}  {'MAT max':>9} {'PY max':>9}  {'Agree?':>7}")
    n_agree = 0
    for mi, bi in zip(mat_mi, bv_mi):
        mat_m = _b(mat_rej.get("bad_channels_mean"), mi)
        py_m  = _b(py_rej.get("bad_channels_mean"),  bi)
        mat_x = _b(mat_rej.get("bad_channels_max"),  mi)
        py_x  = _b(py_rej.get("bad_channels_max"),   bi)
        mat_any = mat_m == "YES" or mat_x == "YES"
        py_any  = py_m  == "YES" or py_x  == "YES"
        agree = "✓" if mat_any == py_any else "✗"
        if mat_any == py_any:
            n_agree += 1
        print(f"  {mat_channels[mi]:<20} {bv_channels[bi]:<16}"
              f"  {mat_m:>9} {py_m:>9}  {mat_x:>9} {py_x:>9}  {agree:>7}")
    print(f"  → {n_agree}/{len(mat_mi)} matched channels agree on overall bad-channel status")

    # ── Channel SD values (numeric) ────────────────────────────────────────────
    mat_sd  = mat_rej.get("allchannels_sd")
    py_sd   = py_rej.get("allchannels_sd")
    mat_sdx = mat_rej.get("allchannels_sd_max")
    py_sdx  = py_rej.get("allchannels_sd_max")

    if any(x is not None for x in [mat_sd, py_sd, mat_sdx, py_sdx]):
        print("\n\u25b8 CHANNEL-LEVEL SD VALUES  (spread of per-trial means/maxes across trials)")
        print(f"  {'MATLAB ch':<20} {'BV ch':<16}"
              f"  {'MAT sd':>10} {'PY sd':>10}  {'\u0394 sd':>9}"
              f"  {'MAT sd_max':>10} {'PY sd_max':>10}  {'\u0394 sd_max':>9}")

        def _fv(arr: np.ndarray | None, idx: int) -> str:
            if arr is None or idx >= len(arr):
                return "n/a"
            v = arr[idx]
            return f"{v:.5f}" if np.isfinite(v) else "nan"

        def _fd(a1: np.ndarray | None, a2: np.ndarray | None, i1: int, i2: int) -> str:
            if a1 is None or a2 is None or i1 >= len(a1) or i2 >= len(a2):
                return "n/a"
            v1, v2 = a1[i1], a2[i2]
            return f"{v1 - v2:+.5f}" if (np.isfinite(v1) and np.isfinite(v2)) else "nan"

        for mi, bi in zip(mat_mi, bv_mi):
            ms  = _fv(mat_sd,  mi);  ps  = _fv(py_sd,  bi);  ds  = _fd(mat_sd,  py_sd,  mi, bi)
            msx = _fv(mat_sdx, mi);  psx = _fv(py_sdx, bi);  dsx = _fd(mat_sdx, py_sdx, mi, bi)
            print(f"  {mat_channels[mi]:<20} {bv_channels[bi]:<16}"
                  f"  {ms:>10} {ps:>10}  {ds:>9}"
                  f"  {msx:>10} {psx:>10}  {dsx:>9}")

    # ── Trial-level (outlier mean/max) ────────────────────────────────────────
    mat_otm = mat_rej.get("outlier_trials_mean")
    mat_otx = mat_rej.get("outlier_trials_max")
    py_otm  = py_rej.get("outlier_trials_mean")
    py_otx  = py_rej.get("outlier_trials_max")

    if mat_otm is not None or py_otm is not None:
        print("\n▸ TRIAL-LEVEL OUTLIER REJECTIONS  (discrepancies only; 1-based trial numbers)")
        n_disc = 0
        for mi, bi in zip(mat_mi, bv_mi):
            def _trial_set(mask: np.ndarray | None, ch: int) -> set[int]:
                if mask is None or mask.ndim < 2 or ch >= mask.shape[1]:
                    return set()
                n = min(n_trials, mask.shape[0])
                return set(int(t) for t in np.flatnonzero(mask[:n, ch]))

            mat_all = _trial_set(mat_otm, mi) | _trial_set(mat_otx, mi)
            py_all  = _trial_set(py_otm,  bi) | _trial_set(py_otx,  bi)
            only_mat = sorted(mat_all - py_all)
            only_py  = sorted(py_all  - mat_all)
            if only_mat or only_py:
                n_disc += 1
                print(f"  {mat_channels[mi]} → {bv_channels[bi]}:")
                if only_mat:
                    print(f"    Only MATLAB rejected: trials {[t + 1 for t in only_mat]}")
                if only_py:
                    print(f"    Only Python rejected: trials {[t + 1 for t in only_py]}")
        if n_disc == 0:
            print(f"  → All {len(mat_mi)} matched channels agree on trial-level outlier rejections ✓")
        else:
            print(f"  → {len(mat_mi) - n_disc}/{len(mat_mi)} matched channels agree on trial-level rejections")

    # ── Negative ratings ──────────────────────────────────────────────────────
    mat_neg  = mat_rej.get("neg_rating_indices")
    py_neg_m = py_rej.get("neg_rating_mask")
    print("\n▸ NEGATIVE RATING TRIAL REJECTIONS")
    mat_neg_s: set[int] = (set(int(i) for i in mat_neg if 0 <= int(i) < n_trials)
                           if mat_neg is not None else set())
    py_neg_s:  set[int] = (set(int(i) for i in np.flatnonzero(py_neg_m[:n_trials]))
                           if py_neg_m is not None else set())
    print(f"  MATLAB ({len(mat_neg_s)} trials): {sorted(t + 1 for t in mat_neg_s) or '(none)'}")
    print(f"  Python ({len(py_neg_s)} trials): {sorted(t + 1 for t in py_neg_s) or '(none)'}")
    if mat_neg_s == py_neg_s:
        print("  → Perfect agreement ✓")
    else:
        only_mat = sorted(mat_neg_s - py_neg_s)
        only_py  = sorted(py_neg_s  - mat_neg_s)
        if only_mat:
            print(f"  → Only MATLAB rejected: {[t + 1 for t in only_mat]}")
        if only_py:
            print(f"  → Only Python rejected: {[t + 1 for t in only_py]}")

    print(sep + "\n")


def print_bv_start_offset_diagnostics(
    mat_epochs: np.ndarray,
    bv_epochs: np.ndarray,
    bv_time: np.ndarray,
    channel_mapping: dict[int, int],
    *,
    max_offset: int = SHIFT_DIAGNOSTIC_MAX_OFFSET,
) -> None:
    """Report which BV start offset best matches the MATLAB matrix."""
    extra = int(bv_epochs.shape[2] - mat_epochs.shape[1])
    if extra < 0 or not channel_mapping:
        return

    max_offset = min(max_offset, extra)
    print("\nAlignment diagnostic — BV start offset vs MATLAB")
    results: list[tuple[int, float, float, float, float]] = []
    for start in range(max_offset + 1):
        end = start + mat_epochs.shape[1]
        mean_abs_values: list[float] = []
        corrs: list[float] = []
        for mat_idx, bv_idx in sorted(channel_mapping.items()):
            diff = bv_epochs[:, bv_idx, start:end] - mat_epochs[:, :, mat_idx]
            abs_diff = np.abs(diff)
            if np.isfinite(abs_diff).any():
                mean_abs_values.append(float(np.nanmean(abs_diff)))
            a = mat_epochs[:, :, mat_idx].ravel()
            b = bv_epochs[:, bv_idx, start:end].ravel()
            finite = np.isfinite(a) & np.isfinite(b)
            if np.sum(finite) > 2:
                a = a[finite] - np.mean(a[finite])
                b = b[finite] - np.mean(b[finite])
                denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
                if denom > 0.0:
                    corrs.append(float(np.sum(a * b) / denom))
        if mean_abs_values:
            results.append((
                start,
                float(bv_time[start]),
                float(bv_time[end - 1]),
                float(np.mean(mean_abs_values)),
                float(np.nanmedian(corrs)) if corrs else np.nan,
            ))

    for start, t0, t1, mean_abs, med_corr in sorted(results, key=lambda row: row[3]):
        print(
            f"  start={start}  BV window {t0:.3f}→{t1:.3f} s  "
            f"mean|Δ|={mean_abs:.6f}  median corr={med_corr:.6f}"
        )


def _keep_mask_from_trial_means(
    trial_means: np.ndarray,
    *,
    method: str = BASELINE_OUTLIER_METHOD,
) -> tuple[np.ndarray, np.ndarray]:
    """Return keep/outlier masks from per-trial baseline means.

    Parameters
    ----------
    trial_means:
        Array with shape (n_trials, n_channels).
    method:
        "median_mad" or "mean", mirroring the supported MATLAB/Python modes.
    """
    arr = np.asarray(trial_means, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            "trial_means must be 2-D with shape (n_trials, n_channels), "
            f"got {arr.shape!r}."
        )
    outlier = np.zeros(arr.shape, dtype=bool)
    for ch_idx in range(arr.shape[1]):
        values = arr[:, ch_idx]
        finite = np.isfinite(values)
        finite_values = values[finite]
        if finite_values.size < 3:
            continue
        if method == "mean":
            center = float(np.mean(finite_values))
            spread = float(np.std(finite_values, ddof=1))
            if not np.isfinite(spread) or spread <= 0.0:
                continue
            threshold = 3.0 * spread
        else:
            center = float(np.median(finite_values))
            mad = float(np.median(np.abs(finite_values - center)))
            if not np.isfinite(mad) or mad <= 0.0:
                continue
            threshold = 3.0 * 1.4826 * mad
        bad = np.abs(finite_values - center) > threshold
        if np.any(bad):
            finite_indices = np.flatnonzero(finite)
            outlier[finite_indices[bad], ch_idx] = True
    keep = np.isfinite(arr) & ~outlier
    return keep, outlier


def _map_matlab_keep_mask_to_bv(
    mat_keep_mask: np.ndarray,
    channel_mapping: dict[int, int],
    *,
    n_trials: int,
    n_bv_channels: int,
) -> np.ndarray:
    """Map a MATLAB keep-mask (n_trials, n_mat_channels) to BV channel space."""
    mat_keep = np.asarray(mat_keep_mask, dtype=bool)
    if mat_keep.ndim != 2:
        raise ValueError(
            "mat_keep_mask must be 2-D with shape (n_trials, n_mat_channels), "
            f"got {mat_keep.shape!r}."
        )
    mapped = np.zeros((n_trials, n_bv_channels), dtype=bool)
    for mat_idx, bv_idx in channel_mapping.items():
        if mat_idx >= mat_keep.shape[1] or bv_idx >= n_bv_channels:
            continue
        n = min(n_trials, mat_keep.shape[0])
        mapped[:n, bv_idx] = mat_keep[:n, mat_idx]
    return mapped


def _mask_agreement_summary(
    reference_keep: np.ndarray,
    candidate_keep: np.ndarray,
    channel_indices: list[int],
    channel_names: list[str],
    *,
    top_n: int = BASELINE_EXPERIMENT_TOP_N,
) -> dict[str, object]:
    """Summarize mask agreement on the selected BV channels."""
    ref = np.asarray(reference_keep, dtype=bool)
    cand = np.asarray(candidate_keep, dtype=bool)
    if ref.shape != cand.shape:
        raise ValueError(f"Mask-shape mismatch: {ref.shape!r} vs {cand.shape!r}.")
    if not channel_indices:
        return {
            "pair_agreement_pct": float("nan"),
            "exact_channel_matches": 0,
            "n_channels": 0,
            "mismatched_pairs": 0,
            "worst_channels": [],
        }

    ref_sel = ref[:, channel_indices]
    cand_sel = cand[:, channel_indices]
    same = ref_sel == cand_sel
    mismatches_per_channel = np.sum(~same, axis=0).astype(int)
    worst_rows: list[tuple[int, str, int]] = []
    for local_idx, bv_idx in enumerate(channel_indices):
        mism = int(mismatches_per_channel[local_idx])
        if mism > 0:
            worst_rows.append((int(bv_idx), channel_names[bv_idx], mism))
    worst_rows.sort(key=lambda row: row[2], reverse=True)
    return {
        "pair_agreement_pct": 100.0 * float(np.mean(same)),
        "exact_channel_matches": int(np.sum(mismatches_per_channel == 0)),
        "n_channels": len(channel_indices),
        "mismatched_pairs": int(np.sum(mismatches_per_channel)),
        "worst_channels": worst_rows[:top_n],
    }


def _zscore_epochs_from_keep_mask(
    epochs: np.ndarray,
    baseline_mask: np.ndarray,
    keep_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply a global baseline z-score using an explicit keep-mask."""
    arr = np.asarray(epochs, dtype=np.float64)
    keep = np.asarray(keep_mask, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        trial_means = np.nanmean(arr[:, :, baseline_mask], axis=2)
    if keep.shape != trial_means.shape:
        raise ValueError(
            "keep_mask must match trial baseline means shape, got "
            f"{keep.shape!r} vs {trial_means.shape!r}."
        )
    clean = np.full_like(trial_means, np.nan)
    clean[keep] = trial_means[keep]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref_mean = np.nanmean(clean, axis=0)
        ref_std = np.nanstd(clean, axis=0, ddof=1)
    scale = np.where(np.isfinite(ref_std) & (ref_std > 0.0), ref_std, 1.0)
    z_epochs = (arr - ref_mean[np.newaxis, :, np.newaxis]) / scale[np.newaxis, :, np.newaxis]
    z_epochs[~np.isfinite(arr)] = np.nan
    n_clean = np.sum(np.isfinite(clean), axis=0).astype(int)
    return z_epochs, ref_mean, ref_std, n_clean


def _matched_epoch_metrics(
    mat_epochs: np.ndarray,
    bv_epochs: np.ndarray,
    mat_time: np.ndarray,
    bv_time: np.ndarray,
    channel_mapping: dict[int, int],
) -> tuple[float, float]:
    """Return aggregate mean|diff| and median corr on matched channels."""
    if not channel_mapping:
        return float("nan"), float("nan")
    mat_time_arr = np.asarray(mat_time, dtype=np.float64).ravel()
    bv_time_arr = np.asarray(bv_time, dtype=np.float64).ravel()
    bv_mask = (
        (bv_time_arr >= float(mat_time_arr[0]) - 1e-12)
        & (bv_time_arr <= float(mat_time_arr[-1]) + 1e-12)
    )
    mean_abs_values: list[float] = []
    corrs: list[float] = []
    for mat_idx, bv_idx in sorted(channel_mapping.items()):
        mat_arr = np.asarray(mat_epochs[:, :, mat_idx], dtype=np.float64)
        bv_arr = np.asarray(bv_epochs[:, bv_idx, :][:, bv_mask], dtype=np.float64)
        n_common = min(mat_arr.shape[1], bv_arr.shape[1])
        if n_common <= 0:
            continue
        mat_arr = mat_arr[:, :n_common]
        bv_arr = bv_arr[:, :n_common]
        diff = bv_arr - mat_arr
        if np.isfinite(diff).any():
            mean_abs_values.append(float(np.nanmean(np.abs(diff))))
        a = mat_arr.ravel()
        b = bv_arr.ravel()
        finite = np.isfinite(a) & np.isfinite(b)
        if np.sum(finite) > 2:
            a = a[finite] - np.mean(a[finite])
            b = b[finite] - np.mean(b[finite])
            denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
            if denom > 0.0:
                corrs.append(float(np.sum(a * b) / denom))
    mean_abs = float(np.mean(mean_abs_values)) if mean_abs_values else float("nan")
    med_corr = float(np.nanmedian(corrs)) if corrs else float("nan")
    return mean_abs, med_corr


def _sigma_error_summary(
    mat_std: np.ndarray,
    bv_std: np.ndarray,
    channel_mapping: dict[int, int],
) -> tuple[float, float]:
    """Return median and max absolute sigma error (%) across matched channels."""
    pct_errors: list[float] = []
    for mat_idx, bv_idx in sorted(channel_mapping.items()):
        if mat_idx >= len(mat_std) or bv_idx >= len(bv_std):
            continue
        mat_s = float(mat_std[mat_idx])
        bv_s = float(bv_std[bv_idx])
        if not (np.isfinite(mat_s) and np.isfinite(bv_s) and mat_s > 0.0):
            continue
        pct_errors.append(abs(bv_s / mat_s - 1.0) * 100.0)
    if not pct_errors:
        return float("nan"), float("nan")
    return float(np.median(pct_errors)), float(np.max(pct_errors))


def print_baseline_strategy_experiments(
    *,
    mat_epochs: np.ndarray,
    mat_time: np.ndarray,
    bv_epochs_pre_zscore: np.ndarray,
    bv_time: np.ndarray,
    mat_channels: list[str],
    bv_channels: list[str],
    channel_mapping: dict[int, int],
    mat_bsl_info: dict | None,
    top_n: int = BASELINE_EXPERIMENT_TOP_N,
) -> None:
    """Benchmark mask and z-score strategies against the MATLAB reference."""
    if (
        mat_bsl_info is None
        or mat_bsl_info.get("per_trial_bl_means") is None
        or mat_bsl_info.get("baseline_keep_mask") is None
        or not channel_mapping
    ):
        return

    mat_trial_means_raw = np.asarray(mat_bsl_info["per_trial_bl_means"], dtype=np.float64)
    mat_keep_raw = np.asarray(mat_bsl_info["baseline_keep_mask"], dtype=bool)
    if mat_trial_means_raw.ndim != 2 or mat_keep_raw.ndim != 2:
        return

    n_trials = min(mat_epochs.shape[0], bv_epochs_pre_zscore.shape[0], mat_trial_means_raw.shape[1], mat_keep_raw.shape[1])
    if n_trials <= 0:
        return

    matched_bv_indices = [channel_mapping[mi] for mi in sorted(channel_mapping)]
    mat_trial_means = mat_trial_means_raw[:, :n_trials].T
    mat_keep = mat_keep_raw[:, :n_trials].T
    py_keep_on_mat, _ = _keep_mask_from_trial_means(mat_trial_means, method=BASELINE_OUTLIER_METHOD)
    mat_mask_summary = _mask_agreement_summary(
        mat_keep,
        py_keep_on_mat,
        list(range(mat_keep.shape[1])),
        mat_channels,
        top_n=top_n,
    )

    bl_mask, _, _ = _matlab_baseline_mask(
        bv_time,
        BASELINE_TMIN_S,
        BASELINE_TMAX_S,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bv_trial_means = np.nanmean(
            np.asarray(bv_epochs_pre_zscore[:n_trials], dtype=np.float64)[:, :, bl_mask],
            axis=2,
        )
    mapped_mat_keep = _map_matlab_keep_mask_to_bv(
        mat_keep,
        channel_mapping,
        n_trials=n_trials,
        n_bv_channels=bv_trial_means.shape[1],
    )

    experiments: list[tuple[str, np.ndarray]] = []
    py_keep_bv, _ = _keep_mask_from_trial_means(bv_trial_means, method=BASELINE_OUTLIER_METHOD)
    experiments.append(("BV python keep-mask", py_keep_bv))
    py_keep_bv_f32, _ = _keep_mask_from_trial_means(
        np.asarray(bv_trial_means, dtype=np.float32),
        method=BASELINE_OUTLIER_METHOD,
    )
    experiments.append(("BV python keep-mask (float32 means)", py_keep_bv_f32))
    experiments.append(("BV oracle MATLAB keep-mask", mapped_mat_keep))

    print("\n--- Baseline strategy experiments ---")
    print(
        "  Python mask on MATLAB saved baseline means: "
        f"pair agreement={mat_mask_summary['pair_agreement_pct']:.4f}%  "
        f"exact channels={mat_mask_summary['exact_channel_matches']}/{mat_mask_summary['n_channels']}  "
        f"mismatched pairs={mat_mask_summary['mismatched_pairs']}"
    )

    print(
        f"  {'Strategy':<34}  {'pair agree':>10}  {'exact ch':>9}  "
        f"{'med |sigma|':>11}  {'max |sigma|':>11}  {'mean|Δ|':>9}  {'med corr':>9}"
    )
    for label, keep_mask in experiments:
        mask_summary = _mask_agreement_summary(
            mapped_mat_keep,
            keep_mask,
            matched_bv_indices,
            bv_channels,
            top_n=top_n,
        )
        z_epochs, _ref_mean, ref_std, _n_clean = _zscore_epochs_from_keep_mask(
            bv_epochs_pre_zscore[:n_trials],
            bl_mask,
            keep_mask[:n_trials],
        )
        sigma_median_pct, sigma_max_pct = _sigma_error_summary(
            np.asarray(mat_bsl_info["std"], dtype=np.float64),
            ref_std,
            channel_mapping,
        )
        mean_abs, med_corr = _matched_epoch_metrics(
            mat_epochs[:n_trials],
            z_epochs,
            mat_time,
            bv_time,
            channel_mapping,
        )
        print(
            f"  {label:<34}  "
            f"{mask_summary['pair_agreement_pct']:10.4f}%  "
            f"{mask_summary['exact_channel_matches']:4d}/{mask_summary['n_channels']:<4d}  "
            f"{sigma_median_pct:11.4f}%  {sigma_max_pct:11.4f}%  "
            f"{mean_abs:9.6f}  {med_corr:9.6f}"
        )
        worst = mask_summary["worst_channels"]
        if worst and label == "BV python keep-mask":
            print("    Worst BV keep-mask mismatches:")
            for bv_idx, name, mism in worst:
                print(f"      [{bv_idx:3d}] {name:<12} mismatched trials: {mism}")
    print("---")


# ===========================================================================
# Interactive viewer
# ===========================================================================

class ComparisonViewer:
    """Matplotlib-based viewer to compare MATLAB vs BrainVision epoch traces.

    Controls:
        "Channel" slider / buttons : iterate over matched channels
        "Trial" slider / buttons   : iterate over trials (1-indexed display)
    """

    def __init__(
        self,
        mat_epochs: np.ndarray,    # (N_trials_mat, N_samples_mat, N_bipoles)
        mat_time: np.ndarray,      # (N_samples_mat,)
        mat_channels: list[str],
        mat_sfreq: float,
        bv_epochs: np.ndarray,     # (N_trials_bv, N_ch_bv, N_samples_bv)
        bv_time: np.ndarray,       # (N_samples_bv,)
        bv_channels: list[str],
        bv_sfreq: float,
        channel_mapping: dict[int, int],   # mat_idx → bv_idx
        bv_ref_std: np.ndarray | None = None,  # (N_bv_channels,) actual z-score σ per channel
        bv_ref_mean: np.ndarray | None = None,  # (N_bv_channels,) actual z-score mean per channel
        bv_n_clean: np.ndarray | None = None,  # (N_bv_channels,) # clean trials used for reference
        bv_diag: dict | None = None,  # raw cleaning test values for always-on display
        mat_rej: dict | None = None,  # MATLAB opts_log rejection info
        py_rej: dict | None = None,   # Python pipeline rejection masks
        mat_bsl_info: dict | None = None,  # MATLAB bsl_info {avg, std} per channel if loaded
    ) -> None:
        if not channel_mapping:
            raise ValueError("No channels matched — cannot display comparison.")

        self.mat_epochs = mat_epochs
        self.mat_time = mat_time
        self.mat_channels = mat_channels
        self.mat_sfreq = mat_sfreq
        self.bv_epochs = bv_epochs
        self.bv_time = bv_time
        self.bv_channels = bv_channels
        self.bv_sfreq = bv_sfreq

        # Sorted list of matched MATLAB indices + corresponding BV indices
        self.mat_indices = sorted(channel_mapping)
        self.bv_indices = [channel_mapping[mi] for mi in self.mat_indices]
        self.matched_names = [mat_channels[mi] for mi in self.mat_indices]

        self.n_channels = len(self.mat_indices)
        self.n_trials = min(mat_epochs.shape[0], bv_epochs.shape[0])
        self.bv_ref_std = bv_ref_std
        self.bv_ref_mean = bv_ref_mean
        self.bv_n_clean = bv_n_clean
        self.bv_diag = bv_diag or {}
        self.mat_rej = mat_rej or {}
        self.py_rej  = py_rej  or {}
        self.mat_bsl_info = mat_bsl_info

        self._ch = 0
        self._trial = 0

        self._build_figure()

    # ------------------------------------------------------------------
    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(15, 7))
        self.fig.suptitle(
            "MATLAB vs BrainVision Gamma Comparison",
            fontsize=13, fontweight="bold",
        )

        # Main plot area  [left, bottom, width, height]
        self.ax = self.fig.add_axes([0.07, 0.32, 0.90, 0.58])

        # ---- Channel slider ----
        ax_sl_ch = self.fig.add_axes([0.07, 0.20, 0.50, 0.03])
        self.sl_ch = Slider(
            ax_sl_ch, "Channel", 0, self.n_channels - 1,
            valinit=0, valstep=1, color="steelblue",
        )
        self.sl_ch.on_changed(self._on_ch_changed)

        # ---- Trial slider ----
        ax_sl_tr = self.fig.add_axes([0.07, 0.13, 0.50, 0.03])
        self.sl_tr = Slider(
            ax_sl_tr, "Trial", 0, self.n_trials - 1,
            valinit=0, valstep=1, color="darkorange",
        )
        self.sl_tr.on_changed(self._on_trial_changed)

        # ---- Channel prev/next buttons ----
        ax_prev_ch = self.fig.add_axes([0.63, 0.185, 0.07, 0.04])
        ax_next_ch = self.fig.add_axes([0.72, 0.185, 0.07, 0.04])
        self.btn_prev_ch = Button(ax_prev_ch, "◀ Ch", color="lightblue")
        self.btn_next_ch = Button(ax_next_ch, "Ch ▶", color="lightblue")
        self.btn_prev_ch.on_clicked(lambda _e: self._step(channel_delta=-1))
        self.btn_next_ch.on_clicked(lambda _e: self._step(channel_delta=+1))

        # ---- Trial prev/next buttons ----
        ax_prev_tr = self.fig.add_axes([0.63, 0.115, 0.07, 0.04])
        ax_next_tr = self.fig.add_axes([0.72, 0.115, 0.07, 0.04])
        self.btn_prev_tr = Button(ax_prev_tr, "◀ Trial", color="moccasin")
        self.btn_next_tr = Button(ax_next_tr, "Trial ▶", color="moccasin")
        self.btn_prev_tr.on_clicked(lambda _e: self._step(trial_delta=-1))
        self.btn_next_tr.on_clicked(lambda _e: self._step(trial_delta=+1))

        # ---- Info text ----
        self.info_text = self.fig.text(
            0.82, 0.15, "",
            fontsize=9, verticalalignment="center",
            bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
        )

        # ---- Key bindings ----
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # ---- Suppression status text (bottom left — Python diagnostic values) ----
        self.status_text = self.fig.text(
            0.07, 0.03, "",
            fontsize=8, verticalalignment="bottom",
            color="darkred",
        )

        # ---- Rejection comparison text (bottom right — MATLAB vs Python flags) ----
        self.rej_compare_text = self.fig.text(
            0.55, 0.03, "",
            fontsize=8, verticalalignment="bottom",
            color="navy",
        )

        self._render()

    # ------------------------------------------------------------------
    def _step(self, channel_delta: int = 0, trial_delta: int = 0) -> None:
        new_ch = int(np.clip(self._ch + channel_delta, 0, self.n_channels - 1))
        new_tr = int(np.clip(self._trial + trial_delta, 0, self.n_trials - 1))
        changed = False
        if new_ch != self._ch:
            self._ch = new_ch
            self.sl_ch.set_val(new_ch)
            changed = True
        if new_tr != self._trial:
            self._trial = new_tr
            self.sl_tr.set_val(new_tr)
            changed = True
        if changed:
            self._render()

    def _on_ch_changed(self, val: float) -> None:
        new_ch = int(val)
        if new_ch != self._ch:
            self._ch = new_ch
            self._render()

    def _on_trial_changed(self, val: float) -> None:
        new_tr = int(val)
        if new_tr != self._trial:
            self._trial = new_tr
            self._render()

    def _on_key(self, event: object) -> None:
        key = getattr(event, "key", None)
        if key == "right":
            self._step(trial_delta=+1)
        elif key == "left":
            self._step(trial_delta=-1)
        elif key == "up":
            self._step(channel_delta=+1)
        elif key == "down":
            self._step(channel_delta=-1)

    # ------------------------------------------------------------------
    def _render(self) -> None:
        self.ax.clear()

        mat_ch_i = self.mat_indices[self._ch]
        bv_ch_i = self.bv_indices[self._ch]
        mat_ch_name = self.mat_channels[mat_ch_i]
        bv_ch_name = self.bv_channels[bv_ch_i]
        trial_i = self._trial

        # MATLAB: (N_trials, N_samples, N_bipoles)
        mat_trace = self.mat_epochs[trial_i, :, mat_ch_i]
        # BrainVision: (N_trials, N_channels, N_samples) — trim to MATLAB display range
        bv_display_mask = (self.bv_time >= self.mat_time[0]) & (self.bv_time <= self.mat_time[-1])
        bv_time_display = self.bv_time[bv_display_mask] + BV_DISPLAY_OFFSET_S
        bv_trace = self.bv_epochs[trial_i, bv_ch_i, bv_display_mask]
        bv_nan = np.all(np.isnan(bv_trace))
        mat_nan = np.all(np.isnan(mat_trace))

        bv_label = f"BrainVision ({self.bv_sfreq:.0f} Hz)"
        if ZSCORE_BV:
            bv_label += f"  z-score [{BASELINE_TMIN_S},{BASELINE_TMAX_S}]s"

        self.ax.plot(
            self.mat_time, mat_trace,
            label=f"MATLAB ({self.mat_sfreq:.0f} Hz)",
            color="steelblue", linewidth=4.5, alpha=0.85,
        )
        self.ax.plot(
            bv_time_display, bv_trace,
            label=bv_label,
            color="darkorange", linewidth=1.5, alpha=0.85,
        )
        self.ax.axvline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)

        self.ax.set_xlabel("Time relative to anchor onset (s)")
        self.ax.set_ylabel("Amplitude")
        title = (
            f"Trial {trial_i + 1} / {self.n_trials}  —  "
            f"MATLAB: {mat_ch_name}  →  BV: {bv_ch_name}"
        )
        self.ax.set_title(title, fontsize=11)
        self.ax.legend(loc="upper right", fontsize=9)
        self.ax.grid(True, alpha=0.3)

        # Info box — includes cross-trial correlation and scale ratio to diagnose
        # whether residual differences are a fixed per-channel scale offset (z-score
        # reference drift) or a trial-varying shape difference (filter / normalization).
        first = _first_contact(mat_ch_name)

        # Compute stats across all trials for this channel pair.
        mat_all = self.mat_epochs[:self.n_trials, :, mat_ch_i]          # (N_trials, N_mat_samp)
        bv_all  = self.bv_epochs[:self.n_trials, bv_ch_i, :][:, bv_display_mask]  # (N_trials, N_bv_disp)
        # Trim mat to same length as bv_display if needed
        n_common = min(mat_all.shape[1], bv_all.shape[1])
        mat_all = mat_all[:, :n_common]
        bv_all  = bv_all[:, :n_common]

        # Per-trial std ratio (BV std / MATLAB std) — a consistent ratio > 1 means
        # BV has higher variance, likely due to a smaller z-score reference_std.
        with np.errstate(divide="ignore", invalid="ignore"):
            std_ratio_per_trial = (
                np.nanstd(bv_all, axis=1) / np.nanstd(mat_all, axis=1)
            )  # (N_trials,)
        median_ratio = float(np.nanmedian(std_ratio_per_trial))

        ref_std_str  = "n/a"
        ref_mean_str = "n/a"
        n_clean_str  = "n/a"
        mat_std_str  = "n/a"
        mat_mean_str = "n/a"
        if self.bv_ref_std is not None:
            ref_std_str = f"{self.bv_ref_std[bv_ch_i]:.4f}"
        if self.bv_ref_mean is not None:
            ref_mean_str = f"{self.bv_ref_mean[bv_ch_i]:.4f}"
        if self.bv_n_clean is not None:
            n_clean_str = f"{int(self.bv_n_clean[bv_ch_i])}/{self.n_trials}"
        mat_n_finite_str = "n/a"   # non-NaN trials in MATLAB's data_fix_wdw_allchans (pre-rmoutliers)
        mat_n_clean_str  = "n/a"   # trials remaining after rmoutliers
        mat_std_recomp_str = "n/a"
        if self.mat_bsl_info is not None:
            _mstd = self.mat_bsl_info["std"]
            _mavg = self.mat_bsl_info["avg"]
            if mat_ch_i < len(_mstd):
                mat_std_str  = f"{_mstd[mat_ch_i]:.4f}"
                mat_mean_str = f"{_mavg[mat_ch_i]:.4f}"
            # Prefer the exact MATLAB keep-mask when available; otherwise fall
            # back to a Python recomputation from the saved baseline means.
            _ptbl = self.mat_bsl_info.get("per_trial_bl_means")
            _mk = self.mat_bsl_info.get("baseline_keep_mask")
            if _ptbl is not None and _ptbl.ndim == 2 and mat_ch_i < _ptbl.shape[0]:
                _ch_bl = _ptbl[mat_ch_i, :].astype(np.float64)  # (n_trials,)
                _finite_bl = _ch_bl[np.isfinite(_ch_bl)]
                mat_n_finite_str = f"{len(_finite_bl)}/{len(_ch_bl)}"  # pre-rmoutliers
                if _mk is not None and _mk.ndim == 2 and mat_ch_i < _mk.shape[0]:
                    _keep = np.asarray(_mk[mat_ch_i, :], dtype=bool)
                    _n = min(len(_keep), len(_ch_bl))
                    _keep = _keep[:_n] & np.isfinite(_ch_bl[:_n])
                    _clean_bl = _ch_bl[:_n][_keep]
                    mat_n_clean_str = f"{len(_clean_bl)}/{len(_ch_bl)}"
                    if len(_clean_bl) >= 2:
                        mat_std_recomp_str = f"{float(np.std(_clean_bl, ddof=1)):.4f}"
                elif len(_finite_bl) >= 3:
                    _ctr = float(np.median(_finite_bl))
                    _mad = float(np.median(np.abs(_finite_bl - _ctr)))
                    if _mad > 0.0:
                        _thresh = 3.0 * 1.4826 * _mad
                        _clean_bl = _finite_bl[np.abs(_finite_bl - _ctr) <= _thresh]
                    else:
                        _clean_bl = _finite_bl
                    mat_n_clean_str    = f"{len(_clean_bl)}/{len(_ch_bl)}"
                    mat_std_recomp_str = f"{float(np.std(_clean_bl, ddof=1)):.4f}"

        # BV n_finite before outlier removal (non-NaN baseline means before median/MAD step)
        bv_n_finite_str = "n/a"
        if ZSCORE_BV:
            _bl_mask_info, _, _ = _matlab_baseline_mask(
                self.bv_time,
                BASELINE_TMIN_S,
                BASELINE_TMAX_S,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                _bv_bl_means = np.nanmean(self.bv_epochs[:, :, _bl_mask_info], axis=2)  # (N_trials, N_ch)
            _bv_ch_bl = _bv_bl_means[:, bv_ch_i]
            _bv_n_finite = int(np.sum(np.isfinite(_bv_ch_bl)))
            bv_n_finite_str = f"{_bv_n_finite}/{len(_bv_ch_bl)}"

        info = (
            f"Ch {self._ch + 1}/{self.n_channels}\n"
            f"MATLAB bipole: {mat_ch_name}\n"
            f"First contact: {first}\n"
            f"BV channel:    {bv_ch_name}\n"
            f"MATLAB idx:    {mat_ch_i}\n"
            f"BV idx:        {bv_ch_i}\n"
            f"Trial:         {trial_i + 1}/{self.n_trials}\n"
            f"BV  n_finite(pre-rmout): {bv_n_finite_str}  n_clean: {n_clean_str}\n"
            f"MAT n_finite(pre-rmout): {mat_n_finite_str}  n_clean: {mat_n_clean_str}\n"
            f"BV  zscore μ/σ: {ref_mean_str} / {ref_std_str}\n"
            f"MAT zscore μ/σ: {mat_mean_str} / {mat_std_str}\n"
            f"MAT σ recomp (py-rmout): {mat_std_recomp_str}\n"
            f"Median BV/MAT std ratio: {median_ratio:.3f}"
        )
        self.info_text.set_text(info)

        # ---- Always-on diagnostic panel ----
        diag_lines: list[str] = []
        d = self.bv_diag
        if d:
            # PRECLEAN mean
            if "trial_means_pre" in d:
                val = d["trial_means_pre"][trial_i, bv_ch_i]
                lo = d["ch_mean_m"][bv_ch_i] - PRECLEAN_THRESHOLD * d["ch_std_m"][bv_ch_i]
                hi = d["ch_mean_m"][bv_ch_i] + PRECLEAN_THRESHOLD * d["ch_std_m"][bv_ch_i]
                mark = "✗" if (val < lo or val > hi) else "✓"
                diag_lines.append(f"PRECLEAN mean: {val:.3f}  [{lo:.3f}…{hi:.3f}] {mark}")
            # PRECLEAN max
            if "trial_maxes_pre" in d:
                val = d["trial_maxes_pre"][trial_i, bv_ch_i]
                lo = d["ch_mean_x"][bv_ch_i] - PRECLEAN_THRESHOLD * d["ch_std_x"][bv_ch_i]
                hi = d["ch_mean_x"][bv_ch_i] + PRECLEAN_THRESHOLD * d["ch_std_x"][bv_ch_i]
                mark = "✗" if (val < lo or val > hi) else "✓"
                diag_lines.append(f"PRECLEAN max:  {val:.3f}  [{lo:.3f}…{hi:.3f}] {mark}")
            # Bad channel (channel-level, same for all trials)
            if "ch_spread_m" in d:
                val_m = d["ch_spread_m"][bv_ch_i]
                lo_m = d["spm_mean"] - REJECT_BAD_CHANNELS_SD_THRESHOLD * d["spm_std"]
                hi_m = d["spm_mean"] + REJECT_BAD_CHANNELS_SD_THRESHOLD * d["spm_std"]
                mark_m = "✗" if (val_m < lo_m or val_m > hi_m) else "✓"
                val_x = d["ch_spread_x"][bv_ch_i]
                lo_x = d["spx_mean"] - REJECT_BAD_CHANNELS_SD_THRESHOLD * d["spx_std"]
                hi_x = d["spx_mean"] + REJECT_BAD_CHANNELS_SD_THRESHOLD * d["spx_std"]
                mark_x = "✗" if (val_x < lo_x or val_x > hi_x) else "✓"
                diag_lines.append(
                    f"Ch spread mean: {val_m:.3f}  [{lo_m:.3f}…{hi_m:.3f}] {mark_m}  "
                    f"max: {val_x:.3f}  [{lo_x:.3f}…{hi_x:.3f}] {mark_x}"
                )
            # Negative rating
            if "ratings" in d and trial_i < len(d["ratings"]):
                rating = d["ratings"][trial_i]
                r_str = f"{rating:.3f}" if np.isfinite(rating) else "n/a"
                mark = "✗" if (np.isfinite(rating) and rating < MIN_RATING) else "✓"
                diag_lines.append(f"Rating: {r_str} (min={MIN_RATING:.1f}) {mark}")
        if mat_nan:
            diag_lines.append("MATLAB: NaN")
        self.status_text.set_text("\n".join(diag_lines))
        self.status_text.set_color("darkred" if (bv_nan or mat_nan) else "darkgreen")

        # ---- Rejection comparison (MATLAB opts_log vs Python pipeline) ----
        cmp_lines: list[str] = ["MATLAB vs Python  (current trial / channel):"]
        mat_r = self.mat_rej
        py_r  = self.py_rej

        def _flag_1d(d: dict, key: str, idx: int) -> str:
            m = d.get(key)
            if m is None or idx >= len(m):
                return "n/a "
            return "YES " if bool(m[idx]) else "no  "

        def _flag_2d(d: dict, key: str, t: int, ch: int) -> str:
            m = d.get(key)
            if m is None or m.ndim < 2 or ch >= m.shape[1] or t >= m.shape[0]:
                return "n/a "
            return "YES " if bool(m[t, ch]) else "no  "

        neg_idx_mat = mat_r.get("neg_rating_indices")
        neg_mask_py = py_r.get("neg_rating_mask")
        mat_neg_flag = ("YES " if trial_i in set(neg_idx_mat) else "no  ") if neg_idx_mat is not None else "n/a "
        py_neg_flag  = ("YES " if (trial_i < len(neg_mask_py) and bool(neg_mask_py[trial_i])) else "no  ") if neg_mask_py is not None else "n/a "

        rows = [
            ("Trial rej mean",   _flag_2d(mat_r, "outlier_trials_mean", trial_i, mat_ch_i),
                                 _flag_2d(py_r,  "outlier_trials_mean", trial_i, bv_ch_i)),
            ("Trial rej max",    _flag_2d(mat_r, "outlier_trials_max",  trial_i, mat_ch_i),
                                 _flag_2d(py_r,  "outlier_trials_max",  trial_i, bv_ch_i)),
            ("Channel bad mean", _flag_1d(mat_r, "bad_channels_mean", mat_ch_i),
                                 _flag_1d(py_r,  "bad_channels_mean", bv_ch_i)),
            ("Channel bad max",  _flag_1d(mat_r, "bad_channels_max",  mat_ch_i),
                                 _flag_1d(py_r,  "bad_channels_max",  bv_ch_i)),
            ("Neg rating",       mat_neg_flag, py_neg_flag),
        ]
        for label, mv, pv in rows:
            agree = "" if "n/a" in mv or "n/a" in pv else (" ✓" if mv == pv else " ✗")
            cmp_lines.append(f"  {label:<18} MAT={mv} PY={pv}{agree}")

        def _num_2d(d: dict, key: str, t: int, ch: int) -> str:
            m = d.get(key)
            if m is None or m.ndim < 2 or ch >= m.shape[1] or t >= m.shape[0]:
                return "n/a    "
            v = float(m[t, ch])
            return f"{v:.4f} " if np.isfinite(v) else "nan    "

        def _num_1d(d: dict, key: str, idx: int) -> str:
            m = d.get(key)
            if m is None or idx >= len(m):
                return "n/a    "
            v = float(m[idx])
            return f"{v:.4f} " if np.isfinite(v) else "nan    "

        num_rows = [
            ("Mean trial val",
             _num_2d(mat_r, "mean_trial_vals", trial_i, mat_ch_i),
             _num_2d(py_r,  "mean_trial_vals", trial_i, bv_ch_i)),
            ("Max trial val",
             _num_2d(mat_r, "max_trial_vals",  trial_i, mat_ch_i),
             _num_2d(py_r,  "max_trial_vals",  trial_i, bv_ch_i)),
            ("Ch SD (mean)",
             _num_1d(mat_r, "allchannels_sd",     mat_ch_i),
             _num_1d(py_r,  "allchannels_sd",     bv_ch_i)),
            ("Ch SD (max)",
             _num_1d(mat_r, "allchannels_sd_max", mat_ch_i),
             _num_1d(py_r,  "allchannels_sd_max", bv_ch_i)),
        ]
        for label, mv, pv in num_rows:
            cmp_lines.append(f"  {label:<18} MAT={mv} PY={pv}")
        self.rej_compare_text.set_text("\n".join(cmp_lines))
        self.rej_compare_text.set_color("navy")

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def show(self) -> None:
        plt.show()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print(
        "Comparison mode: "
        f"python_rejections_only={not USE_MATLAB_REJECTION_MASKS}, "
        f"python_baseline_mask_only={not USE_MATLAB_BASELINE_KEEP_MASK}"
    )
    # ------------------------------------------------------------------
    # 1. Load MATLAB
    # ------------------------------------------------------------------
    print(f"Loading MATLAB file: {MATLAB_PATH}")
    if not MATLAB_PATH.exists():
        sys.exit(f"ERROR: MATLAB file not found: {MATLAB_PATH}")

    mat_data, mat_channels, mat_sfreq, mat_rej_info, mat_time = load_matlab(MATLAB_PATH)
    print(f"  alldata shape : {mat_data.shape}  (N_trials, N_samples, N_bipoles)")
    print(f"  sfreq         : {mat_sfreq} Hz")
    print(f"  channels      : {len(mat_channels)}")
    if mat_channels:
        print(f"  first 5       : {mat_channels[:5]}")
    else:
        print("  WARNING: no channel names found in hdr.label")

    n_mat_samples = mat_data.shape[1]
    if mat_time is None or len(mat_time) != n_mat_samples:
        print("  WARNING: hdr.timelist unavailable — falling back to synthetic MATLAB time axis.")
        mat_time = np.arange(n_mat_samples) / mat_sfreq + MAT_TMIN_S
    else:
        mat_time = np.asarray(mat_time, dtype=np.float64).ravel()
    print(f"  time range    : {mat_time[0]:.3f} s → {mat_time[-1]:.3f} s")

    # ------------------------------------------------------------------
    # 2. Load BrainVision and epoch
    # ------------------------------------------------------------------
    print(f"\nLoading BrainVision file: {BV_VHDR_PATH}")
    if not BV_VHDR_PATH.exists():
        sys.exit(f"ERROR: BrainVision .vhdr not found: {BV_VHDR_PATH}")

    mne.set_log_level("WARNING")
    raw = mne.io.read_raw_brainvision(str(BV_VHDR_PATH), preload=True, verbose=False)
    bv_sfreq = float(raw.info["sfreq"])
    print(f"  BV channels   : {len(raw.ch_names)}, sfreq: {bv_sfreq} Hz")
    print(f"  first 5       : {raw.ch_names[:5]}")

    print("\nExtracting BV epochs...")
    bv_epochs, bv_time, bv_channels = epoch_brainvision(raw)
    print(f"  BV epochs shape: {bv_epochs.shape}  (N_trials, N_channels, N_samples)")
    print(f"  time range     : {bv_time[0]:.3f} s → {bv_time[-1]:.3f} s")

    # Diagnostic arrays stored for viewer display (raw test values, always shown).
    bv_diag: dict = {}
    py_rej:  dict = {}  # Python pipeline rejection masks (captured during cleaning)

    # ------------------------------------------------------------------
    # Pre-load MATLAB bsl_info (needed for NaN alignment before z-scoring).
    # ------------------------------------------------------------------
    mat_bsl_info: dict | None = None
    if BSL_INFO_PATH is not None:
        if BSL_INFO_PATH.exists():
            try:
                _bi = scipy.io.loadmat(str(BSL_INFO_PATH), squeeze_me=True, struct_as_record=False)
                _bsl = _bi.get("bsl_info")
                if _bsl is not None:
                    _per_trial_raw = getattr(_bsl, "data_fix_wdw_allchans", None)
                    if _per_trial_raw is not None:
                        _ptbl = np.atleast_2d(np.asarray(_per_trial_raw, dtype=np.float64))
                    else:
                        _ptbl = None
                    _keep_mask_raw = getattr(_bsl, "baseline_keep_mask", None)
                    _out_mask_raw = getattr(_bsl, "baseline_outlier_mask", None)
                    _keep_mask = (
                        np.atleast_2d(np.asarray(_keep_mask_raw, dtype=bool))
                        if _keep_mask_raw is not None
                        else None
                    )
                    _out_mask = (
                        np.atleast_2d(np.asarray(_out_mask_raw, dtype=bool))
                        if _out_mask_raw is not None
                        else None
                    )
                    mat_bsl_info = {
                        "avg": np.atleast_1d(np.asarray(getattr(_bsl, "avgdata_perelec", []), dtype=np.float64)),
                        "std": np.atleast_1d(np.asarray(getattr(_bsl, "stddata_perelec", []), dtype=np.float64)),
                        "per_trial_bl_means": _ptbl,  # (n_chan, n_trials) or None
                        "baseline_keep_mask": _keep_mask,       # (n_chan, n_trials) or None
                        "baseline_outlier_mask": _out_mask,     # (n_chan, n_trials) or None
                    }
                    print(f"  Loaded MATLAB bsl_info: {BSL_INFO_PATH.name} "
                          f"({len(mat_bsl_info['std'])} channels"
                          f"{', per-trial BL means: ' + str(_ptbl.shape) if _ptbl is not None else ''}"
                          f"{', keep-mask: ' + str(_keep_mask.shape) if _keep_mask is not None else ''})"
                    )
            except Exception as _e:
                print(f"  WARNING: could not load BSL_INFO_PATH: {_e}")
        else:
            print(f"  WARNING: BSL_INFO_PATH not found: {BSL_INFO_PATH}")

    # -----------------------------------------------------------------------
    # Optional: apply MATLAB opts_log masks instead of recomputing PRECLEAN
    # and removebadchannelsSd in Python.  This lets us isolate the z-scoring
    # step as the only remaining comparison variable.
    # -----------------------------------------------------------------------
    _mat_masks_applied = False
    if USE_MATLAB_REJECTION_MASKS and mat_rej_info:
        print("\nApplying MATLAB opts_log rejection masks to BV data...")
        _early_mapping = match_channels(mat_channels, bv_channels)
        if not _early_mapping:
            print("  WARNING: no channel mapping found — cannot apply MATLAB masks.")
        else:
            _n_bt, _n_bc = bv_epochs.shape[0], bv_epochs.shape[1]
            # --- Trial-level masks (mean then max, same order as MATLAB) ---
            for _mat_key, _py_key in [("outlier_trials_mean", "outlier_trials_mean"),
                                       ("outlier_trials_max",  "outlier_trials_max")]:
                _mat_m = mat_rej_info.get(_mat_key)
                if _mat_m is not None and _mat_m.ndim == 2:
                    _bv_m = np.zeros((_n_bt, _n_bc), dtype=bool)
                    for _mi, _bi in _early_mapping.items():
                        _n = min(_n_bt, _mat_m.shape[0])
                        _bv_m[:_n, _bi] = _mat_m[:_n, _mi]
                    bv_epochs = apply_trial_nan_mask(bv_epochs, _bv_m)
                    py_rej[_py_key] = _bv_m
                    print(f"  {_mat_key}: {int(np.sum(_bv_m))} (trial, ch) pairs masked")
            # Store MATLAB numeric trial values for viewer display
            for _mat_key, _py_key in [("mean_trial_vals", "mean_trial_vals"),
                                       ("max_trial_vals",  "max_trial_vals")]:
                _mat_m = mat_rej_info.get(_mat_key)
                if _mat_m is not None and _mat_m.ndim == 2:
                    _bv_m_f = np.full((_n_bt, _n_bc), np.nan)
                    for _mi, _bi in _early_mapping.items():
                        _n = min(_n_bt, _mat_m.shape[0])
                        _bv_m_f[:_n, _bi] = _mat_m[:_n, _mi]
                    py_rej[_py_key] = _bv_m_f
            # --- Channel-level masks ---
            _bad_mean_bv = np.zeros(_n_bc, dtype=bool)
            _bad_max_bv  = np.zeros(_n_bc, dtype=bool)
            for _mat_key, _bv_bad in [("bad_channels_mean", _bad_mean_bv),
                                       ("bad_channels_max",  _bad_max_bv)]:
                _mat_m1d = mat_rej_info.get(_mat_key)
                if _mat_m1d is not None:
                    for _mi, _bi in _early_mapping.items():
                        if _mi < len(_mat_m1d) and _mat_m1d[_mi]:
                            _bv_bad[_bi] = True
            for _bv_bad in [_bad_mean_bv, _bad_max_bv]:
                if np.any(_bv_bad):
                    bv_epochs[:, _bv_bad, :] = np.nan
            py_rej["bad_channels_mean"] = _bad_mean_bv
            py_rej["bad_channels_max"]  = _bad_max_bv
            _bad_combined = _bad_mean_bv | _bad_max_bv
            if np.any(_bad_combined):
                print(f"  Channel masks: NaN'd {int(np.sum(_bad_combined))} BV channel(s): "
                      f"{[bv_channels[i] for i in np.flatnonzero(_bad_combined)]}")
            else:
                print("  Channel masks: no channels to NaN.")
            # Map MATLAB SD spread values to BV-indexed arrays for display
            for _mat_key, _py_key in [("allchannels_sd",     "allchannels_sd"),
                                       ("allchannels_sd_max", "allchannels_sd_max")]:
                _mat_v = mat_rej_info.get(_mat_key)
                if _mat_v is not None:
                    _bv_v = np.full(_n_bc, np.nan)
                    for _mi, _bi in _early_mapping.items():
                        if _mi < len(_mat_v):
                            _bv_v[_bi] = _mat_v[_mi]
                    py_rej[_py_key] = _bv_v
            # --- RT outlier mask from opts_log (applied to ALL channels, same order as MATLAB) ---
            _rt_indices = mat_rej_info.get("outlier_rt_indices")
            if _rt_indices is not None and len(_rt_indices) > 0:
                _valid_rt = _rt_indices[(_rt_indices >= 0) & (_rt_indices < _n_bt)]
                if len(_valid_rt) > 0:
                    bv_epochs[_valid_rt, :, :] = np.nan
                    _rt_mask_mat = np.zeros(_n_bt, dtype=bool)
                    _rt_mask_mat[_valid_rt] = True
                    py_rej["outlier_rt_mask"] = _rt_mask_mat
                    print(f"  RT outlier mask (from opts_log): NaN'd {len(_valid_rt)} trial(s) for all channels")
                else:
                    print("  RT outlier mask: no valid RT outlier trials in opts_log.")
            else:
                print("  RT outlier mask: opts_log.outlierRTs absent or empty.")
            _high_nan_idx = mat_rej_info.get("high_nan_channel_indices")
            if _high_nan_idx is not None:
                _high_nan_idx = _high_nan_idx[
                    (_high_nan_idx >= 0) & (_high_nan_idx < len(mat_channels))
                ]
                _high_nan_bv = np.zeros(_n_bc, dtype=bool)
                for _mi in _high_nan_idx:
                    _bi = _early_mapping.get(int(_mi))
                    if _bi is not None:
                        _high_nan_bv[_bi] = True
                if np.any(_high_nan_bv):
                    bv_epochs[:, _high_nan_bv, :] = np.nan
                    print(f"  25%-NaN mask (from opts_log): NaN'd {int(np.sum(_high_nan_bv))} BV channel(s)")
                else:
                    print("  25%-NaN mask (from opts_log): no valid matched channels.")
                py_rej["high_nan_channels"] = _high_nan_bv
            # --- Align bsl_info NaN pattern (catches pre-existing b1 NaN not in opts_log) ---
            if mat_bsl_info is not None:
                _ptbl_align = mat_bsl_info.get("per_trial_bl_means")
                if _ptbl_align is not None and _ptbl_align.ndim == 2:
                    _n_nan_added = 0
                    for _mi, _bi in _early_mapping.items():
                        if _mi < _ptbl_align.shape[0]:
                            _mat_nan_trials = np.flatnonzero(~np.isfinite(_ptbl_align[_mi]))
                            for _ti in _mat_nan_trials:
                                if _ti < _n_bt and np.any(np.isfinite(bv_epochs[_ti, _bi, :])):
                                    bv_epochs[_ti, _bi, :] = np.nan
                                    _n_nan_added += 1
                    if _n_nan_added > 0:
                        print(f"  bsl_info NaN alignment: added {_n_nan_added} (trial,ch) NaN from pre-existing MATLAB NaN (b1).")
                    else:
                        print("  bsl_info NaN alignment: no additional NaN needed.")
            _mat_masks_applied = True
            print("  MATLAB masks applied — PRECLEAN, removebadchannelsSd, RT removal and 25%-NaN recomputation will be skipped.")

    if PRECLEAN_BV and not _mat_masks_applied:
        print(f"  Pre-cleaning: NaN-masking outlier trials per channel (threshold={PRECLEAN_THRESHOLD}σ)…")
        _ep_f64 = np.asarray(bv_epochs, dtype=np.float64)
        # Diagnostic stats computed on original (pre-NaN) data for display purposes.
        _tr_means_pre = np.mean(_ep_f64, axis=2)          # (n_trials, n_ch)
        _ch_means_m = np.mean(_tr_means_pre, axis=0)
        _ch_stds_m  = np.std(_tr_means_pre, axis=0, ddof=1)
        _tr_maxes_pre = np.max(np.abs(_ep_f64), axis=2)   # (n_trials, n_ch)
        _ch_means_x = np.mean(_tr_maxes_pre, axis=0)
        _ch_stds_x  = np.std(_tr_maxes_pre, axis=0, ddof=1)
        bv_diag.update({
            "trial_means_pre": _tr_means_pre,
            "ch_mean_m": _ch_means_m, "ch_std_m": _ch_stds_m,
            "trial_maxes_pre": _tr_maxes_pre,
            "ch_mean_x": _ch_means_x, "ch_std_x": _ch_stds_x,
        })
        # Step 1: mean outlier removal on original data.
        # Matches MATLAB's first rmHGAOutlierTrials block.
        mask_mean = detect_outlier_trial_channel_pairs_by_mean(bv_epochs, PRECLEAN_THRESHOLD)
        bv_epochs = apply_trial_nan_mask(bv_epochs, mask_mean)
        # Step 2: max outlier removal on mean-cleaned data.
        # MATLAB recomputes nanmax after step 1, so some previously NaN'd trials
        # are excluded from the reference statistics here.  Use nan-aware stats.
        _ep_cln = np.asarray(bv_epochs, dtype=np.float64)
        _tr_maxes_cln = np.nanmax(np.abs(_ep_cln), axis=2)  # (n_trials, n_ch) — NaN for NaN'd trials
        _ch_mean_xc = np.nanmean(_tr_maxes_cln, axis=0)
        _ch_std_xc  = np.nanstd(_tr_maxes_cln, axis=0, ddof=1)
        _valid_xc   = _ch_std_xc > 0.0
        _dev_xc     = np.abs(_tr_maxes_cln - _ch_mean_xc[np.newaxis, :])
        mask_max = np.zeros(_tr_maxes_cln.shape, dtype=bool)
        mask_max[:, _valid_xc] = (
            _dev_xc[:, _valid_xc] > PRECLEAN_THRESHOLD * _ch_std_xc[np.newaxis, _valid_xc]
        )
        mask_max &= ~mask_mean  # don't re-flag pairs already NaN'd in step 1
        bv_epochs = apply_trial_nan_mask(bv_epochs, mask_max)
        py_rej["outlier_trials_mean"] = mask_mean
        py_rej["outlier_trials_max"]  = mask_max
        py_rej["mean_trial_vals"]     = _tr_means_pre
        py_rej["max_trial_vals"]      = _tr_maxes_pre
        n_masked = int(np.sum(mask_mean)) + int(np.sum(mask_max))
        print(f"    Masked {n_masked} (trial, channel) pairs "
              f"({int(np.sum(mask_mean))} by mean, {int(np.sum(mask_max))} by max).")

    # MATLAB b2 order: removeoutlierRTs comes BEFORE removebadchannelsSd so that
    # RT-outlier trials are excluded when computing the per-channel trial-mean spread.
    if REMOVE_OUTLIER_RTS and not _mat_masks_applied:
        print(f"  Removing outlier-RT trials (RT > {MAX_RT_S}s, column '{RT_COLUMN}' from {BEH_TSV_PATH.name})…")
        if not BEH_TSV_PATH.exists():
            print(f"    WARNING: beh TSV not found, skipping: {BEH_TSV_PATH}")
        else:
            import csv as _csv  # noqa: PLC0415 (local import reuse)
            with BEH_TSV_PATH.open(newline="", encoding="utf-8-sig") as _f:
                _rows_rt = list(_csv.DictReader(_f, delimiter="\t"))
            _rts = np.array(
                [float(r[RT_COLUMN]) if r.get(RT_COLUMN, "").strip() not in ("", "n/a", "nan") else np.nan
                 for r in _rows_rt],
                dtype=np.float64,
            )
            n_bv = bv_epochs.shape[0]
            if len(_rts) < n_bv:
                print(f"    WARNING: beh has {len(_rts)} rows but BV has {n_bv} trials — skipping.")
            else:
                _rt_mask = _rts[:n_bv] > MAX_RT_S
                n_rt = int(np.sum(_rt_mask & np.isfinite(_rts[:n_bv])))
                if n_rt:
                    bv_epochs[_rt_mask, :, :] = np.nan
                    print(f"    NaN'd {n_rt} trial(s) with RT > {MAX_RT_S}s.")
                else:
                    print(f"    No trials with RT > {MAX_RT_S}s.")
                py_rej["outlier_rt_mask"] = _rt_mask

    if REJECT_BAD_CHANNELS_SD and not _mat_masks_applied:
        print(f"  Rejecting bad channels by trial-mean spread (threshold={REJECT_BAD_CHANNELS_SD_THRESHOLD}σ)…")
        # --- Step 1: mean-spread criterion (matching MATLAB's first rmoutliers pass) ---
        _tr_means_bc = np.nanmean(bv_epochs, axis=2)      # (n_trials, n_ch)
        _ch_spread_m = np.nanstd(_tr_means_bc, axis=0, ddof=1)
        _spm_mean = float(np.nanmean(_ch_spread_m[np.isfinite(_ch_spread_m)]))
        _spm_std  = float(np.nanstd(_ch_spread_m[np.isfinite(_ch_spread_m)], ddof=1))
        bad_ch_mean = reject_channels_by_trial_mean_spread(bv_epochs, REJECT_BAD_CHANNELS_SD_THRESHOLD)
        py_rej["bad_channels_mean"] = bad_ch_mean
        py_rej["allchannels_sd"]    = _ch_spread_m
        if np.any(bad_ch_mean):
            bv_epochs[:, bad_ch_mean, :] = np.nan
        # --- Step 2: max-spread criterion on epochs already updated by step 1 ---
        # (matching MATLAB: max_alldata_per_trial is recomputed after NaN-ing mean-bad channels)
        _tr_maxes_bc = np.nanmax(np.abs(bv_epochs), axis=2)
        _ch_spread_x = np.nanstd(_tr_maxes_bc, axis=0, ddof=1)
        _spx_mean = float(np.nanmean(_ch_spread_x[np.isfinite(_ch_spread_x)]))
        _spx_std  = float(np.nanstd(_ch_spread_x[np.isfinite(_ch_spread_x)], ddof=1))
        bv_diag.update({
            "ch_spread_m": _ch_spread_m, "spm_mean": _spm_mean, "spm_std": _spm_std,
            "ch_spread_x": _ch_spread_x, "spx_mean": _spx_mean, "spx_std": _spx_std,
        })
        bad_ch_max  = reject_channels_by_trial_max_spread(bv_epochs, REJECT_BAD_CHANNELS_SD_THRESHOLD)
        py_rej["bad_channels_max"]   = bad_ch_max
        py_rej["allchannels_sd_max"] = _ch_spread_x
        bad_ch_mask = bad_ch_mean | bad_ch_max
        n_bad_ch = int(np.sum(bad_ch_mask))
        if n_bad_ch:
            # NaN channels flagged by max-spread that weren't already NaN'd in step 1
            if np.any(bad_ch_max & ~bad_ch_mean):
                bv_epochs[:, bad_ch_max & ~bad_ch_mean, :] = np.nan
            bad_names = [bv_channels[i] for i in np.flatnonzero(bad_ch_mask)]
            print(f"    NaN'd {n_bad_ch} channel(s): {bad_names}")
        else:
            print("    No bad channels detected.")

    if REMOVE_HIGH_NAN_CHANNELS and not _mat_masks_applied:
        print(
            f"  Removing channels with ≥{HIGH_NAN_CHANNEL_THRESHOLD*100:.0f}% NaN trials "
            "(MATLAB b2: remove channels with >25 % trials deleted)…"
        )
        _high_nan_mask = reject_channels_by_nan_trial_ratio(bv_epochs, HIGH_NAN_CHANNEL_THRESHOLD)
        n_high_nan = int(np.sum(_high_nan_mask))
        if n_high_nan:
            _high_nan_names = [bv_channels[i] for i in np.flatnonzero(_high_nan_mask)]
            bv_epochs[:, _high_nan_mask, :] = np.nan
            print(f"    NaN'd {n_high_nan} channel(s): {_high_nan_names}")
        else:
            print("    No channels above threshold.")
        py_rej["high_nan_channels"] = _high_nan_mask

    if REMOVE_NEGATIVE_RATINGS:
        print(f"  Removing negative-rating trials (rating < {MIN_RATING}, from {BEH_TSV_PATH.name})…")
        if not BEH_TSV_PATH.exists():
            print(f"    WARNING: beh TSV not found, skipping: {BEH_TSV_PATH}")
        else:
            import csv as _csv
            with BEH_TSV_PATH.open(newline="", encoding="utf-8-sig") as _f:
                _rows = list(_csv.DictReader(_f, delimiter="\t"))
            _ratings = np.array(
                [float(r["rating"]) if r.get("rating", "").strip() not in ("", "n/a", "nan") else np.nan
                 for r in _rows],
                dtype=np.float64,
            )
            n_bv = bv_epochs.shape[0]
            if len(_ratings) < n_bv:
                print(f"    WARNING: beh has {len(_ratings)} rows but BV has {n_bv} trials — skipping.")
            else:
                _neg_mask = _ratings[:n_bv] < MIN_RATING  # (n_trials,)
                py_rej["neg_rating_mask"] = _neg_mask
                n_neg = int(np.sum(_neg_mask))
                if n_neg:
                    # NaN all channels for those trials (matching MATLAB alldata(trialsToRemove,:,:) = NaN)
                    bv_epochs[_neg_mask, :, :] = np.nan
                    print(f"    NaN'd {n_neg} trial(s) with rating < {MIN_RATING}.")
                else:
                    print(f"    No negative-rating trials found.")
                bv_diag["ratings"] = _ratings[:n_bv]

    # Snapshot the exact pre-zscore BV epochs used for baseline-mask experiments.
    bv_epochs_pre_zscore = np.asarray(bv_epochs, dtype=np.float64).copy()

    # Compute z-score reference stats per channel (actual post-outlier-removal values).
    # These are recomputed to match exactly what zscore_activity_by_baseline uses internally.
    bv_ref_std: np.ndarray | None = None
    bv_ref_mean: np.ndarray | None = None
    bv_n_clean: np.ndarray | None = None

    if ZSCORE_BV:
        print(
            f"  Applying global z-score (baseline {BASELINE_TMIN_S} – {BASELINE_TMAX_S} s, "
            f"remove_outliers={BASELINE_REMOVE_OUTLIERS})…"
        )
        _bl_mask, _bl_idx_start, _bl_idx_stop = _matlab_baseline_mask(
            bv_time,
            BASELINE_TMIN_S,
            BASELINE_TMAX_S,
        )
        baseline_tmax_inclusive = float(bv_time[_bl_idx_stop])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _trial_bl_means = np.nanmean(bv_epochs[:, :, _bl_mask], axis=2)  # (N_trials, N_ch)

        n_ch, n_samp = bv_epochs.shape[1], bv_epochs.shape[2]
        empty_b = np.empty((0, n_ch, n_samp), dtype=np.float64)

        # Compute the ACTUAL reference mean and σ. In oracle mode we can use
        # the exact MATLAB keep-mask saved in bsl_info; otherwise fall back to
        # Python's own outlier detection on the BV baseline means.
        _used_matlab_keep_mask = False
        _mat_keep_mask = mat_bsl_info.get("baseline_keep_mask") if mat_bsl_info is not None else None
        if (
            USE_MATLAB_BASELINE_KEEP_MASK
            and _mat_keep_mask is not None
            and _mat_keep_mask.ndim == 2
        ):
            _baseline_mapping = match_channels(mat_channels, bv_channels)
            _bv_keep_mask = np.zeros(_trial_bl_means.shape, dtype=bool)
            for _mi, _bi in _baseline_mapping.items():
                if _mi < _mat_keep_mask.shape[0]:
                    _n = min(_trial_bl_means.shape[0], _mat_keep_mask.shape[1])
                    _bv_keep_mask[:_n, _bi] = _mat_keep_mask[_mi, :_n]
            _bl_means_clean = np.full_like(_trial_bl_means, np.nan)
            _bl_means_clean[_bv_keep_mask] = _trial_bl_means[_bv_keep_mask]
            _used_matlab_keep_mask = True
            py_rej["baseline_keep_mask_matlab"] = _bv_keep_mask
            print("  Using MATLAB baseline_keep_mask directly for the BV z-score reference.")
        elif BASELINE_REMOVE_OUTLIERS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                _outlier_mask, _ = compute_baseline_outlier_mask(
                    bv_epochs,
                    empty_b,
                    bv_time,
                    baseline_tmin_s=BASELINE_TMIN_S,
                    baseline_tmax_s=baseline_tmax_inclusive,
                    baseline_scope="global",
                    remove_outlier_trial_means=True,
                    outlier_method=BASELINE_OUTLIER_METHOD,
                )
            _bl_means_clean = _trial_bl_means.copy()
            _bl_means_clean[_outlier_mask] = np.nan
        else:
            _bl_means_clean = _trial_bl_means

        bv_n_clean = np.sum(np.isfinite(_bl_means_clean), axis=0).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bv_ref_mean = np.nanmean(_bl_means_clean, axis=0)
            bv_ref_std = np.nanstd(_bl_means_clean, axis=0, ddof=1)

        if _used_matlab_keep_mask:
            _scale = np.where(
                np.isfinite(bv_ref_std) & (bv_ref_std > 0.0),
                bv_ref_std,
                1.0,
            )
            bv_epochs = (
                bv_epochs - bv_ref_mean[np.newaxis, :, np.newaxis]
            ) / _scale[np.newaxis, :, np.newaxis]
            bv_epochs[~np.isfinite(np.asarray(bv_epochs, dtype=np.float64))] = np.nan
        else:
            # zscore_activity_by_baseline expects (n_trials, n_features, n_times).
            # Pass all trials as condition A and an empty array as condition B so
            # the global scope pools baseline means from all trials (single group).
            n_ch, n_samp = bv_epochs.shape[1], bv_epochs.shape[2]
            empty_b = np.empty((0, n_ch, n_samp), dtype=np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                bv_epochs_z, _ = zscore_activity_by_baseline(
                    bv_epochs,
                    empty_b,
                    bv_time,
                    baseline_tmin_s=BASELINE_TMIN_S,
                    baseline_tmax_s=baseline_tmax_inclusive,
                    baseline_scope="global",
                    remove_outlier_trial_means=BASELINE_REMOVE_OUTLIERS,
                    outlier_method=BASELINE_OUTLIER_METHOD,
                )
            bv_epochs = bv_epochs_z
        print(
            "  Z-scoring done "
            f"(effective baseline={bv_time[_bl_idx_start]:.4f}→{baseline_tmax_inclusive:.4f} s, "
            f"{int(np.sum(_bl_mask))} samples)."
        )

    # bsl_info is already loaded above (before the mask block).
    # ------------------------------------------------------------------
    print("\nMatching channels...")
    channel_mapping = match_channels(mat_channels, bv_channels)
    print(f"  Matched: {len(channel_mapping)} / {len(mat_channels)} MATLAB channels")

    if not channel_mapping:
        print("\nDiagnostic — no channels matched.")
        print(f"  MATLAB bipole names (first 10): {mat_channels[:10]}")
        print(f"  BV channels        (first 10): {bv_channels[:10]}")
        print("\nFirst contacts extracted from MATLAB (first 10):",
              [_first_contact(c) for c in mat_channels[:10]])
        print("Normalized first contacts (first 10):",
              [_norm(_first_contact(c)) for c in mat_channels[:10]])
        print("Normalized BV (first 10):",
              [_norm(c) for c in bv_channels[:10]])
        sys.exit("ERROR: No channel match found. Check channel naming conventions.")

    # Print a sample of matched pairs for inspection
    print("  Sample matches (MATLAB bipole → first contact → BV):")
    for mi, bi in list(channel_mapping.items())[:8]:
        first = _first_contact(mat_channels[mi])
        print(f"    [{mi:3d}] {mat_channels[mi]!r:20s}  first={first!r:12s} → [{bi:3d}] {bv_channels[bi]!r}")

    # ------------------------------------------------------------------
    # 4. Trial count alignment
    # ------------------------------------------------------------------
    n_trials_mat = mat_data.shape[0]
    n_trials_bv = bv_epochs.shape[0]
    print(f"\nTrials — MATLAB: {n_trials_mat}, BrainVision: {n_trials_bv}")
    if n_trials_mat != n_trials_bv:
        print(
            f"  WARNING: trial counts differ. "
            f"Comparison will use the first {min(n_trials_mat, n_trials_bv)} trials."
        )

    # Print MATLAB opts_log vs Python pipeline rejection comparison
    print_rejection_comparison(
        mat_rej_info, py_rej, mat_channels, bv_channels,
        channel_mapping, min(n_trials_mat, n_trials_bv),
    )
    print_bv_start_offset_diagnostics(
        mat_data[:min(n_trials_mat, n_trials_bv)],
        bv_epochs[:min(n_trials_mat, n_trials_bv)],
        bv_time,
        channel_mapping,
    )
    if RUN_BASELINE_STRATEGY_EXPERIMENTS:
        print_baseline_strategy_experiments(
            mat_epochs=mat_data[:min(n_trials_mat, n_trials_bv)],
            mat_time=mat_time,
            bv_epochs_pre_zscore=bv_epochs_pre_zscore[:min(n_trials_mat, n_trials_bv)],
            bv_time=bv_time,
            mat_channels=mat_channels,
            bv_channels=bv_channels,
            channel_mapping=channel_mapping,
            mat_bsl_info=mat_bsl_info,
            top_n=BASELINE_EXPERIMENT_TOP_N,
        )

    # ------------------------------------------------------------------
    # 4b. Per-channel σ summary (MATLAB bsl_info vs BV)
    # ------------------------------------------------------------------
    if mat_bsl_info is not None and bv_ref_std is not None:
        _ptbl = mat_bsl_info.get("per_trial_bl_means")
        _mstd_all = mat_bsl_info["std"]
        print("\n--- Per-channel zscore σ summary (MAT bsl_info vs BV) ---")
        print(f"  {'MAT ch':>6}  {'BV ch':>5}  {'BV name':<14}  {'BV σ':>8}  {'MAT σ':>8}  {'ratio BV/MAT':>12}  {'BV n_fin':>8}  {'MAT n_fin':>9}  {'BV n_cln':>8}  {'MAT n_cln':>9}")
        ratios = []
        for mi, bi in sorted(channel_mapping.items()):
            bv_s = float(bv_ref_std[bi]) if bi < len(bv_ref_std) else float("nan")
            mat_s = float(_mstd_all[mi]) if mi < len(_mstd_all) else float("nan")
            ratio = bv_s / mat_s if (mat_s > 0 and np.isfinite(mat_s) and np.isfinite(bv_s)) else float("nan")

            # BV n_finite (pre-rmoutliers) for this channel
            _bl_m_s, _, _ = _matlab_baseline_mask(
                bv_time,
                BASELINE_TMIN_S,
                BASELINE_TMAX_S,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                _bv_bl = np.nanmean(bv_epochs[:, :, _bl_m_s], axis=2)[:, bi]
            bv_n_fin = int(np.sum(np.isfinite(_bv_bl)))
            bv_n_cln = int(bv_n_clean[bi]) if bv_n_clean is not None else -1

            # MAT n_finite + n_clean from saved bsl_info masks when available.
            mat_n_fin = mat_n_cln = -1
            if _ptbl is not None and _ptbl.ndim == 2 and mi < _ptbl.shape[0]:
                _ch_bl = _ptbl[mi, :].astype(np.float64)
                _fin = _ch_bl[np.isfinite(_ch_bl)]
                mat_n_fin = len(_fin)
                _mk = mat_bsl_info.get("baseline_keep_mask")
                if _mk is not None and _mk.ndim == 2 and mi < _mk.shape[0]:
                    _keep = np.asarray(_mk[mi, :], dtype=bool)
                    _n = min(len(_keep), len(_ch_bl))
                    mat_n_cln = int(np.sum(_keep[:_n] & np.isfinite(_ch_bl[:_n])))
                elif len(_fin) >= 3:
                    _ctr = float(np.median(_fin))
                    _mad = float(np.median(np.abs(_fin - _ctr)))
                    if _mad > 0.0:
                        mat_n_cln = int(np.sum(np.abs(_fin - _ctr) <= 3.0 * 1.4826 * _mad))
                    else:
                        mat_n_cln = len(_fin)

            if np.isfinite(ratio):
                ratios.append(ratio)
            print(f"  [{mi:4d}]  [{bi:4d}]  {bv_channels[bi]:<14}  {bv_s:8.4f}  {mat_s:8.4f}  {ratio:12.4f}  {bv_n_fin:8d}  {mat_n_fin:9d}  {bv_n_cln:8d}  {mat_n_cln:9d}")
        if ratios:
            print(f"\n  Median BV/MAT σ ratio across {len(ratios)} channels: {float(np.median(ratios)):.4f}")
            print(f"  Mean   BV/MAT σ ratio: {float(np.mean(ratios)):.4f}  std: {float(np.std(ratios, ddof=1)):.4f}")
        print("---")

    # ------------------------------------------------------------------
    # 5. Launch viewer
    # ------------------------------------------------------------------
    print("\nLaunching comparison viewer…")
    print("  Keyboard shortcuts: ← → to step trials, ↑ ↓ to step channels")

    viewer = ComparisonViewer(
        mat_epochs=mat_data,
        mat_time=mat_time,
        mat_channels=mat_channels,
        mat_sfreq=mat_sfreq,
        bv_epochs=bv_epochs,
        bv_time=bv_time,
        bv_channels=bv_channels,
        bv_sfreq=bv_sfreq,
        channel_mapping=channel_mapping,
        bv_ref_std=bv_ref_std,
        bv_ref_mean=bv_ref_mean,
        bv_n_clean=bv_n_clean,
        bv_diag=bv_diag,
        mat_rej=mat_rej_info,
        py_rej=py_rej,
        mat_bsl_info=mat_bsl_info,
    )
    viewer.show()


if __name__ == "__main__":
    main()
