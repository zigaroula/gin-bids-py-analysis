#!/usr/bin/env python3
"""
Raw comparison: MATLAB b1 output vs BrainVision (no corrections, no z-score).

MATLAB file  : alldata from b1_BPF_computations (N_trials × N_samples × N_bipoles).
               Epochs are around events 11/12, starting from event 5, tmin=-1 s, tmax=10 s.
BrainVision  : continuous Hilbert-processed signal with embedded events.
               Re-epoched here with the same parameters.

Both signals are displayed as-is (raw amplitude).  Because the physical units may
differ, an optional per-trace normalization (divide by per-trial std) can be
toggled with the "Norm" button to compare shapes independently of scale.

Usage:
    .venv\\Scripts\\python scripts\\compare_b1_vs_bids_raw.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.io
from matplotlib.widgets import Button, Slider

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from compare_subject_config import BV_EEG_PATH, BV_EVENT_SAMPLE_SHIFT_SAMPLES, MATLAB_B1_PATH  # noqa: E402

# ---------------------------------------------------------------------------
# File paths  (see compare_subject_config.py to change the subject)
# ---------------------------------------------------------------------------

# MNE needs the .vhdr header — derive it from the .eeg path.
BV_VHDR_PATH = BV_EEG_PATH.with_suffix(".vhdr")

# Epoching window applied to the BrainVision recording.
TMIN_S: float = -1   # epoch start relative to anchor
TMAX_S: float = 6    # epoch end relative to anchor
BV_DISPLAY_OFFSET_S: float = 0.0  # shift BV time axis by this amount for display
# Same anchor / experiment-start codes as the b2 comparison script.
ANCHOR_CODES: set[str] = {"11", "12"}
EXPERIMENT_START_CODE: str = "5"
# BV_EVENT_SAMPLE_SHIFT_SAMPLES is imported from compare_subject_config.py.
# Set to -1 for Micromed/TRC subjects (Grenoble, Lyon) or 0 for Prague Matlab
# subjects.  See compare_subject_config.py for the full explanation.

# MATLAB epoch window (for display trimming of BV data).
MAT_TMIN_S: float = -1
MAT_TMAX_S: float = 6
SHIFT_DIAGNOSTIC_RANGE: tuple[int, ...] = (-2, -1, 0, 1, 2)
CHANNEL_DIAGNOSTIC_TOP_N: int = 15
BASELINE_TMIN_S: float = -0.25
BASELINE_TMAX_S: float = -0.05
BASELINE_OUTLIER_METHOD: str = "median_mad"
BASELINE_DIAGNOSTIC_TOP_N: int = 10


# ===========================================================================
# MATLAB loading  (same helpers as compare_matlab_vs_bids.py)
# ===========================================================================

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
    m = re.search(r"_sf(\d+)", path.stem, re.IGNORECASE)
    return float(m.group(1)) if m else fallback


# --- h5py helpers -----------------------------------------------------------

def _hdf5_cell_strings(f: object, ds: object) -> list[str]:
    import h5py  # noqa: PLC0415
    data = ds[()]
    if h5py.check_string_dtype(ds.dtype):
        raw = ds.asstr()[()]
        return [str(s).strip() for s in np.asarray(raw).flatten()]
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
    import h5py  # noqa: PLC0415
    data = np.asarray(ds[()])
    if data.ndim == 1:
        refs = data
    elif data.ndim == 2:
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


def _load_hdf5_matlab(path: Path) -> tuple[np.ndarray, list[str], float]:
    import h5py  # noqa: PLC0415
    with h5py.File(str(path), "r") as f:
        raw_data = f["alldata"][()]
        alldata: np.ndarray = raw_data.T  # (N_trials, N_samples, N_bipoles)
        channels: list[str] = []
        sfreq: float = _sfreq_from_filename(path)
        if "hdr" in f:
            hdr_group = f["hdr"]
            if "chan_info" in hdr_group:
                channels = _hdf5_chan_info_column(f, hdr_group["chan_info"], col=1)
            elif "label" in hdr_group:
                channels = _hdf5_cell_strings(f, hdr_group["label"])
            for fname in ("Fs", "fsample", "fs", "srate", "sfreq"):
                if fname in hdr_group:
                    val = _hdf5_scalar(f, hdr_group[fname])
                    if val is not None:
                        sfreq = val
                        break
    return alldata, channels, sfreq


# ---------------------------------------------------------------------------
# SPM MEEG format loader  (variable 'D' + binary '.dat' sidecar)
# ---------------------------------------------------------------------------

def _spm_str(val: object) -> str:
    """Extract a plain Python string from a scipy-loaded MATLAB char array or string."""
    if isinstance(val, str):
        return val.strip()
    arr = np.asarray(val).flatten()
    return "".join(chr(int(c)) for c in arr).strip()


def _spm_channel_labels(D: object) -> list[str]:
    """Extract channel labels from an SPM MEEG D struct."""
    ch_field = getattr(D, "channels", None)
    if ch_field is None:
        return []
    ch_arr = np.asarray(ch_field).flatten()
    labels: list[str] = []
    for ch in ch_arr:
        lbl = getattr(ch, "label", None)
        if lbl is not None:
            labels.append(_spm_str(lbl))
    return labels


def _load_spm_meeg(mat_path: Path, D: object) -> tuple[np.ndarray, list[str], float]:
    """Load data from an SPM MEEG object (D) and its binary .dat sidecar.

    SPM stores data as a flat binary file (float32, Fortran order).
    Dimensions are inferred from D metadata with multiple fallbacks, including
    reading D.data.dim / D.data.size if present, and inferring the missing
    dimension from the actual file size.

    Returns:
        alldata : (N_trials, N_samples, N_channels)  float64
                  For continuous files: N_trials == 1.
        channels: list of channel label strings
        sfreq   : sampling frequency in Hz
    """
    # --- Sampling frequency ---
    sfreq: float = _sfreq_from_filename(mat_path)
    for attr in ("Fsample", "fsample", "Fs", "fs"):
        val = getattr(D, attr, None)
        if val is not None:
            try:
                sfreq = float(np.asarray(val).flat[0])
                break
            except (TypeError, ValueError):
                pass

    # --- Channel labels ---
    channels = _spm_channel_labels(D)
    n_ch_from_labels = len(channels)

    # --- Try D.data sub-struct for explicit stored dimensions ---
    # SPM12 keeps D.data.dim = [n_channels_stored, n_samples, n_trials]
    dim_from_data: list[int] | None = None
    data_sub = getattr(D, "data", None)
    if data_sub is not None:
        for dim_attr in ("dim", "size", "Dim", "Size"):
            dval = getattr(data_sub, dim_attr, None)
            if dval is not None:
                arr = np.asarray(dval, dtype=float).flatten()
                if len(arr) >= 2:
                    dim_from_data = [int(x) for x in arr[:3]]
                    break

    # --- Nsamples and n_trials from D fields ---
    n_samples_d: int = 0
    for attr in ("Nsamples", "nsamples"):
        val = getattr(D, attr, None)
        if val is not None:
            n_samples_d = int(np.asarray(val).flat[0])
            break
    trials_field = getattr(D, "trials", None)
    n_trials_d: int = 1
    if trials_field is not None:
        n_trials_d = max(1, int(np.asarray(trials_field).flatten().size))

    print(
        f"  D metadata → channels_labels={n_ch_from_labels}, "
        f"Nsamples={n_samples_d}, n_trials(struct)={n_trials_d}, "
        f"sfreq={sfreq}"
        + (f", D.data.dim={dim_from_data}" if dim_from_data else "")
    )

    # --- Locate the .dat binary file ---
    # SPM stores the binary sidecar as <same_stem>.dat next to the .mat.
    # D.fname typically holds the .mat filename; D.data.fname may hold the .dat name.
    dat_path: Path | None = None

    # 1. Try D.data.fname (most reliable in SPM12)
    data_sub_for_fname = getattr(D, "data", None)
    if data_sub_for_fname is not None:
        dat_fname_field = getattr(data_sub_for_fname, "fname", None)
        if dat_fname_field is not None:
            dat_fname_str = _spm_str(dat_fname_field)
            if dat_fname_str and not dat_fname_str.endswith(".mat"):
                candidate = mat_path.parent / Path(dat_fname_str).name
                if candidate.exists():
                    dat_path = candidate

    # 2. Try same stem as .mat with .dat extension
    if dat_path is None:
        candidate = mat_path.with_suffix(".dat")
        if candidate.exists():
            dat_path = candidate

    # 3. Try D.fname (only if it doesn't point to a .mat file)
    if dat_path is None:
        fname_field = getattr(D, "fname", None)
        if fname_field is not None:
            dat_name = _spm_str(fname_field)
            if dat_name and not dat_name.endswith(".mat"):
                candidate = mat_path.parent / Path(dat_name).name
                if candidate.exists():
                    dat_path = candidate

    # 4. Scan directory for any .dat file (pick the one with matching stem prefix)
    if dat_path is None:
        dat_candidates = sorted(mat_path.parent.glob("*.dat"))
        if dat_candidates:
            # Prefer one whose stem matches the .mat stem
            stem = mat_path.stem
            best = next((p for p in dat_candidates if p.stem == stem), dat_candidates[0])
            dat_path = best
            if len(dat_candidates) > 1:
                print(f"  Multiple .dat files found, using: {best.name}")
                print(f"  Others: {[p.name for p in dat_candidates if p != best]}")

    if dat_path is None:
        raise FileNotFoundError(
            f"SPM .dat sidecar not found in {mat_path.parent}.\n"
            f"  Expected: {mat_path.stem}.dat\n"
            f"  Place the .dat file next to {mat_path.name}"
        )
    print(f"  SPM .dat sidecar: {dat_path.name}")

    # --- Read binary data (try float32 then float64) ---
    raw_f32 = np.fromfile(str(dat_path), dtype=np.float32)
    raw_f64 = np.fromfile(str(dat_path), dtype=np.float64)
    n_elem_f32 = raw_f32.size
    n_elem_f64 = raw_f64.size
    print(f"  .dat size: {dat_path.stat().st_size} bytes "
          f"({n_elem_f32} float32 elements / {n_elem_f64} float64 elements)")

    # --- Determine n_channels, n_samples, n_trials ---
    # Priority 1: D.data.dim (most reliable)
    if dim_from_data and len(dim_from_data) >= 2:
        n_ch_stored = dim_from_data[0]
        n_samp_stored = dim_from_data[1]
        n_trials_stored = dim_from_data[2] if len(dim_from_data) >= 3 else 1
        for raw, dtype_name in [(raw_f32, "float32"), (raw_f64, "float64")]:
            if raw.size == n_ch_stored * n_samp_stored * n_trials_stored:
                print(f"  Using D.data.dim dimensions ({dtype_name}).")
                raw_flat = raw
                n_channels, n_samples, n_trials = n_ch_stored, n_samp_stored, n_trials_stored
                break
        else:
            print(f"  WARNING: D.data.dim={dim_from_data} doesn't match file size — falling through.")
            dim_from_data = None

    if not dim_from_data:
        # Priority 2: infer from D.Nsamples + D.trials, adjusting n_trials or n_channels
        # from file size.  Try float32 first.
        found = False
        for raw, dtype_name in [(raw_f32, "float32"), (raw_f64, "float64")]:
            n_elem = raw.size
            n_channels = n_ch_from_labels
            n_samples = n_samples_d
            n_trials = n_trials_d

            # Try to infer n_trials from file size given n_channels and n_samples.
            if n_channels > 0 and n_samples > 0 and n_elem % (n_channels * n_samples) == 0:
                n_trials = n_elem // (n_channels * n_samples)
                raw_flat = raw
                found = True
                if n_trials != n_trials_d:
                    print(f"  Adjusted n_trials {n_trials_d}→{n_trials} from file size ({dtype_name}).")
                else:
                    print(f"  Dimensions confirmed from file size ({dtype_name}).")
                break
            # Try to infer n_channels from file size given n_samples and n_trials.
            if n_samples > 0 and n_trials_d > 1 and n_elem % (n_samples * n_trials_d) == 0:
                n_channels = n_elem // (n_samples * n_trials_d)
                n_trials = n_trials_d
                raw_flat = raw
                found = True
                print(f"  Inferred n_channels={n_channels} (D.channels had {n_ch_from_labels}) "
                      f"from file size ({dtype_name}).")
                break
            # Try treating as continuous: (n_channels, n_total_samples)
            if n_channels > 0 and n_elem % n_channels == 0:
                n_samples_cont = n_elem // n_channels
                n_trials = 1
                n_samples = n_samples_cont
                raw_flat = raw
                found = True
                print(f"  Treating as continuous: {n_channels}ch × {n_samples}samp ({dtype_name}).")
                break

        if not found:
            # Last resort: dump all possible 3-factor combinations
            print(
                f"  ERROR: cannot infer shape from file ({n_elem_f32} float32 elements).\n"
                f"  D says: {n_ch_from_labels}ch × {n_samples_d}samp × {n_trials_d}trials.\n"
                f"  All integer factor-pairs of {n_elem_f32}: "
                + str(sorted({(a, n_elem_f32 // a) for a in range(1, int(n_elem_f32**0.5) + 1)
                              if n_elem_f32 % a == 0}))
            )
            raise ValueError(
                f"Cannot infer SPM data shape from .dat file size ({n_elem_f32} elements). "
                f"See diagnostic output above."
            )

    # --- Reshape ---
    # SPM layout: (n_channels, n_samples, n_trials) Fortran order.
    # For continuous files n_trials==1 and data is (n_channels, n_total_samples).
    expected = n_channels * n_samples * n_trials
    data_3d = raw_flat[:expected].reshape((n_channels, n_samples, n_trials), order="F")
    # Transpose to (N_trials, N_samples, N_channels)
    alldata = np.asarray(data_3d.transpose(2, 1, 0), dtype=np.float64)
    print(f"  SPM data loaded: {alldata.shape}  (N_trials, N_samples, N_channels)")

    # Trim/pad channel labels to match stored dimension
    if len(channels) != n_channels:
        print(f"  NOTE: D.channels has {len(channels)} labels but {n_channels} channels stored "
              f"— using indices.")
        channels = [f"ch{i:03d}" for i in range(n_channels)]

    return alldata, channels, sfreq


def load_matlab_b1(path: Path) -> tuple[np.ndarray, list[str], float]:
    """Load (alldata, channel_names, sfreq) from a MATLAB b1 file.

    Supports:
    - MATLAB v5  with alldata/data/epochs variable
    - MATLAB v5  with SPM MEEG 'D' variable (+ binary .dat sidecar)
    - MATLAB v7.3 (HDF5) with alldata variable

    alldata shape: (N_trials, N_samples, N_channels)
    """
    # If path points to a directory, find the .mat file inside it.
    if path.is_dir():
        candidates = sorted(path.glob("*.mat"))
        if not candidates:
            sys.exit(f"ERROR: no .mat files found in directory: {path}")
        if len(candidates) > 1:
            print(f"  Multiple .mat files found, using: {candidates[0].name}")
            print(f"  Others: {[c.name for c in candidates[1:]]}")
            print("  Set MATLAB_B1_PATH to the exact file to override.")
        path = candidates[0]

    print(f"  Using file: {path}")
    try:
        mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        print("  (MATLAB v7.3 detected, using h5py loader)")
        return _load_hdf5_matlab(path)

    mat_keys = [k for k in mat.keys() if not k.startswith("_")]
    print(f"  Variables in file: {mat_keys}")

    # --- SPM MEEG format: single 'D' variable with binary .dat sidecar ---
    if mat_keys == ["D"] or (len(mat_keys) == 1 and "D" in mat_keys):
        print("  Detected SPM MEEG format (D + .dat sidecar).")
        return _load_spm_meeg(path, mat["D"])

    # --- Classic alldata / data / epochs variable ---
    _DATA_CANDIDATES = ["alldata", "data", "epochs", "allData", "Data", "EpochData"]
    data_key = next((k for k in _DATA_CANDIDATES if k in mat), None)
    if data_key is None:
        # Fall back to first non-scalar array
        for k, v in mat.items():
            if k.startswith("_"):
                continue
            arr = np.asarray(v)
            if arr.ndim >= 2:
                data_key = k
                break
    if data_key is None:
        raise KeyError(
            f"Could not find epoch data array in {path.name}. "
            f"Available keys: {mat_keys}"
        )
    if data_key != "alldata":
        print(f"  NOTE: using variable '{data_key}' as epoch data.")

    alldata = np.asarray(mat[data_key])
    if alldata.ndim == 2:
        alldata = alldata[np.newaxis, ...]
    if alldata.ndim != 3:
        raise ValueError(f"Unexpected alldata shape: {alldata.shape}")

    hdr = mat.get("hdr")
    channels: list[str] = []
    sfreq: float = _sfreq_from_filename(path)
    if hdr is not None:
        channels = _extract_channels_scipy(hdr)
        sfreq = _extract_sfreq_scipy(hdr, fallback=sfreq)

    return alldata, channels, sfreq


# ===========================================================================
# BrainVision epoching
# ===========================================================================

def _annotation_code(description: str) -> str | None:
    m = re.search(r"S\s*(\d+)", str(description))
    return m.group(1) if m else None


def _collect_anchor_onsets(
    raw: mne.io.BaseRaw,
    anchor_codes: set[str] = ANCHOR_CODES,
    experiment_start_code: str = EXPERIMENT_START_CODE,
) -> tuple[list[float], float]:
    """Return sorted anchor onsets after the experiment-start marker."""
    t_start = -np.inf
    for ann in raw.annotations:
        if _annotation_code(str(ann["description"])) == experiment_start_code:
            t_start = float(ann["onset"])
            break

    anchor_onsets: list[float] = []
    for ann in raw.annotations:
        code = _annotation_code(str(ann["description"]))
        if code in anchor_codes:
            onset = float(ann["onset"])
            if onset > t_start:
                anchor_onsets.append(onset)
    anchor_onsets.sort()
    return anchor_onsets, t_start


def epoch_brainvision(
    raw: mne.io.BaseRaw,
    tmin_s: float = TMIN_S,
    tmax_s: float = TMAX_S,
    anchor_codes: set[str] = ANCHOR_CODES,
    experiment_start_code: str = EXPERIMENT_START_CODE,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Epoch the BrainVision file around anchor events.

    Returns:
        epochs     : (N_trials, N_channels, N_samples)  float64
        time_axis  : (N_samples,)  seconds relative to anchor
        ch_names   : list of channel names
    """
    sfreq = float(raw.info["sfreq"])
    anchor_onsets, t_start = _collect_anchor_onsets(
        raw,
        anchor_codes=anchor_codes,
        experiment_start_code=experiment_start_code,
    )

    if np.isinf(t_start):
        print(f"  WARNING: experiment start event '{experiment_start_code}' not found.")

    print(f"  Found {len(anchor_onsets)} anchor events [codes {sorted(anchor_codes)}]"
          f" after t_start={t_start:.3f}s")
    if not anchor_onsets:
        raise RuntimeError("No anchor events found in BrainVision file.")

    n_trials = len(anchor_onsets)
    events = np.column_stack([
        np.round(np.array(anchor_onsets) * sfreq).astype(np.int64) + BV_EVENT_SAMPLE_SHIFT_SAMPLES,
        np.zeros(n_trials, dtype=np.int64),
        np.ones(n_trials, dtype=np.int64),
    ])

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
    data = epochs_mne.get_data()
    time_axis = epochs_mne.times

    dropped = n_trials - data.shape[0]
    if dropped:
        print(f"  WARNING: {dropped} trials were dropped by MNE (edge effects).")

    return data, time_axis, list(raw.ch_names)


# ===========================================================================
# Channel matching
# ===========================================================================

_FIRST_CONTACT_RE = re.compile(r"^([A-Za-z]+\d+)")


def _first_contact(bipole_name: str) -> str:
    m = _FIRST_CONTACT_RE.match(bipole_name)
    return m.group(1) if m else bipole_name


def _norm(name: str) -> str:
    return re.sub(r"[\s\-_\.]", "", name).casefold()


def match_channels(
    matlab_channels: list[str],
    bv_channels: list[str],
) -> dict[int, int]:
    """Return {matlab_idx: bv_idx} matched by first-contact name."""
    bv_by_norm: dict[str, int] = {_norm(ch): i for i, ch in enumerate(bv_channels)}
    mapping: dict[int, int] = {}
    for mi, mc in enumerate(matlab_channels):
        norm_mc = _norm(_first_contact(mc))
        if norm_mc in bv_by_norm:
            mapping[mi] = bv_by_norm[norm_mc]
    return mapping


def _matlab_nearest_index(time_axis_s: np.ndarray, target_s: float) -> int:
    """Return the nearest-sample index, matching MATLAB's min(abs(t-target))."""
    time_axis = np.asarray(time_axis_s, dtype=np.float64).ravel()
    return int(np.argmin(np.abs(time_axis - float(target_s))))


def _matlab_baseline_mask(
    time_axis_s: np.ndarray,
    baseline_tmin_s: float,
    baseline_tmax_s: float,
) -> tuple[np.ndarray, int, int]:
    """Return the exact baseline mask used by MATLAB's nearest-index logic."""
    time_axis = np.asarray(time_axis_s, dtype=np.float64).ravel()
    idx_start = _matlab_nearest_index(time_axis, baseline_tmin_s)
    idx_stop = _matlab_nearest_index(time_axis, baseline_tmax_s) - 1
    if idx_stop < idx_start:
        raise ValueError(
            "Empty baseline window after applying MATLAB's exclusive upper bound: "
            f"{baseline_tmin_s=} {baseline_tmax_s=}."
        )
    mask = np.zeros(time_axis.shape, dtype=bool)
    mask[idx_start:idx_stop + 1] = True
    return mask, idx_start, idx_stop


def _keep_mask_from_trial_means(
    trial_means: np.ndarray,
    *,
    method: str = BASELINE_OUTLIER_METHOD,
) -> tuple[np.ndarray, np.ndarray]:
    """Return keep/outlier masks from per-trial baseline means."""
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


def _baseline_trial_means(
    mat_epochs: np.ndarray,
    bv_epochs: np.ndarray,
    mat_time: np.ndarray,
    bv_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    """Return per-trial baseline means for MATLAB and BV epochs."""
    mat_mask, mat_i0, mat_i1 = _matlab_baseline_mask(
        mat_time,
        BASELINE_TMIN_S,
        BASELINE_TMAX_S,
    )
    bv_mask, bv_i0, bv_i1 = _matlab_baseline_mask(
        bv_time,
        BASELINE_TMIN_S,
        BASELINE_TMAX_S,
    )
    mat_trial_means = np.nanmean(
        np.asarray(mat_epochs, dtype=np.float64)[:, mat_mask, :],
        axis=1,
    )  # (n_trials, n_mat_ch)
    bv_trial_means = np.nanmean(
        np.asarray(bv_epochs, dtype=np.float64)[:, :, bv_mask],
        axis=2,
    )  # (n_trials, n_bv_ch)
    return mat_trial_means, bv_trial_means, (mat_i0, mat_i1), (bv_i0, bv_i1)


def print_sample_shift_diagnostics(
    mat_epochs: np.ndarray,
    raw: mne.io.BaseRaw,
    channel_mapping: dict[int, int],
    *,
    tmin_s: float = TMIN_S,
    shift_range: tuple[int, ...] = SHIFT_DIAGNOSTIC_RANGE,
) -> None:
    """Report which integer sample shift best aligns BV raw epochs to MATLAB b1."""
    if not channel_mapping:
        return

    sfreq = float(raw.info["sfreq"])
    anchor_onsets, _ = _collect_anchor_onsets(raw)
    anchor_samples = np.round(np.array(anchor_onsets) * sfreq).astype(np.int64)
    full = np.asarray(raw._data, dtype=np.float64)  # raw is already preloaded in main()
    base_start = int(round(tmin_s * sfreq)) + BV_EVENT_SAMPLE_SHIFT_SAMPLES
    n_trials = min(mat_epochs.shape[0], len(anchor_samples))
    n_samples = mat_epochs.shape[1]

    print("\nAlignment diagnostic — extra integer sample shifts around the configured BV epoch shift")
    results: list[tuple[int, float, float]] = []
    for shift in shift_range:
        mean_abs_values: list[float] = []
        corrs: list[float] = []
        for mat_idx, bv_idx in sorted(channel_mapping.items()):
            bv_trials: list[np.ndarray] = []
            for sample in anchor_samples[:n_trials]:
                start = int(sample + base_start + shift)
                stop = start + n_samples
                if start < 0 or stop > full.shape[1]:
                    continue
                bv_trials.append(full[bv_idx, start:stop])
            if len(bv_trials) != n_trials:
                continue
            bv_arr = np.asarray(bv_trials, dtype=np.float64)
            mat_arr = mat_epochs[:n_trials, :, mat_idx]
            diff = bv_arr - mat_arr
            mean_abs = float(np.nanmean(np.abs(diff)))
            if np.isfinite(mean_abs):
                mean_abs_values.append(mean_abs)
            a = mat_arr.ravel()
            b = bv_arr.ravel()
            finite = np.isfinite(a) & np.isfinite(b)
            if np.sum(finite) > 2:
                a = a[finite] - np.mean(a[finite])
                b = b[finite] - np.mean(b[finite])
                denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
                if denom > 0.0:
                    corrs.append(float(np.sum(a * b) / denom))
        if mean_abs_values:
            results.append((
                shift,
                float(np.mean(mean_abs_values)),
                float(np.nanmedian(corrs)) if corrs else np.nan,
            ))

    for shift, mean_abs, med_corr in sorted(results, key=lambda row: row[1]):
        print(f"  shift={shift:+d} sample(s)  mean|Δ|={mean_abs:.6f}  median corr={med_corr:.6f}")


def print_baseline_shift_diagnostics(
    mat_epochs: np.ndarray,
    raw: mne.io.BaseRaw,
    mat_time: np.ndarray,
    mat_channels: list[str],
    bv_channels: list[str],
    channel_mapping: dict[int, int],
    *,
    tmin_s: float = TMIN_S,
    shift_range: tuple[int, ...] = SHIFT_DIAGNOSTIC_RANGE,
) -> None:
    """Report how integer BV shifts affect baseline-mean agreement and keep masks."""
    if not channel_mapping:
        return

    sfreq = float(raw.info["sfreq"])
    anchor_onsets, _ = _collect_anchor_onsets(raw)
    anchor_samples = np.round(np.array(anchor_onsets) * sfreq).astype(np.int64)
    full = np.asarray(raw._data, dtype=np.float64)
    base_start = int(round(tmin_s * sfreq)) + BV_EVENT_SAMPLE_SHIFT_SAMPLES
    n_trials = min(mat_epochs.shape[0], len(anchor_samples))
    n_samples = mat_epochs.shape[1]
    mat_mask, mat_i0, mat_i1 = _matlab_baseline_mask(
        mat_time,
        BASELINE_TMIN_S,
        BASELINE_TMAX_S,
    )
    mat_trial_means = np.nanmean(
        np.asarray(mat_epochs[:n_trials], dtype=np.float64)[:, mat_mask, :],
        axis=1,
    )
    mat_keep, _ = _keep_mask_from_trial_means(mat_trial_means, method=BASELINE_OUTLIER_METHOD)

    print(
        "\nBaseline shift diagnostic — baseline-trial means and keep-mask agreement "
        f"[{BASELINE_TMIN_S},{BASELINE_TMAX_S}]s"
    )
    print(
        f"  {'extra':>7}  {'total':>7}  {'mean|ΔBL|':>10}  {'pair agree':>10}  "
        f"{'exact ch':>9}  {'mismatched':>11}"
    )
    results: list[tuple[int, int, float, float, int, int]] = []
    for shift in shift_range:
        mean_abs_values: list[float] = []
        mismatched_pairs = 0
        exact_channels = 0
        for mat_idx, bv_idx in sorted(channel_mapping.items()):
            mat_vec = mat_trial_means[:, mat_idx]
            bv_means: list[float] = []
            valid = True
            for sample in anchor_samples[:n_trials]:
                start = int(sample + base_start + shift)
                stop = start + n_samples
                if start < 0 or stop > full.shape[1]:
                    valid = False
                    break
                baseline = full[bv_idx, start + mat_i0: start + mat_i1 + 1]
                bv_means.append(float(np.mean(baseline)))
            if not valid or len(bv_means) != n_trials:
                continue
            bv_vec = np.asarray(bv_means, dtype=np.float64)
            finite = np.isfinite(mat_vec) & np.isfinite(bv_vec)
            if np.any(finite):
                mean_abs_values.append(float(np.mean(np.abs(bv_vec[finite] - mat_vec[finite]))))
            mat_keep_ch = mat_keep[:, mat_idx]
            bv_keep_ch, _ = _keep_mask_from_trial_means(
                bv_vec[:, np.newaxis],
                method=BASELINE_OUTLIER_METHOD,
            )
            bv_keep_vec = bv_keep_ch[:, 0]
            mism = int(np.sum(mat_keep_ch != bv_keep_vec))
            mismatched_pairs += mism
            if mism == 0:
                exact_channels += 1
        if mean_abs_values:
            pair_agree = 100.0 * (
                1.0 - mismatched_pairs / float(n_trials * len(channel_mapping))
            )
            results.append((
                shift,
                BV_EVENT_SAMPLE_SHIFT_SAMPLES + shift,
                float(np.mean(mean_abs_values)),
                pair_agree,
                exact_channels,
                mismatched_pairs,
            ))

    for extra_shift, total_shift, mean_abs, pair_agree, exact_channels, mismatched_pairs in sorted(
        results,
        key=lambda row: (-row[3], row[2]),
    ):
        print(
            f"  {extra_shift:+7d}  {total_shift:+7d}  {mean_abs:10.6f}  {pair_agree:9.4f}%  "
            f"{exact_channels:4d}/{len(channel_mapping):<4d}  {mismatched_pairs:11d}"
        )


def print_baseline_diagnostics(
    mat_epochs: np.ndarray,
    bv_epochs: np.ndarray,
    mat_time: np.ndarray,
    bv_time: np.ndarray,
    mat_channels: list[str],
    bv_channels: list[str],
    channel_mapping: dict[int, int],
    *,
    top_n: int = BASELINE_DIAGNOSTIC_TOP_N,
) -> None:
    """Print baseline-mean diagnostics relevant for b2 rmoutliers behavior."""
    if not channel_mapping:
        return

    (
        mat_trial_means,
        bv_trial_means,
        (mat_i0, mat_i1),
        (bv_i0, bv_i1),
    ) = _baseline_trial_means(mat_epochs, bv_epochs, mat_time, bv_time)
    mat_keep, _ = _keep_mask_from_trial_means(mat_trial_means, method=BASELINE_OUTLIER_METHOD)
    bv_keep, _ = _keep_mask_from_trial_means(bv_trial_means, method=BASELINE_OUTLIER_METHOD)

    rows: list[dict[str, object]] = []
    pair_matches = 0
    pair_total = 0
    for mat_idx, bv_idx in sorted(channel_mapping.items()):
        mat_vec = np.asarray(mat_trial_means[:, mat_idx], dtype=np.float64)
        bv_vec = np.asarray(bv_trial_means[:, bv_idx], dtype=np.float64)
        finite = np.isfinite(mat_vec) & np.isfinite(bv_vec)
        if int(np.sum(finite)) < 3:
            continue
        a = mat_vec[finite]
        b = bv_vec[finite]
        mean_abs = float(np.mean(np.abs(b - a)))
        rms_diff = float(np.sqrt(np.mean((b - a) ** 2)))
        a_mean = float(np.mean(a))
        b_mean = float(np.mean(b))
        a_centered = a - a_mean
        b_centered = b - b_mean
        denom_corr = float(np.sqrt(np.sum(a_centered * a_centered) * np.sum(b_centered * b_centered)))
        corr = float(np.sum(a_centered * b_centered) / denom_corr) if denom_corr > 0.0 else np.nan
        denom_slope = float(np.sum(a_centered * a_centered))
        slope = float(np.sum(a_centered * b_centered) / denom_slope) if denom_slope > 0.0 else np.nan
        mat_std = float(np.std(a, ddof=1)) if a.size > 1 else np.nan
        bv_std = float(np.std(b, ddof=1)) if b.size > 1 else np.nan
        std_ratio = bv_std / mat_std if (np.isfinite(mat_std) and mat_std > 0.0 and np.isfinite(bv_std)) else np.nan

        mat_keep_vec = np.asarray(mat_keep[:, mat_idx], dtype=bool)
        bv_keep_vec = np.asarray(bv_keep[:, bv_idx], dtype=bool)
        pair_total += len(mat_keep_vec)
        pair_matches += int(np.sum(mat_keep_vec == bv_keep_vec))
        only_mat = np.flatnonzero(~mat_keep_vec & bv_keep_vec)
        only_bv = np.flatnonzero(mat_keep_vec & ~bv_keep_vec)
        rows.append(
            {
                "mat_idx": int(mat_idx),
                "bv_idx": int(bv_idx),
                "mat_name": str(mat_channels[mat_idx]),
                "bv_name": str(bv_channels[bv_idx]),
                "mean_abs": mean_abs,
                "rms_diff": rms_diff,
                "corr": corr,
                "slope": slope,
                "std_ratio": std_ratio,
                "mismatched_trials": int(len(only_mat) + len(only_bv)),
                "only_mat": only_mat,
                "only_bv": only_bv,
            }
        )

    if not rows:
        print("\nBaseline diagnostic — no finite matched channels to rank.")
        return

    exact_channels = int(sum(int(row["mismatched_trials"]) == 0 for row in rows))
    print(
        "\nBaseline diagnostic — raw b1 trial means driving b2 rmoutliers "
        f"[{BASELINE_TMIN_S},{BASELINE_TMAX_S}]s "
        f"(MAT idx {mat_i0}:{mat_i1}, BV idx {bv_i0}:{bv_i1})"
    )
    print(
        f"  pair agreement={100.0 * pair_matches / float(pair_total):.4f}%  "
        f"exact channels={exact_channels}/{len(rows)}  "
        f"mismatched pairs={pair_total - pair_matches}"
    )

    def _print_table(title: str, *, sort_key: object, reverse: bool) -> list[dict[str, object]]:
        print(f"\n{title}")
        print(
            f"  {'MAT':>5}  {'BV':>5}  {'MATLAB bipole':<16}  {'BV':<10}  "
            f"{'mism':>5}  {'mean|ΔBL|':>10}  {'RMSΔBL':>10}  {'corr':>8}  {'slope':>8}  {'σBV/σMAT':>9}"
        )
        ordered = sorted(rows, key=sort_key, reverse=reverse)
        for row in ordered[:top_n]:
            print(
                f"  [{int(row['mat_idx']):4d}]  [{int(row['bv_idx']):4d}]  "
                f"{str(row['mat_name'])[:16]:<16}  {str(row['bv_name'])[:10]:<10}  "
                f"{int(row['mismatched_trials']):5d}  {float(row['mean_abs']):10.5f}  "
                f"{float(row['rms_diff']):10.5f}  {float(row['corr']):8.4f}  "
                f"{float(row['slope']):8.4f}  {float(row['std_ratio']):9.4f}"
            )
        return ordered

    ordered_mism = _print_table(
        "Baseline diagnostic — most keep-mask mismatches",
        sort_key=lambda row: (
            int(row["mismatched_trials"]),
            float(row["mean_abs"]) if np.isfinite(float(row["mean_abs"])) else -np.inf,
        ),
        reverse=True,
    )
    _print_table(
        "Baseline diagnostic — largest baseline-mean differences",
        sort_key=lambda row: float(row["mean_abs"]) if np.isfinite(float(row["mean_abs"])) else -np.inf,
        reverse=True,
    )

    print("\n  Trial lists for the worst keep-mask mismatches:")
    shown = 0
    for row in ordered_mism:
        mism = int(row["mismatched_trials"])
        if mism <= 0 or shown >= top_n:
            break
        shown += 1
        mat_idx = int(row["mat_idx"])
        bv_idx = int(row["bv_idx"])
        only_mat = [int(t) + 1 for t in np.asarray(row["only_mat"], dtype=int)]
        only_bv = [int(t) + 1 for t in np.asarray(row["only_bv"], dtype=int)]
        print(f"  {row['mat_name']} → {row['bv_name']}:")
        print(f"    Only MATLAB baseline-outlier: {only_mat or '(none)'}")
        print(f"    Only BV baseline-outlier:     {only_bv or '(none)'}")
        interesting_trials = sorted(
            set(int(t) for t in np.asarray(row["only_mat"], dtype=int))
            | set(int(t) for t in np.asarray(row["only_bv"], dtype=int))
        )
        if interesting_trials:
            print("    Trial baseline means:")
            for trial_idx in interesting_trials[:6]:
                mat_val = float(mat_trial_means[trial_idx, mat_idx])
                bv_val = float(bv_trial_means[trial_idx, bv_idx])
                print(
                    f"      trial {trial_idx + 1:3d}: "
                    f"MAT={mat_val:10.5f}  BV={bv_val:10.5f}  Δ={bv_val - mat_val:+10.5f}"
                )


def print_channel_divergence_ranking(
    mat_epochs: np.ndarray,
    bv_epochs: np.ndarray,
    mat_time: np.ndarray,
    bv_time: np.ndarray,
    mat_channels: list[str],
    bv_channels: list[str],
    channel_mapping: dict[int, int],
    *,
    top_n: int = CHANNEL_DIAGNOSTIC_TOP_N,
) -> None:
    """Print per-channel raw-signal divergence rankings.

    The goal is to highlight channels where the b1 MATLAB output and the
    Hilbert/BrainVision output are still not identical despite the global
    epoch alignment.  Metrics are computed over all matched trials and all
    samples in the MATLAB display window.
    """
    if not channel_mapping:
        return

    mat_time_arr = np.asarray(mat_time, dtype=np.float64).ravel()
    bv_time_arr = np.asarray(bv_time, dtype=np.float64).ravel()
    bv_mask = (
        (bv_time_arr >= float(mat_time_arr[0]) - 1e-12)
        & (bv_time_arr <= float(mat_time_arr[-1]) + 1e-12)
    )
    if int(np.sum(bv_mask)) != mat_epochs.shape[1]:
        raise ValueError(
            "BV display mask does not match the MATLAB epoch length: "
            f"{int(np.sum(bv_mask))} vs {mat_epochs.shape[1]}."
        )

    rows: list[dict[str, float | int | str]] = []
    for mat_idx, bv_idx in sorted(channel_mapping.items()):
        mat_arr = np.asarray(mat_epochs[:, :, mat_idx], dtype=np.float64)
        bv_arr = np.asarray(bv_epochs[:, bv_idx, :][:, bv_mask], dtype=np.float64)
        if mat_arr.shape != bv_arr.shape:
            continue

        a = mat_arr.ravel()
        b = bv_arr.ravel()
        finite = np.isfinite(a) & np.isfinite(b)
        if int(np.sum(finite)) < 3:
            continue
        a = a[finite]
        b = b[finite]

        mean_abs = float(np.mean(np.abs(b - a)))
        rms_diff = float(np.sqrt(np.mean((b - a) ** 2)))

        a_mean = float(np.mean(a))
        b_mean = float(np.mean(b))
        a_centered = a - a_mean
        b_centered = b - b_mean
        denom_corr = float(np.sqrt(np.sum(a_centered * a_centered) * np.sum(b_centered * b_centered)))
        corr = float(np.sum(a_centered * b_centered) / denom_corr) if denom_corr > 0.0 else np.nan

        denom_slope = float(np.sum(a_centered * a_centered))
        slope = float(np.sum(a_centered * b_centered) / denom_slope) if denom_slope > 0.0 else np.nan
        intercept = float(b_mean - slope * a_mean) if np.isfinite(slope) else np.nan

        mat_std = float(np.std(a, ddof=1)) if a.size > 1 else np.nan
        bv_std = float(np.std(b, ddof=1)) if b.size > 1 else np.nan
        std_ratio = bv_std / mat_std if (np.isfinite(mat_std) and mat_std > 0.0 and np.isfinite(bv_std)) else np.nan

        trial_corrs: list[float] = []
        trial_slopes: list[float] = []
        for trial_i in range(mat_arr.shape[0]):
            at = mat_arr[trial_i]
            bt = bv_arr[trial_i]
            finite_t = np.isfinite(at) & np.isfinite(bt)
            if int(np.sum(finite_t)) < 3:
                continue
            at = at[finite_t]
            bt = bt[finite_t]
            atc = at - np.mean(at)
            btc = bt - np.mean(bt)
            denom_t = float(np.sqrt(np.sum(atc * atc) * np.sum(btc * btc)))
            if denom_t > 0.0:
                trial_corrs.append(float(np.sum(atc * btc) / denom_t))
            denom_st = float(np.sum(atc * atc))
            if denom_st > 0.0:
                trial_slopes.append(float(np.sum(atc * btc) / denom_st))

        rows.append(
            {
                "mat_idx": int(mat_idx),
                "bv_idx": int(bv_idx),
                "mat_name": str(mat_channels[mat_idx]),
                "bv_name": str(bv_channels[bv_idx]),
                "mean_abs": mean_abs,
                "rms_diff": rms_diff,
                "corr": corr,
                "slope": slope,
                "intercept": intercept,
                "std_ratio": std_ratio,
                "median_trial_corr": float(np.nanmedian(trial_corrs)) if trial_corrs else np.nan,
                "median_trial_slope": float(np.nanmedian(trial_slopes)) if trial_slopes else np.nan,
            }
        )

    if not rows:
        print("\nRaw-channel diagnostic — no finite matched channels to rank.")
        return

    def _print_table(title: str, *, sort_key: object, reverse: bool) -> None:
        print(f"\n{title}")
        print(
            f"  {'MAT':>5}  {'BV':>5}  {'MATLAB bipole':<16}  {'BV':<10}  "
            f"{'mean|Δ|':>9}  {'RMSΔ':>9}  {'corr':>8}  {'slope':>8}  {'σBV/σMAT':>9}"
        )
        ordered = sorted(
            rows,
            key=sort_key,
            reverse=reverse,
        )
        for row in ordered[:top_n]:
            print(
                f"  [{int(row['mat_idx']):4d}]  [{int(row['bv_idx']):4d}]  "
                f"{str(row['mat_name'])[:16]:<16}  {str(row['bv_name'])[:10]:<10}  "
                f"{float(row['mean_abs']):9.5f}  {float(row['rms_diff']):9.5f}  "
                f"{float(row['corr']):8.4f}  {float(row['slope']):8.4f}  {float(row['std_ratio']):9.4f}"
            )

    _print_table(
        "Raw-channel diagnostic — largest amplitude mismatches (|slope-1|)",
        sort_key=lambda row: (
            abs(float(row["slope"]) - 1.0) if np.isfinite(float(row["slope"])) else -np.inf
        ),
        reverse=True,
    )
    _print_table(
        "Raw-channel diagnostic — lowest waveform similarity (corr)",
        sort_key=lambda row: float(row["corr"]) if np.isfinite(float(row["corr"])) else np.inf,
        reverse=False,
    )
    _print_table(
        "Raw-channel diagnostic — largest absolute differences (mean|Δ|)",
        sort_key=lambda row: float(row["mean_abs"]) if np.isfinite(float(row["mean_abs"])) else -np.inf,
        reverse=True,
    )

    abs_slope_dev = np.array(
        [abs(float(row["slope"]) - 1.0) for row in rows if np.isfinite(float(row["slope"]))],
        dtype=np.float64,
    )
    corrs_all = np.array(
        [float(row["corr"]) for row in rows if np.isfinite(float(row["corr"]))],
        dtype=np.float64,
    )
    if abs_slope_dev.size and corrs_all.size:
        print(
            "\n  Summary:"
            f" median |slope-1|={float(np.median(abs_slope_dev)):.4f},"
            f" max |slope-1|={float(np.max(abs_slope_dev)):.4f},"
            f" median corr={float(np.median(corrs_all)):.4f},"
            f" min corr={float(np.min(corrs_all)):.4f}"
        )


# ===========================================================================
# Interactive viewer
# ===========================================================================

class RawComparisonViewer:
    """Simple trial-by-trial viewer: MATLAB b1 vs BrainVision (raw amplitude).

    Controls:
        ← → arrows / buttons  : step through trials
        ↑ ↓ arrows / buttons  : step through matched channels
        "Norm" button          : toggle per-trace std normalization for shape comparison
    """

    def __init__(
        self,
        mat_epochs: np.ndarray,      # (N_trials, N_samples, N_bipoles)
        mat_time: np.ndarray,        # (N_samples,)
        mat_channels: list[str],
        mat_sfreq: float,
        bv_epochs: np.ndarray,       # (N_trials, N_ch, N_samples)
        bv_time: np.ndarray,         # (N_samples,)
        bv_channels: list[str],
        bv_sfreq: float,
        channel_mapping: dict[int, int],
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

        self.mat_indices = sorted(channel_mapping)
        self.bv_indices = [channel_mapping[mi] for mi in self.mat_indices]

        self.n_channels = len(self.mat_indices)
        self.n_trials = min(mat_epochs.shape[0], bv_epochs.shape[0])

        self._ch = 0
        self._trial = 0
        self._normalize = False

        self._build_figure()

    # ------------------------------------------------------------------
    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(15, 7))
        self.fig.suptitle(
            "MATLAB b1 vs BrainVision — Raw HGA (no corrections, no z-score)",
            fontsize=13, fontweight="bold",
        )

        self.ax = self.fig.add_axes([0.07, 0.30, 0.90, 0.60])

        # Channel slider
        ax_sl_ch = self.fig.add_axes([0.07, 0.19, 0.50, 0.03])
        self.sl_ch = Slider(
            ax_sl_ch, "Channel", 0, self.n_channels - 1,
            valinit=0, valstep=1, color="steelblue",
        )
        self.sl_ch.on_changed(self._on_ch_changed)

        # Trial slider
        ax_sl_tr = self.fig.add_axes([0.07, 0.12, 0.50, 0.03])
        self.sl_tr = Slider(
            ax_sl_tr, "Trial", 0, self.n_trials - 1,
            valinit=0, valstep=1, color="darkorange",
        )
        self.sl_tr.on_changed(self._on_trial_changed)

        # Channel prev/next buttons
        ax_prev_ch = self.fig.add_axes([0.63, 0.175, 0.07, 0.04])
        ax_next_ch = self.fig.add_axes([0.72, 0.175, 0.07, 0.04])
        self.btn_prev_ch = Button(ax_prev_ch, "◀ Ch",    color="lightblue")
        self.btn_next_ch = Button(ax_next_ch, "Ch ▶",    color="lightblue")
        self.btn_prev_ch.on_clicked(lambda _e: self._step(channel_delta=-1))
        self.btn_next_ch.on_clicked(lambda _e: self._step(channel_delta=+1))

        # Trial prev/next buttons
        ax_prev_tr = self.fig.add_axes([0.63, 0.105, 0.07, 0.04])
        ax_next_tr = self.fig.add_axes([0.72, 0.105, 0.07, 0.04])
        self.btn_prev_tr = Button(ax_prev_tr, "◀ Trial", color="moccasin")
        self.btn_next_tr = Button(ax_next_tr, "Trial ▶", color="moccasin")
        self.btn_prev_tr.on_clicked(lambda _e: self._step(trial_delta=-1))
        self.btn_next_tr.on_clicked(lambda _e: self._step(trial_delta=+1))

        # Normalize toggle button
        ax_norm = self.fig.add_axes([0.82, 0.14, 0.10, 0.04])
        self.btn_norm = Button(ax_norm, "Norm: OFF", color="lightgray")
        self.btn_norm.on_clicked(self._toggle_norm)

        # Info text box
        self.info_text = self.fig.text(
            0.82, 0.36, "",
            fontsize=9, verticalalignment="center",
            bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._render()

    # ------------------------------------------------------------------
    def _toggle_norm(self, _event: object) -> None:
        self._normalize = not self._normalize
        label = "Norm: ON" if self._normalize else "Norm: OFF"
        color = "lightgreen" if self._normalize else "lightgray"
        self.btn_norm.label.set_text(label)
        self.btn_norm.ax.set_facecolor(color)
        self._render()

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
        elif key == "n":
            self._toggle_norm(None)

    # ------------------------------------------------------------------
    def _render(self) -> None:
        self.ax.clear()

        mat_ch_i = self.mat_indices[self._ch]
        bv_ch_i  = self.bv_indices[self._ch]
        mat_ch_name = self.mat_channels[mat_ch_i]
        bv_ch_name  = self.bv_channels[bv_ch_i]
        trial_i = self._trial

        # MATLAB: (N_trials, N_samples, N_bipoles)
        mat_trace = self.mat_epochs[trial_i, :, mat_ch_i].copy().astype(np.float64)

        # BV: (N_trials, N_ch, N_samples) — trim to MATLAB display window
        bv_display_mask = (self.bv_time >= MAT_TMIN_S) & (self.bv_time <= MAT_TMAX_S)
        bv_time_display = self.bv_time[bv_display_mask] + BV_DISPLAY_OFFSET_S
        bv_trace = self.bv_epochs[trial_i, bv_ch_i, bv_display_mask].copy().astype(np.float64)

        mat_std = float(np.nanstd(mat_trace))
        bv_std  = float(np.nanstd(bv_trace))

        if self._normalize:
            if mat_std > 0:
                mat_trace = mat_trace / mat_std
            if bv_std > 0:
                bv_trace = bv_trace / bv_std
            ylabel = "Amplitude / std  (normalized per trial)"
        else:
            ylabel = "Amplitude (raw)"

        self.ax.plot(
            self.mat_time, mat_trace,
            label=f"MATLAB b1 ({self.mat_sfreq:.0f} Hz)",
            color="steelblue", linewidth=4.5, alpha=0.85,
        )
        self.ax.plot(
            bv_time_display, bv_trace,
            label=f"BrainVision ({self.bv_sfreq:.0f} Hz)",
            color="darkorange", linewidth=1.5, alpha=0.85,
        )
        self.ax.axvline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)

        self.ax.set_xlabel("Time relative to anchor onset (s)")
        self.ax.set_ylabel(ylabel)
        title = (
            f"Trial {trial_i + 1} / {self.n_trials}  —  "
            f"MATLAB: {mat_ch_name}  →  BV: {bv_ch_name}"
        )
        self.ax.set_title(title, fontsize=11)
        self.ax.legend(loc="upper right", fontsize=9)
        self.ax.grid(True, alpha=0.3)

        # Per-trial cross-correlation (shape similarity independent of scale)
        n_common = min(len(mat_trace), len(bv_trace))
        mt = mat_trace[:n_common]
        bt = bv_trace[:n_common]
        corr = np.nan
        scale = np.nan
        if np.isfinite(mt).all() and np.isfinite(bt).all() and n_common > 1:
            mt_z = mt - np.nanmean(mt)
            bt_z = bt - np.nanmean(bt)
            denom = np.sqrt(np.sum(mt_z**2) * np.sum(bt_z**2))
            if denom > 0:
                corr = float(np.sum(mt_z * bt_z) / denom)
            # Linear scale factor: bv ≈ scale × mat (least-squares)
            if np.sum(mt_z**2) > 0:
                scale = float(np.sum(mt_z * bt_z) / np.sum(mt_z**2))

        # Cross-trial median correlation for this channel
        mat_all = self.mat_epochs[:self.n_trials, :, mat_ch_i]
        bv_all  = self.bv_epochs[:self.n_trials, bv_ch_i, :][:, bv_display_mask]
        n_c = min(mat_all.shape[1], bv_all.shape[1])
        mat_all = mat_all[:, :n_c]
        bv_all  = bv_all[:, :n_c]
        corrs = []
        for t in range(self.n_trials):
            mt2 = mat_all[t] - np.nanmean(mat_all[t])
            bt2 = bv_all[t]  - np.nanmean(bv_all[t])
            d2 = np.sqrt(np.sum(mt2**2) * np.sum(bt2**2))
            if d2 > 0 and np.isfinite(mt2).all() and np.isfinite(bt2).all():
                corrs.append(float(np.sum(mt2 * bt2) / d2))
        med_corr = float(np.nanmedian(corrs)) if corrs else np.nan

        with np.errstate(divide="ignore", invalid="ignore"):
            std_ratio_per_trial = np.nanstd(bv_all, axis=1) / np.nanstd(mat_all, axis=1)
        med_ratio = float(np.nanmedian(std_ratio_per_trial))

        first = _first_contact(mat_ch_name)
        info = (
            f"Ch {self._ch + 1}/{self.n_channels}\n"
            f"MATLAB bipole: {mat_ch_name}\n"
            f"First contact: {first}\n"
            f"BV channel:    {bv_ch_name}\n"
            f"MATLAB idx:    {mat_ch_i}\n"
            f"BV idx:        {bv_ch_i}\n"
            f"Trial:         {trial_i + 1}/{self.n_trials}\n"
            f"Trial corr:    {corr:.4f}\n"
            f"BV/MAT scale:  {scale:.4f}\n"
            f"Median corr:   {med_corr:.4f}\n"
            f"Median BV/MAT std ratio: {med_ratio:.4f}"
        )
        self.info_text.set_text(info)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def show(self) -> None:
        plt.show()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load MATLAB b1
    # ------------------------------------------------------------------
    print(f"Loading MATLAB b1 file: {MATLAB_B1_PATH}")
    if not MATLAB_B1_PATH.exists():
        sys.exit(f"ERROR: MATLAB b1 path not found: {MATLAB_B1_PATH}")

    mat_data, mat_channels, mat_sfreq = load_matlab_b1(MATLAB_B1_PATH)
    print(f"  alldata shape : {mat_data.shape}  (N_trials, N_samples, N_bipoles)")
    print(f"  sfreq         : {mat_sfreq} Hz")
    print(f"  channels      : {len(mat_channels)}")
    if mat_channels:
        print(f"  first 5       : {mat_channels[:5]}")
    else:
        print("  WARNING: no channel names found — channel matching will not work.")

    n_mat_samples = mat_data.shape[1]
    mat_time = np.arange(n_mat_samples) / mat_sfreq + MAT_TMIN_S
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

    print("\nExtracting BV epochs…")
    bv_epochs, bv_time, bv_channels = epoch_brainvision(raw)
    print(f"  BV epochs shape: {bv_epochs.shape}  (N_trials, N_channels, N_samples)")
    print(f"  time range     : {bv_time[0]:.3f} s → {bv_time[-1]:.3f} s")

    # ------------------------------------------------------------------
    # 3. Channel matching
    # ------------------------------------------------------------------
    print("\nMatching channels…")
    channel_mapping = match_channels(mat_channels, bv_channels)
    n_matched = len(channel_mapping)
    print(f"  Matched {n_matched} / {len(mat_channels)} MATLAB channels to BV channels.")
    if n_matched == 0:
        sys.exit("ERROR: No channels matched — check channel names in both files.")

    # Align trial counts
    n_trials_mat = mat_data.shape[0]
    n_trials_bv  = bv_epochs.shape[0]
    if n_trials_mat != n_trials_bv:
        print(
            f"  WARNING: MATLAB has {n_trials_mat} trials, BV has {n_trials_bv}. "
            f"Using first {min(n_trials_mat, n_trials_bv)}."
        )
    n_trials = min(n_trials_mat, n_trials_bv)
    mat_data  = mat_data[:n_trials]
    bv_epochs = bv_epochs[:n_trials]
    print_sample_shift_diagnostics(
        mat_data,
        raw,
        channel_mapping,
    )
    print_baseline_shift_diagnostics(
        mat_data,
        raw,
        mat_time,
        mat_channels,
        bv_channels,
        channel_mapping,
    )
    print_baseline_diagnostics(
        mat_data,
        bv_epochs,
        mat_time,
        bv_time,
        mat_channels,
        bv_channels,
        channel_mapping,
    )
    print_channel_divergence_ranking(
        mat_data,
        bv_epochs,
        mat_time,
        bv_time,
        mat_channels,
        bv_channels,
        channel_mapping,
    )

    # ------------------------------------------------------------------
    # 4. Launch viewer
    # ------------------------------------------------------------------
    print("\nLaunching raw comparison viewer…")
    print("  Keyboard: ← → trials,  ↑ ↓ channels,  n = toggle normalization")

    viewer = RawComparisonViewer(
        mat_epochs=mat_data,
        mat_time=mat_time,
        mat_channels=mat_channels,
        mat_sfreq=mat_sfreq,
        bv_epochs=bv_epochs,
        bv_time=bv_time,
        bv_channels=bv_channels,
        bv_sfreq=bv_sfreq,
        channel_mapping=channel_mapping,
    )
    viewer.show()


if __name__ == "__main__":
    main()

