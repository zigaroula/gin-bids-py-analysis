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
    reject_channels_by_trial_max_spread,
    reject_channels_by_trial_mean_spread,
)
from gin_bids_py_analysis.processing.utils.statistics import zscore_activity_by_baseline

# ---------------------------------------------------------------------------
# File paths  (edit if needed)
# ---------------------------------------------------------------------------

MATLAB_PATH = Path(
    r"D:/data_clarissa/transfer_12408284_files_bbf26a92"
    r"/edGRE_2021_AICb_MD_CHOICE_Partie1_BPF_f50f150_sf100_sm250_onset.mat"
)

# MNE needs the .vhdr header — derive it from the .eeg path.
BV_EEG_PATH = Path(
    r"D:/data_clarissa/valuation/bids/derivatives/hilbert"
    r"/sub-GRE2021AICb/ieeg/sub-GRE2021AICb_task-MDCHOICE_desc-bgasm250_ieeg.eeg"
)
BV_VHDR_PATH = BV_EEG_PATH.with_suffix(".vhdr")

# Epoching window to apply to the BrainVision file.
# BV is epoched wider than the MATLAB window so the pre-anchor baseline is
# available for z-scoring, even though MATLAB data starts at 0 s.
MAT_TMIN_S: float = -0.5   # MATLAB epoch start relative to anchor
MAT_TMAX_S: float = 9.5  # MATLAB epoch end relative to anchor
TMIN_S: float = -0.5   # BV epoching start (wider, covers baseline)
TMAX_S: float = 9.5   # BV epoching end
BV_DISPLAY_OFFSET_S: float = 0.0  # shift BV time axis by this amount for display
ANCHOR_CODES: set[str] = {"11", "12"}
EXPERIMENT_START_CODE: str = "5"

# Global z-score baseline applied to BrainVision epochs.
ZSCORE_BV: bool = True
BASELINE_TMIN_S: float = -0.25
BASELINE_TMAX_S: float = -0.05
# MATLAB f_baseline_normalization uses rmoutliers() with default = median/MAD criterion,
# applied on per-trial baseline means.  Set True to replicate this step exactly.
BASELINE_REMOVE_OUTLIERS: bool = True
# Pre-z-score NaN masking using mean±3σ and max±3σ over the full epoch.
# NOT present in f_baseline_normalization — only enable if MATLAB b2 does a separate
# pre-cleaning step before calling f_baseline_normalization.
# Current evidence: keeping this False (no PRECLEAN) + BASELINE_REMOVE_OUTLIERS=True
# is the closest match to MATLAB's f_baseline_normalization alone.
PRECLEAN_BV: bool = True
PRECLEAN_THRESHOLD: float = 3.0
# Replicate MATLAB's removebadchannelsSd step: NaN-ise entire channels whose
# across-trial spread of mean HGA is an outlier (threshold=1σ, 'mean' method).
# Applied after PRECLEAN, before z-scoring, matching b2_BPF_apply_options.m.
REJECT_BAD_CHANNELS_SD: bool = True
REJECT_BAD_CHANNELS_SD_THRESHOLD: float = 1.0
# Replicate MATLAB b2_BPF_apply_options_R1 opts.removenegratings: NaN-ise trials
# where the behavioral rating is negative (rating < MIN_RATING) across all channels.
# Requires a beh TSV with a 'rating' column in EEG trial order.
REMOVE_NEGATIVE_RATINGS: bool = True
MIN_RATING: float = 0.0
BEH_TSV_PATH = Path(
    r"D:/data_clarissa/valuation/bids"
    r"/sub-GRE2021AICb/beh/sub-GRE2021AICb_task-MDCHOICE_beh.tsv"
)


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


def _load_hdf5_matlab(path: Path) -> tuple[np.ndarray, list[str], float]:
    """Load alldata, channel names, and sfreq from a MATLAB v7.3 (HDF5) file."""
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


def load_matlab(path: Path) -> tuple[np.ndarray, list[str], float]:
    """Load (alldata, channel_names, sfreq) from a MATLAB file (v5 or v7.3).

    alldata shape: (N_trials, N_samples, N_bipoles)
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

    if hdr is not None:
        channels = _extract_channels_scipy(hdr)
        sfreq = _extract_sfreq_scipy(hdr, fallback=sfreq)

    return alldata, channels, sfreq


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
        np.round(np.array(anchor_onsets) * sfreq).astype(np.int64),
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
        bv_ref_std: np.ndarray | None = None,  # (N_bv_channels,) z-score σ used per channel
        bv_n_clean: np.ndarray | None = None,  # (N_bv_channels,) # clean trials per channel
        bv_diag: dict | None = None,  # raw cleaning test values for always-on display
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
        self.bv_ref_std = bv_ref_std  # may be None if ZSCORE_BV is False
        self.bv_n_clean = bv_n_clean
        self.bv_diag = bv_diag or {}

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

        # ---- Suppression status text (bottom of figure) ----
        self.status_text = self.fig.text(
            0.07, 0.03, "",
            fontsize=8, verticalalignment="bottom",
            color="darkred",
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
        bv_display_mask = (self.bv_time >= MAT_TMIN_S) & (self.bv_time <= MAT_TMAX_S)
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
            color="steelblue", linewidth=1.5, alpha=0.85,
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

        ref_std_str = "n/a"
        n_clean_str = "n/a"
        if self.bv_ref_std is not None:
            ref_std_str = f"{self.bv_ref_std[bv_ch_i]:.4f}"
        if self.bv_n_clean is not None:
            n_clean_str = f"{int(self.bv_n_clean[bv_ch_i])}/{self.n_trials}"

        info = (
            f"Ch {self._ch + 1}/{self.n_channels}\n"
            f"MATLAB bipole: {mat_ch_name}\n"
            f"First contact: {first}\n"
            f"BV channel:    {bv_ch_name}\n"
            f"MATLAB idx:    {mat_ch_i}\n"
            f"BV idx:        {bv_ch_i}\n"
            f"Trial:         {trial_i + 1}/{self.n_trials}\n"
            f"BV zscore σ:   {ref_std_str}  (n={n_clean_str})\n"
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

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def show(self) -> None:
        plt.show()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load MATLAB
    # ------------------------------------------------------------------
    print(f"Loading MATLAB file: {MATLAB_PATH}")
    if not MATLAB_PATH.exists():
        sys.exit(f"ERROR: MATLAB file not found: {MATLAB_PATH}")

    mat_data, mat_channels, mat_sfreq = load_matlab(MATLAB_PATH)
    print(f"  alldata shape : {mat_data.shape}  (N_trials, N_samples, N_bipoles)")
    print(f"  sfreq         : {mat_sfreq} Hz")
    print(f"  channels      : {len(mat_channels)}")
    if mat_channels:
        print(f"  first 5       : {mat_channels[:5]}")
    else:
        print("  WARNING: no channel names found in hdr.label")

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

    print("\nExtracting BV epochs...")
    bv_epochs, bv_time, bv_channels = epoch_brainvision(raw)
    print(f"  BV epochs shape: {bv_epochs.shape}  (N_trials, N_channels, N_samples)")
    print(f"  time range     : {bv_time[0]:.3f} s → {bv_time[-1]:.3f} s")

    # Diagnostic arrays stored for viewer display (raw test values, always shown).
    bv_diag: dict = {}

    if PRECLEAN_BV:
        print(f"  Pre-cleaning: NaN-masking outlier trials per channel (threshold={PRECLEAN_THRESHOLD}σ)…")
        _ep_f64 = np.asarray(bv_epochs, dtype=np.float64)
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
        mask_mean = detect_outlier_trial_channel_pairs_by_mean(bv_epochs, PRECLEAN_THRESHOLD)
        mask_max  = detect_outlier_trial_channel_pairs_by_max(bv_epochs, PRECLEAN_THRESHOLD)
        combined_mask = mask_mean | mask_max
        n_masked = int(np.sum(combined_mask))
        bv_epochs = apply_trial_nan_mask(bv_epochs, combined_mask)
        print(f"    Masked {n_masked} (trial, channel) pairs "
              f"({int(np.sum(mask_mean))} by mean, {int(np.sum(mask_max))} by max).")

    if REJECT_BAD_CHANNELS_SD:
        print(f"  Rejecting bad channels by trial-mean spread (threshold={REJECT_BAD_CHANNELS_SD_THRESHOLD}σ)…")
        _tr_means_bc = np.nanmean(bv_epochs, axis=2)      # (n_trials, n_ch)
        _ch_spread_m = np.nanstd(_tr_means_bc, axis=0, ddof=1)
        _spm_mean = float(np.nanmean(_ch_spread_m[np.isfinite(_ch_spread_m)]))
        _spm_std  = float(np.nanstd(_ch_spread_m[np.isfinite(_ch_spread_m)], ddof=1))
        _tr_maxes_bc = np.nanmax(np.abs(bv_epochs), axis=2)
        _ch_spread_x = np.nanstd(_tr_maxes_bc, axis=0, ddof=1)
        _spx_mean = float(np.nanmean(_ch_spread_x[np.isfinite(_ch_spread_x)]))
        _spx_std  = float(np.nanstd(_ch_spread_x[np.isfinite(_ch_spread_x)], ddof=1))
        bv_diag.update({
            "ch_spread_m": _ch_spread_m, "spm_mean": _spm_mean, "spm_std": _spm_std,
            "ch_spread_x": _ch_spread_x, "spx_mean": _spx_mean, "spx_std": _spx_std,
        })
        bad_ch_mean = reject_channels_by_trial_mean_spread(bv_epochs, REJECT_BAD_CHANNELS_SD_THRESHOLD)
        bad_ch_max  = reject_channels_by_trial_max_spread(bv_epochs, REJECT_BAD_CHANNELS_SD_THRESHOLD)
        bad_ch_mask = bad_ch_mean | bad_ch_max
        n_bad_ch = int(np.sum(bad_ch_mask))
        if n_bad_ch:
            # NaN entire channel across all trials (matching MATLAB alldata(:,:,chans) = NaN)
            bv_epochs[:, bad_ch_mask, :] = np.nan
            bad_names = [bv_channels[i] for i in np.flatnonzero(bad_ch_mask)]
            print(f"    NaN'd {n_bad_ch} channel(s): {bad_names}")
        else:
            print("    No bad channels detected.")

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
                n_neg = int(np.sum(_neg_mask))
                if n_neg:
                    # NaN all channels for those trials (matching MATLAB alldata(trialsToRemove,:,:) = NaN)
                    bv_epochs[_neg_mask, :, :] = np.nan
                    print(f"    NaN'd {n_neg} trial(s) with rating < {MIN_RATING}.")
                else:
                    print(f"    No negative-rating trials found.")
                bv_diag["ratings"] = _ratings[:n_bv]

    # Compute z-score reference stats per channel (on pre-zscore data; NaN from PRECLEAN
    # already propagate).  Stored for display in the viewer infobox.
    bv_ref_std: np.ndarray | None = None
    bv_n_clean: np.ndarray | None = None

    if ZSCORE_BV:
        print(
            f"  Applying global z-score (baseline {BASELINE_TMIN_S} – {BASELINE_TMAX_S} s, "
            f"remove_outliers={BASELINE_REMOVE_OUTLIERS})…"
        )
        # Replicate MATLAB's exclusive upper endpoint: tmax_idx = tmax_idx - 1
        # i.e. the sample at exactly BASELINE_TMAX_S is excluded.
        baseline_tmax_exclusive = BASELINE_TMAX_S - 1.0 / bv_sfreq

        # Capture the z-score reference (σ per channel) from the pre-zscore epochs.
        # trial_baseline_means shape: (N_trials, N_ch) — NaN for PRECLEAN-masked trials.
        _bl_mask = (bv_time >= BASELINE_TMIN_S) & (bv_time < baseline_tmax_exclusive)
        _trial_bl_means = np.nanmean(bv_epochs[:, :, _bl_mask], axis=2)  # (N_trials, N_ch)
        bv_n_clean = np.sum(np.isfinite(_trial_bl_means), axis=0).astype(float)  # (N_ch,)
        bv_ref_std = np.nanstd(_trial_bl_means, axis=0, ddof=1)  # (N_ch,) — approx (no MAD step)

        # zscore_activity_by_baseline expects (n_trials, n_features, n_times).
        # Pass all trials as condition A and an empty array as condition B so
        # the global scope pools baseline means from all trials (single group).
        n_ch, n_samp = bv_epochs.shape[1], bv_epochs.shape[2]
        empty_b = np.empty((0, n_ch, n_samp), dtype=np.float64)
        bv_epochs_z, _ = zscore_activity_by_baseline(
            bv_epochs,
            empty_b,
            bv_time,
            baseline_tmin_s=BASELINE_TMIN_S,
            baseline_tmax_s=baseline_tmax_exclusive,
            baseline_scope="global",
            remove_outlier_trial_means=BASELINE_REMOVE_OUTLIERS,
        )
        bv_epochs = bv_epochs_z
        print(f"  Z-scoring done (effective tmax={baseline_tmax_exclusive:.4f} s, exclusive).")

    # ------------------------------------------------------------------
    # 3. Channel matching
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
        bv_n_clean=bv_n_clean,
        bv_diag=bv_diag,
    )
    viewer.show()


if __name__ == "__main__":
    main()
