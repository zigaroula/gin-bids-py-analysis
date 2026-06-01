#!/usr/bin/env python3
"""
Compare MATLAB b1_TF output with Python time-frequency derivatives.

MATLAB reference:
    C:\\GRE\\dev\\clarissa\\seeg\\b1_TF_data\\*_TFR.mat
    C:\\GRE\\dev\\clarissa\\seeg\\b1_TF_data\\*_TFR.wya
    C:\\GRE\\dev\\clarissa\\seeg\\b1_TF_data\\*_TFR_baseline.mat

Python reference:
    derivatives/time_frequency/sub-*/ieeg/*_desc-tf_tfr.h5

Usage:
    python scripts/debug/compare_b1_TF.py
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import sys
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
import scipy.io

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
for path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bidsforge.processing.utils.hdf5 import decode_str_array  # noqa: E402
from compare_subject_config import (  # noqa: E402
    BIDS_ROOT,
    BIDS_SUBJECT,
    MATLAB_B1_TF_BASELINE_PATH,
    MATLAB_B1_TF_HEADER_PATH,
    MATLAB_B1_TF_WYA_PATH,
    MATLAB_ROOT,
    PYTHON_TIME_FREQUENCY_PATH,
    SUBJECT,
)

SHOW_BASELINE_CORRECTED: bool = True
SUMMARY_MODES: tuple[bool, ...] = (False, True)
INITIAL_FREQUENCY_HZ: float = 50.0
DIAGNOSTIC_TOP_N: int = 12
SUMMARY_SAMPLE_TRIALS: int = 24
SUMMARY_SAMPLE_CHANNELS: int = 24


@dataclass
class MatlabB1TF:
    path_header: Path
    path_wya: Path
    path_baseline: Path
    power: np.memmap
    baseline: np.ndarray
    channels: list[str]
    frequency_hz: np.ndarray
    time_s: np.ndarray
    dims: tuple[int, int, int, int]
    baseline_is_usable: bool

    @classmethod
    def load(
        cls,
        path_header: Path,
        path_wya: Path,
        path_baseline: Path,
    ) -> "MatlabB1TF":
        mat = scipy.io.loadmat(str(path_header), squeeze_me=True, struct_as_record=False)
        hdr = mat["hdr"]
        dims = tuple(int(v) for v in np.asarray(hdr.dimsiz, dtype=int).ravel())
        if len(dims) != 4:
            raise ValueError(f"Unexpected MATLAB hdr.dimsiz: {dims!r}")
        n_trial, n_time, n_freq, n_chan = dims
        channels = [str(ch).strip() for ch in np.asarray(hdr.chanlist, dtype=object).ravel()]
        frequency_hz = np.asarray(hdr.freqlist, dtype=np.float64).ravel()
        time_s = np.asarray(hdr.timelist, dtype=np.float64).ravel()

        expected_values = n_trial * n_time * n_freq * n_chan
        expected_bytes = expected_values * np.dtype("<f4").itemsize
        actual_bytes = path_wya.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{path_wya.name}: size mismatch. Expected {expected_bytes} bytes "
                f"from hdr.dimsiz={dims}, got {actual_bytes}."
            )

        raw = np.memmap(path_wya, dtype="<f4", mode="r", shape=dims, order="F")
        baseline_mat = scipy.io.loadmat(
            str(path_baseline),
            squeeze_me=True,
            struct_as_record=False,
        )
        baseline = np.asarray(baseline_mat["powbaseline"], dtype=np.float32)
        if baseline.shape == (n_trial, n_freq, n_chan):
            baseline = np.transpose(baseline, (0, 2, 1))
        elif baseline.shape == (n_trial, 1, n_freq, n_chan):
            baseline = np.transpose(baseline[:, 0, :, :], (0, 2, 1))
        else:
            raise ValueError(f"Unexpected MATLAB baseline shape: {baseline.shape}")
        baseline_is_usable = bool(np.any(np.isfinite(baseline)))

        return cls(
            path_header=path_header,
            path_wya=path_wya,
            path_baseline=path_baseline,
            power=raw,
            baseline=baseline,
            channels=channels,
            frequency_hz=frequency_hz,
            time_s=time_s,
            dims=dims,
            baseline_is_usable=baseline_is_usable,
        )

    def slice_trial_channel(
        self,
        trial_i: int,
        channel_i: int,
        *,
        baseline_corrected: bool = SHOW_BASELINE_CORRECTED,
    ) -> np.ndarray:
        # MATLAB wya is [trial x time x freq x channel].
        arr = np.asarray(self.power[trial_i, :, :, channel_i], dtype=np.float32).T
        if baseline_corrected:
            if self.baseline_is_usable:
                baseline = self.baseline[trial_i, channel_i, :]
            else:
                baseline = self._compute_display_baseline(arr)
            arr = arr - baseline[:, np.newaxis]
        return arr

    def _compute_display_baseline(self, freq_time: np.ndarray) -> np.ndarray:
        mask = (self.time_s >= -1.3) & (self.time_s <= -0.7)
        if not np.any(mask):
            return np.full((freq_time.shape[0],), np.nan, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(freq_time[:, mask], axis=1).astype(np.float32)


@dataclass
class PythonTF:
    path: Path
    h5: h5py.File | None
    power: object
    baseline: np.ndarray
    channels: list[str]
    frequency_hz: np.ndarray
    time_s: np.ndarray
    power_is_baseline_corrected: bool

    @classmethod
    def load(cls, path: Path) -> "PythonTF":
        if path.suffix.lower() == ".mat":
            from bidsforge.processing.time_frequency.result_loader import (
                load_time_frequency_result,
            )

            result = load_time_frequency_result(path)
            return cls(
                path=path,
                h5=None,
                power=result.power_db,
                baseline=result.baseline_db,
                channels=result.channel_names,
                frequency_hz=result.frequency_hz,
                time_s=result.time_s,
                power_is_baseline_corrected=bool(result.metadata.get("apply_baseline", False)),
            )

        h5 = h5py.File(path, "r")
        channels = decode_str_array(np.asarray(h5["axes/channel"][:], dtype=object))
        power_is_baseline_corrected = False
        if "meta/apply_baseline" in h5:
            power_is_baseline_corrected = bool(np.asarray(h5["meta/apply_baseline"][()]).item())
        return cls(
            path=path,
            h5=h5,
            power=h5["data/power_db"],
            baseline=np.asarray(h5["data/baseline_db"][:], dtype=np.float32),
            channels=channels,
            frequency_hz=np.asarray(h5["axes/frequency_hz"][:], dtype=np.float64),
            time_s=np.asarray(h5["axes/time_s"][:], dtype=np.float64),
            power_is_baseline_corrected=power_is_baseline_corrected,
        )

    def close(self) -> None:
        if self.h5 is not None:
            self.h5.close()

    def slice_trial_channel(
        self,
        trial_i: int,
        channel_i: int,
        *,
        baseline_corrected: bool = SHOW_BASELINE_CORRECTED,
    ) -> np.ndarray:
        # Python output is [trial x channel x frequency x time].
        arr = np.asarray(self.power[trial_i, channel_i, :, :], dtype=np.float32)
        baseline = self.baseline[trial_i, channel_i, :, np.newaxis]
        if baseline_corrected and not self.power_is_baseline_corrected:
            return (arr - baseline).astype(np.float32)
        if not baseline_corrected and self.power_is_baseline_corrected:
            return (arr + baseline).astype(np.float32)
        return arr


def _resolve_python_path(path: Path) -> Path:
    if path.exists():
        return path

    search_roots = [
        path.parent,
        BIDS_ROOT / "derivatives" / "time_frequency" / f"sub-{BIDS_SUBJECT}" / "ieeg",
        BIDS_ROOT / "derivatives" / "time_frequency",
    ]
    patterns = ("*_desc-tf_tfr.h5", "*_desc-tf_tfr.mat")
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            globber = root.glob if root == path.parent else root.rglob
            candidates.extend(sorted(globber(pattern)))
        if candidates:
            break

    unique_candidates = sorted(set(candidates))
    if unique_candidates:
        chosen = unique_candidates[0]
        print(f"  Python TF default path missing, using: {chosen}")
        if len(unique_candidates) > 1:
            print("  Other Python TF candidates:")
            for candidate in unique_candidates[1:8]:
                print(f"    {candidate}")
        return chosen

    raise FileNotFoundError(
        f"Python time-frequency output not found: {path}\n"
        "Run scripts/debug/run_time_frequency.py first, or edit "
        "PYTHON_TIME_FREQUENCY_PATH in compare_subject_config.py."
    )


def _resolve_matlab_paths(
    path_header: Path,
    path_wya: Path,
    path_baseline: Path,
) -> tuple[Path, Path, Path]:
    if path_header.exists() and path_wya.exists() and path_baseline.exists():
        return path_header, path_wya, path_baseline

    root = path_header.parent
    if not root.exists():
        root = MATLAB_ROOT / "b1_TF_data"

    patterns = [
        f"de{SUBJECT}_*_TFR.mat",
        f"*{SUBJECT}*_TFR.mat",
        "*_TFR.mat",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates = sorted(root.glob(pattern))
        candidates = [p for p in candidates if not p.name.endswith("_TFR_baseline.mat")]
        if candidates:
            break

    complete: list[tuple[Path, Path, Path]] = []
    for header in candidates:
        wya = header.with_suffix(".wya")
        baseline = header.parent / f"{header.stem}_baseline.mat"
        if wya.exists() and baseline.exists():
            complete.append((header, wya, baseline))

    if complete:
        chosen = complete[0]
        print("  MATLAB TF default paths missing/incomplete, using:")
        print(f"    Header   : {chosen[0]}")
        print(f"    WYA      : {chosen[1]}")
        print(f"    Baseline : {chosen[2]}")
        if len(complete) > 1:
            print("  Other complete MATLAB TF candidates:")
            for header, _, _ in complete[1:8]:
                print(f"    {header}")
        return chosen

    existing = sorted(root.glob("*_TFR*"))
    details = "\n".join(f"    {p.name}" for p in existing[:20])
    raise FileNotFoundError(
        "MATLAB b1_TF files not found as a complete header/WYA/baseline set.\n"
        f"Expected:\n"
        f"  Header   : {path_header}  exists={path_header.exists()}\n"
        f"  WYA      : {path_wya}  exists={path_wya.exists()}\n"
        f"  Baseline : {path_baseline}  exists={path_baseline.exists()}\n"
        f"Search root: {root}\n"
        f"Files seen:\n{details if details else '    <none>'}"
    )


def _norm_channel(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _first_bipolar_contact(name: object) -> str:
    text = str(name).strip()
    match = re.match(r"^([A-Za-z]+0*\d+)", text)
    if match:
        return match.group(1)
    return text


def match_channels(mat_channels: list[str], py_channels: list[str]) -> dict[int, int]:
    py_by_exact = {_norm_channel(ch): idx for idx, ch in enumerate(py_channels)}
    mapping: dict[int, int] = {}
    used: set[int] = set()
    for mat_i, mat_ch in enumerate(mat_channels):
        keys = [
            _norm_channel(mat_ch),
            _norm_channel(_first_bipolar_contact(mat_ch)),
        ]
        for key in keys:
            py_i = py_by_exact.get(key)
            if py_i is not None and py_i not in used:
                mapping[mat_i] = py_i
                used.add(py_i)
                break
    return mapping


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_idx: list[int] = []
    target_idx: list[int] = []
    for i, value in enumerate(source):
        j = int(np.argmin(np.abs(target - value)))
        if np.isclose(value, target[j], rtol=1e-4, atol=1e-4):
            source_idx.append(i)
            target_idx.append(j)
    return np.asarray(source_idx, dtype=np.int64), np.asarray(target_idx, dtype=np.int64)


def _pair_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(finite)) < 3:
        return {
            "corr": np.nan,
            "mean_abs": np.nan,
            "rms": np.nan,
            "bias": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "std_ratio": np.nan,
            "n": float(np.sum(finite)),
        }
    av = np.asarray(a[finite], dtype=np.float64)
    bv = np.asarray(b[finite], dtype=np.float64)
    a_mean = float(np.mean(av))
    b_mean = float(np.mean(bv))
    ac = av - np.mean(av)
    bc = bv - np.mean(bv)
    denom = float(np.sqrt(np.sum(ac * ac) * np.sum(bc * bc)))
    corr = float(np.sum(ac * bc) / denom) if denom > 0 else np.nan
    denom_slope = float(np.sum(ac * ac))
    slope = float(np.sum(ac * bc) / denom_slope) if denom_slope > 0 else np.nan
    intercept = float(b_mean - slope * a_mean) if np.isfinite(slope) else np.nan
    a_std = float(np.std(av, ddof=1)) if av.size > 1 else np.nan
    b_std = float(np.std(bv, ddof=1)) if bv.size > 1 else np.nan
    std_ratio = b_std / a_std if a_std > 0 and np.isfinite(b_std) else np.nan
    diff = bv - av
    return {
        "corr": corr,
        "mean_abs": float(np.mean(np.abs(diff))),
        "rms": float(np.sqrt(np.mean(diff * diff))),
        "bias": float(np.mean(diff)),
        "slope": slope,
        "intercept": intercept,
        "std_ratio": std_ratio,
        "n": float(av.size),
    }


def _finite_percentile(values: np.ndarray, q: float, fallback: float = 1.0) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return fallback
    out = float(np.percentile(finite, q))
    return out if np.isfinite(out) else fallback


def print_summary(
    mat: MatlabB1TF,
    py: PythonTF,
    channel_mapping: dict[int, int],
    mat_time_idx: np.ndarray,
    py_time_idx: np.ndarray,
    mat_freq_idx: np.ndarray,
    py_freq_idx: np.ndarray,
    *,
    baseline_corrected: bool,
) -> None:
    mode_name = "baseline-corrected" if baseline_corrected else "raw dB"
    print(f"\nGlobal summary on a deterministic subset ({mode_name})")
    n_trials = min(mat.dims[0], int(py.power.shape[0]), SUMMARY_SAMPLE_TRIALS)
    pairs = list(channel_mapping.items())[:SUMMARY_SAMPLE_CHANNELS]
    rows: list[dict[str, object]] = []
    freq_chunks_mat: list[list[np.ndarray]] = [[] for _ in mat_freq_idx]
    freq_chunks_py: list[list[np.ndarray]] = [[] for _ in py_freq_idx]
    for mat_ch_i, py_ch_i in pairs:
        mat_chunks = []
        py_chunks = []
        for trial_i in range(n_trials):
            mat_slice = mat.slice_trial_channel(
                trial_i,
                mat_ch_i,
                baseline_corrected=baseline_corrected,
            )[
                np.ix_(mat_freq_idx, mat_time_idx)
            ]
            py_slice = py.slice_trial_channel(
                trial_i,
                py_ch_i,
                baseline_corrected=baseline_corrected,
            )[
                np.ix_(py_freq_idx, py_time_idx)
            ]
            mat_chunks.append(mat_slice.ravel())
            py_chunks.append(py_slice.ravel())
            for idx in range(len(mat_freq_idx)):
                freq_chunks_mat[idx].append(mat_slice[idx])
                freq_chunks_py[idx].append(py_slice[idx])
        stats = _pair_stats(np.concatenate(mat_chunks), np.concatenate(py_chunks))
        rows.append(
            {
                **stats,
                "mat_i": mat_ch_i,
                "py_i": py_ch_i,
                "mat_name": mat.channels[mat_ch_i],
                "py_name": py.channels[py_ch_i],
            }
        )

    if not rows:
        print("  No matched channels to summarize.")
        return
    corrs = np.asarray([float(r["corr"]) for r in rows], dtype=np.float64)
    mean_abs = np.asarray([float(r["mean_abs"]) for r in rows], dtype=np.float64)
    slopes = np.asarray([float(r["slope"]) for r in rows], dtype=np.float64)
    intercepts = np.asarray([float(r["intercept"]) for r in rows], dtype=np.float64)
    std_ratios = np.asarray([float(r["std_ratio"]) for r in rows], dtype=np.float64)
    print(
        f"  Compared {n_trials} trial(s), {len(rows)} channel(s), "
        f"{len(mat_freq_idx)} freq(s), {len(mat_time_idx)} time point(s)."
    )
    print(
        f"  Median corr={np.nanmedian(corrs):.4f}, "
        f"median mean|diff|={np.nanmedian(mean_abs):.4f} dB, "
        f"median slope={np.nanmedian(slopes):.4f}, "
        f"median intercept={np.nanmedian(intercepts):.4f}, "
        f"median std ratio={np.nanmedian(std_ratios):.4f}"
    )
    if not baseline_corrected:
        print(
            "  Acceptance raw dB: "
            f"median corr={np.nanmedian(corrs):.4f}, "
            f"median slope={np.nanmedian(slopes):.4f}, "
            f"median intercept={np.nanmedian(intercepts):.4f}"
        )
    print("\n  Lowest channel correlations")
    print(
        f"  {'MAT':>5}  {'PY':>5}  {'MATLAB':<16}  {'Python':<12}  "
        f"{'corr':>8}  {'slope':>8}  {'int':>8}  {'stdR':>8}  {'mean|d|':>9}"
    )
    ordered = sorted(
        rows,
        key=lambda row: float(row["corr"]) if np.isfinite(float(row["corr"])) else -np.inf,
    )
    for row in ordered[:DIAGNOSTIC_TOP_N]:
        print(
            f"  [{int(row['mat_i']):4d}]  [{int(row['py_i']):4d}]  "
            f"{str(row['mat_name'])[:16]:<16}  {str(row['py_name'])[:12]:<12}  "
            f"{float(row['corr']):8.4f}  {float(row['slope']):8.4f}  "
            f"{float(row['intercept']):8.4f}  {float(row['std_ratio']):8.4f}  "
            f"{float(row['mean_abs']):9.4f}"
        )

    freq_rows: list[dict[str, float]] = []
    for idx, (m_chunks, p_chunks) in enumerate(zip(freq_chunks_mat, freq_chunks_py)):
        if not m_chunks or not p_chunks:
            continue
        stats = _pair_stats(np.concatenate(m_chunks), np.concatenate(p_chunks))
        freq_rows.append(
            {
                **stats,
                "frequency_hz": float(mat.frequency_hz[mat_freq_idx[idx]]),
            }
        )
    if not freq_rows:
        return

    print("\n  Per-frequency diagnostic")
    print(
        f"  {'freq':>8}  {'corr':>8}  {'slope':>8}  {'int':>9}  "
        f"{'stdR':>8}  {'mean|d|':>9}  {'n':>8}"
    )
    for row in freq_rows:
        print(
            f"  {float(row['frequency_hz']):8.2f}  {float(row['corr']):8.4f}  "
            f"{float(row['slope']):8.4f}  {float(row['intercept']):9.4f}  "
            f"{float(row['std_ratio']):8.4f}  {float(row['mean_abs']):9.4f}  "
            f"{int(row['n']):8d}"
        )


class TFComparisonViewer:
    def __init__(
        self,
        mat: MatlabB1TF,
        py: PythonTF,
        channel_mapping: dict[int, int],
        mat_time_idx: np.ndarray,
        py_time_idx: np.ndarray,
        mat_freq_idx: np.ndarray,
        py_freq_idx: np.ndarray,
    ) -> None:
        self.mat = mat
        self.py = py
        self.mat_channels = sorted(channel_mapping)
        self.py_channels = [channel_mapping[i] for i in self.mat_channels]
        self.mat_time_idx = mat_time_idx
        self.py_time_idx = py_time_idx
        self.mat_freq_idx = mat_freq_idx
        self.py_freq_idx = py_freq_idx
        self.time = mat.time_s[mat_time_idx]
        self.freq = mat.frequency_hz[mat_freq_idx]
        self.n_trials = min(mat.dims[0], int(py.power.shape[0]))
        self._trial = 0
        self._channel = 0
        self._freq = int(np.argmin(np.abs(self.freq - INITIAL_FREQUENCY_HZ)))
        self._build()

    def _build(self) -> None:
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.suptitle("MATLAB b1_TF vs Python time_frequency", fontweight="bold")
        self.ax_mat = self.fig.add_axes([0.06, 0.45, 0.27, 0.40])
        self.ax_py = self.fig.add_axes([0.37, 0.45, 0.27, 0.40])
        self.ax_diff = self.fig.add_axes([0.68, 0.45, 0.27, 0.40])
        self.ax_trace = self.fig.add_axes([0.08, 0.18, 0.60, 0.18])
        self.info = self.fig.text(
            0.72,
            0.25,
            "",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.85},
        )

        self.sl_channel = Slider(
            self.fig.add_axes([0.08, 0.10, 0.44, 0.03]),
            "Channel",
            0,
            len(self.mat_channels) - 1,
            valinit=0,
            valstep=1,
        )
        self.sl_trial = Slider(
            self.fig.add_axes([0.08, 0.05, 0.44, 0.03]),
            "Trial",
            0,
            self.n_trials - 1,
            valinit=0,
            valstep=1,
        )
        self.sl_freq = Slider(
            self.fig.add_axes([0.58, 0.10, 0.32, 0.03]),
            "Trace freq",
            0,
            len(self.freq) - 1,
            valinit=self._freq,
            valstep=1,
        )
        self.sl_channel.on_changed(lambda v: self._set(channel=int(v)))
        self.sl_trial.on_changed(lambda v: self._set(trial=int(v)))
        self.sl_freq.on_changed(lambda v: self._set(freq=int(v)))

        self.btn_prev = Button(self.fig.add_axes([0.58, 0.045, 0.08, 0.04]), "< Trial")
        self.btn_next = Button(self.fig.add_axes([0.67, 0.045, 0.08, 0.04]), "Trial >")
        self.btn_prev.on_clicked(lambda _event: self._step(trial_delta=-1))
        self.btn_next.on_clicked(lambda _event: self._step(trial_delta=1))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._render()

    def _set(
        self,
        *,
        trial: int | None = None,
        channel: int | None = None,
        freq: int | None = None,
    ) -> None:
        if trial is not None:
            self._trial = trial
        if channel is not None:
            self._channel = channel
        if freq is not None:
            self._freq = freq
        self._render()

    def _step(self, *, trial_delta: int = 0, channel_delta: int = 0) -> None:
        self._trial = int(np.clip(self._trial + trial_delta, 0, self.n_trials - 1))
        self._channel = int(
            np.clip(self._channel + channel_delta, 0, len(self.mat_channels) - 1)
        )
        self.sl_trial.set_val(self._trial)
        self.sl_channel.set_val(self._channel)
        self._render()

    def _on_key(self, event: object) -> None:
        key = getattr(event, "key", None)
        if key == "right":
            self._step(trial_delta=1)
        elif key == "left":
            self._step(trial_delta=-1)
        elif key == "up":
            self._step(channel_delta=1)
        elif key == "down":
            self._step(channel_delta=-1)

    def _render(self) -> None:
        mat_ch_i = self.mat_channels[self._channel]
        py_ch_i = self.py_channels[self._channel]
        mat_slice = self.mat.slice_trial_channel(
            self._trial,
            mat_ch_i,
            baseline_corrected=SHOW_BASELINE_CORRECTED,
        )[
            np.ix_(self.mat_freq_idx, self.mat_time_idx)
        ]
        py_slice = self.py.slice_trial_channel(
            self._trial,
            py_ch_i,
            baseline_corrected=SHOW_BASELINE_CORRECTED,
        )[
            np.ix_(self.py_freq_idx, self.py_time_idx)
        ]
        diff = py_slice - mat_slice
        stats = _pair_stats(mat_slice, py_slice)

        for ax in (self.ax_mat, self.ax_py, self.ax_diff, self.ax_trace):
            ax.clear()
        extent = [self.time[0], self.time[-1], self.freq[0], self.freq[-1]]
        combined = np.concatenate([mat_slice.ravel(), py_slice.ravel()])
        vmin = _finite_percentile(combined, 2, fallback=-1.0)
        vmax = _finite_percentile(combined, 98, fallback=1.0)
        if not vmax > vmin:
            vmin, vmax = vmin - 1.0, vmax + 1.0
        self.ax_mat.imshow(mat_slice, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
        self.ax_py.imshow(py_slice, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
        dlim = _finite_percentile(np.abs(diff), 98, fallback=1.0)
        if not np.isfinite(dlim) or dlim <= 0:
            dlim = 1.0
        self.ax_diff.imshow(
            diff,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=-dlim,
            vmax=dlim,
        )
        self.ax_mat.set_title("MATLAB b1_TF")
        self.ax_py.set_title("Python time_frequency")
        self.ax_diff.set_title("Python - MATLAB")
        for ax in (self.ax_mat, self.ax_py, self.ax_diff):
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")

        freq_i = self._freq
        self.ax_trace.plot(self.time, mat_slice[freq_i], label="MATLAB", linewidth=2.5)
        self.ax_trace.plot(self.time, py_slice[freq_i], label="Python", linewidth=1.5)
        self.ax_trace.axvline(0.0, color="gray", linestyle="--", linewidth=1)
        self.ax_trace.set_title(f"Trace at {self.freq[freq_i]:.2f} Hz")
        self.ax_trace.set_xlabel("Time (s)")
        self.ax_trace.set_ylabel("Power dB")
        self.ax_trace.grid(True, alpha=0.3)
        self.ax_trace.legend(loc="best")

        self.info.set_text(
            f"Trial: {self._trial + 1}/{self.n_trials}\n"
            f"Channel: {self._channel + 1}/{len(self.mat_channels)}\n"
            f"MATLAB [{mat_ch_i}]: {self.mat.channels[mat_ch_i]}\n"
            f"Python [{py_ch_i}]: {self.py.channels[py_ch_i]}\n"
            f"Baseline corrected: {SHOW_BASELINE_CORRECTED}\n"
            f"corr: {stats['corr']:.4f}\n"
            f"slope: {stats['slope']:.4f}\n"
            f"intercept: {stats['intercept']:.4f}\n"
            f"std ratio: {stats['std_ratio']:.4f}\n"
            f"mean|diff|: {stats['mean_abs']:.4f} dB\n"
            f"RMS diff: {stats['rms']:.4f} dB\n"
            f"bias: {stats['bias']:.4f} dB"
        )
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def main() -> None:
    print("Loading MATLAB b1_TF")
    header_path, wya_path, baseline_path = _resolve_matlab_paths(
        MATLAB_B1_TF_HEADER_PATH,
        MATLAB_B1_TF_WYA_PATH,
        MATLAB_B1_TF_BASELINE_PATH,
    )
    mat = MatlabB1TF.load(
        header_path,
        wya_path,
        baseline_path,
    )
    print(f"  Header : {mat.path_header}")
    print(f"  WYA    : {mat.path_wya}")
    print(f"  Shape  : {mat.dims}  [trial x time x freq x channel]")
    print(f"  Channels={len(mat.channels)}, freqs={len(mat.frequency_hz)}, times={len(mat.time_s)}")
    if not mat.baseline_is_usable:
        print(
            "  WARNING: MATLAB baseline file contains no finite values. "
            "The viewer will recompute display baselines with nanmean over "
            "[-1.3, -0.7] s from the WYA data."
        )

    print("\nLoading Python time_frequency")
    py_path = _resolve_python_path(PYTHON_TIME_FREQUENCY_PATH)
    py = PythonTF.load(py_path)
    try:
        print(f"  Path   : {py.path}")
        print(f"  Shape  : {py.power.shape}  [trial x channel x freq x time]")
        print(f"  Channels={len(py.channels)}, freqs={len(py.frequency_hz)}, times={len(py.time_s)}")
        if py.power_is_baseline_corrected:
            print(
                "  WARNING: Python power_db is baseline-corrected while MATLAB "
                ".wya stores raw dB. Raw diagnostics below reconstruct Python raw "
                "dB by adding baseline_db back."
            )

        channel_mapping = match_channels(mat.channels, py.channels)
        print(f"\nMatched channels: {len(channel_mapping)} / {len(mat.channels)} MATLAB channels")
        if not channel_mapping:
            print(f"  MATLAB first 10: {mat.channels[:10]}")
            print(f"  Python first 10: {py.channels[:10]}")
            raise RuntimeError("No channels matched.")
        print("  First matches:")
        for mat_i, py_i in list(channel_mapping.items())[:8]:
            print(f"    [{mat_i:3d}] {mat.channels[mat_i]:<16} -> [{py_i:3d}] {py.channels[py_i]}")

        mat_time_idx, py_time_idx = _nearest_indices(mat.time_s, py.time_s)
        mat_freq_idx, py_freq_idx = _nearest_indices(mat.frequency_hz, py.frequency_hz)
        if len(mat_time_idx) == 0 or len(mat_freq_idx) == 0:
            raise RuntimeError(
                "No common time/frequency grid points. Check TFR params and epoch window."
            )
        print(
            f"\nCommon grid: {len(mat_freq_idx)} freq(s), {len(mat_time_idx)} time point(s)"
        )
        if len(mat.time_s) != len(py.time_s) or len(mat.frequency_hz) != len(py.frequency_hz):
            print(
                "  NOTE: axis lengths differ; comparison uses exact/near-exact common points."
            )

        for baseline_corrected in SUMMARY_MODES:
            print_summary(
                mat,
                py,
                channel_mapping,
                mat_time_idx,
                py_time_idx,
                mat_freq_idx,
                py_freq_idx,
                baseline_corrected=baseline_corrected,
            )

        print("\nLaunching viewer")
        print("  Keyboard: left/right = trials, up/down = channels")
        viewer = TFComparisonViewer(
            mat,
            py,
            channel_mapping,
            mat_time_idx,
            py_time_idx,
            mat_freq_idx,
            py_freq_idx,
        )
        viewer.show()
    finally:
        py.close()


if __name__ == "__main__":
    main()
