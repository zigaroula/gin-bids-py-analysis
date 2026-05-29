#!/usr/bin/env python3
"""
Interactive comparison: MATLAB pipeline a3 continuous SPM output vs BIDS raw.

MATLAB a3:
    Continuous SPM MEEG file after event correction and channel renaming /
    bipolar montage (a1_convert2spm -> a2_set_spm_events -> a3_set_spm_channels).

BIDS raw:
    Continuous BrainVision recording from the raw BIDS dataset.  The viewer
    reconstructs the MATLAB bipolar channel on the fly from the BIDS monopolar
    contacts, e.g. ``PD02PD01`` -> ``PD02 - PD01``.

Usage:
    .venv\\Scripts\\python scripts\\debug\\compare_a.py
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.io
from matplotlib.widgets import Button, Slider, TextBox

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from compare_subject_config import (  # noqa: E402
    BIDS_RAW_EEG_PATH,
    BIDS_RAW_EVENTS_TSV_PATH,
    COMPARE_A_SPM_EVENT_SAMPLE_SHIFT_SAMPLES,
    MATLAB_A3_PATH,
)


# ---------------------------------------------------------------------------
# Display / alignment knobs
# ---------------------------------------------------------------------------

# MNE returns EEG/iEEG BrainVision data in SI volts.  The MATLAB a3 SPM output
# is not homogeneous across source formats: Micromed/TRC subjects are in volts,
# while Prague Matlab subjects are in microvolts.  Use "auto" to infer the
# display scale from matched channels (TRC -> ~1, Prague -> ~1e6).
BIDS_TO_MATLAB_SCALE: float | str = "auto"

INITIAL_TIME_S: float = 205.0
INITIAL_WINDOW_S: float = 10.0
MIN_WINDOW_S: float = 0.25
MAX_WINDOW_S: float = 120.0
MAX_PLOT_POINTS: int = 12_000
MAX_EVENT_LINES: int = 80
EVENT_CODES_TO_DRAW: set[str] | None = None  # None = draw all events in the window.
EXPERIMENT_START_CODE: str = "5"
# Imported from compare_subject_config so the correct format-specific shift is
# applied automatically when switching subjects.
# Formula: 1 - gin2bids event_sample_offset_samples
#   Micromed (gin2bids offset=0) -> 1
#   Prague   (gin2bids offset=1) -> 0
SPM_EVENT_SAMPLE_SHIFT_SAMPLES: int = COMPARE_A_SPM_EVENT_SAMPLE_SHIFT_SAMPLES


@dataclass(frozen=True)
class Event:
    source: str
    code: str
    onset_s: float
    sample: int


@dataclass(frozen=True)
class SpmContinuous:
    mat_path: Path
    dat_path: Path
    data: np.memmap
    channels: list[str]
    events: list[Event]
    sfreq: float
    n_samples: int
    time_onset_s: float


@dataclass(frozen=True)
class ChannelMatch:
    spm_idx: int
    spm_name: str
    bids_pos_idx: int | None
    bids_neg_idx: int | None
    bids_expr: str


# ---------------------------------------------------------------------------
# SPM helpers
# ---------------------------------------------------------------------------

def _spm_str(val: object) -> str:
    if isinstance(val, str):
        return val.strip()
    arr = np.asarray(val).flatten()
    if arr.size == 0:
        return ""
    if arr.dtype.kind in {"U", "S", "O"}:
        return "".join(str(x) for x in arr).strip()
    if arr.size == 1 and arr.dtype.kind in {"i", "u", "f"}:
        number = float(arr.flat[0])
        return str(int(number)) if number.is_integer() else str(number)
    return "".join(chr(int(c)) for c in arr).strip()


def _spm_scalar(val: object, fallback: float = 0.0) -> float:
    if val is None:
        return fallback
    try:
        return float(np.asarray(val).flat[0])
    except Exception:  # noqa: BLE001
        return fallback


def _spm_channel_labels(D: object) -> list[str]:
    channels = np.asarray(getattr(D, "channels", []), dtype=object).flatten()
    labels: list[str] = []
    for channel in channels:
        label = getattr(channel, "label", None)
        labels.append(_spm_str(label) if label is not None else "")
    return labels


def _spm_events(D: object, *, sfreq: float, time_onset_s: float) -> list[Event]:
    trials = np.asarray(getattr(D, "trials", []), dtype=object).flatten()
    if trials.size == 0:
        return []
    raw_events = np.asarray(getattr(trials[0], "events", []), dtype=object).flatten()
    events: list[Event] = []
    for event in raw_events:
        raw_time = _spm_scalar(getattr(event, "time", None), fallback=np.nan)
        if not np.isfinite(raw_time):
            continue
        onset_s = raw_time - time_onset_s
        sample = int(round(onset_s * sfreq)) + SPM_EVENT_SAMPLE_SHIFT_SAMPLES
        code = _spm_str(getattr(event, "type", None))
        if not code:
            code = _spm_str(getattr(event, "value", None))
        events.append(
            Event(
                source="MATLAB a3",
                code=code,
                onset_s=float(sample / sfreq),
                sample=sample,
            )
        )
    return events


def _spm_dtype(data_sub: object, dat_path: Path, n_channels: int, n_samples: int) -> np.dtype:
    dtype_field = getattr(data_sub, "dtype", None)
    dtype_text = _spm_str(dtype_field).casefold() if dtype_field is not None else ""
    dtype_code = int(_spm_scalar(dtype_field, fallback=-1))

    if "float32" in dtype_text or "single" in dtype_text or dtype_code == 16:
        return np.dtype("<f4")
    if "float64" in dtype_text or "double" in dtype_text or dtype_code == 64:
        return np.dtype("<f8")

    expected_f4 = n_channels * n_samples * np.dtype("<f4").itemsize
    expected_f8 = n_channels * n_samples * np.dtype("<f8").itemsize
    size = dat_path.stat().st_size
    if size == expected_f4:
        return np.dtype("<f4")
    if size == expected_f8:
        return np.dtype("<f8")
    raise ValueError(
        f"Cannot infer SPM .dat dtype from D.data.dtype={dtype_field!r} "
        f"and file size {size} bytes."
    )


def _find_spm_dat_path(mat_path: Path, D: object) -> Path:
    data_sub = getattr(D, "data", None)
    if data_sub is not None:
        fname = getattr(data_sub, "fname", None)
        if fname is not None:
            candidate = mat_path.parent / Path(_spm_str(fname)).name
            if candidate.exists():
                return candidate

    candidate = mat_path.with_suffix(".dat")
    if candidate.exists():
        return candidate

    fname = getattr(D, "fname", None)
    if fname is not None:
        candidate = mat_path.parent / Path(_spm_str(fname)).with_suffix(".dat").name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find SPM .dat sidecar for {mat_path}")


def load_spm_continuous(path: Path) -> SpmContinuous:
    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "D" not in mat:
        raise KeyError(f"{path.name}: expected SPM variable 'D'.")
    D = mat["D"]
    data_sub = getattr(D, "data", None)
    if data_sub is None:
        raise KeyError(f"{path.name}: missing D.data metadata.")

    dim = np.asarray(getattr(data_sub, "dim", []), dtype=int).ravel()
    if dim.size < 2:
        raise ValueError(f"{path.name}: D.data.dim does not expose [n_channels, n_samples].")
    n_channels = int(dim[0])
    n_samples = int(dim[1])
    sfreq = _spm_scalar(getattr(D, "Fsample", None), fallback=512.0)
    time_onset_s = _spm_scalar(getattr(D, "timeOnset", None), fallback=0.0)
    dat_path = _find_spm_dat_path(path, D)
    dtype = _spm_dtype(data_sub, dat_path, n_channels, n_samples)

    data = np.memmap(
        str(dat_path),
        dtype=dtype,
        mode="r",
        shape=(n_channels, n_samples),
        order="F",
    )
    channels = _spm_channel_labels(D)
    if len(channels) != n_channels:
        channels = [f"ch{i:03d}" for i in range(n_channels)]
    events = _spm_events(D, sfreq=sfreq, time_onset_s=time_onset_s)

    return SpmContinuous(
        mat_path=path,
        dat_path=dat_path,
        data=data,
        channels=channels,
        events=events,
        sfreq=sfreq,
        n_samples=n_samples,
        time_onset_s=time_onset_s,
    )


# ---------------------------------------------------------------------------
# BIDS and matching helpers
# ---------------------------------------------------------------------------

def _event_code_from_description(description: str) -> str:
    text = str(description).strip()
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text


def load_bids_events(path: Path, raw: mne.io.BaseRaw) -> list[Event]:
    if path.exists():
        events: list[Event] = []
        with path.open("r", encoding="utf-8-sig", newline="") as tsv_file:
            reader = csv.DictReader(tsv_file, delimiter="\t")
            for row in reader:
                onset = float(row["onset"])
                sample_text = row.get("sample", "")
                sample = (
                    int(float(sample_text))
                    if sample_text.strip() not in {"", "n/a", "nan"}
                    else int(round(onset * float(raw.info["sfreq"])))
                )
                code = row.get("code") or row.get("trial_type") or ""
                events.append(
                    Event(
                        source="BIDS",
                        code=str(code).strip(),
                        onset_s=onset,
                        sample=sample,
                    )
                )
        return events

    sfreq = float(raw.info["sfreq"])
    return [
        Event(
            source="BIDS",
            code=_event_code_from_description(desc),
            onset_s=float(onset),
            sample=int(round(float(onset) * sfreq)),
        )
        for onset, desc in zip(raw.annotations.onset, raw.annotations.description, strict=False)
    ]


def _norm_channel(name: str) -> str:
    return re.sub(r"[\s_\-\.]", "", str(name)).casefold()


def _contact_name(prefix: str, number: str) -> str:
    return f"{prefix.upper()}{int(number):02d}"


def _parse_bipolar_label(label: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)([A-Za-z]+)(\d+)", str(label).strip())
    if match is None:
        return None
    pos = _contact_name(match.group(1), match.group(2))
    neg = _contact_name(match.group(3), match.group(4))
    return pos, neg


def match_a3_channels(spm_channels: list[str], bids_channels: list[str]) -> list[ChannelMatch]:
    bids_by_norm = {_norm_channel(ch): idx for idx, ch in enumerate(bids_channels)}
    matches: list[ChannelMatch] = []

    for spm_idx, spm_name in enumerate(spm_channels):
        direct_idx = bids_by_norm.get(_norm_channel(spm_name))
        if direct_idx is not None:
            matches.append(
                ChannelMatch(
                    spm_idx=spm_idx,
                    spm_name=spm_name,
                    bids_pos_idx=direct_idx,
                    bids_neg_idx=None,
                    bids_expr=bids_channels[direct_idx],
                )
            )
            continue

        pair = _parse_bipolar_label(spm_name)
        if pair is None:
            continue
        pos_name, neg_name = pair
        pos_idx = bids_by_norm.get(_norm_channel(pos_name))
        neg_idx = bids_by_norm.get(_norm_channel(neg_name))
        if pos_idx is None or neg_idx is None:
            continue
        matches.append(
            ChannelMatch(
                spm_idx=spm_idx,
                spm_name=spm_name,
                bids_pos_idx=pos_idx,
                bids_neg_idx=neg_idx,
                bids_expr=f"{bids_channels[pos_idx]} - {bids_channels[neg_idx]}",
            )
        )

    return matches


def _zscore_trace(trace: np.ndarray) -> np.ndarray:
    arr = np.asarray(trace, dtype=np.float64)
    std = float(np.nanstd(arr))
    if not np.isfinite(std) or std <= 0.0:
        return arr - float(np.nanmean(arr))
    return (arr - float(np.nanmean(arr))) / std


def _trace_metrics(mat_trace: np.ndarray, bids_trace: np.ndarray) -> tuple[float, float, float, float]:
    finite = np.isfinite(mat_trace) & np.isfinite(bids_trace)
    if int(np.sum(finite)) < 3:
        return np.nan, np.nan, np.nan, np.nan
    a = np.asarray(mat_trace[finite], dtype=np.float64)
    b = np.asarray(bids_trace[finite], dtype=np.float64)
    mean_abs = float(np.mean(np.abs(b - a)))
    rms = float(np.sqrt(np.mean((b - a) ** 2)))
    ac = a - float(np.mean(a))
    bc = b - float(np.mean(b))
    denom = float(np.sqrt(np.sum(ac * ac) * np.sum(bc * bc)))
    corr = float(np.sum(ac * bc) / denom) if denom > 0.0 else np.nan
    slope_denom = float(np.sum(ac * ac))
    slope = float(np.sum(ac * bc) / slope_denom) if slope_denom > 0.0 else np.nan
    return mean_abs, rms, corr, slope


def _bids_trace_volts(raw: mne.io.BaseRaw, match: ChannelMatch, start: int, stop: int) -> np.ndarray:
    if match.bids_pos_idx is None:
        raise ValueError(f"Channel {match.spm_name!r} has no BIDS positive contact.")
    if match.bids_neg_idx is None:
        return np.asarray(
            raw.get_data(picks=[match.bids_pos_idx], start=start, stop=stop)[0],
            dtype=np.float64,
        )
    data = raw.get_data(
        picks=[match.bids_pos_idx, match.bids_neg_idx],
        start=start,
        stop=stop,
    )
    return np.asarray(data[0] - data[1], dtype=np.float64)


def estimate_bids_to_matlab_scale(
    *,
    spm: SpmContinuous,
    raw: mne.io.BaseRaw,
    matches: list[ChannelMatch],
) -> float:
    if isinstance(BIDS_TO_MATLAB_SCALE, (int, float)):
        return float(BIDS_TO_MATLAB_SCALE)

    duration_s = min(spm.n_samples / spm.sfreq, raw.n_times / float(raw.info["sfreq"]))
    window_s = min(max(INITIAL_WINDOW_S, 2.0), duration_s)
    center_s = min(max(INITIAL_TIME_S, window_s / 2.0), max(window_s / 2.0, duration_s - window_s / 2.0))
    start = int(round(max(0.0, center_s - window_s / 2.0) * spm.sfreq))
    stop = int(round(min(duration_s, center_s + window_s / 2.0) * spm.sfreq))
    stop = max(start + 1, min(stop, spm.n_samples, raw.n_times))

    ratios: list[float] = []
    for match in matches[: min(30, len(matches))]:
        mat_trace = np.asarray(spm.data[match.spm_idx, start:stop], dtype=np.float64)
        bids_trace = _bids_trace_volts(raw, match, start, stop)
        mat_std = float(np.nanstd(mat_trace))
        bids_std = float(np.nanstd(bids_trace))
        if np.isfinite(mat_std) and np.isfinite(bids_std) and mat_std > 0.0 and bids_std > 0.0:
            ratios.append(mat_std / bids_std)

    if not ratios:
        print("  WARNING: could not infer BIDS display scale; using 1.0.")
        return 1.0

    median_ratio = float(np.nanmedian(np.asarray(ratios, dtype=np.float64)))
    for candidate in (1.0, 1_000.0, 1_000_000.0, 1_000_000_000.0):
        if candidate / 5.0 <= median_ratio <= candidate * 5.0:
            return candidate
    return median_ratio


def print_event_alignment_diagnostics(spm_events: list[Event], bids_events: list[Event]) -> None:
    def _from_first_start(events: list[Event]) -> list[Event]:
        start = next((i for i, ev in enumerate(events) if ev.code == EXPERIMENT_START_CODE), 0)
        return events[start:]

    spm_tail = _from_first_start(spm_events)
    bids_tail = _from_first_start(bids_events)
    n = min(len(spm_tail), len(bids_tail))
    if n == 0:
        print("  Event diagnostics: no comparable events found.")
        return

    same_code = [spm_tail[i].code == bids_tail[i].code for i in range(n)]
    sample_delta = np.array([spm_tail[i].sample - bids_tail[i].sample for i in range(n)], dtype=int)
    print("\nEvent alignment after first code 5:")
    print(f"  comparable events: {n}")
    print(f"  same code sequence: {int(np.sum(same_code))}/{n}")
    print(
        "  sample delta MATLAB-BIDS: "
        f"median={float(np.median(sample_delta)):.2f}, "
        f"min={int(np.min(sample_delta))}, max={int(np.max(sample_delta))}"
    )
    print("  first comparable pairs:")
    for i in range(min(8, n)):
        print(
            f"    {i:02d}: code {spm_tail[i].code:>3s}/{bids_tail[i].code:<3s} "
            f"samples {spm_tail[i].sample:>8d}/{bids_tail[i].sample:<8d} "
            f"delta={spm_tail[i].sample - bids_tail[i].sample:+d}"
        )


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------

class ContinuousComparisonViewer:
    def __init__(
        self,
        *,
        spm: SpmContinuous,
        raw: mne.io.BaseRaw,
        bids_events: list[Event],
        matches: list[ChannelMatch],
        bids_to_matlab_scale: float,
    ) -> None:
        if not matches:
            raise ValueError("No matched channels to display.")
        self.spm = spm
        self.raw = raw
        self.bids_events = bids_events
        self.matches = matches
        self.bids_to_matlab_scale = float(bids_to_matlab_scale)
        self.duration_s = min(
            spm.n_samples / spm.sfreq,
            raw.n_times / float(raw.info["sfreq"]),
        )
        self._channel = 0
        self._center_s = min(max(INITIAL_TIME_S, 0.0), self.duration_s)
        self._window_s = min(max(INITIAL_WINDOW_S, MIN_WINDOW_S), min(MAX_WINDOW_S, self.duration_s))
        self._normalize = False
        self._flip_matlab = False
        self._flip_bids = False
        self._build_figure()

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(15, 8))
        self.fig.suptitle(
            "MATLAB a3 continuous output vs BIDS raw",
            fontsize=13,
            fontweight="bold",
        )
        self.ax = self.fig.add_axes([0.07, 0.32, 0.72, 0.58])

        ax_ch = self.fig.add_axes([0.07, 0.22, 0.48, 0.03])
        self.sl_ch = Slider(ax_ch, "Channel", 0, len(self.matches) - 1, valinit=0, valstep=1)
        self.sl_ch.on_changed(self._on_channel_slider)

        ax_time = self.fig.add_axes([0.07, 0.15, 0.48, 0.03])
        self.sl_time = Slider(
            ax_time,
            "Time center (s)",
            0.0,
            self.duration_s,
            valinit=self._center_s,
            valstep=1.0 / self.spm.sfreq,
            color="darkorange",
        )
        self.sl_time.on_changed(self._on_time_slider)

        ax_win = self.fig.add_axes([0.07, 0.08, 0.48, 0.03])
        self.sl_window = Slider(
            ax_win,
            "Window (s)",
            MIN_WINDOW_S,
            min(MAX_WINDOW_S, self.duration_s),
            valinit=self._window_s,
            valstep=0.25,
            color="seagreen",
        )
        self.sl_window.on_changed(self._on_window_slider)

        ax_prev_ch = self.fig.add_axes([0.60, 0.205, 0.07, 0.045])
        ax_next_ch = self.fig.add_axes([0.69, 0.205, 0.07, 0.045])
        self.btn_prev_ch = Button(ax_prev_ch, "< Ch", color="lightblue")
        self.btn_next_ch = Button(ax_next_ch, "Ch >", color="lightblue")
        self.btn_prev_ch.on_clicked(lambda _event: self._step_channel(-1))
        self.btn_next_ch.on_clicked(lambda _event: self._step_channel(+1))

        ax_prev_t = self.fig.add_axes([0.60, 0.135, 0.07, 0.045])
        ax_next_t = self.fig.add_axes([0.69, 0.135, 0.07, 0.045])
        self.btn_prev_t = Button(ax_prev_t, "< Time", color="moccasin")
        self.btn_next_t = Button(ax_next_t, "Time >", color="moccasin")
        self.btn_prev_t.on_clicked(lambda _event: self._step_time(-0.5))
        self.btn_next_t.on_clicked(lambda _event: self._step_time(+0.5))

        ax_norm = self.fig.add_axes([0.60, 0.065, 0.16, 0.045])
        self.btn_norm = Button(ax_norm, "Norm window: OFF", color="lightgray")
        self.btn_norm.on_clicked(self._toggle_norm)

        ax_flip_mat = self.fig.add_axes([0.78, 0.135, 0.10, 0.045])
        ax_flip_bids = self.fig.add_axes([0.89, 0.135, 0.10, 0.045])
        self.btn_flip_mat = Button(ax_flip_mat, "Flip MAT: OFF", color="lightgray")
        self.btn_flip_bids = Button(ax_flip_bids, "Flip BIDS: OFF", color="lightgray")
        self.btn_flip_mat.on_clicked(self._toggle_flip_matlab)
        self.btn_flip_bids.on_clicked(self._toggle_flip_bids)

        ax_search = self.fig.add_axes([0.82, 0.22, 0.13, 0.04])
        self.tb_search = TextBox(ax_search, "Go to", initial="")
        self.tb_search.on_submit(self._submit_channel_search)

        self.info_text = self.fig.text(
            0.82,
            0.58,
            "",
            ha="left",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.9},
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._redraw()

    def _window_bounds(self) -> tuple[int, int, np.ndarray]:
        half = self._window_s / 2.0
        start_s = max(0.0, self._center_s - half)
        stop_s = min(self.duration_s, start_s + self._window_s)
        start_s = max(0.0, stop_s - self._window_s)
        start = int(round(start_s * self.spm.sfreq))
        stop = int(round(stop_s * self.spm.sfreq))
        stop = max(start + 1, min(stop, self.spm.n_samples, self.raw.n_times))
        time = np.arange(start, stop, dtype=np.float64) / self.spm.sfreq
        return start, stop, time

    def _current_bids_trace(self, match: ChannelMatch, start: int, stop: int) -> np.ndarray:
        if match.bids_pos_idx is None:
            raise ValueError(f"Channel {match.spm_name!r} has no BIDS positive contact.")
        return _bids_trace_volts(self.raw, match, start, stop) * self.bids_to_matlab_scale

    def _redraw(self) -> None:
        match = self.matches[self._channel]
        start, stop, time = self._window_bounds()
        mat_trace = np.asarray(self.spm.data[match.spm_idx, start:stop], dtype=np.float64)
        bids_trace = self._current_bids_trace(match, start, stop)
        if self._flip_matlab:
            mat_trace = -mat_trace
        if self._flip_bids:
            bids_trace = -bids_trace

        if self._normalize:
            mat_to_plot = _zscore_trace(mat_trace)
            bids_to_plot = _zscore_trace(bids_trace)
            ylabel = "Window z-score"
        else:
            mat_to_plot = mat_trace
            bids_to_plot = bids_trace
            ylabel = "Amplitude (MATLAB native units)"

        step = max(1, int(np.ceil(time.size / MAX_PLOT_POINTS)))
        plot_slice = slice(None, None, step)

        self.ax.clear()
        self.ax.plot(
            time[plot_slice],
            mat_to_plot[plot_slice],
            color="steelblue",
            linewidth=4.5,
            alpha=0.85,
            label=f"MATLAB a3 ({self.spm.sfreq:.0f} Hz)",
        )
        self.ax.plot(
            time[plot_slice],
            bids_to_plot[plot_slice],
            color="darkorange",
            linewidth=1.35,
            alpha=0.9,
            label="BIDS raw bipolar",
        )
        self._draw_events(float(time[0]), float(time[-1]))

        mean_abs, rms, corr, slope = _trace_metrics(mat_trace, bids_trace)
        self.ax.set_title(
            f"Channel {self._channel + 1}/{len(self.matches)}: "
            f"{match.spm_name}  vs  {match.bids_expr}",
            fontsize=11,
        )
        self.ax.set_xlabel("Time from BIDS recording start (s)")
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True, alpha=0.28)
        self.ax.legend(loc="upper right", fontsize=9)

        info = (
            f"MATLAB idx: {match.spm_idx}\n"
            f"MATLAB ch:  {match.spm_name}\n"
            f"BIDS expr:  {match.bids_expr}\n"
            f"BIDS scale: {self.bids_to_matlab_scale:g}\n"
            f"Samples:    {start} -> {stop - 1}\n"
            f"Window:     {time[0]:.3f} -> {time[-1]:.3f} s\n"
            f"mean|d|:    {mean_abs:.4f}\n"
            f"RMS d:      {rms:.4f}\n"
            f"corr:       {corr:.5f}\n"
            f"slope:      {slope:.5f}\n"
            f"Flip MAT:   {'ON' if self._flip_matlab else 'OFF'}\n"
            f"Flip BIDS:  {'ON' if self._flip_bids else 'OFF'}\n"
            f"Keys: arrows, +/- zoom, n norm, m/b flip"
        )
        self.info_text.set_text(info)
        self.fig.canvas.draw_idle()

    def _draw_events(self, start_s: float, stop_s: float) -> None:
        spm_events = [
            ev for ev in self.spm.events
            if start_s <= ev.onset_s <= stop_s
            and (EVENT_CODES_TO_DRAW is None or ev.code in EVENT_CODES_TO_DRAW)
        ]
        bids_events = [
            ev for ev in self.bids_events
            if start_s <= ev.onset_s <= stop_s
            and (EVENT_CODES_TO_DRAW is None or ev.code in EVENT_CODES_TO_DRAW)
        ]
        drawn = 0
        y_top = 0.98
        for ev in spm_events[:MAX_EVENT_LINES]:
            self.ax.axvline(ev.onset_s, color="steelblue", linewidth=1.1, alpha=0.35)
            if drawn < 12:
                self.ax.text(
                    ev.onset_s,
                    y_top,
                    ev.code,
                    color="steelblue",
                    fontsize=8,
                    rotation=90,
                    va="top",
                    ha="right",
                    transform=self.ax.get_xaxis_transform(),
                )
            drawn += 1
        for ev in bids_events[:MAX_EVENT_LINES]:
            self.ax.axvline(ev.onset_s, color="darkorange", linestyle="--", linewidth=0.9, alpha=0.45)
            if drawn < 24:
                self.ax.text(
                    ev.onset_s,
                    0.88,
                    ev.code,
                    color="darkorange",
                    fontsize=8,
                    rotation=90,
                    va="top",
                    ha="left",
                    transform=self.ax.get_xaxis_transform(),
                )
            drawn += 1

    def _set_channel(self, idx: int) -> None:
        idx = int(np.clip(idx, 0, len(self.matches) - 1))
        if idx != self._channel:
            self._channel = idx
            self.sl_ch.set_val(idx)
        else:
            self._redraw()

    def _step_channel(self, delta: int) -> None:
        self._set_channel(self._channel + delta)

    def _set_center(self, center_s: float) -> None:
        center_s = float(np.clip(center_s, 0.0, self.duration_s))
        if abs(center_s - self._center_s) > 1e-12:
            self._center_s = center_s
            self.sl_time.set_val(center_s)
        else:
            self._redraw()

    def _step_time(self, window_fraction: float) -> None:
        self._set_center(self._center_s + window_fraction * self._window_s)

    def _on_channel_slider(self, val: float) -> None:
        new_idx = int(val)
        if new_idx != self._channel:
            self._channel = new_idx
            self._redraw()

    def _on_time_slider(self, val: float) -> None:
        self._center_s = float(val)
        self._redraw()

    def _on_window_slider(self, val: float) -> None:
        self._window_s = float(val)
        self._redraw()

    def _toggle_norm(self, _event: object) -> None:
        self._normalize = not self._normalize
        self.btn_norm.label.set_text("Norm window: ON" if self._normalize else "Norm window: OFF")
        self.btn_norm.ax.set_facecolor("lightgreen" if self._normalize else "lightgray")
        self._redraw()

    def _toggle_flip_matlab(self, _event: object) -> None:
        self._flip_matlab = not self._flip_matlab
        self.btn_flip_mat.label.set_text("Flip MAT: ON" if self._flip_matlab else "Flip MAT: OFF")
        self.btn_flip_mat.ax.set_facecolor("lightgreen" if self._flip_matlab else "lightgray")
        self._redraw()

    def _toggle_flip_bids(self, _event: object) -> None:
        self._flip_bids = not self._flip_bids
        self.btn_flip_bids.label.set_text("Flip BIDS: ON" if self._flip_bids else "Flip BIDS: OFF")
        self.btn_flip_bids.ax.set_facecolor("lightgreen" if self._flip_bids else "lightgray")
        self._redraw()

    def _submit_channel_search(self, text: str) -> None:
        query = text.strip()
        if not query:
            return
        if query.isdigit():
            self._set_channel(int(query) - 1)
            return
        query_norm = _norm_channel(query)
        for idx, match in enumerate(self.matches):
            if query_norm in _norm_channel(match.spm_name) or query_norm in _norm_channel(match.bids_expr):
                self._set_channel(idx)
                return
        print(f"  Channel search: no match for {query!r}")

    def _on_key(self, event: object) -> None:
        key = getattr(event, "key", None)
        if key == "right":
            self._step_time(+0.5)
        elif key == "left":
            self._step_time(-0.5)
        elif key == "up":
            self._step_channel(+1)
        elif key == "down":
            self._step_channel(-1)
        elif key in {"+", "="}:
            self.sl_window.set_val(max(MIN_WINDOW_S, self._window_s / 1.5))
        elif key in {"-", "_"}:
            self.sl_window.set_val(min(min(MAX_WINDOW_S, self.duration_s), self._window_s * 1.5))
        elif key == "n":
            self._toggle_norm(None)
        elif key == "m":
            self._toggle_flip_matlab(None)
        elif key == "b":
            self._toggle_flip_bids(None)

    def show(self) -> None:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading MATLAB a3 SPM file: {MATLAB_A3_PATH}")
    if not MATLAB_A3_PATH.exists():
        sys.exit(f"ERROR: MATLAB a3 file not found: {MATLAB_A3_PATH}")
    spm = load_spm_continuous(MATLAB_A3_PATH)
    print(
        f"  data: {spm.data.shape[0]} channels x {spm.data.shape[1]} samples, "
        f"sfreq={spm.sfreq:.3f} Hz"
    )
    print(f"  .dat sidecar: {spm.dat_path}")
    print(f"  timeOnset: {spm.time_onset_s:.6f} s")
    print(f"  first channels: {spm.channels[:8]}")
    print(f"  MATLAB events: {len(spm.events)}")

    bv_vhdr_path = BIDS_RAW_EEG_PATH.with_suffix(".vhdr")
    print(f"\nLoading BIDS raw BrainVision file: {bv_vhdr_path}")
    if not bv_vhdr_path.exists():
        sys.exit(f"ERROR: BIDS raw .vhdr not found: {bv_vhdr_path}")
    mne.set_log_level("WARNING")
    raw = mne.io.read_raw_brainvision(str(bv_vhdr_path), preload=False, verbose=False)
    print(
        f"  data: {len(raw.ch_names)} channels x {raw.n_times} samples, "
        f"sfreq={float(raw.info['sfreq']):.3f} Hz"
    )
    print(f"  first channels: {raw.ch_names[:8]}")

    if abs(float(raw.info["sfreq"]) - spm.sfreq) > 1e-6:
        print(
            "  WARNING: sampling frequencies differ. The viewer assumes matching sample grids "
            f"(MATLAB={spm.sfreq}, BIDS={float(raw.info['sfreq'])})."
        )
    if raw.n_times != spm.n_samples:
        print(
            "  WARNING: sample counts differ. The viewer uses the common duration "
            f"(MATLAB={spm.n_samples}, BIDS={raw.n_times})."
        )

    bids_events = load_bids_events(BIDS_RAW_EVENTS_TSV_PATH, raw)
    print(f"  BIDS events: {len(bids_events)}")
    print_event_alignment_diagnostics(spm.events, bids_events)

    matches = match_a3_channels(spm.channels, raw.ch_names)
    print(f"\nChannel matching: {len(matches)} / {len(spm.channels)} MATLAB a3 channels")
    for match in matches[:10]:
        print(f"  [{match.spm_idx:3d}] {match.spm_name:<16s} -> {match.bids_expr}")
    if not matches:
        sys.exit("ERROR: no channel match found between MATLAB a3 and BIDS raw.")

    bids_to_matlab_scale = estimate_bids_to_matlab_scale(
        spm=spm,
        raw=raw,
        matches=matches,
    )
    print(f"\nBIDS display scale: {bids_to_matlab_scale:g} (MNE volts -> MATLAB native units)")

    print("\nLaunching continuous comparison viewer...")
    print("  Keyboard: left/right move time, up/down channels, +/- zoom, n normalize, m/b flip.")
    print("  Search box accepts a 1-based channel number or a name fragment (e.g. PD08).")
    viewer = ContinuousComparisonViewer(
        spm=spm,
        raw=raw,
        bids_events=bids_events,
        matches=matches,
        bids_to_matlab_scale=bids_to_matlab_scale,
    )
    viewer.show()


if __name__ == "__main__":
    main()

