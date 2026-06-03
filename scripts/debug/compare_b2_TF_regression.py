#!/usr/bin/env python3
"""Compare MATLAB b2_TF regression outputs with Python TF regression.

Configuration is centralized in ``compare_subject_config.py``.

Usage:
    .venv\\Scripts\\python scripts\\debug\\compare_b2_TF_regression.py
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
import scipy.io

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
for path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bidsforge.processing.time_frequency_stats.regression import (  # noqa: E402
    load_time_frequency_regression_result,
)
from compare_subject_config import (  # noqa: E402
    BIDS_ROOT,
    BIDS_SUBJECT,
    MATLAB_B2_TF_REGRESSION_PATH,
    MATLAB_ROOT,
    PYTHON_TF_REGRESSION_PATH,
    SUBJECT,
)


MATLAB_REALIGN = "onset"
MATLAB_PYTHON_REGRESSION_MAP = [
    ("P_Rating", "condition_a"),
    ("UP_Rating", "condition_b"),
]
TOP_N = 12
INITIAL_BUNDLE_INDEX = 0
INITIAL_CHANNEL_INDEX = 0


@dataclass
class TFMapBundle:
    label: str
    matlab: np.ndarray
    python: np.ndarray


def _resolve_python_path(path: Path) -> Path:
    if path.exists():
        return path
    roots = [
        path.parent,
        BIDS_ROOT / "derivatives" / "time_frequency_regression" / f"sub-{BIDS_SUBJECT}" / "ieeg",
        BIDS_ROOT / "derivatives" / "time_frequency_regression",
    ]
    patterns = [
        f"sub-{BIDS_SUBJECT}_*_desc-tfregressiononset_stats.h5",
        f"sub-{BIDS_SUBJECT}_*_desc-tfregression*_stats.h5",
        "*_desc-tfregressiononset_stats.h5",
    ]
    candidates = _find_candidates(roots, patterns)
    if candidates:
        chosen = candidates[0]
        print(f"  Python TF regression default path missing, using: {chosen}")
        _print_other_candidates(candidates)
        return chosen
    raise FileNotFoundError(
        f"Python TF regression output not found: {path}\n"
        "Run scripts/debug/run_time_frequency_regression_debug.py first, "
        "or edit PYTHON_TF_REGRESSION_PATH in compare_subject_config.py."
    )


def _resolve_matlab_path(path: Path) -> Path:
    if path.exists():
        return path
    r1_root = MATLAB_ROOT / "b2_TF_single_contrast_R1" / SUBJECT
    roots = [
        path.parent,
        MATLAB_ROOT / "b2_TF_single_contrast" / SUBJECT / "regressions_TF",
        r1_root,
        MATLAB_ROOT / "b2_TF_single_contrast",
        MATLAB_ROOT / "b2_TF_single_contrast_R1",
    ]
    patterns = ["log_data.mat", "regressions_TF_Part-*/log_data.mat"]
    candidates = _find_candidates(roots, patterns)
    if candidates:
        chosen = candidates[0]
        print(f"  MATLAB TF regression default path missing, using: {chosen}")
        _print_other_candidates(candidates)
        return chosen
    raise FileNotFoundError(
        f"MATLAB TF regression output not found: {path}\n"
        "Run b2_TF_subj_contact_levels_CB.m or CB_R1.m for this subject, "
        "or edit MATLAB_B2_TF_REGRESSION_PATH in compare_subject_config.py."
    )


def _find_candidates(roots: list[Path], patterns: list[str]) -> list[Path]:
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            globber = root.glob if root == roots[0] else root.rglob
            candidates.extend(sorted(globber(pattern)))
        if candidates:
            break
    return sorted(set(candidates))


def _print_other_candidates(candidates: list[Path]) -> None:
    if len(candidates) <= 1:
        return
    print("  Other candidates:")
    for candidate in candidates[1:8]:
        print(f"    {candidate}")


def _load_matlab(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, dict[str, np.ndarray]]]:
    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "log" not in mat:
        raise ValueError(f"{path} does not contain a 'log' struct.")
    hdr = mat.get("hdr")
    log = mat["log"]
    channels = _string_list(getattr(hdr, "chanlist", [])) if hdr is not None else []
    freqs = _float_axis(getattr(hdr, "freqlist", [])) if hdr is not None else np.array([])
    times = _float_axis(mat.get("timelist", getattr(hdr, "timelist", []))) if hdr is not None else _float_axis(mat.get("timelist", []))
    maps: dict[str, dict[str, np.ndarray]] = {}
    for regressor, _condition_field in MATLAB_PYTHON_REGRESSION_MAP:
        branch = _matlab_regression_branch(log, regressor)
        maps[regressor] = {
            "slope": _cell_maps(_first_existing_attr(branch, ["dots", "slope"])),
            "tstat": _cell_maps(_first_existing_attr(branch, ["tstat", "t_values"])),
            "pval": _cell_maps(_first_existing_attr(branch, ["pval", "p_value"])),
        }
    if not channels:
        n_channels = next(iter(maps.values()))["slope"].shape[0]
        channels = [f"channel_{idx + 1}" for idx in range(n_channels)]
    return channels, freqs, times, maps


def _matlab_regression_branch(log: object, regressor: str) -> object:
    if not hasattr(log, regressor):
        available = [name for name in dir(log) if not name.startswith("_")]
        raise ValueError(f"Regressor {regressor!r} not found. Available: {available}")
    reg_branch = getattr(log, regressor)
    if hasattr(reg_branch, MATLAB_REALIGN):
        return getattr(reg_branch, MATLAB_REALIGN)
    available = [name for name in dir(reg_branch) if not name.startswith("_")]
    if len(available) == 1:
        return getattr(reg_branch, available[0])
    raise ValueError(
        f"Realign branch {MATLAB_REALIGN!r} not found under {regressor!r}. "
        f"Available: {available}"
    )


def _python_condition_stats(result, condition_field: str) -> dict[str, np.ndarray]:
    stats = getattr(result.regression, condition_field)
    return {
        "slope": np.asarray(stats.slope, dtype=np.float64),
        "tstat": np.asarray(stats.t_values, dtype=np.float64),
        "pval": np.asarray(stats.p_value, dtype=np.float64),
    }


def _compare_regressor(
    regressor: str,
    condition_field: str,
    mat_channels: list[str],
    mat_freq: np.ndarray,
    mat_time: np.ndarray,
    mat_maps: dict[str, np.ndarray],
    py,
) -> None:
    if condition_field == "condition_a":
        condition_name = py.condition_a
    elif condition_field == "condition_b":
        condition_name = py.condition_b
    else:
        condition_name = condition_field
    print(f"\nMATLAB {regressor} -> Python {condition_field} ({condition_name})")
    py_maps = _python_condition_stats(py, condition_field)
    for map_name in ("slope", "tstat", "pval"):
        _compare_map(
            map_name,
            mat_channels,
            mat_maps[map_name],
            mat_freq,
            mat_time,
            py.channel_names,
            py_maps[map_name],
            np.asarray(py.frequency_hz, dtype=np.float64),
            np.asarray(py.time_axis_s, dtype=np.float64),
        )


def _build_bundles(py, mat_maps: dict[str, dict[str, np.ndarray]]) -> list[TFMapBundle]:
    bundles: list[TFMapBundle] = []
    for regressor, condition_field in MATLAB_PYTHON_REGRESSION_MAP:
        if condition_field == "condition_a":
            condition_name = py.condition_a
        elif condition_field == "condition_b":
            condition_name = py.condition_b
        else:
            condition_name = condition_field
        py_maps = _python_condition_stats(py, condition_field)
        for map_name in ("slope", "tstat", "pval"):
            bundles.append(
                TFMapBundle(
                    label=f"{regressor} -> {condition_name} / {map_name}",
                    matlab=mat_maps[regressor][map_name],
                    python=py_maps[map_name],
                )
            )
    return bundles


def _compare_map(
    label: str,
    mat_channels: list[str],
    mat_map: np.ndarray,
    mat_freq: np.ndarray,
    mat_time: np.ndarray,
    py_channels: list[str],
    py_map: np.ndarray,
    py_freq: np.ndarray,
    py_time: np.ndarray,
) -> None:
    mat_f, py_f = _nearest_indices(mat_freq, py_freq)
    mat_t, py_t = _nearest_indices(mat_time, py_time)
    mapping = match_channels(mat_channels, py_channels)
    print(f"  {label}: matched channels {len(mapping)} / {len(mat_channels)}, common grid {len(mat_f)} x {len(mat_t)}")
    if not mapping or len(mat_f) == 0 or len(mat_t) == 0:
        return
    rows = []
    for mat_i, py_i in mapping.items():
        if mat_i >= mat_map.shape[0]:
            continue
        matlab = mat_map[mat_i][np.ix_(mat_f, mat_t)]
        python = py_map[py_i][np.ix_(py_f, py_t)]
        rows.append((mat_i, py_i, mat_channels[mat_i], py_channels[py_i], *_pair_stats(matlab, python)))
    _print_rows(rows)


def _print_rows(rows: list[tuple[int, int, str, str, float, float, float, float]]) -> None:
    if not rows:
        print("    No comparable rows.")
        return
    corrs = np.asarray([row[4] for row in rows], dtype=np.float64)
    mean_abs = np.asarray([row[5] for row in rows], dtype=np.float64)
    max_abs = np.asarray([row[6] for row in rows], dtype=np.float64)
    bias = np.asarray([row[7] for row in rows], dtype=np.float64)
    print(
        f"    median corr={np.nanmedian(corrs):.6f}, "
        f"median mean|diff|={np.nanmedian(mean_abs):.6g}, "
        f"median max|diff|={np.nanmedian(max_abs):.6g}, "
        f"median bias={np.nanmedian(bias):.6g}"
    )
    print("    lowest channel correlations:")
    print(f"    {'MAT':>5}  {'PY':>5}  {'MATLAB':<16}  {'Python':<16}  {'corr':>9}  {'mean|d|':>10}  {'max|d|':>10}  {'bias':>10}")
    for mat_i, py_i, mat_name, py_name, corr, mean_abs, max_abs, bias in sorted(rows, key=lambda row: row[4])[:TOP_N]:
        print(
            f"    [{mat_i:3d}]  [{py_i:3d}]  {mat_name[:16]:<16}  {py_name[:16]:<16}  "
            f"{corr:9.6f}  {mean_abs:10.6g}  {max_abs:10.6g}  {bias:10.6g}"
        )


class TFMapComparisonViewer:
    def __init__(
        self,
        *,
        title: str,
        bundles: list[TFMapBundle],
        mat_channels: list[str],
        py_channels: list[str],
        channel_mapping: dict[int, int],
        mat_time: np.ndarray,
        py_time: np.ndarray,
        mat_freq: np.ndarray,
        py_freq: np.ndarray,
    ) -> None:
        self.title = title
        self.bundles = bundles
        self.all_mat_channels = mat_channels
        self.all_py_channels = py_channels
        self.mat_channels = sorted(channel_mapping)
        self.py_channels = [channel_mapping[idx] for idx in self.mat_channels]
        self.mat_time_idx, self.py_time_idx = _nearest_indices(mat_time, py_time)
        self.mat_freq_idx, self.py_freq_idx = _nearest_indices(mat_freq, py_freq)
        self.time = mat_time[self.mat_time_idx]
        self.freq = mat_freq[self.mat_freq_idx]
        self._bundle = min(INITIAL_BUNDLE_INDEX, max(0, len(bundles) - 1))
        self._channel = min(INITIAL_CHANNEL_INDEX, max(0, len(self.mat_channels) - 1))
        if not self.bundles:
            raise ValueError("No comparison bundles available.")
        if not self.mat_channels:
            raise ValueError("No matched channels available for viewer.")
        if len(self.time) == 0 or len(self.freq) == 0:
            raise ValueError("No common time/frequency grid points available for viewer.")
        self._build()

    def _build(self) -> None:
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.suptitle(self.title, fontweight="bold")
        self.ax_mat = self.fig.add_axes([0.06, 0.42, 0.27, 0.43])
        self.ax_py = self.fig.add_axes([0.37, 0.42, 0.27, 0.43])
        self.ax_diff = self.fig.add_axes([0.68, 0.42, 0.27, 0.43])
        self.info = self.fig.text(
            0.70,
            0.20,
            "",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.85},
        )
        self.sl_channel = Slider(
            self.fig.add_axes([0.08, 0.14, 0.50, 0.03]),
            "Channel",
            0,
            len(self.mat_channels) - 1,
            valinit=self._channel,
            valstep=1,
        )
        self.sl_bundle = Slider(
            self.fig.add_axes([0.08, 0.08, 0.50, 0.03]),
            "Map",
            0,
            len(self.bundles) - 1,
            valinit=self._bundle,
            valstep=1,
        )
        self.sl_channel.on_changed(lambda value: self._set(channel=int(value)))
        self.sl_bundle.on_changed(lambda value: self._set(bundle=int(value)))
        self.btn_prev_channel = Button(self.fig.add_axes([0.62, 0.135, 0.08, 0.04]), "< Chan")
        self.btn_next_channel = Button(self.fig.add_axes([0.71, 0.135, 0.08, 0.04]), "Chan >")
        self.btn_prev_map = Button(self.fig.add_axes([0.62, 0.075, 0.08, 0.04]), "< Map")
        self.btn_next_map = Button(self.fig.add_axes([0.71, 0.075, 0.08, 0.04]), "Map >")
        self.btn_prev_channel.on_clicked(lambda _event: self._step(channel_delta=-1))
        self.btn_next_channel.on_clicked(lambda _event: self._step(channel_delta=1))
        self.btn_prev_map.on_clicked(lambda _event: self._step(bundle_delta=-1))
        self.btn_next_map.on_clicked(lambda _event: self._step(bundle_delta=1))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._render()

    def _set(self, *, channel: int | None = None, bundle: int | None = None) -> None:
        if channel is not None:
            self._channel = int(np.clip(channel, 0, len(self.mat_channels) - 1))
        if bundle is not None:
            self._bundle = int(np.clip(bundle, 0, len(self.bundles) - 1))
        self._render()

    def _step(self, *, channel_delta: int = 0, bundle_delta: int = 0) -> None:
        self._channel = int(np.clip(self._channel + channel_delta, 0, len(self.mat_channels) - 1))
        self._bundle = int(np.clip(self._bundle + bundle_delta, 0, len(self.bundles) - 1))
        self.sl_channel.set_val(self._channel)
        self.sl_bundle.set_val(self._bundle)
        self._render()

    def _on_key(self, event: object) -> None:
        key = getattr(event, "key", None)
        if key == "right":
            self._step(channel_delta=1)
        elif key == "left":
            self._step(channel_delta=-1)
        elif key == "up":
            self._step(bundle_delta=1)
        elif key == "down":
            self._step(bundle_delta=-1)

    def _render(self) -> None:
        bundle = self.bundles[self._bundle]
        mat_ch_i = self.mat_channels[self._channel]
        py_ch_i = self.py_channels[self._channel]
        mat_map = bundle.matlab[mat_ch_i][np.ix_(self.mat_freq_idx, self.mat_time_idx)]
        py_map = bundle.python[py_ch_i][np.ix_(self.py_freq_idx, self.py_time_idx)]
        diff = py_map - mat_map
        stats = _pair_stats(mat_map, py_map)
        for ax in (self.ax_mat, self.ax_py, self.ax_diff):
            ax.clear()
        extent = [self.time[0], self.time[-1], self.freq[0], self.freq[-1]]
        combined = np.concatenate([mat_map.ravel(), py_map.ravel()])
        vmin = _finite_percentile(combined, 2, fallback=-1.0)
        vmax = _finite_percentile(combined, 98, fallback=1.0)
        if not vmax > vmin:
            vmin, vmax = vmin - 1.0, vmax + 1.0
        self.ax_mat.imshow(mat_map, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
        self.ax_py.imshow(py_map, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
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
        self.ax_mat.set_title("MATLAB b2_TF")
        self.ax_py.set_title("Python TF stats")
        self.ax_diff.set_title("Python - MATLAB")
        for ax in (self.ax_mat, self.ax_py, self.ax_diff):
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
        self.info.set_text(
            f"Map: {self._bundle + 1}/{len(self.bundles)}\n"
            f"{bundle.label}\n\n"
            f"Channel: {self._channel + 1}/{len(self.mat_channels)}\n"
            f"MATLAB [{mat_ch_i}]: {self.all_mat_channels[mat_ch_i]}\n"
            f"Python [{py_ch_i}]: {self.all_py_channels[py_ch_i]}\n\n"
            f"corr: {stats[0]:.6f}\n"
            f"mean|diff|: {stats[1]:.6g}\n"
            f"max|diff|: {stats[2]:.6g}\n"
            f"bias: {stats[3]:.6g}"
        )
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_idx: list[int] = []
    target_idx: list[int] = []
    for i, value in enumerate(source):
        if target.size == 0:
            break
        j = int(np.argmin(np.abs(target - value)))
        if np.isclose(value, target[j], rtol=1e-4, atol=1e-4):
            source_idx.append(i)
            target_idx.append(j)
    return np.asarray(source_idx, dtype=np.int64), np.asarray(target_idx, dtype=np.int64)


def _finite_percentile(values: np.ndarray, q: float, fallback: float = 1.0) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return fallback
    out = float(np.percentile(finite, q))
    return out if np.isfinite(out) else fallback


def match_channels(matlab_channels: list[str], python_channels: list[str]) -> dict[int, int]:
    py_by_norm = {_norm(ch): idx for idx, ch in enumerate(python_channels)}
    mapping: dict[int, int] = {}
    used: set[int] = set()
    for mat_i, mat_ch in enumerate(matlab_channels):
        keys = [_norm(mat_ch), _norm(_first_bipolar_contact(mat_ch))]
        for key in keys:
            py_i = py_by_norm.get(key)
            if py_i is not None and py_i not in used:
                mapping[mat_i] = py_i
                used.add(py_i)
                break
    return mapping


def _norm(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _first_bipolar_contact(name: object) -> str:
    text = str(name).strip()
    match = re.match(r"^([A-Za-z]+0*\d+)", text)
    return match.group(1) if match else text


def _pair_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    finite = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(finite)) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    av = np.asarray(a[finite], dtype=np.float64).ravel()
    bv = np.asarray(b[finite], dtype=np.float64).ravel()
    corr = float(np.corrcoef(av, bv)[0, 1])
    diff = bv - av
    return corr, float(np.mean(np.abs(diff))), float(np.max(np.abs(diff))), float(np.mean(diff))


def _first_existing_attr(obj: object, names: list[str]) -> object:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise ValueError(f"None of these fields exist on Matlab struct: {names}")


def _string_list(value: object) -> list[str]:
    arr = np.asarray(value, dtype=object).ravel()
    return [str(item).strip() for item in arr]


def _float_axis(value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _cell_maps(value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=object)
    if arr.dtype == object:
        return np.stack([_as_freq_time(item) for item in arr.ravel()], axis=0)
    numeric = np.asarray(value, dtype=np.float64)
    if numeric.ndim == 2:
        return numeric[np.newaxis, :, :]
    if numeric.ndim == 3:
        return np.squeeze(numeric)
    return numeric


def _as_freq_time(value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a frequency x time map, got shape {arr.shape}.")
    return arr


def main() -> None:
    print("Loading MATLAB b2_TF regression")
    matlab_path = _resolve_matlab_path(MATLAB_B2_TF_REGRESSION_PATH)
    mat_channels, mat_freq, mat_time, mat_maps = _load_matlab(matlab_path)
    print(f"  Path: {matlab_path}")
    print(f"  Channels={len(mat_channels)}, freqs={len(mat_freq)}, times={len(mat_time)}")

    print("\nLoading Python TF regression")
    python_path = _resolve_python_path(PYTHON_TF_REGRESSION_PATH)
    py = load_time_frequency_regression_result(python_path)
    print(f"  Path: {python_path}")
    print(
        f"  Conditions: {py.condition_a}={py.condition_a_trial_count}, "
        f"{py.condition_b}={py.condition_b_trial_count}; "
        f"predictor={py.predictor}, zscore={py.predictor_zscore}"
    )
    print(f"  Channels={len(py.channel_names)}, freqs={len(py.frequency_hz)}, times={len(py.time_axis_s)}")

    for regressor, condition_field in MATLAB_PYTHON_REGRESSION_MAP:
        _compare_regressor(
            regressor,
            condition_field,
            mat_channels,
            mat_freq,
            mat_time,
            mat_maps[regressor],
            py,
        )

    channel_mapping = match_channels(mat_channels, py.channel_names)
    bundles = _build_bundles(py, mat_maps)
    print("\nLaunching viewer")
    print("  Keyboard: left/right = channels, up/down = maps")
    viewer = TFMapComparisonViewer(
        title="MATLAB b2_TF regression vs Python time_frequency_regression",
        bundles=bundles,
        mat_channels=mat_channels,
        py_channels=py.channel_names,
        channel_mapping=channel_mapping,
        mat_time=mat_time,
        py_time=np.asarray(py.time_axis_s, dtype=np.float64),
        mat_freq=mat_freq,
        py_freq=np.asarray(py.frequency_hz, dtype=np.float64),
    )
    viewer.show()


if __name__ == "__main__":
    main()
