#!/usr/bin/env python3
"""Compare MATLAB b3_TF regression group maps with Python TF group output.

The stock ``b3_TF_group_parcel_levels_CB.m`` writes figures. For direct
numerical comparison, export a small .mat containing condition/regressor maps
such as ``condition_a_mean_source_t_values`` and
``condition_b_mean_source_t_values`` (or ``P_Rating``/``UP_Rating`` fields).

Usage:
    .venv\\Scripts\\python scripts\\debug\\compare_b3_TF_regression.py
"""

from __future__ import annotations

from dataclasses import dataclass
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

from bidsforge.processing.time_frequency_stats_group.regression import (  # noqa: E402
    load_time_frequency_regression_group_result,
)
from compare_subject_config import (  # noqa: E402
    BIDS_ROOT,
    MATLAB_B3_TF_REGRESSION_PATH,
    PYTHON_TF_REGRESSION_GROUP_PATH,
)


TOP_N = 12
PYTHON_BUNDLES = [
    ("condition_a", "condition_a"),
    ("condition_b", "condition_b"),
]
MATLAB_KEYS = {
    "condition_a": ["condition_a_mean_source_t_values", "P_Rating", "pleasant", "accepted"],
    "condition_b": ["condition_b_mean_source_t_values", "UP_Rating", "unpleasant", "rejected"],
}


@dataclass
class MatlabB3TFRegressionExport:
    path: Path
    regions: list[str]
    frequency_hz: np.ndarray
    time_s: np.ndarray
    maps: dict[str, np.ndarray]


def _resolve_python_path(path: Path) -> Path:
    if path.exists():
        return path
    roots = [
        path.parent,
        BIDS_ROOT / "derivatives" / "time_frequency_regression_group",
    ]
    patterns = ["*_desc-tfregressiongroup_stats.h5", "*_desc-tfregressiongroup_stats.mat"]
    candidates = _find_candidates(roots, patterns)
    if candidates:
        print(f"  Python TF regression group default path missing, using: {candidates[0]}")
        _print_other_candidates(candidates)
        return candidates[0]
    raise FileNotFoundError(
        f"Python TF regression group output not found: {path}\n"
        "Run scripts/run_time_frequency_regression_group.py first, or edit "
        "PYTHON_TF_REGRESSION_GROUP_PATH in compare_subject_config.py."
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


def _load_matlab_export(path: Path, py_regions: list[str], py_freq: np.ndarray, py_time: np.ndarray) -> MatlabB3TFRegressionExport | None:
    if not path.exists():
        print("\nMATLAB b3 TF regression numeric export not found.")
        print(f"  Expected optional export: {path}")
        print("  The stock b3_TF_group_parcel_levels_CB.m writes PNGs; export mean tstat maps to compare numerically.")
        return None
    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    root = _first_user_value(mat)
    regions = _string_list(_first_existing(root, mat, ["region", "regions", "roi", "rois"])) or py_regions
    freqs = _float_axis(_first_existing(root, mat, ["frequency_hz", "freqlist", "freqs"]))
    times = _float_axis(_first_existing(root, mat, ["time_s", "timelist", "times"]))
    if freqs.size == 0:
        freqs = py_freq
    if times.size == 0:
        times = py_time
    maps: dict[str, np.ndarray] = {}
    for bundle, names in MATLAB_KEYS.items():
        raw = _first_existing(root, mat, names)
        if raw is not None:
            maps[bundle] = _coerce_group_tf(raw, len(regions), len(freqs), len(times))
    if not maps:
        raise ValueError(
            f"{path} exists but no supported regression map was found. Expected keys like "
            "condition_a_mean_source_t_values, condition_b_mean_source_t_values, P_Rating, UP_Rating."
        )
    return MatlabB3TFRegressionExport(
        path=path,
        regions=regions,
        frequency_hz=np.asarray(freqs, dtype=np.float64),
        time_s=np.asarray(times, dtype=np.float64),
        maps=maps,
    )


def _first_user_value(mat: dict[str, object]) -> object:
    for key, value in mat.items():
        if not key.startswith("__"):
            return value
    return mat


def _first_existing(root: object, mat: dict[str, object], names: list[str]) -> object | None:
    for name in names:
        if hasattr(root, name):
            value = getattr(root, name)
            if _looks_like_struct_with_tstat(value):
                return _struct_map_value(value)
            return value
        if isinstance(root, dict) and name in root:
            return root[name]
        if name in mat:
            return mat[name]
    return None


def _looks_like_struct_with_tstat(value: object) -> bool:
    return hasattr(value, "mean_source_t_values") or hasattr(value, "dots") or hasattr(value, "tstat")


def _struct_map_value(value: object) -> object:
    for name in ("mean_source_t_values", "dots", "tstat", "mean_tstat"):
        if hasattr(value, name):
            return getattr(value, name)
    return value


def _string_list(value: object | None) -> list[str]:
    if value is None:
        return []
    arr = np.asarray(value, dtype=object).ravel()
    return [str(item).strip() for item in arr if str(item).strip()]


def _float_axis(value: object | None) -> np.ndarray:
    if value is None:
        return np.array([])
    arr = np.asarray(value, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _coerce_group_tf(value: object, n_rois: int, n_freqs: int, n_times: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    expected = (n_rois, n_freqs, n_times)
    if arr.shape == expected:
        return arr
    if n_rois == 1 and arr.shape == (n_freqs, n_times):
        return arr.reshape(expected)
    if arr.shape == (n_freqs, n_times, n_rois):
        return np.transpose(arr, (2, 0, 1))
    if arr.shape == (n_times, n_freqs, n_rois):
        return np.transpose(arr, (2, 1, 0))
    if arr.size == n_rois * n_freqs * n_times:
        return arr.reshape(expected)
    raise ValueError(f"Cannot coerce MATLAB map shape {arr.shape!r} to {expected!r}.")


def _python_map(py, bundle: str) -> np.ndarray:
    if bundle == "condition_a":
        return np.asarray(py.source_metric.condition_a.mean, dtype=np.float64)
    return np.asarray(py.source_metric.condition_b.mean, dtype=np.float64)


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_idx: list[int] = []
    tgt_idx: list[int] = []
    for i, value in enumerate(source):
        j = int(np.argmin(np.abs(target - value)))
        if np.isclose(value, target[j], rtol=1e-4, atol=1e-4):
            src_idx.append(i)
            tgt_idx.append(j)
    return np.asarray(src_idx, dtype=np.int64), np.asarray(tgt_idx, dtype=np.int64)


def _pair_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, int]:
    finite = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(finite)) < 3:
        return np.nan, np.nan, np.nan, int(np.sum(finite))
    av = np.asarray(a[finite], dtype=np.float64)
    bv = np.asarray(b[finite], dtype=np.float64)
    corr = float(np.corrcoef(av, bv)[0, 1]) if np.std(av) > 0 and np.std(bv) > 0 else np.nan
    diff = bv - av
    return corr, float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff * diff))), int(av.size)


def _print_summary(py, mat: MatlabB3TFRegressionExport | None) -> None:
    print("\nPython TF regression group")
    print(f"  Regions={len(py.region_names)}, freqs={len(py.frequency_hz)}, times={len(py.time_axis_s)}")
    print(f"  Metric={py.primary_regression_metric}, correction={py.p_value_correction_method}")
    print("  ROI channel counts:")
    for roi, n_ch, n_sub in zip(py.region_names[:TOP_N], py.roi_channel_counts, py.roi_subject_counts, strict=False):
        print(f"    {roi:<24} channels={int(n_ch):>3} subjects={int(n_sub):>3}")
    if mat is None:
        return
    print("\nMATLAB numeric export")
    print(f"  Path={mat.path}")
    print(f"  Regions={len(mat.regions)}, freqs={len(mat.frequency_hz)}, times={len(mat.time_s)}")
    mat_f, py_f = _nearest_indices(mat.frequency_hz, py.frequency_hz)
    mat_t, py_t = _nearest_indices(mat.time_s, py.time_axis_s)
    common_rois = [roi for roi in mat.regions if roi in py.region_names]
    print(f"  Common regions={len(common_rois)}, common grid={len(mat_f)} x {len(mat_t)}")
    for bundle, _label in PYTHON_BUNDLES:
        if bundle not in mat.maps:
            continue
        print(f"  Bundle {bundle}:")
        py_map = _python_map(py, bundle)
        for roi in common_rois[:TOP_N]:
            mi = mat.regions.index(roi)
            pi = py.region_names.index(roi)
            corr, mean_abs, rms, n = _pair_stats(
                mat.maps[bundle][mi][np.ix_(mat_f, mat_t)],
                py_map[pi][np.ix_(py_f, py_t)],
            )
            print(f"    {roi:<24} corr={corr:7.4f} mean|diff|={mean_abs:9.4g} rms={rms:9.4g} n={n}")


class ROIMapViewer:
    def __init__(self, py, mat: MatlabB3TFRegressionExport | None) -> None:
        self.py = py
        self.mat = mat
        self.roi = 0
        self.bundle = 0
        self.fig = plt.figure(figsize=(15, 8))
        self.fig.suptitle("b3 TF regression: mean source t-values")
        self.ax_py = self.fig.add_axes([0.06, 0.22, 0.40, 0.66])
        self.ax_mat = self.fig.add_axes([0.54, 0.22, 0.40, 0.66])
        self.info = self.fig.text(0.06, 0.07, "", fontsize=10)
        self.slider = Slider(
            self.fig.add_axes([0.18, 0.13, 0.62, 0.03]),
            "ROI",
            0,
            max(len(self.py.region_names) - 1, 0),
            valinit=0,
            valstep=1,
        )
        self.slider.on_changed(lambda value: self._set_roi(int(value)))
        self.prev_button = Button(self.fig.add_axes([0.80, 0.12, 0.05, 0.05]), "<")
        self.next_button = Button(self.fig.add_axes([0.86, 0.12, 0.05, 0.05]), ">")
        self.bundle_button = Button(self.fig.add_axes([0.92, 0.12, 0.06, 0.05]), "Map")
        self.prev_button.on_clicked(lambda _event: self.slider.set_val(max(self.roi - 1, 0)))
        self.next_button.on_clicked(lambda _event: self.slider.set_val(min(self.roi + 1, len(self.py.region_names) - 1)))
        self.bundle_button.on_clicked(lambda _event: self._set_bundle((self.bundle + 1) % len(PYTHON_BUNDLES)))
        self._render()

    def _set_roi(self, roi: int) -> None:
        self.roi = roi
        self._render()

    def _set_bundle(self, bundle: int) -> None:
        self.bundle = bundle
        self._render()

    def _render(self) -> None:
        for ax in (self.ax_py, self.ax_mat):
            ax.clear()
        bundle, label = PYTHON_BUNDLES[self.bundle]
        roi_name = self.py.region_names[self.roi]
        py_map = _python_map(self.py, bundle)[self.roi]
        extent = [self.py.time_axis_s[0], self.py.time_axis_s[-1], self.py.frequency_hz[0], self.py.frequency_hz[-1]]
        self.ax_py.imshow(py_map, origin="lower", aspect="auto", extent=extent)
        self.ax_py.set_title(f"Python {label} / {roi_name}")
        self.ax_py.set_xlabel("Time (s)")
        self.ax_py.set_ylabel("Frequency (Hz)")
        if self.mat is not None and roi_name in self.mat.regions and bundle in self.mat.maps:
            mat_i = self.mat.regions.index(roi_name)
            mat_map = self.mat.maps[bundle][mat_i]
            mat_extent = [self.mat.time_s[0], self.mat.time_s[-1], self.mat.frequency_hz[0], self.mat.frequency_hz[-1]]
            self.ax_mat.imshow(mat_map, origin="lower", aspect="auto", extent=mat_extent)
            self.ax_mat.set_title(f"MATLAB {label} / {roi_name}")
        else:
            self.ax_mat.text(0.5, 0.5, "No MATLAB numeric map", ha="center", va="center")
            self.ax_mat.set_title("MATLAB")
        self.ax_mat.set_xlabel("Time (s)")
        self.ax_mat.set_ylabel("Frequency (Hz)")
        self.info.set_text(
            f"ROI {self.roi + 1}/{len(self.py.region_names)}: {roi_name}   "
            f"bundle={label}   channels={int(self.py.roi_channel_counts[self.roi])}   "
            f"subjects={int(self.py.roi_subject_counts[self.roi])}"
        )
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def main() -> None:
    print("Loading Python b3 TF regression group")
    py_path = _resolve_python_path(PYTHON_TF_REGRESSION_GROUP_PATH)
    py = load_time_frequency_regression_group_result(py_path)
    print(f"  Path: {py_path}")
    mat = _load_matlab_export(
        MATLAB_B3_TF_REGRESSION_PATH,
        py.region_names,
        np.asarray(py.frequency_hz, dtype=np.float64),
        np.asarray(py.time_axis_s, dtype=np.float64),
    )
    _print_summary(py, mat)
    print("\nLaunching ROI viewer")
    ROIMapViewer(py, mat).show()


if __name__ == "__main__":
    main()
