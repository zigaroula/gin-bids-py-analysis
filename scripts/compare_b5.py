#!/usr/bin/env python3
"""Compare MATLAB b5 ROI group regression curves against Python outputs.

This script targets the valuation part-1 vmPFC/PFCvm mean-slope figure.
It reads the MATLAB b5 parcel file, extracts the post-b5 ``dots`` matrices
(including the UP sign flip already applied by b5), then loads the matching
Python subject regression HDF5 rows and compares:

- per-channel slope matrices
- ROI mean time courses
- one-sample t-tests vs zero
- paired P-vs-UP t-tests

Run from the repository root:

    .venv\\Scripts\\python scripts\\compare_b5.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable

import h5py
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel

from gin_bids_py_analysis.processing.utils.hdf5 import decode_str_array

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from compare_b3 import _read_h5_char_dataset  # noqa: E402
from trial_slope_shared import BIDS_ROOT  # noqa: E402


MATLAB_B5_PATH = Path(
    r"C:\GRE\dev\clarissa\seeg\b5_BPF_group_analyses_UPlow\PFCvm\log_data.mat"
)
MATLAB_B5_ELECS_CSV = Path(
    r"C:\GRE\dev\clarissa\seeg\b5_BPF_group_analyses_UPlow\PFCvm\PFCvm_elecs_tbl.csv"
)

MATLAB_ROI = "PFCvm"
MATLAB_REALIGN = "onset"
MATLAB_BAND = "f50f150"
MATLAB_SMOOTHING = "sm250"

PYTHON_DESC = "correlation"


@dataclass(frozen=True)
class ComparisonMetrics:
    n: int
    mean_abs: float
    max_abs: float
    rms: float
    corr: float


@dataclass(frozen=True)
class B5Data:
    time_s: np.ndarray
    labels: list[str]
    condition_a: np.ndarray
    condition_b: np.ndarray


def _subject_to_bids(subject: str) -> str:
    return str(subject).strip().replace("_", "")


def _subject_to_matlab(subject: str) -> str:
    text = str(subject).strip()
    if "_" in text:
        return text
    match = re.match(r"^(GRE|LYO|PRA|REN|TOU)(\d{4})([A-Za-z]+)$", text)
    if match is None:
        return text
    return f"{match.group(1)}_{match.group(2)}_{match.group(3)}"


def _first_contact(channel: str) -> str:
    match = re.match(r"^([A-Za-z]+)(\d+)", str(channel).strip())
    if match is None:
        return str(channel).strip()
    prefix, number = match.groups()
    return f"{prefix}{int(number):02d}"


def _norm_channel(channel: str) -> str:
    return re.sub(r"[\s\-_.]", "", _first_contact(channel)).casefold()


def _format_label(subject: str, channel: str) -> str:
    return f"{_subject_to_bids(subject)}/{_first_contact(channel)}"


def _load_b5_labels(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "subname" not in reader.fieldnames or "channel" not in reader.fieldnames:
            raise ValueError(f"{csv_path}: expected columns 'subname' and 'channel'.")
        return [_format_label(row["subname"], row["channel"]) for row in reader]


def _load_b5_data(
    path: Path,
    *,
    labels_csv: Path,
    roi: str,
    realign: str,
    band: str,
    smoothing: str,
) -> B5Data:
    labels = _load_b5_labels(labels_csv)
    with h5py.File(path, "r") as fh:
        base = f"parcel_log/{roi}"
        time_s = np.asarray(
            fh[f"{base}/P_Rating/{realign}/time/timelist"][()],
            dtype=np.float64,
        ).ravel()
        condition_a = np.asarray(
            fh[f"{base}/P_Rating/{realign}/{band}/{smoothing}/dots"][()],
            dtype=np.float64,
        ).T
        condition_b = np.asarray(
            fh[f"{base}/UP_Rating/{realign}/{band}/{smoothing}/dots"][()],
            dtype=np.float64,
        ).T

    if condition_a.shape != condition_b.shape:
        raise ValueError(
            f"MATLAB condition shape mismatch: {condition_a.shape} vs {condition_b.shape}."
        )
    if condition_a.shape[0] != len(labels):
        raise ValueError(
            f"MATLAB b5 rows ({condition_a.shape[0]}) do not match CSV rows ({len(labels)})."
        )
    if condition_a.shape[1] != time_s.size:
        raise ValueError(
            f"MATLAB time length ({time_s.size}) does not match dots width ({condition_a.shape[1]})."
        )

    return B5Data(
        time_s=time_s,
        labels=labels,
        condition_a=condition_a,
        condition_b=condition_b,
    )


def _python_regression_path(subject: str, *, bids_root: Path, desc: str) -> Path:
    bids_subject = _subject_to_bids(subject)
    return (
        bids_root
        / "derivatives"
        / "regression"
        / f"sub-{bids_subject}"
        / "ieeg"
        / f"sub-{bids_subject}_task-MDCHOICE_desc-{desc}_stats.h5"
    )


def _load_python_rows(
    labels: Iterable[str],
    target_time_s: np.ndarray,
    *,
    bids_root: Path,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], float]:
    labels_in = list(labels)
    rows_a: list[np.ndarray] = []
    rows_b: list[np.ndarray] = []
    kept_labels: list[str] = []
    missing_labels: list[str] = []
    max_time_error = 0.0
    cache: dict[str, tuple[np.ndarray, list[str], np.ndarray, np.ndarray]] = {}

    for label in labels_in:
        subject, channel = label.split("/", 1)
        if subject not in cache:
            path = _python_regression_path(subject, bids_root=bids_root, desc=desc)
            if not path.exists():
                missing_labels.append(label)
                continue
            with h5py.File(path, "r") as fh:
                channels = decode_str_array(np.asarray(fh["axes/channel"][:], dtype=object))
                time_s = np.asarray(fh["axes/time_s"][:], dtype=np.float64)
                slope_a = np.asarray(fh["regression/condition_a/slope"][:], dtype=np.float64)
                slope_b = np.asarray(fh["regression/condition_b/slope"][:], dtype=np.float64)
            cache[subject] = (time_s, channels, slope_a, slope_b)

        time_s, channels, slope_a, slope_b = cache[subject]
        channel_index = {
            _norm_channel(name): idx for idx, name in enumerate(channels)
        }.get(_norm_channel(channel))
        if channel_index is None:
            missing_labels.append(label)
            continue

        time_index = np.array(
            [int(np.argmin(np.abs(time_s - t))) for t in target_time_s],
            dtype=int,
        )
        if target_time_s.size:
            max_time_error = max(
                max_time_error,
                float(np.max(np.abs(time_s[time_index] - target_time_s))),
            )
        rows_a.append(slope_a[channel_index, time_index])
        rows_b.append(slope_b[channel_index, time_index])
        kept_labels.append(label)

    if not rows_a:
        n_times = len(target_time_s)
        return (
            np.empty((0, n_times), dtype=np.float64),
            np.empty((0, n_times), dtype=np.float64),
            kept_labels,
            missing_labels,
            max_time_error,
        )

    return (
        np.stack(rows_a, axis=0),
        np.stack(rows_b, axis=0),
        kept_labels,
        missing_labels,
        max_time_error,
    )


def _finite_row_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        return np.array([], dtype=bool)
    mask = np.zeros(arrays[0].shape[0], dtype=bool)
    for arr in arrays:
        mask |= np.isfinite(arr).any(axis=1)
    return mask


def _metrics(a: np.ndarray, b: np.ndarray) -> ComparisonMetrics:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(arr_a) & np.isfinite(arr_b)
    if not valid.any():
        return ComparisonMetrics(0, np.nan, np.nan, np.nan, np.nan)
    diff = arr_a[valid] - arr_b[valid]
    a_flat = arr_a[valid].ravel()
    b_flat = arr_b[valid].ravel()
    corr = np.nan
    if a_flat.size >= 2 and np.nanstd(a_flat) > 0 and np.nanstd(b_flat) > 0:
        corr = float(np.corrcoef(a_flat, b_flat)[0, 1])
    return ComparisonMetrics(
        n=int(valid.sum()),
        mean_abs=float(np.nanmean(np.abs(diff))),
        max_abs=float(np.nanmax(np.abs(diff))),
        rms=float(np.sqrt(np.nanmean(diff * diff))),
        corr=corr,
    )


def _nansem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    count = np.sum(np.isfinite(arr), axis=axis)
    std = np.nanstd(arr, axis=axis, ddof=1)
    sem = std / np.sqrt(np.maximum(count, 1))
    return np.where(count >= 2, sem, np.nan)


def _one_sample_t(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stats = ttest_1samp(rows, popmean=0.0, axis=0, nan_policy="omit")
    return (
        np.asarray(stats.statistic, dtype=np.float64),
        np.asarray(stats.pvalue, dtype=np.float64),
    )


def _paired_t(rows_a: np.ndarray, rows_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t_values = np.full(rows_a.shape[1], np.nan, dtype=np.float64)
    p_values = np.full(rows_a.shape[1], np.nan, dtype=np.float64)
    for time_idx in range(rows_a.shape[1]):
        a_col = rows_a[:, time_idx]
        b_col = rows_b[:, time_idx]
        valid = np.isfinite(a_col) & np.isfinite(b_col)
        if int(valid.sum()) < 2:
            continue
        stats = ttest_rel(a_col[valid], b_col[valid], nan_policy="omit")
        t_values[time_idx] = float(np.asarray(stats.statistic, dtype=np.float64))
        p_values[time_idx] = float(np.asarray(stats.pvalue, dtype=np.float64))
    return t_values, p_values


def _print_metrics(label: str, metrics: ComparisonMetrics) -> None:
    print(
        f"{label:<28} n={metrics.n:<8d} "
        f"mean|d|={metrics.mean_abs:.8g}  "
        f"max|d|={metrics.max_abs:.8g}  "
        f"rms={metrics.rms:.8g}  corr={metrics.corr:.8g}"
    )


def _print_removed_channel_diagnostic() -> None:
    """Print why the known REN channels are absent from MATLAB b5 PFCvm."""
    b3_path = (
        Path(r"C:\GRE\dev\clarissa\seeg\b3_BPF_indiv_analyses")
        / "REN_2021_SIDs"
        / "regressions"
        / "P_Rating"
        / "log_data.mat"
    )
    targets = {
        "Op02",
        "Op03",
        "Yp02",
        "Yp03",
        "Yp04",
        "Yp05",
        "Yp06",
        "Yp07",
        "Yp08",
        "Yp09",
        "Yp10",
    }
    with h5py.File(b3_path, "r") as fh:
        chan_info_refs = np.asarray(fh["hdr/chan_info"][()]).T
        roi_refs = np.asarray(fh["hdr/ROI_chans"][()]).T
        roi_members: set[str] = set()
        target_keys = {_norm_channel(target) for target in targets}
        for row in roi_refs:
            roi_name = _read_h5_char_dataset(fh[row[0]])
            if roi_name != MATLAB_ROI:
                continue
            indices = np.asarray(fh[row[1]][()]).ravel()
            labels = [_read_h5_char_dataset(fh[chan_info_refs[int(i) - 1, 1]]) for i in indices]
            roi_members.update(_norm_channel(label) for label in labels)

        print("\nREN_2021_SIDs ROI-channel diagnostic:")
        for idx, row in enumerate(chan_info_refs, start=1):
            label = _read_h5_char_dataset(fh[row[1]])
            first = _first_contact(label)
            if _norm_channel(first) not in target_keys:
                continue
            roi_value = _read_h5_char_dataset(fh[row[2]]).replace("\x00", "")
            in_roi = _norm_channel(label) in roi_members
            shown_roi = roi_value if roi_value else "<empty>"
            print(
                f"  {first:<5} ({label:<10}) idx={idx:<3d} "
                f"hdr.chan_info ROI={shown_roi:<8} in hdr.ROI_chans={in_roi}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab-b5", type=Path, default=MATLAB_B5_PATH)
    parser.add_argument("--matlab-elecs-csv", type=Path, default=MATLAB_B5_ELECS_CSV)
    parser.add_argument("--bids-root", type=Path, default=BIDS_ROOT)
    parser.add_argument("--desc", default=PYTHON_DESC)
    parser.add_argument("--include-all-matlab-rows", action="store_true")
    parser.add_argument("--diagnose-ren-roi", action="store_true")
    args = parser.parse_args()

    b5 = _load_b5_data(
        args.matlab_b5,
        labels_csv=args.matlab_elecs_csv,
        roi=MATLAB_ROI,
        realign=MATLAB_REALIGN,
        band=MATLAB_BAND,
        smoothing=MATLAB_SMOOTHING,
    )
    finite_matlab_rows = _finite_row_mask(b5.condition_a, b5.condition_b)
    row_mask = (
        np.ones(len(b5.labels), dtype=bool)
        if args.include_all_matlab_rows
        else finite_matlab_rows
    )
    selected_labels = [label for label, keep in zip(b5.labels, row_mask) if bool(keep)]

    py_a, py_b, kept_labels, missing_labels, max_time_error = _load_python_rows(
        selected_labels,
        b5.time_s,
        bids_root=args.bids_root,
        desc=args.desc,
    )
    kept_lookup = {label: idx for idx, label in enumerate(kept_labels)}
    keep_both = [idx for idx, label in enumerate(selected_labels) if label in kept_lookup]
    py_order = [kept_lookup[label] for label in selected_labels if label in kept_lookup]

    mat_a = b5.condition_a[row_mask][keep_both]
    mat_b = b5.condition_b[row_mask][keep_both]
    py_a = py_a[py_order]
    py_b = py_b[py_order]

    print(f"MATLAB b5 file : {args.matlab_b5}")
    print(f"MATLAB rows    : {len(b5.labels)} total, {int(finite_matlab_rows.sum())} finite")
    print(f"Compared rows  : {mat_a.shape[0]} ({len(missing_labels)} Python-missing)")
    print(f"Time axis      : {b5.time_s[0]:.3f} -> {b5.time_s[-1]:.3f}s ({len(b5.time_s)} samples)")
    print(f"Max time error : {max_time_error:.9g}s")
    if missing_labels:
        print("Missing Python labels:")
        for label in missing_labels:
            print(f"  {label}")

    print("\nMatrix comparisons:")
    _print_metrics("P rows", _metrics(mat_a, py_a))
    _print_metrics("UP rows", _metrics(mat_b, py_b))

    mat_mean_a = np.nanmean(mat_a, axis=0)
    mat_mean_b = np.nanmean(mat_b, axis=0)
    py_mean_a = np.nanmean(py_a, axis=0)
    py_mean_b = np.nanmean(py_b, axis=0)
    mat_sem_a = _nansem(mat_a, axis=0)
    mat_sem_b = _nansem(mat_b, axis=0)
    py_sem_a = _nansem(py_a, axis=0)
    py_sem_b = _nansem(py_b, axis=0)

    print("\nMean/SEM comparisons:")
    _print_metrics("P mean", _metrics(mat_mean_a, py_mean_a))
    _print_metrics("UP mean", _metrics(mat_mean_b, py_mean_b))
    _print_metrics("P SEM", _metrics(mat_sem_a, py_sem_a))
    _print_metrics("UP SEM", _metrics(mat_sem_b, py_sem_b))

    mat_t_a, mat_p_a = _one_sample_t(mat_a)
    mat_t_b, mat_p_b = _one_sample_t(mat_b)
    py_t_a, py_p_a = _one_sample_t(py_a)
    py_t_b, py_p_b = _one_sample_t(py_b)
    mat_t_contrast, mat_p_contrast = _paired_t(mat_a, mat_b)
    py_t_contrast, py_p_contrast = _paired_t(py_a, py_b)

    print("\nT-test comparisons:")
    _print_metrics("P vs zero t", _metrics(mat_t_a, py_t_a))
    _print_metrics("P vs zero p", _metrics(mat_p_a, py_p_a))
    _print_metrics("UP vs zero t", _metrics(mat_t_b, py_t_b))
    _print_metrics("UP vs zero p", _metrics(mat_p_b, py_p_b))
    _print_metrics("P vs UP paired t", _metrics(mat_t_contrast, py_t_contrast))
    _print_metrics("P vs UP paired p", _metrics(mat_p_contrast, py_p_contrast))

    print("\nSelected time points:")
    for target in (0.0, 0.5, 1.0, 2.0, 3.0, 3.5):
        idx = int(np.argmin(np.abs(b5.time_s - target)))
        print(
            f"  t={b5.time_s[idx]:>4.2f}s  "
            f"P matlab={mat_mean_a[idx]: .8f} python={py_mean_a[idx]: .8f}  "
            f"UP matlab={mat_mean_b[idx]: .8f} python={py_mean_b[idx]: .8f}"
        )

    if args.diagnose_ren_roi:
        _print_removed_channel_diagnostic()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
