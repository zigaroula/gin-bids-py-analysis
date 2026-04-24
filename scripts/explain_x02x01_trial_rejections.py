#!/usr/bin/env python3
"""Explain trial rejection decisions for one channel in MATLAB b3 vs Python.

This is a focused diagnostic for the REN_2021_SIDs / X02X01 mismatch seen in
``compare_b3.py``.  It reports, for each b3 regression condition, which trials
are unusable for the target channel and which comparison caused that decision:

* MATLAB side: reads b2 ``opts_log`` plus b3 ``log_data.mat`` regressor values.
* Python side: reruns the regression processor with an instrumented subclass
  that captures epoch-cleaning mean/max values and thresholds before masking.

Usage:
    .venv\\Scripts\\python scripts\\explain_x02x01_trial_rejections.py
    .venv\\Scripts\\python scripts\\explain_x02x01_trial_rejections.py --channel X02X01
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import sys
from pathlib import Path
from typing import Any
import warnings

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from compare_b2 import load_matlab as load_matlab_b2_with_opts  # noqa: E402
from compare_b3 import (  # noqa: E402
    MATLAB_LOG_DATA_FILENAME,
    load_matlab_b3_regressor,
)
from compare_subject_config import (  # noqa: E402
    BEHAVIOR_TSV_PATH,
    MATLAB_B2_PATH,
    MATLAB_B3_ROOT,
    PYTHON_REGRESSION_PATH,
)
from gin_bids_py_analysis.bids import BIDSDataset  # noqa: E402
from gin_bids_py_analysis.processing.trial_stats.regression import RegressionProcessing  # noqa: E402
from gin_bids_py_analysis.processing.utils.epoch_quality import (  # noqa: E402
    apply_trial_nan_mask,
    detect_outlier_trial_channel_pairs_by_mean,
    reject_channels_by_nan_trial_ratio,
    reject_channels_by_trial_max_spread,
    reject_channels_by_trial_mean_spread,
)
from trial_slope_shared import (  # noqa: E402
    BIDS_ROOT as PYTHON_SOURCE_BIDS_ROOT,
    PARAMS as PYTHON_REGRESSION_PARAMS,
    RESOLVER as PYTHON_REGRESSION_RESOLVER,
    build_trial_annotators,
    build_trial_slope_groups,
    load_roi_channels_from_csv,
)


REGRESSIONS = [
    ("P_Rating", "pleasant", "condition_a", 1),
    ("UP_Rating", "unpleasant", "condition_b", 2),
]


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).casefold() if ch.isalnum())


def _first_contact(name: str) -> str:
    import re

    match = re.match(r"^([A-Za-z]+\d+)", str(name).strip())
    return match.group(1) if match else str(name).strip()


def _fmt(value: object, ndigits: int = 6) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "NaN"
    return f"{val:.{ndigits}g}"


def _bounds(values: np.ndarray, threshold: float) -> tuple[float, float, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return np.nan, np.nan, np.nan, np.nan
    center = float(np.nanmean(finite))
    spread = float(np.nanstd(finite, ddof=1))
    return center, spread, center - threshold * spread, center + threshold * spread


def _trial_desc(trial: Any, *, condition_local_index: int | None = None) -> str:
    metadata = getattr(trial, "metadata", {}) or {}
    label_row = metadata.get("label_row_index")
    event_row = metadata.get("event_row_index")
    parts = []
    if condition_local_index is not None:
        parts.append(f"cond#{condition_local_index + 1}")
    if label_row is not None:
        parts.append(f"beh_row#{int(label_row) + 1}")
    if event_row is not None:
        parts.append(f"event_row#{int(event_row) + 1}")
    if getattr(trial, "trial_id", None):
        parts.append(f"trial_id={trial.trial_id}")
    parts.append(f"anchor#{int(getattr(trial, 'anchor_event_index', -1)) + 1}")
    return " ".join(parts)


def _find_channel_index(channels: list[str], target: str) -> int:
    target_norm = _norm(target)
    target_first_norm = _norm(_first_contact(target))
    for idx, name in enumerate(channels):
        if _norm(name) == target_norm:
            return idx
    for idx, name in enumerate(channels):
        if _norm(_first_contact(name)) == target_first_norm or _norm(name) == target_first_norm:
            return idx
    raise ValueError(f"Could not find channel {target!r}. Available examples: {channels[:12]!r}")


def _load_behavior_pleasantness() -> np.ndarray:
    import csv

    values: list[int] = []
    with BEHAVIOR_TSV_PATH.open("r", encoding="utf-8-sig", newline="") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")
        for row in reader:
            values.append(int(float(row["pleasant"])))
    return np.asarray(values, dtype=int)


@dataclass
class PythonCleaningDebug:
    feature_names_before: list[str]
    target_feature_before: str
    target_idx_before: int
    pooled_trials_before: list[Any]
    n_a_before: int
    mean_values: np.ndarray
    mean_center: float
    mean_std: float
    mean_low: float
    mean_high: float
    mean_mask: np.ndarray
    max_values_after_mean: np.ndarray
    max_center: float
    max_std: float
    max_low: float
    max_high: float
    max_mask: np.ndarray
    channel_mean_spread: float
    channel_max_spread: float
    channel_reasons: list[str]
    feature_names_after: list[str]
    target_idx_after: int | None
    kept_trials_a_after: list[Any]
    kept_trials_b_after: list[Any]


class DebugRegressionProcessing(RegressionProcessing):
    def __init__(self, *args: Any, target_channel: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.target_channel = target_channel
        self.debug: PythonCleaningDebug | None = None

    def _prepare_epochs_before_activity_zscore(self, **kwargs: Any):  # type: ignore[override]
        epochs_a = np.asarray(kwargs["epochs_a"], dtype=np.float64)
        epochs_b = np.asarray(kwargs["epochs_b"], dtype=np.float64)
        kept_trials_a = list(kwargs["kept_trials_a"])
        kept_trials_b = list(kwargs["kept_trials_b"])
        feature_names = list(kwargs["feature_names"])
        cfg = self.params.epoch_cleaning

        n_a = int(epochs_a.shape[0])
        pooled = np.concatenate([epochs_a, epochs_b], axis=0)
        target_idx = _find_channel_index(feature_names, self.target_channel)

        mean_values = np.full(pooled.shape[0], np.nan, dtype=np.float64)
        mean_center = mean_std = mean_low = mean_high = np.nan
        mean_mask = np.zeros(pooled.shape[0], dtype=bool)
        all_mean_mask = np.zeros((pooled.shape[0], pooled.shape[1]), dtype=bool)
        max_values = np.full(pooled.shape[0], np.nan, dtype=np.float64)
        max_center = max_std = max_low = max_high = np.nan
        max_mask = np.zeros(pooled.shape[0], dtype=bool)
        channel_mean_spread = np.nan
        channel_max_spread = np.nan
        channel_reasons: list[str] = []

        working = pooled.copy()
        if cfg.reject_trials_by_epoch_mean and pooled.size:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                all_means = np.nanmean(working, axis=2)
            mean_values = np.asarray(all_means[:, target_idx], dtype=np.float64)
            mean_center, mean_std, mean_low, mean_high = _bounds(
                mean_values,
                float(cfg.epoch_mean_threshold_factor),
            )
            all_mean_mask = detect_outlier_trial_channel_pairs_by_mean(
                working,
                threshold_factor=float(cfg.epoch_mean_threshold_factor),
            )
            mean_mask = np.asarray(all_mean_mask[:, target_idx], dtype=bool)
            working = apply_trial_nan_mask(working, all_mean_mask)

        if cfg.reject_trials_by_epoch_max and pooled.size:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                all_max = np.nanmax(np.abs(working), axis=2)
            max_values = np.asarray(all_max[:, target_idx], dtype=np.float64)
            max_center, max_std, max_low, max_high = _bounds(
                max_values,
                float(cfg.epoch_max_threshold_factor),
            )
            valid_std = np.nanstd(all_max, axis=0, ddof=1) > 0.0
            all_max_center = np.nanmean(all_max, axis=0)
            all_max_std = np.nanstd(all_max, axis=0, ddof=1)
            all_max_mask = np.zeros(all_max.shape, dtype=bool)
            all_max_mask[:, valid_std] = (
                np.abs(all_max[:, valid_std] - all_max_center[np.newaxis, valid_std])
                > float(cfg.epoch_max_threshold_factor) * all_max_std[np.newaxis, valid_std]
            )
            if cfg.reject_trials_by_epoch_mean:
                all_max_mask &= ~all_mean_mask
            max_mask = np.asarray(all_max_mask[:, target_idx], dtype=bool)
            working = apply_trial_nan_mask(working, all_max_mask)

        if pooled.size:
            if cfg.reject_by_trial_mean_spread:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    trial_means = np.nanmean(working, axis=2)
                spreads = np.nanstd(trial_means, axis=0, ddof=1)
                channel_mean_spread = float(spreads[target_idx])
                if reject_channels_by_trial_mean_spread(
                    working,
                    threshold_factor=float(cfg.trial_mean_spread_threshold),
                )[target_idx]:
                    channel_reasons.append("trial_mean_spread")
                    working[:, target_idx, :] = np.nan
            if cfg.reject_by_trial_max_spread:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    trial_maxes = np.nanmax(np.abs(working), axis=2)
                spreads = np.nanstd(trial_maxes, axis=0, ddof=1)
                channel_max_spread = float(spreads[target_idx])
                if reject_channels_by_trial_max_spread(
                    working,
                    threshold_factor=float(cfg.trial_max_spread_threshold),
                )[target_idx]:
                    channel_reasons.append("trial_max_spread")
            if cfg.max_nan_trial_ratio is not None:
                if reject_channels_by_nan_trial_ratio(working, float(cfg.max_nan_trial_ratio))[target_idx]:
                    channel_reasons.append("nan_trial_ratio")

        out = super()._prepare_epochs_before_activity_zscore(**kwargs)
        feature_names_after = list(out[2])
        try:
            target_idx_after = _find_channel_index(feature_names_after, self.target_channel)
        except ValueError:
            target_idx_after = None

        self.debug = PythonCleaningDebug(
            feature_names_before=feature_names,
            target_feature_before=feature_names[target_idx],
            target_idx_before=target_idx,
            pooled_trials_before=kept_trials_a + kept_trials_b,
            n_a_before=n_a,
            mean_values=mean_values,
            mean_center=mean_center,
            mean_std=mean_std,
            mean_low=mean_low,
            mean_high=mean_high,
            mean_mask=mean_mask,
            max_values_after_mean=max_values,
            max_center=max_center,
            max_std=max_std,
            max_low=max_low,
            max_high=max_high,
            max_mask=max_mask,
            channel_mean_spread=channel_mean_spread,
            channel_max_spread=channel_max_spread,
            channel_reasons=channel_reasons,
            feature_names_after=feature_names_after,
            target_idx_after=target_idx_after,
            kept_trials_a_after=list(kwargs["kept_trials_a"]),
            kept_trials_b_after=list(kwargs["kept_trials_b"]),
        )
        return out


def _load_fresh_python_result(target_channel: str) -> tuple[Any, PythonCleaningDebug]:
    manual_region_channels = load_roi_channels_from_csv()
    annotators = build_trial_annotators(manual_region_channels)
    ds = BIDSDataset(PYTHON_SOURCE_BIDS_ROOT)
    target_subject = PYTHON_REGRESSION_PATH.parent.parent.name.removeprefix("sub-")
    for group in build_trial_slope_groups(ds):
        if str(group.primary.get("subject", "")).strip() != target_subject:
            continue
        processor = DebugRegressionProcessing(
            PYTHON_REGRESSION_PARAMS,
            resolver=PYTHON_REGRESSION_RESOLVER,
            annotators=annotators,
            target_channel=target_channel,
        )
        result = processor.process_group(group)
        if processor.debug is None:
            raise RuntimeError("Python debug hook did not capture epoch cleaning.")
        return result, processor.debug
    raise RuntimeError(f"Could not find Python BIDS group for subject {target_subject!r}.")


def _print_matlab_section(channel: str) -> None:
    print("\n=== MATLAB b2/b3 effective trial use ===")
    mat_data, mat_channels, _sfreq, mat_rej, _timelist = load_matlab_b2_with_opts(MATLAB_B2_PATH)
    mat_ch = _find_channel_index(mat_channels, channel)
    print(f"MATLAB channel: {mat_channels[mat_ch]} (index {mat_ch + 1})")

    mean_vals = np.asarray(mat_rej.get("mean_trial_vals", np.empty((0, 0))), dtype=np.float64)
    max_vals = np.asarray(mat_rej.get("max_trial_vals", np.empty((0, 0))), dtype=np.float64)
    mean_mask = np.asarray(mat_rej.get("outlier_trials_mean", np.zeros((mat_data.shape[0], len(mat_channels)))), dtype=bool)
    max_mask = np.asarray(mat_rej.get("outlier_trials_max", np.zeros((mat_data.shape[0], len(mat_channels)))), dtype=bool)
    neg = set(int(i) for i in np.asarray(mat_rej.get("neg_rating_indices", []), dtype=int).ravel())
    rt = set(int(i) for i in np.asarray(mat_rej.get("outlier_rt_indices", []), dtype=int).ravel())
    pleasantness = _load_behavior_pleasantness()

    if mean_vals.size:
        c, s, lo, hi = _bounds(mean_vals[:, mat_ch], 3.0)
        print(f"Mean criterion: center={_fmt(c)} std={_fmt(s)} bounds=[{_fmt(lo)}, {_fmt(hi)}]")
    if max_vals.size:
        c, s, lo, hi = _bounds(max_vals[:, mat_ch], 3.0)
        print(f"Max criterion : center={_fmt(c)} std={_fmt(s)} bounds=[{_fmt(lo)}, {_fmt(hi)}]")

    for reg_name, _label, _field, pleasant_value in REGRESSIONS:
        path = MATLAB_B3_ROOT / reg_name / MATLAB_LOG_DATA_FILENAME
        regressor = load_matlab_b3_regressor(path, regression_name=reg_name)
        # MATLAB b3 builds trial_Idx = find(pleasantness == 1/2); the regressor
        # vector keeps that condition order.  Map condition rows back to original
        # behavior/b2 trial indices before reading opts_log masks.
        original_indices = np.flatnonzero(pleasantness == pleasant_value)
        if original_indices.size != regressor.size:
            print(
                f"\n{reg_name}: WARNING condition rows mismatch "
                f"(behavior={original_indices.size}, regressor={regressor.size})"
            )
            n_rows = min(original_indices.size, regressor.size)
            original_indices = original_indices[:n_rows]
            regressor = regressor[:n_rows]
        print(f"\n{reg_name}: {regressor.size} condition rows in MATLAB log_data")
        rejected_any = False
        for cond_i, orig_i in enumerate(original_indices):
            data_is_nan = orig_i >= mat_data.shape[0] or not np.isfinite(mat_data[orig_i, 0, mat_ch])
            reg_is_nan = not np.isfinite(regressor[cond_i])
            reasons: list[str] = []
            if int(orig_i) in neg:
                reasons.append("opts_log.negratings")
            if int(orig_i) in rt:
                reasons.append("opts_log.outlierRTs")
            if orig_i < mean_mask.shape[0] and bool(mean_mask[orig_i, mat_ch]):
                reasons.append("OutlierTrialsMean")
            if orig_i < max_mask.shape[0] and bool(max_mask[orig_i, mat_ch]):
                reasons.append("OutlierTrialsMax")
            if reg_is_nan:
                reasons.append("regressor=NaN")
            if data_is_nan:
                reasons.append("BGA=NaN")
            if not reasons:
                continue
            rejected_any = True
            mv = mean_vals[orig_i, mat_ch] if mean_vals.size and orig_i < mean_vals.shape[0] else np.nan
            xv = max_vals[orig_i, mat_ch] if max_vals.size and orig_i < max_vals.shape[0] else np.nan
            print(
                f"  cond#{cond_i + 1} beh_row#{int(orig_i) + 1}: reject {', '.join(reasons)} | "
                f"mean={_fmt(mv)} max={_fmt(xv)} regressor={_fmt(regressor[cond_i])}"
            )
        if not rejected_any:
            print("  no rejected rows detected for this channel/regressor")


def _print_python_section(channel: str) -> None:
    print("\n=== Python regression effective trial use ===")
    result, dbg = _load_fresh_python_result(channel)
    print(
        f"Python target feature before cleaning: {dbg.target_feature_before} "
        f"(index {dbg.target_idx_before + 1})"
    )
    print(
        "Mean criterion: "
        f"center={_fmt(dbg.mean_center)} std={_fmt(dbg.mean_std)} "
        f"bounds=[{_fmt(dbg.mean_low)}, {_fmt(dbg.mean_high)}]"
    )
    print(
        "Max criterion : "
        f"center={_fmt(dbg.max_center)} std={_fmt(dbg.max_std)} "
        f"bounds=[{_fmt(dbg.max_low)}, {_fmt(dbg.max_high)}]"
    )
    if dbg.channel_reasons:
        print(
            "Channel-level exclusion for target: "
            f"{', '.join(dbg.channel_reasons)} "
            f"(mean_spread={_fmt(dbg.channel_mean_spread)}, "
            f"max_spread={_fmt(dbg.channel_max_spread)})"
        )
    else:
        print(
            "Channel-level exclusion for target: none "
            f"(mean_spread={_fmt(dbg.channel_mean_spread)}, "
            f"max_spread={_fmt(dbg.channel_max_spread)})"
        )

    for reg_name, label, field, _pleasant_value in REGRESSIONS:
        print(f"\n{reg_name} / Python {field} ({label})")
        start = 0 if field == "condition_a" else dbg.n_a_before
        stop = dbg.n_a_before if field == "condition_a" else len(dbg.pooled_trials_before)
        rejected_any = False
        for pooled_i in range(start, stop):
            reasons: list[str] = []
            if bool(dbg.mean_mask[pooled_i]):
                reasons.append("epoch_mean")
            if bool(dbg.max_mask[pooled_i]):
                reasons.append("epoch_max")
            if not reasons:
                continue
            rejected_any = True
            cond_i = pooled_i - start
            print(
                f"  {_trial_desc(dbg.pooled_trials_before[pooled_i], condition_local_index=cond_i)}: "
                f"reject {', '.join(reasons)} | "
                f"mean={_fmt(dbg.mean_values[pooled_i])} "
                f"[{_fmt(dbg.mean_low)}, {_fmt(dbg.mean_high)}], "
                f"max={_fmt(dbg.max_values_after_mean[pooled_i])} "
                f"[{_fmt(dbg.max_low)}, {_fmt(dbg.max_high)}]"
            )
        if not rejected_any:
            print("  no target-channel epoch mean/max rejections")

        epochs = np.asarray(getattr(result, f"{field}_epochs"), dtype=np.float64)
        predictor = np.asarray(getattr(result, f"{field}_predictor_values"), dtype=np.float64).ravel()
        if dbg.target_idx_after is None:
            print("  target feature is absent after channel exclusion, so no rows reach regression")
            continue
        if epochs.size == 0:
            print("  no epochs reach regression")
            continue
        valid = np.isfinite(predictor) & np.isfinite(epochs[:, dbg.target_idx_after, 0])
        print(
            f"  rows reaching regression for target: {int(np.sum(valid))}/{valid.size}; "
            f"condition epochs stored={epochs.shape[0]}"
        )

    excluded = [
        trial
        for trial in getattr(result, "resolved_trials", [])
        if not getattr(trial, "keep", True)
    ]
    behavioral = [
        trial for trial in excluded
        if getattr(trial, "exclusion_reason", None) == "behavioral_threshold"
    ]
    if behavioral:
        print("\nPython trials excluded before epoch cleaning by behavioral_threshold:")
        for trial in behavioral:
            md = getattr(trial, "metadata", {}) or {}
            print(
                f"  label={getattr(trial, 'label', None)} {_trial_desc(trial)} | "
                f"rating={_fmt(md.get('rating'))} >= 0, RT={_fmt(md.get('RT'))} <= 20"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="X02X01")
    args = parser.parse_args()

    print(f"Diagnostic target channel: {args.channel}")
    _print_matlab_section(args.channel)
    _print_python_section(args.channel)


if __name__ == "__main__":
    main()
