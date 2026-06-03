from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from bidsforge.bids import BIDSDataset, BIDSFile, BIDSFileGroup, build_subject_groups
from bidsforge.bids.helpers import normalize_subject_value
from bidsforge.processing.time_frequency_stats.condition_test import (
    TimeFrequencyConditionTestParams,
    TimeFrequencyConditionTestWriter,
    TimeFrequencyConditionTestWriterParams,
)
from bidsforge.processing.time_frequency_stats.regression import (
    TimeFrequencyRegressionParams,
    TimeFrequencyRegressionWriter,
    TimeFrequencyRegressionWriterParams,
)
from bidsforge.processing.utils.events import AnnotationEvent
from bidsforge.processing.utils.trial_resolver import ResolvedTrial

from trial_slope_shared import (
    ANCHOR_EVENT_CODES,
    BIDS_ROOT,
    MATLAB_ZSCORES_PATH,
    MAX_RT_S,
    MIN_RATING,
    RESOLVER,
    SECONDARY_FILTERS,
    SUBJECT,
    TIME_FREQUENCY_OUTPUT_DESCRIPTION,
    TIME_FREQUENCY_OUTPUT_FORMAT,
    TRIAL_SLOPE_CONDITION_A,
    TRIAL_SLOPE_CONDITION_B,
    TRIAL_SLOPE_P_VALUE_CORRECTION_METHOD,
    TRIAL_SLOPE_PREDICTOR_WITH_MATLAB_ZSCORES,
    TRIAL_SLOPE_PREDICTOR_WITHOUT_MATLAB_ZSCORES,
    TRIAL_SLOPE_PREDICTOR_ZSCORE,
    TRIAL_SLOPE_SIGNIFICANCE_ALPHA,
    USE_MATLAB_ZSCORES,
    load_matlab_zscores,
    map_bids_subject_to_source,
)


MATLAB_SEEG_ROOT = Path(r"C:\GRE\dev\clarissa\seeg")

CONDITION_TEST_OUTPUT_DESCRIPTION = "tfconditiontestonset"
REGRESSION_OUTPUT_DESCRIPTION = "tfregressiononset"
OUTPUT_FORMAT = "hdf5"

# MATLAB b2_TF CB scripts use strict boundaries and trim 0.5 s at both ends.
TIME_WINDOW_S = (-0.5, 5.5)

# Direct b2_TF CB comparison: the inspected MATLAB TF regression branch keeps
# all 120 pleasant and all 120 unpleasant trials. It does not apply the BGA b2
# negative-rating/RT filters before TF glmfit.
APPLY_BEHAVIORAL_EXCLUSIONS = False

MATLAB_CONTRAST_NAME = os.environ.get("MATLAB_TF_CONTRAST_NAME", "T_swh_nswh")
MATLAB_CONDITION_REALIGN_NAME = os.environ.get("MATLAB_TF_CONDITION_REALIGN_NAME", "stimulus")
MATLAB_REGRESSION_REALIGN_NAME = os.environ.get("MATLAB_TF_REGRESSION_REALIGN_NAME", "onset")

MATLAB_REGRESSOR_BY_CONDITION = {
    TRIAL_SLOPE_CONDITION_A: os.environ.get("MATLAB_TF_REGRESSOR_A", "P_Rating"),
    TRIAL_SLOPE_CONDITION_B: os.environ.get("MATLAB_TF_REGRESSOR_B", "UP_Rating"),
}

# Direct CB/CB_R1 Matlab comparison: both branches use rating_Zsc directly.
MATLAB_CB_PREDICTOR_TRANSFORM_BY_CONDITION = {
    TRIAL_SLOPE_CONDITION_A: {"scale": 1.0, "offset": 0.0},
    TRIAL_SLOPE_CONDITION_B: {"scale": 1.0, "offset": 0.0},
}


class TrialSlopeConditionResolver:
    @property
    def condition_labels(self) -> tuple[str, ...]:
        return tuple(RESOLVER.condition_labels)

    def resolve_trials(
        self,
        group: BIDSFileGroup,
        ieeg_file: BIDSFile,
        anchor_events: Sequence[AnnotationEvent],
    ) -> list[ResolvedTrial]:
        trials = RESOLVER.resolve_trials(group, ieeg_file, anchor_events)
        if not APPLY_BEHAVIORAL_EXCLUSIONS:
            return trials
        return [_apply_behavioral_exclusions(trial) for trial in trials]


class TrialSlopeTFRegressionResolver:
    def __init__(self) -> None:
        self.zscores_by_subject = (
            load_matlab_zscores(MATLAB_ZSCORES_PATH, bids_root=BIDS_ROOT)
            if USE_MATLAB_ZSCORES
            else {}
        )

    @property
    def condition_labels(self) -> tuple[str, ...]:
        return tuple(RESOLVER.condition_labels)

    def resolve_trials(
        self,
        group: BIDSFileGroup,
        ieeg_file: BIDSFile,
        anchor_events: Sequence[AnnotationEvent],
    ) -> list[ResolvedTrial]:
        trials = RESOLVER.resolve_trials(group, ieeg_file, anchor_events)
        subject_id = normalize_subject_value(ieeg_file.get("subject", ""))
        zscores = self.zscores_by_subject.get(subject_id)
        out: list[ResolvedTrial] = []
        kept_order = 0
        for trial in trials:
            metadata = dict(trial.metadata)
            if USE_MATLAB_ZSCORES:
                if trial.keep and zscores is not None and kept_order < len(zscores):
                    metadata[TRIAL_SLOPE_PREDICTOR_WITH_MATLAB_ZSCORES] = float(
                        zscores[kept_order]
                    )
                else:
                    metadata[TRIAL_SLOPE_PREDICTOR_WITH_MATLAB_ZSCORES] = np.nan
                if trial.keep:
                    kept_order += 1
            resolved = replace(trial, metadata=metadata)
            if APPLY_BEHAVIORAL_EXCLUSIONS:
                resolved = _apply_behavioral_exclusions(resolved)
            out.append(resolved)
        return out


def build_tfr_stats_groups(dataset: BIDSDataset) -> list[BIDSFileGroup]:
    filters = {
        "scope": "time_frequency",
        "datatype": "ieeg",
        "suffix": "tfr",
        "extension": _tfr_extension(),
        "desc": TIME_FREQUENCY_OUTPUT_DESCRIPTION,
    }
    if SUBJECT is not None:
        filters["subject"] = normalize_subject_value(SUBJECT)
    return build_subject_groups(dataset, filters, SECONDARY_FILTERS, aggregate_runs=False)


def build_condition_test_params() -> TimeFrequencyConditionTestParams:
    return TimeFrequencyConditionTestParams(
        anchor_event_codes=list(ANCHOR_EVENT_CODES),
        time_window_s=TIME_WINDOW_S,
        time_selection="strict_matlab",
        power_mode="stored",
        condition_a=TRIAL_SLOPE_CONDITION_A,
        condition_b=TRIAL_SLOPE_CONDITION_B,
        min_trials_per_condition=2,
        equal_var=False,
        compute_grand_average=True,
        p_value_correction_method=TRIAL_SLOPE_P_VALUE_CORRECTION_METHOD,
        significance_alpha=TRIAL_SLOPE_SIGNIFICANCE_ALPHA,
        n_permutations=0,
    )


def build_regression_params() -> TimeFrequencyRegressionParams:
    return TimeFrequencyRegressionParams(
        anchor_event_codes=list(ANCHOR_EVENT_CODES),
        time_window_s=TIME_WINDOW_S,
        time_selection="strict_matlab",
        power_mode="stored",
        condition_a=TRIAL_SLOPE_CONDITION_A,
        condition_b=TRIAL_SLOPE_CONDITION_B,
        min_trials_per_condition=3,
        predictor=(
            TRIAL_SLOPE_PREDICTOR_WITH_MATLAB_ZSCORES
            if USE_MATLAB_ZSCORES
            else TRIAL_SLOPE_PREDICTOR_WITHOUT_MATLAB_ZSCORES
        ),
        predictor_transform_by_condition=MATLAB_CB_PREDICTOR_TRANSFORM_BY_CONDITION,
        predictor_zscore=TRIAL_SLOPE_PREDICTOR_ZSCORE,
        p_value_correction_method=TRIAL_SLOPE_P_VALUE_CORRECTION_METHOD,
        significance_alpha=TRIAL_SLOPE_SIGNIFICANCE_ALPHA,
        n_permutations=0,
    )


def build_condition_test_writer() -> TimeFrequencyConditionTestWriter:
    return TimeFrequencyConditionTestWriter(
        TimeFrequencyConditionTestWriterParams(
            bids_root=BIDS_ROOT,
            output_format=OUTPUT_FORMAT,
            output_description=CONDITION_TEST_OUTPUT_DESCRIPTION,
            include_epochs=False,
        )
    )


def build_regression_writer() -> TimeFrequencyRegressionWriter:
    return TimeFrequencyRegressionWriter(
        TimeFrequencyRegressionWriterParams(
            bids_root=BIDS_ROOT,
            output_format=OUTPUT_FORMAT,
            output_description=REGRESSION_OUTPUT_DESCRIPTION,
            include_epochs=False,
        )
    )


def condition_test_reference_candidates(subject_id: str) -> list[Path]:
    explicit = os.environ.get("MATLAB_B2_TF_CONDITION_TEST_PATH")
    if explicit:
        return [Path(explicit)]
    source_subject = map_bids_subject_to_source(subject_id, bids_root=BIDS_ROOT)
    filename = f"TF_Ttest_{source_subject}_{MATLAB_CONDITION_REALIGN_NAME}.mat"
    return [
        MATLAB_SEEG_ROOT / "b2_TF_single_contrast" / source_subject / "Ttest_TF" / filename,
        MATLAB_SEEG_ROOT / "b2_TF_single_contrast_R1" / source_subject / "Ttest_TF" / filename,
    ]


def regression_reference_candidates(subject_id: str) -> list[Path]:
    explicit = os.environ.get("MATLAB_B2_TF_REGRESSION_PATH")
    if explicit:
        return [Path(explicit)]
    source_subject = map_bids_subject_to_source(subject_id, bids_root=BIDS_ROOT)
    r1_root = MATLAB_SEEG_ROOT / "b2_TF_single_contrast_R1" / source_subject
    candidates = [
        MATLAB_SEEG_ROOT
        / "b2_TF_single_contrast"
        / source_subject
        / "regressions_TF"
        / "log_data.mat",
    ]
    candidates.extend(sorted(r1_root.glob("regressions_TF_Part-*/log_data.mat")))
    candidates.append(r1_root / "regressions_TF" / "log_data.mat")
    return candidates


def first_existing_path(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def compare_condition_test_result(result, matlab_path: Path) -> None:
    mat_channels, mat_t, mat_freq, mat_time = load_matlab_ttest(
        matlab_path,
        MATLAB_CONTRAST_NAME,
    )
    freq_i_mat, freq_i_py, max_freq_delta = axis_pairs(
        mat_freq,
        np.asarray(result.frequency_hz, dtype=np.float64).ravel(),
        tolerance=1e-6,
    )
    time_i_mat, time_i_py, max_time_delta = axis_pairs(
        mat_time,
        np.asarray(result.time_axis_s, dtype=np.float64).ravel(),
        tolerance=1e-6,
    )
    py_t = np.asarray(result.contrast.t_values, dtype=np.float64)
    py_by_channel = {norm_channel(name): idx for idx, name in enumerate(result.channel_names)}
    rows = []
    for mat_idx, mat_name in enumerate(mat_channels):
        py_idx = py_by_channel.get(norm_channel(mat_name))
        if py_idx is None or mat_idx >= mat_t.shape[0]:
            continue
        mat_map = mat_t[mat_idx][np.ix_(freq_i_mat, time_i_mat)]
        py_map = py_t[py_idx][np.ix_(freq_i_py, time_i_py)]
        rows.append((mat_name, result.channel_names[py_idx], *pair_stats(mat_map, py_map)))

    print(f"MATLAB reference: {matlab_path}")
    print(f"Contrast: {MATLAB_CONTRAST_NAME}")
    print(
        "Matched axes: "
        f"{len(freq_i_py)} freq bins (max delta={max_freq_delta:g}), "
        f"{len(time_i_py)} time bins (max delta={max_time_delta:g})"
    )
    print(f"Matched channels: {len(rows)} / {len(mat_channels)}")
    if not rows:
        return
    print(f"Median corr(t): {np.nanmedian([row[2] for row in rows]):.6f}")
    print(f"Median mean|diff t|: {np.nanmedian([row[3] for row in rows]):.6g}")
    print("Lowest channel correlations:")
    for mat_name, py_name, corr, mean_abs, max_abs in sorted(rows, key=lambda row: row[2])[:12]:
        print(
            f"  {mat_name:<16} -> {py_name:<16} "
            f"corr={corr:.6f} mean|d|={mean_abs:.6g} max|d|={max_abs:.6g}"
        )


def compare_regression_result(result, matlab_path: Path) -> None:
    print(f"MATLAB reference: {matlab_path}")
    for condition in (result.condition_a, result.condition_b):
        compare_regression_condition(result, matlab_path, condition)


def compare_regression_condition(result, matlab_path: Path, condition: str) -> None:
    regressor = MATLAB_REGRESSOR_BY_CONDITION[condition]
    mat_channels, mat_maps, mat_freq, mat_time = load_matlab_regression_maps(
        matlab_path,
        regressor,
    )
    py_maps = python_regression_maps_for_condition(result, condition)
    freq_i_mat, freq_i_py, max_freq_delta = axis_pairs(
        mat_freq,
        np.asarray(result.frequency_hz, dtype=np.float64).ravel(),
        tolerance=1e-6,
    )
    time_i_mat, time_i_py, max_time_delta = axis_pairs(
        mat_time,
        np.asarray(result.time_axis_s, dtype=np.float64).ravel(),
        tolerance=1e-6,
    )
    py_by_channel = {norm_channel(name): idx for idx, name in enumerate(result.channel_names)}

    print(f"\nMATLAB regressor: {regressor} -> Python condition: {condition}")
    print(
        "Matched axes: "
        f"{len(freq_i_py)} freq bins (max delta={max_freq_delta:g}), "
        f"{len(time_i_py)} time bins (max delta={max_time_delta:g})"
    )
    for map_name in ("slope", "tstat", "pval"):
        rows = []
        mat_map_all = mat_maps[map_name]
        py_map_all = py_maps[map_name]
        for mat_idx, mat_name in enumerate(mat_channels):
            py_idx = py_by_channel.get(norm_channel(mat_name))
            if py_idx is None or mat_idx >= mat_map_all.shape[0]:
                continue
            mat_map = mat_map_all[mat_idx][np.ix_(freq_i_mat, time_i_mat)]
            py_map = py_map_all[py_idx][np.ix_(freq_i_py, time_i_py)]
            rows.append((mat_name, result.channel_names[py_idx], *pair_stats(mat_map, py_map)))
        print(f"  {map_name}: matched channels {len(rows)} / {len(mat_channels)}")
        if not rows:
            continue
        print(
            f"    median corr={np.nanmedian([row[2] for row in rows]):.6f}, "
            f"median mean|diff|={np.nanmedian([row[3] for row in rows]):.6g}"
        )
        print("    lowest correlations:")
        for mat_name, py_name, corr, mean_abs, max_abs in sorted(rows, key=lambda row: row[2])[:8]:
            print(
                f"      {mat_name:<16} -> {py_name:<16} "
                f"corr={corr:.6f} mean|d|={mean_abs:.6g} max|d|={max_abs:.6g}"
            )


def load_matlab_ttest(path: Path, contrast_name: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    import scipy.io

    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    hdr = mat["hdr"]
    store = np.asarray(mat["store"], dtype=object).ravel()
    selected = None
    for item in store:
        if str(getattr(item, "contrastname", "")).strip() == contrast_name:
            selected = item
            break
    if selected is None:
        available = [str(getattr(item, "contrastname", "")).strip() for item in store]
        raise ValueError(
            f"Contrast {contrast_name!r} not found in {path}. Available: {available}"
        )
    return (
        string_list(getattr(hdr, "chanlist", [])),
        cell_maps(getattr(selected, "tstat")),
        float_axis(getattr(hdr, "freqlist", [])),
        float_axis(getattr(hdr, "timelist", [])),
    )


def load_matlab_regression_maps(path: Path, regressor: str) -> tuple[list[str], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    import scipy.io

    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "log" not in mat:
        raise ValueError(f"{path} does not contain a 'log' struct.")
    log = mat["log"]
    hdr = mat.get("hdr")
    branch = matlab_regression_branch(log, regressor)
    maps = {
        "slope": cell_maps(first_existing_attr(branch, ["dots", "slope"])),
        "tstat": cell_maps(first_existing_attr(branch, ["tstat", "t_values"])),
        "pval": cell_maps(first_existing_attr(branch, ["pval", "p_value"])),
    }
    channels = string_list(getattr(hdr, "chanlist", [])) if hdr is not None else []
    if not channels:
        channels = [f"channel_{idx + 1}" for idx in range(maps["slope"].shape[0])]
    freq = float_axis(getattr(hdr, "freqlist", [])) if hdr is not None else np.array([])
    time = float_axis(getattr(hdr, "timelist", [])) if hdr is not None else np.array([])
    return channels, maps, freq, time


def matlab_regression_branch(log: object, regressor: str) -> object:
    if not hasattr(log, regressor):
        available = [name for name in dir(log) if not name.startswith("_")]
        raise ValueError(f"Regressor {regressor!r} not found. Available: {available}")
    reg_branch = getattr(log, regressor)
    if hasattr(reg_branch, MATLAB_REGRESSION_REALIGN_NAME):
        return getattr(reg_branch, MATLAB_REGRESSION_REALIGN_NAME)
    available = [name for name in dir(reg_branch) if not name.startswith("_")]
    if len(available) == 1:
        return getattr(reg_branch, available[0])
    raise ValueError(
        f"Realign branch {MATLAB_REGRESSION_REALIGN_NAME!r} not found under {regressor!r}. "
        f"Available: {available}"
    )


def python_regression_maps_for_condition(result, condition: str) -> dict[str, np.ndarray]:
    stats = (
        result.regression.condition_a
        if condition == result.condition_a
        else result.regression.condition_b
    )
    return {
        "slope": np.asarray(stats.slope, dtype=np.float64),
        "tstat": np.asarray(stats.t_values, dtype=np.float64),
        "pval": np.asarray(stats.p_value, dtype=np.float64),
    }


def first_existing_attr(obj: object, names: list[str]) -> object:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise ValueError(f"None of these fields exist on Matlab struct: {names}")


def axis_pairs(
    matlab_axis: np.ndarray,
    python_axis: np.ndarray,
    *,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if matlab_axis.size == 0:
        idx = np.arange(len(python_axis), dtype=np.int64)
        return idx, idx, 0.0
    mat_idx: list[int] = []
    py_idx: list[int] = []
    deltas: list[float] = []
    used: set[int] = set()
    for j, value in enumerate(python_axis):
        i = int(np.argmin(np.abs(matlab_axis - value)))
        delta = float(abs(matlab_axis[i] - value))
        if delta <= tolerance and i not in used:
            mat_idx.append(i)
            py_idx.append(j)
            deltas.append(delta)
            used.add(i)
    max_delta = max(deltas) if deltas else float("nan")
    return np.asarray(mat_idx), np.asarray(py_idx), max_delta


def pair_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    finite = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(finite)) < 3:
        return float("nan"), float("nan"), float("nan")
    av = a[finite].ravel()
    bv = b[finite].ravel()
    corr = float(np.corrcoef(av, bv)[0, 1])
    mean_abs = float(np.mean(np.abs(bv - av)))
    max_abs = float(np.max(np.abs(bv - av)))
    return corr, mean_abs, max_abs


def norm_channel(name: str) -> str:
    return "".join(ch for ch in str(name).casefold() if ch.isalnum())


def string_list(value: object) -> list[str]:
    arr = np.asarray(value, dtype=object).ravel()
    return [str(item).strip() for item in arr]


def float_axis(value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def cell_maps(value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=object)
    if arr.dtype == object:
        maps = [np.asarray(item, dtype=np.float64) for item in arr.ravel()]
        return np.stack(maps, axis=0)
    numeric = np.asarray(value, dtype=np.float64)
    if numeric.ndim == 2:
        return numeric[np.newaxis, :, :]
    return numeric


def _apply_behavioral_exclusions(trial: ResolvedTrial) -> ResolvedTrial:
    if not trial.keep:
        return trial
    metadata = dict(trial.metadata)
    rt = _float_or_nan(metadata.get("RT"))
    rating = _float_or_nan(metadata.get("rating"))
    if np.isfinite(rt) and rt > MAX_RT_S:
        return replace(trial, keep=False, exclusion_reason="outlier_rt", metadata=metadata)
    if np.isfinite(rating) and rating < MIN_RATING:
        return replace(trial, keep=False, exclusion_reason="negative_rating", metadata=metadata)
    return replace(trial, metadata=metadata)


def _float_or_nan(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _tfr_extension() -> str:
    if TIME_FREQUENCY_OUTPUT_FORMAT == "hdf5":
        return ".h5"
    if TIME_FREQUENCY_OUTPUT_FORMAT == "matlab":
        return ".mat"
    raise ValueError(f"Unsupported TIME_FREQUENCY_OUTPUT_FORMAT={TIME_FREQUENCY_OUTPUT_FORMAT!r}")
