"""Load a pre-computed TrialSlopeStatsProcessingResult from disk."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.utils.hdf5 import (
    dataset_or_none,
    decode_str_array,
    float_scalar,
    int_scalar,
    str_scalar,
)
from gin_bids_py_analysis.processing.utils.matlab import (
    mat_float,
    mat_int,
    mat_str,
    mat_str_list,
    matlab_safe_name,
)

from .result import TrialSlopeStatsProcessingResult


def load_trial_slope_stats_result(path: Path | str) -> TrialSlopeStatsProcessingResult:
    """Load a pre-computed TrialSlopeStatsProcessingResult from *path*."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trial slope stats file not found: {path}")
    ext = path.suffix.lower()
    if ext == ".mat":
        return _load_from_matlab(path)
    return _load_from_hdf5(path)


def _load_from_hdf5(path: Path) -> TrialSlopeStatsProcessingResult:
    with h5py.File(path, "r") as fh:
        analysis_type = str_scalar(dataset_or_none(fh, "meta/analysis_type"), default="")
        if analysis_type and analysis_type != "slope_regression":
            raise ValueError(
                f"{path.name}: not a trial_slope_stats file (analysis_type={analysis_type!r})."
            )

        analysis_level = str_scalar(dataset_or_none(fh, "meta/analysis_level"), default="channel")
        axis_name = "region" if analysis_level == "roi" else "channel"
        if "axes" not in fh or axis_name not in fh["axes"]:
            raise ValueError(f"{path.name}: axes/{axis_name} dataset is required.")
        channel_names = decode_str_array(np.asarray(fh["axes"][axis_name][:]))
        time_axis_s = np.asarray(fh["axes"]["time_s"][:], dtype=np.float64)
        n_features = len(channel_names)
        n_times = len(time_axis_s)
        empty = np.full((n_features, n_times), np.nan, dtype=np.float64)

        labels_ds = dataset_or_none(fh, "meta/trial_count_labels")
        if labels_ds is not None:
            labels = decode_str_array(np.asarray(labels_ds[:], dtype=object))
            condition_a = labels[0] if len(labels) >= 1 else "condition_a"
            condition_b = labels[1] if len(labels) >= 2 else "condition_b"
        elif "means" in fh:
            mean_keys = list(fh["means"].keys())
            condition_a = mean_keys[0] if len(mean_keys) >= 1 else "condition_a"
            condition_b = mean_keys[1] if len(mean_keys) >= 2 else "condition_b"
        else:
            condition_a, condition_b = "condition_a", "condition_b"

        trial_counts_ds = dataset_or_none(fh, "meta/trial_counts")
        if trial_counts_ds is not None:
            trial_counts = np.asarray(trial_counts_ds[:], dtype=np.int64).ravel()
            condition_a_trial_count = int(trial_counts[0]) if len(trial_counts) >= 1 else 0
            condition_b_trial_count = int(trial_counts[1]) if len(trial_counts) >= 2 else 0
        else:
            condition_a_trial_count = 0
            condition_b_trial_count = 0

        def _read_2d(path_key: str) -> np.ndarray:
            ds = dataset_or_none(fh, path_key)
            return np.asarray(ds[:], dtype=np.float64) if ds is not None else empty.copy()

        condition_a_slope = _read_2d("regression/condition_a/slope")
        condition_a_slope_standardized_predictor = _read_2d(
            "regression/condition_a/slope_standardized_predictor"
        )
        condition_a_slope_standardized_full = _read_2d(
            "regression/condition_a/slope_standardized_full"
        )
        condition_a_intercept = _read_2d("regression/condition_a/intercept")
        condition_a_r_value = _read_2d("regression/condition_a/r_value")
        condition_a_p_value = _read_2d("regression/condition_a/p_value")
        condition_a_p_value_corrected = _read_2d("regression/condition_a/p_value_corrected")
        condition_a_significant_mask_ds = dataset_or_none(fh, "regression/condition_a/significant_mask")
        condition_a_significant_mask = (
            np.asarray(condition_a_significant_mask_ds[:], dtype=bool)
            if condition_a_significant_mask_ds is not None
            else (np.isfinite(condition_a_p_value_corrected) & (condition_a_p_value_corrected < 0.05))
        )

        condition_b_slope = _read_2d("regression/condition_b/slope")
        condition_b_slope_standardized_predictor = _read_2d(
            "regression/condition_b/slope_standardized_predictor"
        )
        condition_b_slope_standardized_full = _read_2d(
            "regression/condition_b/slope_standardized_full"
        )
        condition_b_intercept = _read_2d("regression/condition_b/intercept")
        condition_b_r_value = _read_2d("regression/condition_b/r_value")
        condition_b_p_value = _read_2d("regression/condition_b/p_value")
        condition_b_p_value_corrected = _read_2d("regression/condition_b/p_value_corrected")
        condition_b_significant_mask_ds = dataset_or_none(fh, "regression/condition_b/significant_mask")
        condition_b_significant_mask = (
            np.asarray(condition_b_significant_mask_ds[:], dtype=bool)
            if condition_b_significant_mask_ds is not None
            else (np.isfinite(condition_b_p_value_corrected) & (condition_b_p_value_corrected < 0.05))
        )

        condition_a_trials_used = int_scalar(dataset_or_none(fh, "regression/condition_a/n_trials_used"), default=0)
        condition_b_trials_used = int_scalar(dataset_or_none(fh, "regression/condition_b/n_trials_used"), default=0)
        condition_a_stats_valid = bool(dataset_or_none(fh, "regression/condition_a/stats_valid")[()]) if dataset_or_none(fh, "regression/condition_a/stats_valid") is not None else False
        condition_b_stats_valid = bool(dataset_or_none(fh, "regression/condition_b/stats_valid")[()]) if dataset_or_none(fh, "regression/condition_b/stats_valid") is not None else False

        condition_a_mean = _read_2d(f"means/{condition_a}")
        condition_b_mean = _read_2d(f"means/{condition_b}")
        condition_a_sem = _read_2d(f"uncertainty/{condition_a}_sem")
        condition_b_sem = _read_2d(f"uncertainty/{condition_b}_sem")

        predictor_a_raw_ds = dataset_or_none(fh, "predictor/condition_a_raw_values")
        predictor_b_raw_ds = dataset_or_none(fh, "predictor/condition_b_raw_values")
        condition_a_predictor_raw_values = (
            np.asarray(predictor_a_raw_ds[:], dtype=np.float64)
            if predictor_a_raw_ds is not None
            else np.array([], dtype=np.float64)
        )
        condition_b_predictor_raw_values = (
            np.asarray(predictor_b_raw_ds[:], dtype=np.float64)
            if predictor_b_raw_ds is not None
            else np.array([], dtype=np.float64)
        )
        predictor_a_transformed_ds = dataset_or_none(fh, "predictor/condition_a_transformed_values")
        predictor_b_transformed_ds = dataset_or_none(fh, "predictor/condition_b_transformed_values")
        condition_a_predictor_transformed_values = (
            np.asarray(predictor_a_transformed_ds[:], dtype=np.float64)
            if predictor_a_transformed_ds is not None
            else np.array([], dtype=np.float64)
        )
        condition_b_predictor_transformed_values = (
            np.asarray(predictor_b_transformed_ds[:], dtype=np.float64)
            if predictor_b_transformed_ds is not None
            else np.array([], dtype=np.float64)
        )
        predictor_a_ds = dataset_or_none(fh, "predictor/condition_a_values")
        predictor_b_ds = dataset_or_none(fh, "predictor/condition_b_values")
        condition_a_predictor_values = (
            np.asarray(predictor_a_ds[:], dtype=np.float64) if predictor_a_ds is not None else np.array([], dtype=np.float64)
        )
        condition_b_predictor_values = (
            np.asarray(predictor_b_ds[:], dtype=np.float64) if predictor_b_ds is not None else np.array([], dtype=np.float64)
        )

        sfreq = float_scalar(dataset_or_none(fh, "meta/sampling_frequency_hz"), default=0.0)
        predictor = str_scalar(dataset_or_none(fh, "meta/predictor"), default="")
        predictor_scaling = str_scalar(dataset_or_none(fh, "meta/predictor_scaling"), default="none")
        predictor_transform_raw = str_scalar(
            dataset_or_none(fh, "meta/predictor_transform_by_condition_json"),
            default="{}",
        )
        try:
            predictor_transform_by_condition = json.loads(predictor_transform_raw) if predictor_transform_raw else {}
        except json.JSONDecodeError:
            predictor_transform_by_condition = {}
        p_value_correction_method = str_scalar(dataset_or_none(fh, "meta/p_value_correction_method"), default="fdr_bh")
        significance_alpha = float_scalar(dataset_or_none(fh, "meta/significance_alpha"), default=0.05)
        stats_valid = bool(dataset_or_none(fh, "meta/stats_valid")[()]) if dataset_or_none(fh, "meta/stats_valid") is not None else bool(condition_a_stats_valid or condition_b_stats_valid)
        activity_scaling = str_scalar(
            dataset_or_none(fh, "meta/activity_scaling"),
            default="none",
        )
        activity_baseline_tmin_s = float_scalar(
            dataset_or_none(fh, "meta/activity_baseline_tmin_s"),
            default=-0.2,
        )
        activity_baseline_tmax_s = float_scalar(
            dataset_or_none(fh, "meta/activity_baseline_tmax_s"),
            default=0.0,
        )

        atlas_name_raw = str_scalar(dataset_or_none(fh, "meta/atlas_name"), default="")
        atlas_name = atlas_name_raw.strip() or None
        atlas_regions_ds = dataset_or_none(fh, "meta/atlas_regions")
        atlas_regions = decode_str_array(np.asarray(atlas_regions_ds[:], dtype=object)) if atlas_regions_ds is not None else []

        region_channels: dict[str, list[str]] = {}
        atlas_map = dataset_or_none(fh, "meta/atlas_region_channel_map/region")
        atlas_map_channel = dataset_or_none(fh, "meta/atlas_region_channel_map/channel")
        if atlas_map is not None and atlas_map_channel is not None:
            regions = decode_str_array(np.asarray(atlas_map[:], dtype=object))
            channels = decode_str_array(np.asarray(atlas_map_channel[:], dtype=object))
            for region, channel in zip(regions, channels):
                region_channels.setdefault(region, []).append(channel)

        window_ms = float_scalar(dataset_or_none(fh, "meta/window_ms"), default=0.0)
        n_bins = int_scalar(dataset_or_none(fh, "meta/n_bins"), default=0)

        if "epochs" in fh:
            eg = fh["epochs"]
            condition_a_epochs = np.asarray(eg["condition_a"][:], dtype=np.float64) if "condition_a" in eg else np.array([])
            condition_b_epochs = np.asarray(eg["condition_b"][:], dtype=np.float64) if "condition_b" in eg else np.array([])
        else:
            condition_a_epochs = np.array([])
            condition_b_epochs = np.array([])

        if "scatter" in fh:
            sg = fh["scatter"]
            condition_a_epoch_means = (
                np.asarray(sg["condition_a_epoch_means"][:], dtype=np.float64)
                if "condition_a_epoch_means" in sg
                else np.array([])
            )
            condition_b_epoch_means = (
                np.asarray(sg["condition_b_epoch_means"][:], dtype=np.float64)
                if "condition_b_epoch_means" in sg
                else np.array([])
            )
        else:
            condition_a_epoch_means = np.array([])
            condition_b_epoch_means = np.array([])

        source_ieeg_files = []
        source_table_files = []
        source_electrodes_files = []
        if "provenance" in fh:
            prov = fh["provenance"]
            if "source_ieeg_files" in prov:
                source_ieeg_files = decode_str_array(np.asarray(prov["source_ieeg_files"][:], dtype=object))
            if "source_table_files" in prov:
                source_table_files = decode_str_array(np.asarray(prov["source_table_files"][:], dtype=object))
            if "source_electrodes_files" in prov:
                source_electrodes_files = decode_str_array(np.asarray(prov["source_electrodes_files"][:], dtype=object))

    source_group = BIDSFileGroup(primary=BIDSFile.from_path(path))
    return TrialSlopeStatsProcessingResult(
        source_group=source_group,
        metadata={
            "activity_scaling": activity_scaling,
            "activity_baseline_tmin_s": activity_baseline_tmin_s,
            "activity_baseline_tmax_s": activity_baseline_tmax_s,
        },
        condition_a_slope=condition_a_slope,
        condition_a_slope_standardized_predictor=condition_a_slope_standardized_predictor,
        condition_a_slope_standardized_full=condition_a_slope_standardized_full,
        condition_a_intercept=condition_a_intercept,
        condition_a_r_value=condition_a_r_value,
        condition_a_p_value=condition_a_p_value,
        condition_a_p_value_corrected=condition_a_p_value_corrected,
        condition_a_significant_mask=condition_a_significant_mask,
        condition_b_slope=condition_b_slope,
        condition_b_slope_standardized_predictor=condition_b_slope_standardized_predictor,
        condition_b_slope_standardized_full=condition_b_slope_standardized_full,
        condition_b_intercept=condition_b_intercept,
        condition_b_r_value=condition_b_r_value,
        condition_b_p_value=condition_b_p_value,
        condition_b_p_value_corrected=condition_b_p_value_corrected,
        condition_b_significant_mask=condition_b_significant_mask,
        condition_a_mean=condition_a_mean,
        condition_b_mean=condition_b_mean,
        condition_a_sem=condition_a_sem,
        condition_b_sem=condition_b_sem,
        time_axis_s=time_axis_s,
        channel_names=channel_names,
        condition_a=condition_a,
        condition_b=condition_b,
        condition_a_trial_count=condition_a_trial_count,
        condition_b_trial_count=condition_b_trial_count,
        condition_a_trials_used=condition_a_trials_used,
        condition_b_trials_used=condition_b_trials_used,
        sfreq=sfreq,
        condition_a_predictor_raw_values=condition_a_predictor_raw_values,
        condition_b_predictor_raw_values=condition_b_predictor_raw_values,
        condition_a_predictor_transformed_values=condition_a_predictor_transformed_values,
        condition_b_predictor_transformed_values=condition_b_predictor_transformed_values,
        condition_a_predictor_values=condition_a_predictor_values,
        condition_b_predictor_values=condition_b_predictor_values,
        source_ieeg_files=source_ieeg_files,
        source_table_files=source_table_files,
        source_electrodes_files=source_electrodes_files,
        analysis_level=analysis_level,
        analysis_type="slope_regression",
        atlas_name=atlas_name,
        atlas_regions=atlas_regions,
        region_channels=region_channels,
        window_ms=window_ms,
        n_bins=n_bins,
        activity_scaling=activity_scaling,
        activity_baseline_tmin_s=activity_baseline_tmin_s,
        activity_baseline_tmax_s=activity_baseline_tmax_s,
        predictor=predictor,
        predictor_scaling=predictor_scaling,
        predictor_transform_by_condition=predictor_transform_by_condition,
        p_value_correction_method=p_value_correction_method,
        significance_alpha=significance_alpha,
        condition_a_stats_valid=condition_a_stats_valid,
        condition_b_stats_valid=condition_b_stats_valid,
        stats_valid=stats_valid,
        condition_a_epochs=condition_a_epochs,
        condition_b_epochs=condition_b_epochs,
        condition_a_epoch_means=condition_a_epoch_means,
        condition_b_epoch_means=condition_b_epoch_means,
    )


def _load_from_matlab(path: Path) -> TrialSlopeStatsProcessingResult:
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    data = mat["data"]
    regression = data.regression
    means = data.means
    uncertainty = getattr(data, "uncertainty", None)
    predictor = getattr(data, "predictor", None)
    scatter = getattr(data, "scatter", None)
    axes = data.axes
    meta = data.meta
    prov = getattr(data, "provenance", None)

    analysis_type = mat_str(getattr(meta, "analysis_type", None), default="")
    if analysis_type and analysis_type != "slope_regression":
        raise ValueError(
            f"{path.name}: not a trial_slope_stats file (analysis_type={analysis_type!r})."
        )

    analysis_level = mat_str(getattr(meta, "analysis_level", None), default="channel")
    axis_attr = "region" if analysis_level == "roi" else "channel"
    channel_names = mat_str_list(getattr(axes, axis_attr, None))
    if not channel_names:
        raise ValueError(f"{path.name}: axes.{axis_attr} is required in .mat file.")
    time_axis_s = np.asarray(axes.time_s, dtype=np.float64).ravel()
    n_features = len(channel_names)
    n_times = len(time_axis_s)
    empty = np.full((n_features, n_times), np.nan, dtype=np.float64)

    labels = mat_str_list(getattr(meta, "trial_count_labels", None))
    condition_a = labels[0] if len(labels) >= 1 else "condition_a"
    condition_b = labels[1] if len(labels) >= 2 else "condition_b"
    safe_a = matlab_safe_name(condition_a)
    safe_b = matlab_safe_name(condition_b)

    def _mat_2d(obj: object, attr: str) -> np.ndarray:
        raw = getattr(obj, attr, None) if obj is not None else None
        if raw is None:
            return empty.copy()
        arr = np.asarray(raw, dtype=np.float64)
        return arr.reshape(n_features, n_times)

    def _mat_feature_trial_2d(obj: object, attr: str) -> np.ndarray:
        raw = getattr(obj, attr, None) if obj is not None else None
        if raw is None:
            return np.empty((0, 0), dtype=np.float64)
        arr = np.asarray(raw, dtype=np.float64)
        if arr.size == 0:
            return np.empty((n_features, 0), dtype=np.float64)
        arr = np.atleast_2d(arr)
        if arr.shape[0] == n_features:
            return arr.reshape(n_features, -1)
        if arr.shape[1] == n_features:
            return arr.T.reshape(n_features, -1)
        if n_features == 1:
            return arr.reshape(1, -1)
        return np.empty((0, 0), dtype=np.float64)

    reg_a = getattr(regression, "condition_a", None)
    reg_b = getattr(regression, "condition_b", None)

    condition_a_slope = _mat_2d(reg_a, "slope")
    condition_a_slope_standardized_predictor = _mat_2d(reg_a, "slope_standardized_predictor")
    condition_a_slope_standardized_full = _mat_2d(reg_a, "slope_standardized_full")
    condition_a_intercept = _mat_2d(reg_a, "intercept")
    condition_a_r_value = _mat_2d(reg_a, "r_value")
    condition_a_p_value = _mat_2d(reg_a, "p_value")
    condition_a_p_value_corrected = _mat_2d(reg_a, "p_value_corrected")
    sig_a_raw = getattr(reg_a, "significant_mask", None)
    condition_a_significant_mask = (
        np.asarray(sig_a_raw, dtype=bool).reshape(n_features, n_times)
        if sig_a_raw is not None
        else np.zeros((n_features, n_times), dtype=bool)
    )

    condition_b_slope = _mat_2d(reg_b, "slope")
    condition_b_slope_standardized_predictor = _mat_2d(reg_b, "slope_standardized_predictor")
    condition_b_slope_standardized_full = _mat_2d(reg_b, "slope_standardized_full")
    condition_b_intercept = _mat_2d(reg_b, "intercept")
    condition_b_r_value = _mat_2d(reg_b, "r_value")
    condition_b_p_value = _mat_2d(reg_b, "p_value")
    condition_b_p_value_corrected = _mat_2d(reg_b, "p_value_corrected")
    sig_b_raw = getattr(reg_b, "significant_mask", None)
    condition_b_significant_mask = (
        np.asarray(sig_b_raw, dtype=bool).reshape(n_features, n_times)
        if sig_b_raw is not None
        else np.zeros((n_features, n_times), dtype=bool)
    )

    condition_a_trials_used = mat_int(getattr(reg_a, "n_trials_used", None), default=0)
    condition_b_trials_used = mat_int(getattr(reg_b, "n_trials_used", None), default=0)
    condition_a_stats_valid = bool(mat_int(getattr(reg_a, "stats_valid", None), default=0))
    condition_b_stats_valid = bool(mat_int(getattr(reg_b, "stats_valid", None), default=0))

    condition_a_mean = _mat_2d(means, safe_a)
    condition_b_mean = _mat_2d(means, safe_b)
    condition_a_sem = _mat_2d(uncertainty, safe_a + "_sem")
    condition_b_sem = _mat_2d(uncertainty, safe_b + "_sem")

    condition_a_predictor_raw_values = np.asarray(
        getattr(predictor, "condition_a_raw_values", np.array([], dtype=np.float64)),
        dtype=np.float64,
    ).ravel()
    condition_b_predictor_raw_values = np.asarray(
        getattr(predictor, "condition_b_raw_values", np.array([], dtype=np.float64)),
        dtype=np.float64,
    ).ravel()
    condition_a_predictor_transformed_values = np.asarray(
        getattr(predictor, "condition_a_transformed_values", np.array([], dtype=np.float64)),
        dtype=np.float64,
    ).ravel()
    condition_b_predictor_transformed_values = np.asarray(
        getattr(predictor, "condition_b_transformed_values", np.array([], dtype=np.float64)),
        dtype=np.float64,
    ).ravel()
    condition_a_predictor_values = np.asarray(
        getattr(predictor, "condition_a_values", np.array([], dtype=np.float64)),
        dtype=np.float64,
    ).ravel()
    condition_b_predictor_values = np.asarray(
        getattr(predictor, "condition_b_values", np.array([], dtype=np.float64)),
        dtype=np.float64,
    ).ravel()
    condition_a_epoch_means = _mat_feature_trial_2d(scatter, "condition_a_epoch_means")
    condition_b_epoch_means = _mat_feature_trial_2d(scatter, "condition_b_epoch_means")

    counts_raw = getattr(meta, "trial_counts", None)
    counts = np.asarray(counts_raw, dtype=np.int64).ravel() if counts_raw is not None else np.array([], dtype=np.int64)
    condition_a_trial_count = int(counts[0]) if len(counts) >= 1 else 0
    condition_b_trial_count = int(counts[1]) if len(counts) >= 2 else 0

    sfreq = mat_float(getattr(meta, "sampling_frequency_hz", None), default=0.0)
    predictor = mat_str(getattr(meta, "predictor", None), default="")
    predictor_scaling = mat_str(getattr(meta, "predictor_scaling", None), default="none")
    predictor_transform_raw = mat_str(
        getattr(meta, "predictor_transform_by_condition_json", None),
        default="{}",
    )
    try:
        predictor_transform_by_condition = json.loads(predictor_transform_raw) if predictor_transform_raw else {}
    except json.JSONDecodeError:
        predictor_transform_by_condition = {}
    p_value_correction_method = mat_str(getattr(meta, "p_value_correction_method", None), default="fdr_bh")
    significance_alpha = mat_float(getattr(meta, "significance_alpha", None), default=0.05)
    stats_valid = bool(mat_int(getattr(meta, "stats_valid", None), default=int(condition_a_stats_valid or condition_b_stats_valid)))
    activity_scaling = mat_str(getattr(meta, "activity_scaling", None), default="none")
    activity_baseline_tmin_s = mat_float(
        getattr(meta, "activity_baseline_tmin_s", None),
        default=-0.2,
    )
    activity_baseline_tmax_s = mat_float(
        getattr(meta, "activity_baseline_tmax_s", None),
        default=0.0,
    )

    atlas_name_raw = mat_str(getattr(meta, "atlas_name", None), default="")
    atlas_name = atlas_name_raw.strip() or None
    atlas_regions = mat_str_list(getattr(meta, "atlas_regions", None))
    window_ms = mat_float(getattr(meta, "window_ms", None), default=0.0)
    n_bins = mat_int(getattr(meta, "n_bins", None), default=0)

    source_ieeg_files = mat_str_list(getattr(prov, "source_ieeg_files", None)) if prov is not None else []
    source_table_files = mat_str_list(getattr(prov, "source_table_files", None)) if prov is not None else []
    source_electrodes_files = mat_str_list(getattr(prov, "source_electrodes_files", None)) if prov is not None else []

    source_group = BIDSFileGroup(primary=BIDSFile.from_path(path))
    return TrialSlopeStatsProcessingResult(
        source_group=source_group,
        metadata={
            "activity_scaling": activity_scaling,
            "activity_baseline_tmin_s": activity_baseline_tmin_s,
            "activity_baseline_tmax_s": activity_baseline_tmax_s,
        },
        condition_a_slope=condition_a_slope,
        condition_a_slope_standardized_predictor=condition_a_slope_standardized_predictor,
        condition_a_slope_standardized_full=condition_a_slope_standardized_full,
        condition_a_intercept=condition_a_intercept,
        condition_a_r_value=condition_a_r_value,
        condition_a_p_value=condition_a_p_value,
        condition_a_p_value_corrected=condition_a_p_value_corrected,
        condition_a_significant_mask=condition_a_significant_mask,
        condition_b_slope=condition_b_slope,
        condition_b_slope_standardized_predictor=condition_b_slope_standardized_predictor,
        condition_b_slope_standardized_full=condition_b_slope_standardized_full,
        condition_b_intercept=condition_b_intercept,
        condition_b_r_value=condition_b_r_value,
        condition_b_p_value=condition_b_p_value,
        condition_b_p_value_corrected=condition_b_p_value_corrected,
        condition_b_significant_mask=condition_b_significant_mask,
        condition_a_mean=condition_a_mean,
        condition_b_mean=condition_b_mean,
        condition_a_sem=condition_a_sem,
        condition_b_sem=condition_b_sem,
        time_axis_s=time_axis_s,
        channel_names=channel_names,
        condition_a=condition_a,
        condition_b=condition_b,
        condition_a_trial_count=condition_a_trial_count,
        condition_b_trial_count=condition_b_trial_count,
        condition_a_trials_used=condition_a_trials_used,
        condition_b_trials_used=condition_b_trials_used,
        sfreq=sfreq,
        condition_a_predictor_raw_values=condition_a_predictor_raw_values,
        condition_b_predictor_raw_values=condition_b_predictor_raw_values,
        condition_a_predictor_transformed_values=condition_a_predictor_transformed_values,
        condition_b_predictor_transformed_values=condition_b_predictor_transformed_values,
        condition_a_predictor_values=condition_a_predictor_values,
        condition_b_predictor_values=condition_b_predictor_values,
        source_ieeg_files=source_ieeg_files,
        source_table_files=source_table_files,
        source_electrodes_files=source_electrodes_files,
        analysis_level=analysis_level,
        analysis_type="slope_regression",
        atlas_name=atlas_name,
        atlas_regions=atlas_regions,
        region_channels={},
        window_ms=window_ms,
        n_bins=n_bins,
        activity_scaling=activity_scaling,
        activity_baseline_tmin_s=activity_baseline_tmin_s,
        activity_baseline_tmax_s=activity_baseline_tmax_s,
        predictor=predictor,
        predictor_scaling=predictor_scaling,
        predictor_transform_by_condition=predictor_transform_by_condition,
        p_value_correction_method=p_value_correction_method,
        significance_alpha=significance_alpha,
        condition_a_stats_valid=condition_a_stats_valid,
        condition_b_stats_valid=condition_b_stats_valid,
        stats_valid=stats_valid,
        condition_a_epochs=np.array([]),
        condition_b_epochs=np.array([]),
        condition_a_epoch_means=condition_a_epoch_means,
        condition_b_epoch_means=condition_b_epoch_means,
    )
