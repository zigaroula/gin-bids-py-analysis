"""Load a pre-computed RegressionProcessingResult from disk."""

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
    mat_root,
    mat_str,
    mat_str_list,
)

from ..params import normalize_trial_activity_summary_missing_response_policy
from ..result import (
    ConditionEpochs,
    ConditionSignalActivity,
    ConditionTrialSummaryValues,
    SignalActivityEstimate,
)
from .result import (
    ConditionPredictorValues,
    ConditionRegressionStats,
    RegressionPredictor,
    RegressionProcessingResult,
    RegressionStats,
)

_VALID_PREDICTOR_ZSCORE_MODES = frozenset({"none", "condition", "global"})
_VALID_BASELINE_SCOPES = frozenset({"trial", "condition", "global"})
_VALID_TRIAL_ACTIVITY_SUMMARY_KINDS = frozenset({"epoch_mean", "anchor_to_response_mean"})
_VALID_TRIAL_ACTIVITY_SUMMARY_MISSING_RESPONSE_POLICIES = frozenset(
    {"clamp_to_epoch", "nan_if_missing"}
)


def load_regression_result(path: Path | str) -> RegressionProcessingResult:
    """Load a pre-computed RegressionProcessingResult from *path*."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trial slope stats file not found: {path}")
    ext = path.suffix.lower()
    if ext == ".mat":
        return _load_from_matlab(path)
    return _load_from_hdf5(path)


def _load_from_hdf5(path: Path) -> RegressionProcessingResult:
    with h5py.File(path, "r") as fh:
        _require_v2_schema(fh, path.name)
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

        labels_ds = dataset_or_none(fh, "meta/condition_labels")
        if labels_ds is not None:
            labels = decode_str_array(np.asarray(labels_ds[:], dtype=object))
            condition_a = labels[0] if len(labels) >= 1 else "condition_a"
            condition_b = labels[1] if len(labels) >= 2 else "condition_b"
        else:
            condition_a, condition_b = "condition_a", "condition_b"
        available_metrics_ds = dataset_or_none(fh, "meta/available_regression_metrics")
        if available_metrics_ds is not None:
            available_regression_metrics = decode_str_array(
                np.asarray(available_metrics_ds[:], dtype=object)
            )
        else:
            available_regression_metrics = [
                metric
                for metric, present in {
                    "slope": (
                        dataset_or_none(fh, f"stats/regression/{condition_a}/slope")
                        is not None
                        and dataset_or_none(fh, f"stats/regression/{condition_b}/slope")
                        is not None
                    ),
                    "r_value": (
                        dataset_or_none(fh, f"stats/regression/{condition_a}/r_value")
                        is not None
                        and dataset_or_none(fh, f"stats/regression/{condition_b}/r_value")
                        is not None
                    ),
                }.items()
                if present
            ]

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

        condition_a_slope = _read_2d(f"stats/regression/{condition_a}/slope")
        condition_a_intercept = _read_2d(f"stats/regression/{condition_a}/intercept")
        condition_a_r_value = _read_2d(f"stats/regression/{condition_a}/r_value")
        condition_a_p_value = _read_2d(f"stats/regression/{condition_a}/p_value")
        condition_a_p_value_corrected = _read_2d(f"stats/regression/{condition_a}/p_value_corrected")
        condition_a_significant_mask_ds = dataset_or_none(
            fh,
            f"stats/regression/{condition_a}/significant_mask",
        )
        condition_a_significant_mask = (
            np.asarray(condition_a_significant_mask_ds[:], dtype=bool)
            if condition_a_significant_mask_ds is not None
            else (np.isfinite(condition_a_p_value_corrected) & (condition_a_p_value_corrected < 0.05))
        )

        condition_b_slope = _read_2d(f"stats/regression/{condition_b}/slope")
        condition_b_intercept = _read_2d(f"stats/regression/{condition_b}/intercept")
        condition_b_r_value = _read_2d(f"stats/regression/{condition_b}/r_value")
        condition_b_p_value = _read_2d(f"stats/regression/{condition_b}/p_value")
        condition_b_p_value_corrected = _read_2d(f"stats/regression/{condition_b}/p_value_corrected")
        condition_b_significant_mask_ds = dataset_or_none(
            fh,
            f"stats/regression/{condition_b}/significant_mask",
        )
        condition_b_significant_mask = (
            np.asarray(condition_b_significant_mask_ds[:], dtype=bool)
            if condition_b_significant_mask_ds is not None
            else (np.isfinite(condition_b_p_value_corrected) & (condition_b_p_value_corrected < 0.05))
        )

        condition_a_trials_used = int_scalar(dataset_or_none(fh, f"stats/regression/{condition_a}/n_trials_used"), default=0)
        condition_b_trials_used = int_scalar(dataset_or_none(fh, f"stats/regression/{condition_b}/n_trials_used"), default=0)
        condition_a_stats_valid = bool(dataset_or_none(fh, f"stats/regression/{condition_a}/stats_valid")[()]) if dataset_or_none(fh, f"stats/regression/{condition_a}/stats_valid") is not None else False
        condition_b_stats_valid = bool(dataset_or_none(fh, f"stats/regression/{condition_b}/stats_valid")[()]) if dataset_or_none(fh, f"stats/regression/{condition_b}/stats_valid") is not None else False

        perm_a_ds = dataset_or_none(fh, f"stats/regression/{condition_a}/permuted_slopes")
        condition_a_permuted_slopes = (
            np.asarray(perm_a_ds[:], dtype=np.float32) if perm_a_ds is not None else None
        )
        perm_b_ds = dataset_or_none(fh, f"stats/regression/{condition_b}/permuted_slopes")
        condition_b_permuted_slopes = (
            np.asarray(perm_b_ds[:], dtype=np.float32) if perm_b_ds is not None else None
        )

        condition_a_mean = _read_2d(f"data/signal_activity/{condition_a}/mean")
        condition_b_mean = _read_2d(f"data/signal_activity/{condition_b}/mean")
        condition_a_sem = _read_2d(f"data/signal_activity/{condition_a}/sem")
        condition_b_sem = _read_2d(f"data/signal_activity/{condition_b}/sem")

        predictor_a_raw_ds = dataset_or_none(fh, f"predictor/{condition_a}/raw_values") or dataset_or_none(fh, f"predictor/{condition_a}_raw_values")
        predictor_b_raw_ds = dataset_or_none(fh, f"predictor/{condition_b}/raw_values") or dataset_or_none(fh, f"predictor/{condition_b}_raw_values")
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
        predictor_a_transformed_ds = dataset_or_none(fh, f"predictor/{condition_a}/transformed_values") or dataset_or_none(fh, f"predictor/{condition_a}_transformed_values")
        predictor_b_transformed_ds = dataset_or_none(fh, f"predictor/{condition_b}/transformed_values") or dataset_or_none(fh, f"predictor/{condition_b}_transformed_values")
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
        predictor_a_ds = dataset_or_none(fh, f"predictor/{condition_a}/values") or dataset_or_none(fh, f"predictor/{condition_a}_values")
        predictor_b_ds = dataset_or_none(fh, f"predictor/{condition_b}/values") or dataset_or_none(fh, f"predictor/{condition_b}_values")
        condition_a_predictor_values = (
            np.asarray(predictor_a_ds[:], dtype=np.float64) if predictor_a_ds is not None else np.array([], dtype=np.float64)
        )
        condition_b_predictor_values = (
            np.asarray(predictor_b_ds[:], dtype=np.float64) if predictor_b_ds is not None else np.array([], dtype=np.float64)
        )

        sfreq = float_scalar(dataset_or_none(fh, "meta/sampling_frequency_hz"), default=0.0)
        predictor = str_scalar(dataset_or_none(fh, "meta/predictor"), default="")
        predictor_zscore = str_scalar(dataset_or_none(fh, "meta/predictor_zscore"), default="")
        if not predictor_zscore:
            raise ValueError(
                f"{path.name}: meta/predictor_zscore is required; old slope-result schemas are not supported."
            )
        predictor_zscore = _validated_predictor_zscore(predictor_zscore, path.name)
        predictor_transform_raw = str_scalar(
            dataset_or_none(fh, "meta/predictor_transform_by_condition_json"),
            default="{}",
        )
        try:
            predictor_transform_by_condition = json.loads(predictor_transform_raw) if predictor_transform_raw else {}
        except json.JSONDecodeError:
            predictor_transform_by_condition = {}
        trial_activity_summary_kind = _validated_trial_activity_summary_kind(
            str_scalar(
                dataset_or_none(fh, "meta/trial_activity_summary_kind"),
                default="epoch_mean",
            ),
            path.name,
        )
        trial_activity_summary_missing_response_policy = _validated_trial_activity_summary_missing_response_policy(
            str_scalar(
            dataset_or_none(fh, "meta/trial_activity_summary_missing_response_policy"),
            default="nan_if_missing",
            )
            or "nan_if_missing",
            path.name,
        )
        trial_activity_summary_source_raw = str_scalar(
            dataset_or_none(fh, "meta/trial_activity_summary_source_json"),
            default="{}",
        )
        try:
            trial_activity_summary_source = (
                json.loads(trial_activity_summary_source_raw)
                if trial_activity_summary_source_raw
                else {}
            )
        except json.JSONDecodeError:
            trial_activity_summary_source = {}
        epoch_cleaning = _load_json_mapping(
            str_scalar(dataset_or_none(fh, "meta/epoch_cleaning_json"), default="{}")
        )
        epoch_cleaning_audit = _load_json_mapping(
            str_scalar(dataset_or_none(fh, "meta/epoch_cleaning_audit_json"), default="{}")
        )
        trial_activity_summary_label = str_scalar(
            dataset_or_none(fh, "meta/trial_activity_summary_label"),
            default="Epoch mean activity",
        ) or "Epoch mean activity"
        p_value_correction_method = str_scalar(dataset_or_none(fh, "meta/p_value_correction_method"), default="fdr_bh")
        significance_alpha = float_scalar(dataset_or_none(fh, "meta/significance_alpha"), default=0.05)
        stats_valid = bool(dataset_or_none(fh, "meta/stats_valid")[()]) if dataset_or_none(fh, "meta/stats_valid") is not None else bool(condition_a_stats_valid or condition_b_stats_valid)
        activity_zscore = str_scalar(
            dataset_or_none(fh, "meta/activity_zscore"),
            default="",
        )
        if not activity_zscore:
            raise ValueError(
                f"{path.name}: meta/activity_zscore is required; old slope-result schemas are not supported."
            )
        activity_baseline_tmin_s = float_scalar(
            dataset_or_none(fh, "meta/activity_baseline_tmin_s"),
            default=-0.2,
        )
        activity_baseline_tmax_s = float_scalar(
            dataset_or_none(fh, "meta/activity_baseline_tmax_s"),
            default=0.0,
        )
        activity_baseline_scope = str_scalar(
            dataset_or_none(fh, "meta/activity_baseline_scope"),
            default="global",
        )
        activity_baseline_scope = _validated_baseline_scope(activity_baseline_scope, path.name)
        activity_baseline_remove_outlier_trial_means = bool(
            dataset_or_none(fh, "meta/activity_baseline_remove_outlier_trial_means")[()]
        ) if dataset_or_none(fh, "meta/activity_baseline_remove_outlier_trial_means") is not None else False

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

        if "trial_activity_summary" in fh:
            tg = fh["trial_activity_summary"]
            condition_a_trial_activity_summary_values = (
                np.asarray(tg[f"{condition_a}_values"][:], dtype=np.float64)
                if f"{condition_a}_values" in tg
                else np.empty((0, 0), dtype=np.float64)
            )
            condition_b_trial_activity_summary_values = (
                np.asarray(tg[f"{condition_b}_values"][:], dtype=np.float64)
                if f"{condition_b}_values" in tg
                else np.empty((n_features, 0), dtype=np.float64)
            )
            trial_activity_summary_kind = _validated_trial_activity_summary_kind(
                str_scalar(dataset_or_none(tg, "kind"), default=trial_activity_summary_kind),
                path.name,
            )
            trial_activity_summary_missing_response_policy = (
                _validated_trial_activity_summary_missing_response_policy(
                    str_scalar(
                        dataset_or_none(tg, "missing_response_policy"),
                        default=trial_activity_summary_missing_response_policy,
                    )
                    or trial_activity_summary_missing_response_policy,
                    path.name,
                )
            )
            trial_activity_summary_source_raw = str_scalar(
                dataset_or_none(tg, "source_json"),
                default=json.dumps(trial_activity_summary_source, sort_keys=True),
            )
            try:
                trial_activity_summary_source = (
                    json.loads(trial_activity_summary_source_raw)
                    if trial_activity_summary_source_raw
                    else {}
                )
            except json.JSONDecodeError:
                trial_activity_summary_source = {}
            trial_activity_summary_label = str_scalar(
                dataset_or_none(tg, "label"),
                default=trial_activity_summary_label,
            ) or trial_activity_summary_label
        else:
            condition_a_trial_activity_summary_values = np.empty((n_features, 0), dtype=np.float64)
            condition_b_trial_activity_summary_values = np.empty((n_features, 0), dtype=np.float64)
            trial_activity_summary_kind = "epoch_mean"
            trial_activity_summary_missing_response_policy = "nan_if_missing"
            trial_activity_summary_source = {}
            trial_activity_summary_label = "Epoch mean activity"

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
    return RegressionProcessingResult(
        source_group=source_group,
        metadata={
            "activity_zscore": activity_zscore,
            "activity_baseline_tmin_s": activity_baseline_tmin_s,
            "activity_baseline_tmax_s": activity_baseline_tmax_s,
            "activity_baseline_scope": activity_baseline_scope,
            "activity_baseline_remove_outlier_trial_means": activity_baseline_remove_outlier_trial_means,
            "trial_activity_summary_kind": trial_activity_summary_kind,
            "trial_activity_summary_missing_response_policy": trial_activity_summary_missing_response_policy,
            "trial_activity_summary_source": trial_activity_summary_source,
            "trial_activity_summary_label": trial_activity_summary_label,
            "epoch_cleaning": epoch_cleaning,
            "epoch_cleaning_audit": epoch_cleaning_audit,
            "available_regression_metrics": available_regression_metrics,
        },
        regression=RegressionStats(
            condition_a=ConditionRegressionStats(
                slope=condition_a_slope,
                intercept=condition_a_intercept,
                r_value=condition_a_r_value,
                p_value=condition_a_p_value,
                p_value_corrected=condition_a_p_value_corrected,
                significant_mask=condition_a_significant_mask,
                n_trials_used=condition_a_trials_used,
                stats_valid=condition_a_stats_valid,
                permuted_slopes=condition_a_permuted_slopes,
            ),
            condition_b=ConditionRegressionStats(
                slope=condition_b_slope,
                intercept=condition_b_intercept,
                r_value=condition_b_r_value,
                p_value=condition_b_p_value,
                p_value_corrected=condition_b_p_value_corrected,
                significant_mask=condition_b_significant_mask,
                n_trials_used=condition_b_trials_used,
                stats_valid=condition_b_stats_valid,
                permuted_slopes=condition_b_permuted_slopes,
            ),
        ),
        predictor_values=RegressionPredictor(
            condition_a=ConditionPredictorValues(
                raw_values=condition_a_predictor_raw_values,
                transformed_values=condition_a_predictor_transformed_values,
                values=condition_a_predictor_values,
            ),
            condition_b=ConditionPredictorValues(
                raw_values=condition_b_predictor_raw_values,
                transformed_values=condition_b_predictor_transformed_values,
                values=condition_b_predictor_values,
            ),
        ),
        signal_activity=ConditionSignalActivity(
            condition_a=SignalActivityEstimate(mean=condition_a_mean, sem=condition_a_sem),
            condition_b=SignalActivityEstimate(mean=condition_b_mean, sem=condition_b_sem),
        ),
        epochs=ConditionEpochs(
            condition_a=condition_a_epochs,
            condition_b=condition_b_epochs,
        ),
        trial_activity_summary_values=ConditionTrialSummaryValues(
            condition_a=condition_a_trial_activity_summary_values,
            condition_b=condition_b_trial_activity_summary_values,
        ),
        time_axis_s=time_axis_s,
        channel_names=channel_names,
        condition_a=condition_a,
        condition_b=condition_b,
        condition_a_trial_count=condition_a_trial_count,
        condition_b_trial_count=condition_b_trial_count,
        sfreq=sfreq,
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
        activity_zscore=activity_zscore,
        activity_baseline_tmin_s=activity_baseline_tmin_s,
        activity_baseline_tmax_s=activity_baseline_tmax_s,
        activity_baseline_scope=activity_baseline_scope,
        activity_baseline_remove_outlier_trial_means=activity_baseline_remove_outlier_trial_means,
        predictor=predictor,
        predictor_zscore=predictor_zscore,
        predictor_transform_by_condition=predictor_transform_by_condition,
        trial_activity_summary_kind=trial_activity_summary_kind,
        trial_activity_summary_missing_response_policy=trial_activity_summary_missing_response_policy,
        trial_activity_summary_source=trial_activity_summary_source,
        trial_activity_summary_label=trial_activity_summary_label,
        epoch_cleaning_audit=epoch_cleaning_audit,
        p_value_correction_method=p_value_correction_method,
        significance_alpha=significance_alpha,
        stats_valid=stats_valid,
    )


def _load_from_matlab(path: Path) -> RegressionProcessingResult:
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    data = mat_root(mat, "regression")
    _require_v2_schema_mat(data.meta, path.name)
    regression = data.stats.regression
    activity = data.data.signal_activity
    uncertainty = getattr(data, "uncertainty", None)
    predictor = getattr(data, "predictor", None)
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

    labels = mat_str_list(getattr(meta, "condition_labels", None))
    condition_a = labels[0] if len(labels) >= 1 else "condition_a"
    condition_b = labels[1] if len(labels) >= 2 else "condition_b"
    available_regression_metrics = mat_str_list(
        getattr(meta, "available_regression_metrics", None)
    )

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

    reg_a = getattr(regression, condition_a, None)
    reg_b = getattr(regression, condition_b, None)

    condition_a_slope = _mat_2d(reg_a, "slope")
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

    _perm_a_raw = getattr(reg_a, "permuted_slopes", None)
    condition_a_permuted_slopes: np.ndarray | None = (
        np.asarray(_perm_a_raw, dtype=np.float32) if _perm_a_raw is not None and np.asarray(_perm_a_raw).size > 0 else None
    )
    _perm_b_raw = getattr(reg_b, "permuted_slopes", None)
    condition_b_permuted_slopes: np.ndarray | None = (
        np.asarray(_perm_b_raw, dtype=np.float32) if _perm_b_raw is not None and np.asarray(_perm_b_raw).size > 0 else None
    )

    condition_a_mean = _mat_2d(getattr(activity, condition_a, None), "mean")
    condition_b_mean = _mat_2d(getattr(activity, condition_b, None), "mean")
    condition_a_sem = _mat_2d(getattr(activity, condition_a, None), "sem")
    condition_b_sem = _mat_2d(getattr(activity, condition_b, None), "sem")
    if not available_regression_metrics:
        available_regression_metrics = [
            metric
            for metric, present in {
                "slope": (
                    getattr(reg_a, "slope", None) is not None
                    and getattr(reg_b, "slope", None) is not None
                ),
                "r_value": (
                    getattr(reg_a, "r_value", None) is not None
                    and getattr(reg_b, "r_value", None) is not None
                ),
            }.items()
            if present
        ]

    condition_a_predictor_raw_values = np.asarray(
        getattr(getattr(predictor, condition_a, None), "raw_values", None)
        if getattr(predictor, condition_a, None) is not None
        else getattr(predictor, f"{condition_a}_raw_values", np.array([], dtype=np.float64)),
        dtype=np.float64,
    ).ravel()
    condition_b_predictor_raw_values = np.asarray(
        getattr(getattr(predictor, condition_b, None), "raw_values", None)
        if getattr(predictor, condition_b, None) is not None
        else getattr(predictor, f"{condition_b}_raw_values", np.array([], dtype=np.float64)),
        dtype=np.float64,
    ).ravel()
    condition_a_predictor_transformed_values = np.asarray(
        getattr(getattr(predictor, condition_a, None), "transformed_values", None)
        if getattr(predictor, condition_a, None) is not None
        else getattr(predictor, f"{condition_a}_transformed_values", np.array([], dtype=np.float64)),
        dtype=np.float64,
    ).ravel()
    condition_b_predictor_transformed_values = np.asarray(
        getattr(getattr(predictor, condition_b, None), "transformed_values", None)
        if getattr(predictor, condition_b, None) is not None
        else getattr(predictor, f"{condition_b}_transformed_values", np.array([], dtype=np.float64)),
        dtype=np.float64,
    ).ravel()
    condition_a_predictor_values = np.asarray(
        getattr(getattr(predictor, condition_a, None), "values", None)
        if getattr(predictor, condition_a, None) is not None
        else getattr(predictor, f"{condition_a}_values", np.array([], dtype=np.float64)),
        dtype=np.float64,
    ).ravel()
    condition_b_predictor_values = np.asarray(
        getattr(getattr(predictor, condition_b, None), "values", None)
        if getattr(predictor, condition_b, None) is not None
        else getattr(predictor, f"{condition_b}_values", np.array([], dtype=np.float64)),
        dtype=np.float64,
    ).ravel()
    trial_activity_summary_kind = "epoch_mean"
    trial_activity_summary_missing_response_policy = "nan_if_missing"
    trial_activity_summary_source = {}
    trial_activity_summary_label = "Epoch mean activity"
    trial_activity_summary = getattr(data, "trial_activity_summary", None)
    if trial_activity_summary is not None:
        condition_a_trial_activity_summary_values = _mat_feature_trial_2d(
            trial_activity_summary,
            f"{condition_a}_values",
        )
        condition_b_trial_activity_summary_values = _mat_feature_trial_2d(
            trial_activity_summary,
            f"{condition_b}_values",
        )
        trial_activity_summary_kind = _validated_trial_activity_summary_kind(
            mat_str(
                getattr(trial_activity_summary, "kind", None),
                default=trial_activity_summary_kind,
            ),
            path.name,
        )
        trial_activity_summary_missing_response_policy = (
            _validated_trial_activity_summary_missing_response_policy(
                mat_str(
                    getattr(trial_activity_summary, "missing_response_policy", None),
                    default=trial_activity_summary_missing_response_policy,
                )
                or trial_activity_summary_missing_response_policy,
                path.name,
            )
        )
        trial_activity_summary_source_raw = mat_str(
            getattr(trial_activity_summary, "source_json", None),
            default=json.dumps(trial_activity_summary_source, sort_keys=True),
        )
        try:
            trial_activity_summary_source = (
                json.loads(trial_activity_summary_source_raw)
                if trial_activity_summary_source_raw
                else {}
            )
        except json.JSONDecodeError:
            trial_activity_summary_source = {}
        trial_activity_summary_label = mat_str(
            getattr(trial_activity_summary, "label", None),
            default=trial_activity_summary_label,
        ) or trial_activity_summary_label
    else:
        condition_a_trial_activity_summary_values = np.empty((n_features, 0), dtype=np.float64)
        condition_b_trial_activity_summary_values = np.empty((n_features, 0), dtype=np.float64)
        trial_activity_summary_kind = "epoch_mean"
        trial_activity_summary_missing_response_policy = "nan_if_missing"
        trial_activity_summary_source = {}
        trial_activity_summary_label = "Epoch mean activity"

    counts_raw = getattr(meta, "trial_counts", None)
    counts = np.asarray(counts_raw, dtype=np.int64).ravel() if counts_raw is not None else np.array([], dtype=np.int64)
    condition_a_trial_count = int(counts[0]) if len(counts) >= 1 else 0
    condition_b_trial_count = int(counts[1]) if len(counts) >= 2 else 0

    sfreq = mat_float(getattr(meta, "sampling_frequency_hz", None), default=0.0)
    predictor = mat_str(getattr(meta, "predictor", None), default="")
    predictor_zscore = mat_str(getattr(meta, "predictor_zscore", None), default="")
    if not predictor_zscore:
        raise ValueError(
            f"{path.name}: meta.predictor_zscore is required; old slope-result schemas are not supported."
        )
    predictor_zscore = _validated_predictor_zscore(predictor_zscore, path.name)
    predictor_transform_raw = mat_str(
        getattr(meta, "predictor_transform_by_condition_json", None),
        default="{}",
    )
    try:
        predictor_transform_by_condition = json.loads(predictor_transform_raw) if predictor_transform_raw else {}
    except json.JSONDecodeError:
        predictor_transform_by_condition = {}
    trial_activity_summary_kind = _validated_trial_activity_summary_kind(
        mat_str(getattr(meta, "trial_activity_summary_kind", None), default="epoch_mean"),
        path.name,
    )
    trial_activity_summary_missing_response_policy = (
        _validated_trial_activity_summary_missing_response_policy(
            mat_str(
                getattr(meta, "trial_activity_summary_missing_response_policy", None),
                default="nan_if_missing",
            )
            or "nan_if_missing",
            path.name,
        )
    )
    trial_activity_summary_source_raw = mat_str(
        getattr(meta, "trial_activity_summary_source_json", None),
        default="{}",
    )
    try:
        trial_activity_summary_source = (
            json.loads(trial_activity_summary_source_raw)
            if trial_activity_summary_source_raw
            else {}
        )
    except json.JSONDecodeError:
        trial_activity_summary_source = {}
    epoch_cleaning = _load_json_mapping(
        mat_str(getattr(meta, "epoch_cleaning_json", None), default="{}")
    )
    epoch_cleaning_audit = _load_json_mapping(
        mat_str(getattr(meta, "epoch_cleaning_audit_json", None), default="{}")
    )
    trial_activity_summary_label = mat_str(
        getattr(meta, "trial_activity_summary_label", None),
        default="Epoch mean activity",
    ) or "Epoch mean activity"
    p_value_correction_method = mat_str(getattr(meta, "p_value_correction_method", None), default="fdr_bh")
    significance_alpha = mat_float(getattr(meta, "significance_alpha", None), default=0.05)
    stats_valid = bool(mat_int(getattr(meta, "stats_valid", None), default=int(condition_a_stats_valid or condition_b_stats_valid)))
    activity_zscore = mat_str(getattr(meta, "activity_zscore", None), default="")
    if not activity_zscore:
        raise ValueError(
            f"{path.name}: meta.activity_zscore is required; old slope-result schemas are not supported."
        )
    activity_baseline_tmin_s = mat_float(
        getattr(meta, "activity_baseline_tmin_s", None),
        default=-0.2,
    )
    activity_baseline_tmax_s = mat_float(
        getattr(meta, "activity_baseline_tmax_s", None),
        default=0.0,
    )
    activity_baseline_scope = _validated_baseline_scope(
        mat_str(getattr(meta, "activity_baseline_scope", None), default="global"),
        path.name,
    )
    activity_baseline_remove_outlier_trial_means = bool(
        mat_int(
            getattr(meta, "activity_baseline_remove_outlier_trial_means", None),
            default=0,
        )
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
    return RegressionProcessingResult(
        source_group=source_group,
        metadata={
            "activity_zscore": activity_zscore,
            "activity_baseline_tmin_s": activity_baseline_tmin_s,
            "activity_baseline_tmax_s": activity_baseline_tmax_s,
            "activity_baseline_scope": activity_baseline_scope,
            "activity_baseline_remove_outlier_trial_means": activity_baseline_remove_outlier_trial_means,
            "trial_activity_summary_kind": trial_activity_summary_kind,
            "trial_activity_summary_missing_response_policy": trial_activity_summary_missing_response_policy,
            "trial_activity_summary_source": trial_activity_summary_source,
            "trial_activity_summary_label": trial_activity_summary_label,
            "epoch_cleaning": epoch_cleaning,
            "epoch_cleaning_audit": epoch_cleaning_audit,
            "available_regression_metrics": available_regression_metrics,
        },
        regression=RegressionStats(
            condition_a=ConditionRegressionStats(
                slope=condition_a_slope,
                intercept=condition_a_intercept,
                r_value=condition_a_r_value,
                p_value=condition_a_p_value,
                p_value_corrected=condition_a_p_value_corrected,
                significant_mask=condition_a_significant_mask,
                n_trials_used=condition_a_trials_used,
                stats_valid=condition_a_stats_valid,
                permuted_slopes=condition_a_permuted_slopes,
            ),
            condition_b=ConditionRegressionStats(
                slope=condition_b_slope,
                intercept=condition_b_intercept,
                r_value=condition_b_r_value,
                p_value=condition_b_p_value,
                p_value_corrected=condition_b_p_value_corrected,
                significant_mask=condition_b_significant_mask,
                n_trials_used=condition_b_trials_used,
                stats_valid=condition_b_stats_valid,
                permuted_slopes=condition_b_permuted_slopes,
            ),
        ),
        predictor_values=RegressionPredictor(
            condition_a=ConditionPredictorValues(
                raw_values=condition_a_predictor_raw_values,
                transformed_values=condition_a_predictor_transformed_values,
                values=condition_a_predictor_values,
            ),
            condition_b=ConditionPredictorValues(
                raw_values=condition_b_predictor_raw_values,
                transformed_values=condition_b_predictor_transformed_values,
                values=condition_b_predictor_values,
            ),
        ),
        signal_activity=ConditionSignalActivity(
            condition_a=SignalActivityEstimate(mean=condition_a_mean, sem=condition_a_sem),
            condition_b=SignalActivityEstimate(mean=condition_b_mean, sem=condition_b_sem),
        ),
        epochs=ConditionEpochs(
            condition_a=np.array([]),
            condition_b=np.array([]),
        ),
        trial_activity_summary_values=ConditionTrialSummaryValues(
            condition_a=condition_a_trial_activity_summary_values,
            condition_b=condition_b_trial_activity_summary_values,
        ),
        time_axis_s=time_axis_s,
        channel_names=channel_names,
        condition_a=condition_a,
        condition_b=condition_b,
        condition_a_trial_count=condition_a_trial_count,
        condition_b_trial_count=condition_b_trial_count,
        sfreq=sfreq,
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
        activity_zscore=activity_zscore,
        activity_baseline_tmin_s=activity_baseline_tmin_s,
        activity_baseline_tmax_s=activity_baseline_tmax_s,
        activity_baseline_scope=activity_baseline_scope,
        activity_baseline_remove_outlier_trial_means=activity_baseline_remove_outlier_trial_means,
        predictor=predictor,
        predictor_zscore=predictor_zscore,
        predictor_transform_by_condition=predictor_transform_by_condition,
        trial_activity_summary_kind=trial_activity_summary_kind,
        trial_activity_summary_missing_response_policy=trial_activity_summary_missing_response_policy,
        trial_activity_summary_source=trial_activity_summary_source,
        trial_activity_summary_label=trial_activity_summary_label,
        epoch_cleaning_audit=epoch_cleaning_audit,
        p_value_correction_method=p_value_correction_method,
        significance_alpha=significance_alpha,
        stats_valid=stats_valid,
    )


def _validated_predictor_zscore(value: str, path_name: str) -> str:
    cleaned = str(value).strip().lower()
    if cleaned == "within_condition":
        raise ValueError(
            f"{path_name}: meta/predictor_zscore='within_condition' is no longer supported. "
            "Use 'condition' or recompute the result with the updated pipeline."
        )
    if cleaned not in _VALID_PREDICTOR_ZSCORE_MODES:
        raise ValueError(
            f"{path_name}: unsupported predictor_zscore={value!r}. "
            "Valid values are 'none', 'condition', and 'global'."
        )
    return cleaned


def _validated_baseline_scope(value: str, path_name: str) -> str:
    cleaned = str(value).strip().lower() or "global"
    if cleaned not in _VALID_BASELINE_SCOPES:
        raise ValueError(
            f"{path_name}: unsupported activity_baseline_scope={value!r}. "
            "Valid values are 'trial', 'condition', and 'global'."
        )
    return cleaned


def _validated_trial_activity_summary_kind(value: str, path_name: str) -> str:
    cleaned = str(value).strip().lower() or "epoch_mean"
    if cleaned not in _VALID_TRIAL_ACTIVITY_SUMMARY_KINDS:
        raise ValueError(
            f"{path_name}: unsupported trial_activity_summary_kind={value!r}. "
            "Valid values are 'epoch_mean' and 'anchor_to_response_mean'."
        )
    return cleaned


def _validated_trial_activity_summary_missing_response_policy(
    value: str,
    path_name: str,
) -> str:
    cleaned = str(value).strip().lower() or "nan_if_missing"
    if cleaned not in _VALID_TRIAL_ACTIVITY_SUMMARY_MISSING_RESPONSE_POLICIES:
        raise ValueError(
            f"{path_name}: unsupported trial_activity_summary_missing_response_policy="
            f"{value!r}. Valid values are 'clamp_to_epoch' and 'nan_if_missing'."
        )
    try:
        return normalize_trial_activity_summary_missing_response_policy(cleaned)
    except ValueError as exc:
        raise ValueError(f"{path_name}: {exc}") from exc


def _load_json_mapping(raw_value: str) -> dict[str, object]:
    try:
        loaded = json.loads(raw_value) if raw_value else {}
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {str(key): value for key, value in loaded.items()}


def _require_v2_schema(fh: h5py.File, path_name: str) -> None:
    schema_version = str_scalar(dataset_or_none(fh, "meta/schema_version"), default="")
    if schema_version != "3.0":
        raise ValueError(
            f"{path_name}: unsupported trial_stats schema. "
            "schema_version='3.0' is required; regenerate outputs with the v3 writer."
        )


def _require_v2_schema_mat(meta: object, path_name: str) -> None:
    schema_version = mat_str(getattr(meta, "schema_version", None), default="")
    if schema_version != "3.0":
        raise ValueError(
            f"{path_name}: unsupported trial_stats schema. "
            "schema_version='3.0' is required; regenerate outputs with the v3 writer."
        )

