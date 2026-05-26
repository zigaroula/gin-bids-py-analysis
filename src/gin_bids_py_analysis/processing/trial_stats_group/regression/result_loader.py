"""Load a pre-computed ``RegressionGroupProcessingResult`` from disk."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.utils.hdf5 import (
    dataset_or_none,
    decode_str_array,
    float_scalar,
    str_scalar,
)
from gin_bids_py_analysis.processing.trial_stats.params import (
    normalize_trial_activity_summary_missing_response_policy,
)

from ..result import (
    GroupEpochStats,
    GroupEstimate,
    GroupEstimatePair,
    GroupTimecourseStats,
    IndexedConditionContributions,
    ROIChannelContribution,
)
from .result import (
    RegressionGroupProcessingResult,
    RegressionMetricStats,
    ScatterData,
    VsZeroStatsPair,
)


def load_regression_group_result(
    path: Path | str,
) -> RegressionGroupProcessingResult:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Group regression file not found: {path}")
    if path.suffix.lower() == ".mat":
        return _load_from_matlab(path)
    return _load_from_hdf5(path)


def _load_from_hdf5(path: Path) -> RegressionGroupProcessingResult:
    with h5py.File(path, "r") as fh:
        _require_v3_schema(fh, path.name)
        region_names = decode_str_array(np.asarray(fh["axes"]["region"][:]))
        time_axis_s = np.asarray(fh["axes"]["time_s"][:], dtype=np.float64)
        n_rois = len(region_names)
        n_t = len(time_axis_s)

        def _read_2d(key: str, fill: float = 0.0) -> np.ndarray:
            ds = dataset_or_none(fh, key)
            if ds is None:
                return np.full((n_rois, n_t), fill, dtype=np.float64)
            return np.asarray(ds[:], dtype=np.float64).reshape(n_rois, n_t)

        def _read_1d(key: str, fill: float = 0.0) -> np.ndarray:
            ds = dataset_or_none(fh, key)
            if ds is None:
                return np.full(n_rois, fill, dtype=np.float64)
            return np.asarray(ds[:], dtype=np.float64).ravel()

        def _read_1d_int(key: str) -> np.ndarray:
            ds = dataset_or_none(fh, key)
            if ds is None:
                return np.zeros(n_rois, dtype=np.int64)
            return np.asarray(ds[:], dtype=np.int64).ravel()

        significance_alpha = float_scalar(
            dataset_or_none(fh, "meta/significance_alpha"),
            default=0.05,
        )

        _early_labels = decode_str_array(
            np.asarray(fh["meta"]["condition_labels"][:], dtype=object)
        )
        label_a = _early_labels[0] if len(_early_labels) >= 1 else "condition_a"
        label_b = _early_labels[1] if len(_early_labels) >= 2 else "condition_b"

        source_metric_t_values = _read_2d("stats/regression/condition_contrast/t_values")
        source_metric_p_values = _read_2d("stats/regression/condition_contrast/p_values", fill=1.0)
        source_metric_p_values_uncorrected = _read_2d(
            "stats/regression/condition_contrast/p_values_uncorrected",
            fill=1.0,
        )
        sig_metric_ds = dataset_or_none(fh, "stats/regression/condition_contrast/significant_mask")
        if sig_metric_ds is not None:
            source_metric_significant_mask = np.asarray(sig_metric_ds[:], dtype=bool)
        else:
            source_metric_significant_mask = (
                np.isfinite(source_metric_p_values)
                & (source_metric_p_values < significance_alpha)
            )
        condition_a_source_metric_mean = _read_2d(f"data/regression/{label_a}/slope/mean")
        condition_a_source_metric_sem = _read_2d(f"data/regression/{label_a}/slope/sem")
        condition_b_source_metric_mean = _read_2d(f"data/regression/{label_b}/slope/mean")
        condition_b_source_metric_sem = _read_2d(f"data/regression/{label_b}/slope/sem")
        epoch_source_metric_t = _read_1d("stats/regression/condition_contrast/epoch_summary/t")
        epoch_source_metric_p = _read_1d("stats/regression/condition_contrast/epoch_summary/p", fill=1.0)
        epoch_source_metric_df = _read_1d("stats/regression/condition_contrast/epoch_summary/df")

        # Per-condition vs-zero — graceful fallback for old files
        condition_a_vs_zero_t = _read_2d(f"stats/regression/{label_a}_vs_zero/t_values")
        # p_values is the corrected p (new files); fall back to p_values for old files that only
        # had one p dataset (which was uncorrected in the original impl).
        condition_a_vs_zero_p_uncorr_ds = dataset_or_none(
            fh, f"stats/regression/{label_a}_vs_zero/p_values_uncorrected"
        )
        if condition_a_vs_zero_p_uncorr_ds is not None:
            condition_a_vs_zero_p_uncorr = np.asarray(
                condition_a_vs_zero_p_uncorr_ds[:], dtype=np.float64
            ).reshape(n_rois, n_t)
        else:
            condition_a_vs_zero_p_uncorr = _read_2d(
                f"stats/regression/{label_a}_vs_zero/p_values", fill=1.0
            )
        condition_a_vs_zero_p = _read_2d(f"stats/regression/{label_a}_vs_zero/p_values", fill=1.0)
        sig_a_vz_ds = dataset_or_none(fh, f"stats/regression/{label_a}_vs_zero/significant_mask")
        if sig_a_vz_ds is not None:
            condition_a_vs_zero_sig = np.asarray(sig_a_vz_ds[:], dtype=bool).reshape(n_rois, n_t)
        else:
            condition_a_vs_zero_sig = np.isfinite(condition_a_vs_zero_p) & (condition_a_vs_zero_p < significance_alpha)
        # Cluster stats for condition A vs zero
        vz_a_cluster_p: np.ndarray | None = None
        vz_a_cluster_windows: list[list[tuple[float, float]]] | None = None
        vz_a_cluster_null_dists: list[np.ndarray] | None = None
        if f"stats/regression/{label_a}_vs_zero/cluster" in fh:
            cs_a = fh[f"stats/regression/{label_a}_vs_zero/cluster"]
            vz_a_cluster_p = np.asarray(cs_a["p_values"][:], dtype=np.float64)
            starts_a = np.asarray(cs_a["cluster_starts_s"][:], dtype=np.float64)
            ends_a = np.asarray(cs_a["cluster_ends_s"][:], dtype=np.float64)
            if starts_a.ndim == 1:
                starts_a = starts_a[:, np.newaxis]
                ends_a = ends_a[:, np.newaxis]
            vz_a_cluster_windows = [
                [
                    (float(s), float(e))
                    for s, e in zip(row_s, row_e)
                    if np.isfinite(s) and np.isfinite(e)
                ]
                for row_s, row_e in zip(starts_a, ends_a)
            ]
            null_mat_a = np.asarray(cs_a["null_distributions"][:], dtype=np.float64)
            vz_a_cluster_null_dists = [
                null_mat_a[i, np.isfinite(null_mat_a[i])]
                for i in range(null_mat_a.shape[0])
            ]
            # Override sig mask from cluster windows
            condition_a_vs_zero_sig = np.zeros((n_rois, n_t), dtype=bool)
            for roi_idx, roi_windows in enumerate(vz_a_cluster_windows):
                for t_start_s, t_end_s in roi_windows:
                    in_window = (time_axis_s >= t_start_s) & (time_axis_s <= t_end_s)
                    condition_a_vs_zero_sig[roi_idx, in_window] = True

        condition_b_vs_zero_t = _read_2d(f"stats/regression/{label_b}_vs_zero/t_values")
        condition_b_vs_zero_p_uncorr_ds = dataset_or_none(
            fh, f"stats/regression/{label_b}_vs_zero/p_values_uncorrected"
        )
        if condition_b_vs_zero_p_uncorr_ds is not None:
            condition_b_vs_zero_p_uncorr = np.asarray(
                condition_b_vs_zero_p_uncorr_ds[:], dtype=np.float64
            ).reshape(n_rois, n_t)
        else:
            condition_b_vs_zero_p_uncorr = _read_2d(
                f"stats/regression/{label_b}_vs_zero/p_values", fill=1.0
            )
        condition_b_vs_zero_p = _read_2d(f"stats/regression/{label_b}_vs_zero/p_values", fill=1.0)
        sig_b_vz_ds = dataset_or_none(fh, f"stats/regression/{label_b}_vs_zero/significant_mask")
        if sig_b_vz_ds is not None:
            condition_b_vs_zero_sig = np.asarray(sig_b_vz_ds[:], dtype=bool).reshape(n_rois, n_t)
        else:
            condition_b_vs_zero_sig = np.isfinite(condition_b_vs_zero_p) & (condition_b_vs_zero_p < significance_alpha)
        # Cluster stats for condition B vs zero
        vz_b_cluster_p: np.ndarray | None = None
        vz_b_cluster_windows: list[list[tuple[float, float]]] | None = None
        vz_b_cluster_null_dists: list[np.ndarray] | None = None
        if f"stats/regression/{label_b}_vs_zero/cluster" in fh:
            cs_b = fh[f"stats/regression/{label_b}_vs_zero/cluster"]
            vz_b_cluster_p = np.asarray(cs_b["p_values"][:], dtype=np.float64)
            starts_b = np.asarray(cs_b["cluster_starts_s"][:], dtype=np.float64)
            ends_b = np.asarray(cs_b["cluster_ends_s"][:], dtype=np.float64)
            if starts_b.ndim == 1:
                starts_b = starts_b[:, np.newaxis]
                ends_b = ends_b[:, np.newaxis]
            vz_b_cluster_windows = [
                [
                    (float(s), float(e))
                    for s, e in zip(row_s, row_e)
                    if np.isfinite(s) and np.isfinite(e)
                ]
                for row_s, row_e in zip(starts_b, ends_b)
            ]
            null_mat_b = np.asarray(cs_b["null_distributions"][:], dtype=np.float64)
            vz_b_cluster_null_dists = [
                null_mat_b[i, np.isfinite(null_mat_b[i])]
                for i in range(null_mat_b.shape[0])
            ]
            # Override sig mask from cluster windows
            condition_b_vs_zero_sig = np.zeros((n_rois, n_t), dtype=bool)
            for roi_idx, roi_windows in enumerate(vz_b_cluster_windows):
                for t_start_s, t_end_s in roi_windows:
                    in_window = (time_axis_s >= t_start_s) & (time_axis_s <= t_end_s)
                    condition_b_vs_zero_sig[roi_idx, in_window] = True

        activity_t_values = _read_2d("stats/signal_activity/t_values")
        activity_p_values = _read_2d("stats/signal_activity/p_values", fill=1.0)
        activity_p_values_uncorrected = _read_2d(
            "stats/signal_activity/p_values_uncorrected",
            fill=1.0,
        )
        sig_activity_ds = dataset_or_none(fh, "stats/signal_activity/significant_mask")
        if sig_activity_ds is not None:
            activity_significant_mask = np.asarray(sig_activity_ds[:], dtype=bool)
        else:
            activity_significant_mask = np.isfinite(activity_p_values) & (
                activity_p_values < significance_alpha
            )
        epoch_activity_t = _read_1d("stats/signal_activity/epoch_summary/t")
        epoch_activity_p = _read_1d("stats/signal_activity/epoch_summary/p", fill=1.0)
        epoch_activity_df = _read_1d("stats/signal_activity/epoch_summary/df")

        condition_a_signal_activity_summary_mean = _read_2d(f"data/signal_activity/{label_a}/mean")
        condition_a_signal_activity_summary_sem = _read_2d(f"data/signal_activity/{label_a}/sem")
        condition_b_signal_activity_summary_mean = _read_2d(f"data/signal_activity/{label_b}/mean")
        condition_b_signal_activity_summary_sem = _read_2d(f"data/signal_activity/{label_b}/sem")

        condition_a_r_value_mean = _read_2d(f"data/regression/{label_a}/r_value/mean")
        condition_a_r_value_sem = _read_2d(f"data/regression/{label_a}/r_value/sem")
        condition_b_r_value_mean = _read_2d(f"data/regression/{label_b}/r_value/mean")
        condition_b_r_value_sem = _read_2d(f"data/regression/{label_b}/r_value/sem")

        roi_channel_counts = _read_1d_int("meta/roi_channel_counts")
        roi_subject_counts = _read_1d_int("meta/roi_subject_counts")

        labels = decode_str_array(
            np.asarray(fh["meta"]["condition_labels"][:], dtype=object)
        )
        condition_labels: tuple[str, str] = (
            labels[0] if len(labels) >= 1 else "condition_a",
            labels[1] if len(labels) >= 2 else "condition_b",
        )
        primary_regression_metric = str_scalar(
            dataset_or_none(fh, "meta/primary_regression_metric"),
            default="slope",
        )
        contrast_mode = str_scalar(
            dataset_or_none(fh, "meta/contrast_mode"),
            default="paired",
        )
        p_value_correction_method = str_scalar(
            dataset_or_none(fh, "meta/p_value_correction_method"),
            default="none",
        )
        roi_mode = str_scalar(dataset_or_none(fh, "meta/roi_mode"), default="manual")
        atlas_name_raw = str_scalar(dataset_or_none(fh, "meta/atlas_name"), default="")
        atlas_name: str | None = atlas_name_raw.strip() or None

        metadata = {
            "p_value_correction_method": p_value_correction_method,
            "significance_alpha": significance_alpha,
            "primary_regression_metric": primary_regression_metric,
            "contrast_mode": contrast_mode,
            "roi_mode": roi_mode,
            "atlas_name": atlas_name,
            "predictor": str_scalar(dataset_or_none(fh, "meta/predictor"), default=""),
            "predictor_zscore": str_scalar(
                dataset_or_none(fh, "meta/predictor_zscore"),
                default="none",
            ),
            "predictor_transform_by_condition_json": str_scalar(
                dataset_or_none(fh, "meta/predictor_transform_by_condition_json"),
                default="{}",
            ),
            "activity_zscore": str_scalar(
                dataset_or_none(fh, "meta/activity_zscore"),
                default="none",
            ),
            "activity_baseline_tmin_s": float_scalar(
                dataset_or_none(fh, "meta/activity_baseline_tmin_s"),
                default=-0.2,
            ),
            "activity_baseline_tmax_s": float_scalar(
                dataset_or_none(fh, "meta/activity_baseline_tmax_s"),
                default=0.0,
            ),
            "trial_activity_summary_kind": str_scalar(
                dataset_or_none(fh, "meta/trial_activity_summary_kind"),
                default="epoch_mean",
            ),
            "trial_activity_summary_missing_response_policy": (
                normalize_trial_activity_summary_missing_response_policy(
                    str_scalar(
                        dataset_or_none(
                            fh,
                            "meta/trial_activity_summary_missing_response_policy",
                        ),
                        default="nan_if_missing",
                    )
                )
            ),
            "trial_activity_summary_source_json": str_scalar(
                dataset_or_none(fh, "meta/trial_activity_summary_source_json"),
                default="{}",
            ),
            "trial_activity_summary_label": str_scalar(
                dataset_or_none(fh, "meta/trial_activity_summary_label"),
                default="Epoch mean activity",
            ),
            "scatter_aggregation": str_scalar(
                dataset_or_none(fh, "meta/scatter_aggregation"),
                default="trial_pool",
            ),
        }
        for key in ("binning_mode", "window_ms", "n_bins", "effective_n_bins"):
            ds = dataset_or_none(fh, f"meta/{key}")
            if ds is None:
                continue
            value = ds[()]
            if isinstance(value, (bytes, np.bytes_)):
                metadata[key] = value.decode("utf-8")
            elif isinstance(value, (np.floating, float)):
                metadata[key] = float(value)
            else:
                metadata[key] = int(value)

        excluded_rois = _read_excluded_rois_hdf5(fh)
        contributions = _read_contributions_hdf5(fh)
        (
            condition_a_signal_activity_summary_contributions,
            condition_b_signal_activity_summary_contributions,
            contribution_labels,
        ) = _read_indexed_condition_contributions_hdf5(
            fh,
            group_name="signal_activity",
            region_names=region_names,
            n_t=n_t,
            label_a=label_a,
            label_b=label_b,
        )
        (
            condition_a_source_metric_contributions,
            condition_b_source_metric_contributions,
            _,
        ) = _read_indexed_condition_contributions_hdf5(
            fh,
            group_name="regression/slope",
            region_names=region_names,
            n_t=n_t,
            label_a=label_a,
            label_b=label_b,
        )
        (
            condition_a_scatter_predictor,
            condition_a_scatter_activity,
            condition_b_scatter_predictor,
            condition_b_scatter_activity,
        ) = _read_scatter_hdf5(fh, region_names=region_names, label_a=label_a, label_b=label_b)

        source_subject_stats_files: list[str] = []
        source_electrodes_files: list[str] = []
        if "provenance" in fh:
            prov = fh["provenance"]
            if "source_subject_stats_files" in prov:
                source_subject_stats_files = decode_str_array(
                    np.asarray(prov["source_subject_stats_files"][:], dtype=object)
                )
            if "source_electrodes_files" in prov:
                source_electrodes_files = decode_str_array(
                    np.asarray(prov["source_electrodes_files"][:], dtype=object)
                )

        hdf5_cluster_p_values: np.ndarray | None = None
        hdf5_cluster_windows: list[list[tuple[float, float]]] | None = None
        hdf5_cluster_null_dists: list[np.ndarray] | None = None
        if "stats/cluster" in fh:
            cs = fh["stats/cluster"]
            hdf5_cluster_p_values = np.asarray(cs["p_values"][:], dtype=np.float64)
            starts = np.asarray(cs["cluster_starts_s"][:], dtype=np.float64)
            ends = np.asarray(cs["cluster_ends_s"][:], dtype=np.float64)
            if starts.ndim == 1:
                starts = starts[:, np.newaxis]
                ends = ends[:, np.newaxis]
            hdf5_cluster_windows = [
                [
                    (float(s), float(e))
                    for s, e in zip(row_s, row_e)
                    if np.isfinite(s) and np.isfinite(e)
                ]
                for row_s, row_e in zip(starts, ends)
            ]
            null_matrix = np.asarray(cs["null_distributions"][:], dtype=np.float64)
            hdf5_cluster_null_dists = [
                null_matrix[i, np.isfinite(null_matrix[i])]
                for i in range(null_matrix.shape[0])
            ]

    source_group = BIDSFileGroup(primary=BIDSFile.from_path(path))
    return RegressionGroupProcessingResult(
        source_group=source_group,
        metadata=metadata,
        regression_stats=RegressionMetricStats(
            contrast=GroupTimecourseStats(
                t_values=source_metric_t_values,
                p_values=source_metric_p_values,
                p_values_uncorrected=source_metric_p_values_uncorrected,
                significant_mask=source_metric_significant_mask,
            ),
            epoch_summary=GroupEpochStats(
                t=epoch_source_metric_t,
                p=epoch_source_metric_p,
                df=epoch_source_metric_df,
            ),
            vs_zero=VsZeroStatsPair(
                condition_a=GroupTimecourseStats(
                    t_values=condition_a_vs_zero_t,
                    p_values=condition_a_vs_zero_p,
                    p_values_uncorrected=condition_a_vs_zero_p_uncorr,
                    significant_mask=condition_a_vs_zero_sig,
                ),
                condition_b=GroupTimecourseStats(
                    t_values=condition_b_vs_zero_t,
                    p_values=condition_b_vs_zero_p,
                    p_values_uncorrected=condition_b_vs_zero_p_uncorr,
                    significant_mask=condition_b_vs_zero_sig,
                ),
                condition_a_cluster_p_values=vz_a_cluster_p,
                condition_a_cluster_windows_s=vz_a_cluster_windows,
                condition_a_cluster_null_distributions=vz_a_cluster_null_dists,
                condition_b_cluster_p_values=vz_b_cluster_p,
                condition_b_cluster_windows_s=vz_b_cluster_windows,
                condition_b_cluster_null_distributions=vz_b_cluster_null_dists,
            ),
        ),
        signal_activity_stats=GroupTimecourseStats(
            t_values=activity_t_values,
            p_values=activity_p_values,
            p_values_uncorrected=activity_p_values_uncorrected,
            significant_mask=activity_significant_mask,
        ),
        slope=GroupEstimatePair(
            condition_a=GroupEstimate(
                mean=condition_a_source_metric_mean,
                sem=condition_a_source_metric_sem,
            ),
            condition_b=GroupEstimate(
                mean=condition_b_source_metric_mean,
                sem=condition_b_source_metric_sem,
            ),
        ),
        signal_activity=GroupEstimatePair(
            condition_a=GroupEstimate(
                mean=condition_a_signal_activity_summary_mean,
                sem=condition_a_signal_activity_summary_sem,
            ),
            condition_b=GroupEstimate(
                mean=condition_b_signal_activity_summary_mean,
                sem=condition_b_signal_activity_summary_sem,
            ),
        ),
        r_value=GroupEstimatePair(
            condition_a=GroupEstimate(
                mean=condition_a_r_value_mean,
                sem=condition_a_r_value_sem,
            ),
            condition_b=GroupEstimate(
                mean=condition_b_r_value_mean,
                sem=condition_b_r_value_sem,
            ),
        ),
        signal_activity_epoch=GroupEpochStats(
            t=epoch_activity_t,
            p=epoch_activity_p,
            df=epoch_activity_df,
        ),
        time_axis_s=time_axis_s,
        region_names=region_names,
        condition_labels=condition_labels,
        primary_regression_metric=primary_regression_metric,
        contrast_mode=contrast_mode,
        roi_channel_counts=roi_channel_counts,
        roi_subject_counts=roi_subject_counts,
        contributions=contributions,
        slope_contributions=IndexedConditionContributions(
            condition_a=condition_a_source_metric_contributions,
            condition_b=condition_b_source_metric_contributions,
            labels=contribution_labels,
        ),
        signal_activity_contributions=IndexedConditionContributions(
            condition_a=condition_a_signal_activity_summary_contributions,
            condition_b=condition_b_signal_activity_summary_contributions,
            labels=contribution_labels,
        ),
        p_value_correction_method=p_value_correction_method,
        significance_alpha=significance_alpha,
        roi_mode=roi_mode,
        atlas_name=atlas_name,
        source_subject_stats_files=source_subject_stats_files,
        source_electrodes_files=source_electrodes_files,
        excluded_rois=excluded_rois,
        scatter=ScatterData(
            condition_a_predictor=condition_a_scatter_predictor,
            condition_a_signal_activity_summary=condition_a_scatter_activity,
            condition_b_predictor=condition_b_scatter_predictor,
            condition_b_signal_activity_summary=condition_b_scatter_activity,
        ),
        cluster_p_values=hdf5_cluster_p_values,
        cluster_windows_s=hdf5_cluster_windows,
        cluster_null_distributions=hdf5_cluster_null_dists,
    )


def _load_from_matlab(path: Path) -> RegressionGroupProcessingResult:
    from gin_bids_py_analysis.processing.utils.matlab import mat_float, mat_str, mat_str_list
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=False, struct_as_record=False)
    data = mat["data"]
    region_names = mat_str_list(getattr(data.axes, "region", None))
    time_axis_s = np.asarray(data.axes.time_s, dtype=np.float64).ravel()
    n_rois = len(region_names)
    n_t = len(time_axis_s)

    def _mat_2d(obj: object, attr: str, fill: float = 0.0) -> np.ndarray:
        raw = getattr(obj, attr, None) if obj is not None else None
        if raw is None:
            return np.full((n_rois, n_t), fill, dtype=np.float64)
        arr = np.asarray(raw, dtype=np.float64)
        if arr.shape == (n_t, n_rois):
            arr = arr.T
        return arr.reshape(n_rois, n_t)

    def _mat_1d(obj: object, attr: str, fill: float = 0.0) -> np.ndarray:
        raw = getattr(obj, attr, None) if obj is not None else None
        if raw is None:
            return np.full(n_rois, fill, dtype=np.float64)
        return np.asarray(raw, dtype=np.float64).ravel()

    def _mat_1d_int(obj: object, attr: str) -> np.ndarray:
        raw = getattr(obj, attr, None) if obj is not None else None
        if raw is None:
            return np.zeros(n_rois, dtype=np.int64)
        return np.asarray(raw, dtype=np.int64).ravel()

    meta = data.meta
    schema_version = mat_str(getattr(meta, "schema_version", None), default="")
    if schema_version != "3.0":
        raise ValueError(
            "unsupported trial_stats_group schema. "
            "schema_version='3.0' is required; regenerate outputs with the v3 writer."
        )
    significance_alpha = mat_float(getattr(meta, "significance_alpha", None), default=0.05)

    _early_labels = mat_str_list(getattr(meta, "condition_labels", None))
    label_a = _early_labels[0] if len(_early_labels) >= 1 else "condition_a"
    label_b = _early_labels[1] if len(_early_labels) >= 2 else "condition_b"

    stats_root = data.stats
    data_root = getattr(data, "data", data)
    regression_stats = getattr(stats_root, "regression")
    source_metric = getattr(regression_stats, "condition_contrast")
    source_metric_t_values = _mat_2d(source_metric, "t_values")
    source_metric_p_values = _mat_2d(source_metric, "p_values", fill=1.0)
    source_metric_p_values_uncorrected = _mat_2d(
        source_metric,
        "p_values_uncorrected",
        fill=1.0,
    )
    sig_metric_raw = getattr(source_metric, "significant_mask", None)
    if sig_metric_raw is not None:
        source_metric_significant_mask = np.asarray(sig_metric_raw, dtype=bool).reshape(
            n_rois,
            n_t,
        )
    else:
        source_metric_significant_mask = np.isfinite(source_metric_p_values) & (
            source_metric_p_values < significance_alpha
        )

    # Per-condition vs-zero for the primary regression metric.
    vz_a_raw = getattr(regression_stats, f"{label_a}_vs_zero", None)
    condition_a_vs_zero_t = _mat_2d(vz_a_raw, "t_values")
    # p_values_uncorrected: new field; fall back to p_values for old files
    vz_a_p_uncorr_raw = getattr(vz_a_raw, "p_values_uncorrected", None) if vz_a_raw is not None else None
    if vz_a_p_uncorr_raw is not None:
        condition_a_vs_zero_p_uncorr = _mat_2d(vz_a_raw, "p_values_uncorrected", fill=1.0)
    else:
        condition_a_vs_zero_p_uncorr = _mat_2d(vz_a_raw, "p_values", fill=1.0)
    condition_a_vs_zero_p = _mat_2d(vz_a_raw, "p_values", fill=1.0)
    sig_vz_a_raw = getattr(vz_a_raw, "significant_mask", None) if vz_a_raw is not None else None
    if sig_vz_a_raw is not None:
        condition_a_vs_zero_sig = np.asarray(sig_vz_a_raw, dtype=bool).reshape(n_rois, n_t)
    else:
        condition_a_vs_zero_sig = np.isfinite(condition_a_vs_zero_p) & (condition_a_vs_zero_p < significance_alpha)
    # Cluster stats for condition A vs zero
    mat_vz_a_cluster_p: np.ndarray | None = None
    mat_vz_a_cluster_windows: list[list[tuple[float, float]]] | None = None
    mat_vz_a_cluster_null_dists: list[np.ndarray] | None = None
    cs_vz_a_raw = (
        getattr(vz_a_raw, "cluster", getattr(vz_a_raw, "cluster_stats", None))
        if vz_a_raw is not None
        else None
    )
    if cs_vz_a_raw is not None:
        _p_vz_a = getattr(cs_vz_a_raw, "p_values", None)
        if _p_vz_a is not None:
            mat_vz_a_cluster_p = np.asarray(_p_vz_a, dtype=np.float64).ravel()
            _starts_a = np.asarray(
                getattr(cs_vz_a_raw, "cluster_starts_s", np.zeros((len(mat_vz_a_cluster_p), 0))),
                dtype=np.float64,
            )
            _ends_a = np.asarray(
                getattr(cs_vz_a_raw, "cluster_ends_s", np.zeros((len(mat_vz_a_cluster_p), 0))),
                dtype=np.float64,
            )
            if _starts_a.ndim == 1:
                _starts_a = _starts_a[:, np.newaxis]
                _ends_a = _ends_a[:, np.newaxis]
            mat_vz_a_cluster_windows = [
                [
                    (float(s), float(e))
                    for s, e in zip(row_s, row_e)
                    if np.isfinite(s) and np.isfinite(e)
                ]
                for row_s, row_e in zip(_starts_a, _ends_a)
            ]
            _null_vz_a = getattr(cs_vz_a_raw, "null_distributions", None)
            if _null_vz_a is not None:
                _null_m_a = np.asarray(_null_vz_a, dtype=np.float64)
                if _null_m_a.ndim == 2:
                    mat_vz_a_cluster_null_dists = [
                        _null_m_a[i, np.isfinite(_null_m_a[i])]
                        for i in range(_null_m_a.shape[0])
                    ]
                else:
                    mat_vz_a_cluster_null_dists = [
                        np.zeros(0, dtype=np.float64)
                    ] * len(mat_vz_a_cluster_p)
            else:
                mat_vz_a_cluster_null_dists = [
                    np.zeros(0, dtype=np.float64)
                ] * len(mat_vz_a_cluster_p)
            # Override sig mask from cluster windows
            condition_a_vs_zero_sig = np.zeros((n_rois, n_t), dtype=bool)
            for roi_idx, roi_windows in enumerate(mat_vz_a_cluster_windows):
                for t_start_s, t_end_s in roi_windows:
                    in_window = (time_axis_s >= t_start_s) & (time_axis_s <= t_end_s)
                    condition_a_vs_zero_sig[roi_idx, in_window] = True

    vz_b_raw = getattr(regression_stats, f"{label_b}_vs_zero", None)
    condition_b_vs_zero_t = _mat_2d(vz_b_raw, "t_values")
    vz_b_p_uncorr_raw = getattr(vz_b_raw, "p_values_uncorrected", None) if vz_b_raw is not None else None
    if vz_b_p_uncorr_raw is not None:
        condition_b_vs_zero_p_uncorr = _mat_2d(vz_b_raw, "p_values_uncorrected", fill=1.0)
    else:
        condition_b_vs_zero_p_uncorr = _mat_2d(vz_b_raw, "p_values", fill=1.0)
    condition_b_vs_zero_p = _mat_2d(vz_b_raw, "p_values", fill=1.0)
    sig_vz_b_raw = getattr(vz_b_raw, "significant_mask", None) if vz_b_raw is not None else None
    if sig_vz_b_raw is not None:
        condition_b_vs_zero_sig = np.asarray(sig_vz_b_raw, dtype=bool).reshape(n_rois, n_t)
    else:
        condition_b_vs_zero_sig = np.isfinite(condition_b_vs_zero_p) & (condition_b_vs_zero_p < significance_alpha)
    # Cluster stats for condition B vs zero
    mat_vz_b_cluster_p: np.ndarray | None = None
    mat_vz_b_cluster_windows: list[list[tuple[float, float]]] | None = None
    mat_vz_b_cluster_null_dists: list[np.ndarray] | None = None
    cs_vz_b_raw = (
        getattr(vz_b_raw, "cluster", getattr(vz_b_raw, "cluster_stats", None))
        if vz_b_raw is not None
        else None
    )
    if cs_vz_b_raw is not None:
        _p_vz_b = getattr(cs_vz_b_raw, "p_values", None)
        if _p_vz_b is not None:
            mat_vz_b_cluster_p = np.asarray(_p_vz_b, dtype=np.float64).ravel()
            _starts_b = np.asarray(
                getattr(cs_vz_b_raw, "cluster_starts_s", np.zeros((len(mat_vz_b_cluster_p), 0))),
                dtype=np.float64,
            )
            _ends_b = np.asarray(
                getattr(cs_vz_b_raw, "cluster_ends_s", np.zeros((len(mat_vz_b_cluster_p), 0))),
                dtype=np.float64,
            )
            if _starts_b.ndim == 1:
                _starts_b = _starts_b[:, np.newaxis]
                _ends_b = _ends_b[:, np.newaxis]
            mat_vz_b_cluster_windows = [
                [
                    (float(s), float(e))
                    for s, e in zip(row_s, row_e)
                    if np.isfinite(s) and np.isfinite(e)
                ]
                for row_s, row_e in zip(_starts_b, _ends_b)
            ]
            _null_vz_b = getattr(cs_vz_b_raw, "null_distributions", None)
            if _null_vz_b is not None:
                _null_m_b = np.asarray(_null_vz_b, dtype=np.float64)
                if _null_m_b.ndim == 2:
                    mat_vz_b_cluster_null_dists = [
                        _null_m_b[i, np.isfinite(_null_m_b[i])]
                        for i in range(_null_m_b.shape[0])
                    ]
                else:
                    mat_vz_b_cluster_null_dists = [
                        np.zeros(0, dtype=np.float64)
                    ] * len(mat_vz_b_cluster_p)
            else:
                mat_vz_b_cluster_null_dists = [
                    np.zeros(0, dtype=np.float64)
                ] * len(mat_vz_b_cluster_p)
            # Override sig mask from cluster windows
            condition_b_vs_zero_sig = np.zeros((n_rois, n_t), dtype=bool)
            for roi_idx, roi_windows in enumerate(mat_vz_b_cluster_windows):
                for t_start_s, t_end_s in roi_windows:
                    in_window = (time_axis_s >= t_start_s) & (time_axis_s <= t_end_s)
                    condition_b_vs_zero_sig[roi_idx, in_window] = True
    regression_data = getattr(data_root, "regression", None)
    condition_a_regression_data = getattr(regression_data, label_a, None)
    condition_b_regression_data = getattr(regression_data, label_b, None)
    condition_a_source_metric_data = getattr(condition_a_regression_data, "slope", None)
    condition_b_source_metric_data = getattr(condition_b_regression_data, "slope", None)
    condition_a_source_metric_mean = _mat_2d(condition_a_source_metric_data, "mean")
    condition_a_source_metric_sem = _mat_2d(condition_a_source_metric_data, "sem")
    condition_b_source_metric_mean = _mat_2d(condition_b_source_metric_data, "mean")
    condition_b_source_metric_sem = _mat_2d(condition_b_source_metric_data, "sem")
    epoch_summary = getattr(source_metric, "epoch_summary", None)
    epoch_source_metric_t = _mat_1d(epoch_summary, "t")
    epoch_source_metric_p = _mat_1d(epoch_summary, "p", fill=1.0)
    epoch_source_metric_df = _mat_1d(epoch_summary, "df")

    activity = getattr(stats_root, "signal_activity")
    activity_t_values = _mat_2d(activity, "t_values")
    activity_p_values = _mat_2d(activity, "p_values", fill=1.0)
    activity_p_values_uncorrected = _mat_2d(
        activity,
        "p_values_uncorrected",
        fill=1.0,
    )
    sig_activity_raw = getattr(activity, "significant_mask", None)
    if sig_activity_raw is not None:
        activity_significant_mask = np.asarray(sig_activity_raw, dtype=bool).reshape(
            n_rois,
            n_t,
        )
    else:
        activity_significant_mask = np.isfinite(activity_p_values) & (
            activity_p_values < significance_alpha
        )
    epoch_activity = getattr(activity, "epoch_summary", None)
    epoch_activity_t = _mat_1d(epoch_activity, "t")
    epoch_activity_p = _mat_1d(epoch_activity, "p", fill=1.0)
    epoch_activity_df = _mat_1d(epoch_activity, "df")

    activity_data = getattr(data_root, "signal_activity", None)
    condition_a_signal_activity_summary_data = getattr(activity_data, label_a, None)
    condition_b_signal_activity_summary_data = getattr(activity_data, label_b, None)
    condition_a_signal_activity_summary_mean = _mat_2d(condition_a_signal_activity_summary_data, "mean")
    condition_a_signal_activity_summary_sem = _mat_2d(condition_a_signal_activity_summary_data, "sem")
    condition_b_signal_activity_summary_mean = _mat_2d(condition_b_signal_activity_summary_data, "mean")
    condition_b_signal_activity_summary_sem = _mat_2d(condition_b_signal_activity_summary_data, "sem")

    condition_a_r = getattr(condition_a_regression_data, "r_value", None)
    condition_b_r = getattr(condition_b_regression_data, "r_value", None)
    condition_a_r_value_mean = _mat_2d(condition_a_r, "mean")
    condition_a_r_value_sem = _mat_2d(condition_a_r, "sem")
    condition_b_r_value_mean = _mat_2d(condition_b_r, "mean")
    condition_b_r_value_sem = _mat_2d(condition_b_r, "sem")

    labels = mat_str_list(getattr(meta, "condition_labels", None))
    condition_labels: tuple[str, str] = (
        labels[0] if len(labels) >= 1 else "condition_a",
        labels[1] if len(labels) >= 2 else "condition_b",
    )
    primary_regression_metric_name = mat_str(
        getattr(meta, "primary_regression_metric", None),
        default="slope",
    )
    contrast_mode = mat_str(getattr(meta, "contrast_mode", None), default="paired")
    p_value_correction_method = mat_str(
        getattr(meta, "p_value_correction_method", None),
        default="none",
    )
    roi_mode = mat_str(getattr(meta, "roi_mode", None), default="manual")
    atlas_name_raw = mat_str(getattr(meta, "atlas_name", None), default="")
    atlas_name: str | None = atlas_name_raw.strip() or None

    metadata = {
        "p_value_correction_method": p_value_correction_method,
        "significance_alpha": significance_alpha,
        "primary_regression_metric": primary_regression_metric_name,
        "contrast_mode": contrast_mode,
        "roi_mode": roi_mode,
        "atlas_name": atlas_name,
        "predictor": mat_str(getattr(meta, "predictor", None), default=""),
        "predictor_zscore": mat_str(
            getattr(meta, "predictor_zscore", None),
            default="none",
        ),
        "predictor_transform_by_condition_json": mat_str(
            getattr(meta, "predictor_transform_by_condition_json", None),
            default="{}",
        ),
        "activity_zscore": mat_str(getattr(meta, "activity_zscore", None), default="none"),
        "activity_baseline_tmin_s": mat_float(
            getattr(meta, "activity_baseline_tmin_s", None),
            default=-0.2,
        ),
        "activity_baseline_tmax_s": mat_float(
            getattr(meta, "activity_baseline_tmax_s", None),
            default=0.0,
        ),
        "trial_activity_summary_kind": mat_str(
            getattr(meta, "trial_activity_summary_kind", None),
            default="epoch_mean",
        ),
        "trial_activity_summary_missing_response_policy": (
            normalize_trial_activity_summary_missing_response_policy(
                mat_str(
                    getattr(meta, "trial_activity_summary_missing_response_policy", None),
                    default="nan_if_missing",
                )
            )
        ),
        "trial_activity_summary_source_json": mat_str(
            getattr(meta, "trial_activity_summary_source_json", None),
            default="{}",
        ),
        "trial_activity_summary_label": mat_str(
            getattr(meta, "trial_activity_summary_label", None),
            default="Epoch mean activity",
        ),
        "scatter_aggregation": mat_str(
            getattr(meta, "scatter_aggregation", None),
            default="trial_pool",
        ),
    }

    roi_channel_counts = _mat_1d_int(meta, "roi_channel_counts")
    roi_subject_counts = _mat_1d_int(meta, "roi_subject_counts")
    excluded_rois = _read_excluded_rois_mat(getattr(data, "excluded_rois", None))
    contributions_root = getattr(data, "contributions", None)
    contributions = _read_contributions_mat(
        getattr(contributions_root, "summary", contributions_root)
    )
    (
        condition_a_signal_activity_summary_contributions,
        condition_b_signal_activity_summary_contributions,
        contribution_labels,
    ) = _read_indexed_condition_contributions_mat(
        getattr(contributions_root, "signal_activity", None),
        region_names,
    )
    regression_contributions = getattr(contributions_root, "regression", None)
    (
        condition_a_source_metric_contributions,
        condition_b_source_metric_contributions,
        slope_labels,
    ) = _read_indexed_condition_contributions_mat(
        getattr(regression_contributions, "slope", None),
        region_names,
    )
    if not contribution_labels:
        contribution_labels = slope_labels
    (
        condition_a_scatter_predictor,
        condition_a_scatter_activity,
        condition_b_scatter_predictor,
        condition_b_scatter_activity,
    ) = _read_scatter_mat(getattr(data, "scatter", None))

    provenance = getattr(data, "provenance", None)
    source_subject_stats_files = (
        mat_str_list(getattr(provenance, "source_subject_stats_files", None))
        if provenance is not None
        else []
    )
    source_electrodes_files = (
        mat_str_list(getattr(provenance, "source_electrodes_files", None))
        if provenance is not None
        else []
    )

    mat_cluster_p_values: np.ndarray | None = None
    mat_cluster_windows: list[list[tuple[float, float]]] | None = None
    mat_cluster_null_dists: list[np.ndarray] | None = None
    cs_raw = getattr(source_metric, "cluster", None)
    if cs_raw is not None:
        _p = getattr(cs_raw, "p_values", None)
        if _p is not None:
            mat_cluster_p_values = np.asarray(_p, dtype=np.float64).ravel()
            _starts = np.asarray(
                getattr(cs_raw, "cluster_starts_s", np.zeros((len(mat_cluster_p_values), 0))),
                dtype=np.float64,
            )
            _ends = np.asarray(
                getattr(cs_raw, "cluster_ends_s", np.zeros((len(mat_cluster_p_values), 0))),
                dtype=np.float64,
            )
            if _starts.ndim == 1:
                _starts = _starts[:, np.newaxis]
                _ends = _ends[:, np.newaxis]
            mat_cluster_windows = [
                [
                    (float(s), float(e))
                    for s, e in zip(row_s, row_e)
                    if np.isfinite(s) and np.isfinite(e)
                ]
                for row_s, row_e in zip(_starts, _ends)
            ]
            _null_mat_raw = getattr(cs_raw, "null_distributions", None)
            if _null_mat_raw is not None:
                _null_mat = np.asarray(_null_mat_raw, dtype=np.float64)
                if _null_mat.ndim == 2:
                    mat_cluster_null_dists = [
                        _null_mat[i, np.isfinite(_null_mat[i])]
                        for i in range(_null_mat.shape[0])
                    ]
                else:
                    mat_cluster_null_dists = [np.zeros(0, dtype=np.float64)] * len(mat_cluster_p_values)
            else:
                mat_cluster_null_dists = [np.zeros(0, dtype=np.float64)] * len(mat_cluster_p_values)

    source_group = BIDSFileGroup(primary=BIDSFile.from_path(path))
    return RegressionGroupProcessingResult(
        source_group=source_group,
        metadata=metadata,
        regression_stats=RegressionMetricStats(
            contrast=GroupTimecourseStats(
                t_values=source_metric_t_values,
                p_values=source_metric_p_values,
                p_values_uncorrected=source_metric_p_values_uncorrected,
                significant_mask=source_metric_significant_mask,
            ),
            epoch_summary=GroupEpochStats(
                t=epoch_source_metric_t,
                p=epoch_source_metric_p,
                df=epoch_source_metric_df,
            ),
            vs_zero=VsZeroStatsPair(
                condition_a=GroupTimecourseStats(
                    t_values=condition_a_vs_zero_t,
                    p_values=condition_a_vs_zero_p,
                    p_values_uncorrected=condition_a_vs_zero_p_uncorr,
                    significant_mask=condition_a_vs_zero_sig,
                ),
                condition_b=GroupTimecourseStats(
                    t_values=condition_b_vs_zero_t,
                    p_values=condition_b_vs_zero_p,
                    p_values_uncorrected=condition_b_vs_zero_p_uncorr,
                    significant_mask=condition_b_vs_zero_sig,
                ),
                condition_a_cluster_p_values=mat_vz_a_cluster_p,
                condition_a_cluster_windows_s=mat_vz_a_cluster_windows,
                condition_a_cluster_null_distributions=mat_vz_a_cluster_null_dists,
                condition_b_cluster_p_values=mat_vz_b_cluster_p,
                condition_b_cluster_windows_s=mat_vz_b_cluster_windows,
                condition_b_cluster_null_distributions=mat_vz_b_cluster_null_dists,
            ),
        ),
        signal_activity_stats=GroupTimecourseStats(
            t_values=activity_t_values,
            p_values=activity_p_values,
            p_values_uncorrected=activity_p_values_uncorrected,
            significant_mask=activity_significant_mask,
        ),
        slope=GroupEstimatePair(
            condition_a=GroupEstimate(
                mean=condition_a_source_metric_mean,
                sem=condition_a_source_metric_sem,
            ),
            condition_b=GroupEstimate(
                mean=condition_b_source_metric_mean,
                sem=condition_b_source_metric_sem,
            ),
        ),
        signal_activity=GroupEstimatePair(
            condition_a=GroupEstimate(
                mean=condition_a_signal_activity_summary_mean,
                sem=condition_a_signal_activity_summary_sem,
            ),
            condition_b=GroupEstimate(
                mean=condition_b_signal_activity_summary_mean,
                sem=condition_b_signal_activity_summary_sem,
            ),
        ),
        r_value=GroupEstimatePair(
            condition_a=GroupEstimate(
                mean=condition_a_r_value_mean,
                sem=condition_a_r_value_sem,
            ),
            condition_b=GroupEstimate(
                mean=condition_b_r_value_mean,
                sem=condition_b_r_value_sem,
            ),
        ),
        signal_activity_epoch=GroupEpochStats(
            t=epoch_activity_t,
            p=epoch_activity_p,
            df=epoch_activity_df,
        ),
        time_axis_s=time_axis_s,
        region_names=region_names,
        condition_labels=condition_labels,
        primary_regression_metric=primary_regression_metric_name,
        contrast_mode=contrast_mode,
        roi_channel_counts=roi_channel_counts,
        roi_subject_counts=roi_subject_counts,
        contributions=contributions,
        slope_contributions=IndexedConditionContributions(
            condition_a=condition_a_source_metric_contributions,
            condition_b=condition_b_source_metric_contributions,
            labels=contribution_labels,
        ),
        signal_activity_contributions=IndexedConditionContributions(
            condition_a=condition_a_signal_activity_summary_contributions,
            condition_b=condition_b_signal_activity_summary_contributions,
            labels=contribution_labels,
        ),
        p_value_correction_method=p_value_correction_method,
        significance_alpha=significance_alpha,
        roi_mode=roi_mode,
        atlas_name=atlas_name,
        source_subject_stats_files=source_subject_stats_files,
        source_electrodes_files=source_electrodes_files,
        excluded_rois=excluded_rois,
        scatter=ScatterData(
            condition_a_predictor=condition_a_scatter_predictor,
            condition_a_signal_activity_summary=condition_a_scatter_activity,
            condition_b_predictor=condition_b_scatter_predictor,
            condition_b_signal_activity_summary=condition_b_scatter_activity,
        ),
        cluster_p_values=mat_cluster_p_values,
        cluster_windows_s=mat_cluster_windows,
        cluster_null_distributions=mat_cluster_null_dists,
    )


def _read_excluded_rois_hdf5(fh: h5py.File) -> dict[str, str]:
    region_ds = dataset_or_none(fh, "excluded_rois/region")
    reason_ds = dataset_or_none(fh, "excluded_rois/reason")
    if region_ds is None or reason_ds is None:
        return {}
    regions = decode_str_array(np.asarray(region_ds[:], dtype=object))
    reasons = decode_str_array(np.asarray(reason_ds[:], dtype=object))
    return dict(zip(regions, reasons))


def _read_contributions_hdf5(fh: h5py.File) -> list[ROIChannelContribution]:
    if "contributions/summary" not in fh:
        return []
    group = fh["contributions/summary"]
    if not all(key in group for key in ("region", "subject", "channel", "source_stats_file")):
        return []
    regions = decode_str_array(np.asarray(group["region"][:], dtype=object))
    subjects = decode_str_array(np.asarray(group["subject"][:], dtype=object))
    channels = decode_str_array(np.asarray(group["channel"][:], dtype=object))
    source_stats_files = decode_str_array(
        np.asarray(group["source_stats_file"][:], dtype=object)
    )
    return [
        ROIChannelContribution(
            roi=region,
            subject=subject,
            channel=channel,
            source_stats_file=source_stats_file,
        )
        for region, subject, channel, source_stats_file in zip(
            regions,
            subjects,
            channels,
            source_stats_files,
        )
    ]


def _read_indexed_condition_contributions_hdf5(
    fh: h5py.File,
    *,
    group_name: str,
    region_names: list[str],
    n_t: int,
    label_a: str = "condition_a",
    label_b: str = "condition_b",
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[str]]]:
    full_group_name = f"contributions/{group_name}"
    if full_group_name not in fh:
        return [], [], []
    out_a: list[np.ndarray] = []
    out_b: list[np.ndarray] = []
    out_labels: list[list[str]] = []
    group = fh[full_group_name]
    for roi_idx, roi_key in enumerate(region_names):
        if roi_key not in group:
            out_a.append(np.empty((0, n_t), dtype=np.float64))
            out_b.append(np.empty((0, n_t), dtype=np.float64))
            out_labels.append([])
            continue
        roi_group = group[roi_key]
        out_a.append(np.asarray(roi_group[label_a][:], dtype=np.float64))
        out_b.append(np.asarray(roi_group[label_b][:], dtype=np.float64))
        out_labels.append(
            decode_str_array(np.asarray(roi_group["labels"][:], dtype=object))
            if "labels" in roi_group
            else []
        )
    return out_a, out_b, out_labels


def _read_scatter_hdf5(
    fh: h5py.File,
    *,
    region_names: list[str],
    label_a: str = "condition_a",
    label_b: str = "condition_b",
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    if "scatter" not in fh:
        return [], [], [], []
    out_pred_a: list[np.ndarray] = []
    out_act_a: list[np.ndarray] = []
    out_pred_b: list[np.ndarray] = []
    out_act_b: list[np.ndarray] = []
    group = fh["scatter"]
    for roi_idx, roi_key in enumerate(region_names):
        if roi_key not in group:
            out_pred_a.append(np.empty(0, dtype=np.float64))
            out_act_a.append(np.empty(0, dtype=np.float64))
            out_pred_b.append(np.empty(0, dtype=np.float64))
            out_act_b.append(np.empty(0, dtype=np.float64))
            continue
        roi_group = group[roi_key]
        if label_a in roi_group:
            out_pred_a.append(np.asarray(roi_group[label_a]["predictor"][:], dtype=np.float64))
            out_act_a.append(
                np.asarray(
                    roi_group[label_a]["signal_activity_summary"][:],
                    dtype=np.float64,
                )
            )
            out_pred_b.append(np.asarray(roi_group[label_b]["predictor"][:], dtype=np.float64))
            out_act_b.append(
                np.asarray(
                    roi_group[label_b]["signal_activity_summary"][:],
                    dtype=np.float64,
                )
            )
        else:
            out_pred_a.append(np.asarray(roi_group[f"{label_a}_predictor"][:], dtype=np.float64))
            out_act_a.append(np.asarray(roi_group[f"{label_a}_signal_activity_summary"][:], dtype=np.float64))
            out_pred_b.append(np.asarray(roi_group[f"{label_b}_predictor"][:], dtype=np.float64))
            out_act_b.append(np.asarray(roi_group[f"{label_b}_signal_activity_summary"][:], dtype=np.float64))
    return out_pred_a, out_act_a, out_pred_b, out_act_b


def _read_excluded_rois_mat(raw: object) -> dict[str, str]:
    from gin_bids_py_analysis.processing.utils.matlab import mat_str_list

    if raw is None:
        return {}
    return dict(
        zip(
            mat_str_list(getattr(raw, "region", None)),
            mat_str_list(getattr(raw, "reason", None)),
        )
    )


def _read_contributions_mat(raw: object) -> list[ROIChannelContribution]:
    from gin_bids_py_analysis.processing.utils.matlab import mat_str_list

    if raw is None:
        return []
    regions = mat_str_list(getattr(raw, "region", None))
    subjects = mat_str_list(getattr(raw, "subject", None))
    channels = mat_str_list(getattr(raw, "channel", None))
    source_stats_files = mat_str_list(getattr(raw, "source_stats_file", None))
    return [
        ROIChannelContribution(
            roi=region,
            subject=subject,
            channel=channel,
            source_stats_file=source_stats_file,
        )
        for region, subject, channel, source_stats_file in zip(
            regions,
            subjects,
            channels,
            source_stats_files,
        )
    ]


def _read_indexed_condition_contributions_mat(
    raw: object,
    region_names: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[str]]]:
    from gin_bids_py_analysis.processing.utils.matlab import mat_str_list

    if raw is None:
        return [], [], []
    out_a: list[np.ndarray] = []
    out_b: list[np.ndarray] = []
    out_labels: list[list[str]] = []
    for roi_name in region_names:
        roi_raw = getattr(raw, roi_name, None)
        if roi_raw is None:
            out_a.append(np.empty((0,), dtype=np.float64))
            out_b.append(np.empty((0,), dtype=np.float64))
            out_labels.append([])
            continue
        cond_a = getattr(roi_raw, "condition_a", None)
        cond_b = getattr(roi_raw, "condition_b", None)
        labels = getattr(roi_raw, "labels", None)
        out_a.append(np.asarray(cond_a, dtype=np.float64) if cond_a is not None else np.empty((0,), dtype=np.float64))
        out_b.append(np.asarray(cond_b, dtype=np.float64) if cond_b is not None else np.empty((0,), dtype=np.float64))
        out_labels.append(mat_str_list(labels))
    return out_a, out_b, out_labels


def _read_scatter_mat(
    raw: object,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    if raw is None:
        return [], [], [], []
    out_pred_a: list[np.ndarray] = []
    out_act_a: list[np.ndarray] = []
    out_pred_b: list[np.ndarray] = []
    out_act_b: list[np.ndarray] = []
    for attr, target in (
        ("condition_a_predictor", out_pred_a),
        ("condition_a_signal_activity_summary", out_act_a),
        ("condition_b_predictor", out_pred_b),
        ("condition_b_signal_activity_summary", out_act_b),
    ):
        cell = getattr(raw, attr, None)
        if cell is None:
            continue
        for item in np.asarray(cell).ravel():
            target.append(np.asarray(item, dtype=np.float64).ravel())
    return out_pred_a, out_act_a, out_pred_b, out_act_b


def _require_v3_schema(fh: h5py.File, path_name: str) -> None:
    schema_version = str_scalar(dataset_or_none(fh, "meta/schema_version"), default="")
    if schema_version != "3.0":
        raise ValueError(
            f"{path_name}: unsupported trial_stats_group schema. "
            "schema_version='3.0' is required; regenerate outputs with the v3 writer."
        )

