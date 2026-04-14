"""Load a pre-computed ``ConditionTestGroupProcessingResult`` from disk."""

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

from ..result import ROIChannelContribution
from .result import ConditionTestGroupProcessingResult


def load_condition_test_group_result(
    path: Path | str,
) -> ConditionTestGroupProcessingResult:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Group stats file not found: {path}")
    if path.suffix.lower() == ".mat":
        return _load_from_matlab(path)
    return _load_from_hdf5(path)


def _load_from_hdf5(path: Path) -> ConditionTestGroupProcessingResult:
    with h5py.File(path, "r") as fh:
        region_names = decode_str_array(np.asarray(fh["axes"]["region"][:]))
        time_axis_s = np.asarray(fh["axes"]["time_s"][:], dtype=np.float64)
        n_rois = len(region_names)
        n_t = len(time_axis_s)

        def _read_2d(key: str, fill: float = 0.0) -> np.ndarray:
            ds = dataset_or_none(fh, key)
            if ds is None:
                return np.full((n_rois, n_t), fill, dtype=np.float64)
            return np.asarray(ds[:], dtype=np.float64)

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

        t_values = _read_2d("stats/t_values")
        p_values = _read_2d("stats/p_values", fill=1.0)
        p_values_uncorrected = _read_2d("stats/p_values_uncorrected", fill=1.0)
        significance_alpha = float_scalar(
            dataset_or_none(fh, "meta/significance_alpha"),
            default=0.05,
        )
        significant_mask_ds = dataset_or_none(fh, "stats/significant_mask")
        if significant_mask_ds is not None:
            significant_mask = np.asarray(significant_mask_ds[:], dtype=bool)
        else:
            significant_mask = np.isfinite(p_values) & (p_values < significance_alpha)

        metric_mean = _read_2d("means/metric_mean")
        metric_sem = _read_2d("uncertainty/metric_sem")
        condition_a_activity_mean = _read_2d("means/condition_a_mean")
        condition_a_activity_sem = _read_2d("means/condition_a_sem")
        condition_b_activity_mean = _read_2d("means/condition_b_mean")
        condition_b_activity_sem = _read_2d("means/condition_b_sem")

        epoch_mean_t_values = _read_1d("summary_epoch/t_values")
        epoch_mean_p_values = _read_1d("summary_epoch/p_values", fill=1.0)
        epoch_mean_df = _read_1d("summary_epoch/df")
        epoch_mean_metric_mean = _read_1d("summary_epoch/metric_mean")
        epoch_mean_metric_sem = _read_1d("summary_epoch/metric_sem")

        roi_channel_counts = _read_1d_int("meta/roi_channel_counts")
        roi_subject_counts = _read_1d_int("meta/roi_subject_counts")

        source_metric = str_scalar(dataset_or_none(fh, "meta/source_metric"), default="mean_difference")
        labels = decode_str_array(
            np.asarray(fh["meta"]["condition_labels"][:], dtype=object)
        )
        condition_labels: tuple[str, str] = (
            labels[0] if len(labels) >= 1 else "condition_a",
            labels[1] if len(labels) >= 2 else "condition_b",
        )
        p_value_correction_method = str_scalar(
            dataset_or_none(fh, "meta/p_value_correction_method"),
            default="none",
        )
        roi_mode = str_scalar(dataset_or_none(fh, "meta/roi_mode"), default="manual")
        atlas_name_raw = str_scalar(dataset_or_none(fh, "meta/atlas_name"), default="")
        atlas_name: str | None = atlas_name_raw.strip() or None
        activity_zscore = str_scalar(
            dataset_or_none(fh, "meta/activity_zscore"),
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

        excluded_rois = _read_excluded_rois_hdf5(fh)
        contributions = _read_contributions_hdf5(fh)
        (
            condition_a_activity_contributions,
            condition_b_activity_contributions,
            contribution_labels,
        ) = _read_activity_contributions_hdf5(
            fh,
            region_names=region_names,
            n_t=n_t,
        )

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

        cluster_p_values: np.ndarray | None = None
        cluster_windows: list[tuple[float, float] | None] | None = None
        cluster_null_distributions: list[np.ndarray] | None = None
        if "cluster_stats" in fh:
            cluster_stats = fh["cluster_stats"]
            cluster_p_ds = dataset_or_none(cluster_stats, "p_values")
            if cluster_p_ds is not None:
                cluster_p_values = np.asarray(cluster_p_ds[:], dtype=np.float64)
                starts = np.asarray(cluster_stats["best_cluster_start_s"][:], dtype=np.float64)
                ends = np.asarray(cluster_stats["best_cluster_end_s"][:], dtype=np.float64)
                cluster_windows = [
                    (float(start), float(end))
                    if np.isfinite(start) and np.isfinite(end)
                    else None
                    for start, end in zip(starts, ends)
                ]
                null_ds = dataset_or_none(cluster_stats, "null_distributions")
                if null_ds is not None:
                    null_matrix = np.asarray(null_ds[:], dtype=np.float32)
                    cluster_null_distributions = [
                        row[np.isfinite(row)].astype(np.float64)
                        for row in null_matrix
                    ]

    source_group = BIDSFileGroup(primary=BIDSFile.from_path(path))
    return ConditionTestGroupProcessingResult(
        source_group=source_group,
        metadata={
            "source_metric": source_metric,
            "p_value_correction_method": p_value_correction_method,
            "significance_alpha": significance_alpha,
            "roi_mode": roi_mode,
            "activity_zscore": activity_zscore,
            "activity_baseline_tmin_s": activity_baseline_tmin_s,
            "activity_baseline_tmax_s": activity_baseline_tmax_s,
        },
        activity_t_values=t_values,
        activity_p_values=p_values,
        activity_p_values_uncorrected=p_values_uncorrected,
        activity_significant_mask=significant_mask,
        metric_mean=metric_mean,
        metric_sem=metric_sem,
        epoch_activity_t=epoch_mean_t_values,
        epoch_activity_p=epoch_mean_p_values,
        epoch_activity_df=epoch_mean_df,
        epoch_mean_metric_mean=epoch_mean_metric_mean,
        epoch_mean_metric_sem=epoch_mean_metric_sem,
        time_axis_s=time_axis_s,
        region_names=region_names,
        condition_labels=condition_labels,
        roi_channel_counts=roi_channel_counts,
        roi_subject_counts=roi_subject_counts,
        contributions=contributions,
        condition_a_activity_mean=condition_a_activity_mean,
        condition_a_activity_sem=condition_a_activity_sem,
        condition_b_activity_mean=condition_b_activity_mean,
        condition_b_activity_sem=condition_b_activity_sem,
        condition_a_activity_contributions=condition_a_activity_contributions,
        condition_b_activity_contributions=condition_b_activity_contributions,
        contribution_labels=contribution_labels,
        source_metric=source_metric,
        p_value_correction_method=p_value_correction_method,
        significance_alpha=significance_alpha,
        roi_mode=roi_mode,
        atlas_name=atlas_name,
        source_subject_stats_files=source_subject_stats_files,
        source_electrodes_files=source_electrodes_files,
        excluded_rois=excluded_rois,
        cluster_p_values=cluster_p_values,
        cluster_best_cluster_windows_s=cluster_windows,
        cluster_null_distributions=cluster_null_distributions,
    )


def _load_from_matlab(path: Path) -> ConditionTestGroupProcessingResult:
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

    def _mat_1d(obj: object, attr: str, fill: float = 0.0, dtype=np.float64) -> np.ndarray:
        raw = getattr(obj, attr, None) if obj is not None else None
        if raw is None:
            return np.full(n_rois, fill, dtype=dtype)
        return np.asarray(raw, dtype=dtype).ravel()

    stats = data.stats
    means = data.means
    summary = getattr(data, "summary_epoch", None)
    uncertainty = getattr(data, "uncertainty", None)
    meta = data.meta
    provenance = getattr(data, "provenance", None)

    t_values = _mat_2d(stats, "t_values")
    p_values = _mat_2d(stats, "p_values", fill=1.0)
    p_values_uncorrected = _mat_2d(stats, "p_values_uncorrected", fill=1.0)
    significance_alpha = mat_float(getattr(meta, "significance_alpha", None), default=0.05)
    sig_raw = getattr(stats, "significant_mask", None)
    if sig_raw is not None:
        significant_mask = np.asarray(sig_raw, dtype=bool).reshape(n_rois, n_t)
    else:
        significant_mask = np.isfinite(p_values) & (p_values < significance_alpha)

    metric_mean = _mat_2d(means, "metric_mean")
    metric_sem = _mat_2d(uncertainty, "metric_sem")
    condition_a_activity_mean = _mat_2d(means, "condition_a_mean")
    condition_a_activity_sem = _mat_2d(means, "condition_a_sem")
    condition_b_activity_mean = _mat_2d(means, "condition_b_mean")
    condition_b_activity_sem = _mat_2d(means, "condition_b_sem")

    epoch_mean_t_values = _mat_1d(summary, "t_values")
    epoch_mean_p_values = _mat_1d(summary, "p_values", fill=1.0)
    epoch_mean_df = _mat_1d(summary, "df")
    epoch_mean_metric_mean = _mat_1d(summary, "metric_mean")
    epoch_mean_metric_sem = _mat_1d(summary, "metric_sem")

    source_metric = mat_str(getattr(meta, "source_metric", None), default="mean_difference")
    labels = mat_str_list(getattr(meta, "condition_labels", None))
    condition_labels: tuple[str, str] = (
        labels[0] if len(labels) >= 1 else "condition_a",
        labels[1] if len(labels) >= 2 else "condition_b",
    )
    p_value_correction_method = mat_str(
        getattr(meta, "p_value_correction_method", None),
        default="none",
    )
    roi_mode = mat_str(getattr(meta, "roi_mode", None), default="manual")
    atlas_name_raw = mat_str(getattr(meta, "atlas_name", None), default="")
    atlas_name: str | None = atlas_name_raw.strip() or None
    activity_zscore = mat_str(getattr(meta, "activity_zscore", None), default="none")
    activity_baseline_tmin_s = mat_float(
        getattr(meta, "activity_baseline_tmin_s", None),
        default=-0.2,
    )
    activity_baseline_tmax_s = mat_float(
        getattr(meta, "activity_baseline_tmax_s", None),
        default=0.0,
    )

    roi_channel_counts = _mat_1d(meta, "roi_channel_counts", dtype=np.int64)
    roi_subject_counts = _mat_1d(meta, "roi_subject_counts", dtype=np.int64)
    excluded_rois = _read_excluded_rois_mat(getattr(data, "excluded_rois", None))
    contributions = _read_contributions_mat(getattr(data, "contributions", None))
    (
        condition_a_activity_contributions,
        condition_b_activity_contributions,
        contribution_labels,
    ) = _read_activity_contributions_mat(getattr(data, "activity_contributions", None))

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

    cluster_p_values: np.ndarray | None = None
    cluster_windows: list[tuple[float, float] | None] | None = None
    cluster_null_distributions: list[np.ndarray] | None = None
    cluster_stats = getattr(data, "cluster_stats", None)
    if cluster_stats is not None:
        raw = getattr(cluster_stats, "p_values", None)
        if raw is not None:
            cluster_p_values = np.asarray(raw, dtype=np.float64).ravel()
            starts = np.asarray(
                getattr(cluster_stats, "best_cluster_start_s", np.array([])),
                dtype=np.float64,
            ).ravel()
            ends = np.asarray(
                getattr(cluster_stats, "best_cluster_end_s", np.array([])),
                dtype=np.float64,
            ).ravel()
            cluster_windows = [
                (float(start), float(end))
                if np.isfinite(start) and np.isfinite(end)
                else None
                for start, end in zip(starts, ends)
            ]
            raw_null = getattr(cluster_stats, "null_distributions", None)
            if raw_null is not None:
                cell = np.asarray(raw_null).ravel()
                cluster_null_distributions = [
                    np.asarray(cell_item, dtype=np.float64).ravel()
                    for cell_item in cell
                ]

    source_group = BIDSFileGroup(primary=BIDSFile.from_path(path))
    return ConditionTestGroupProcessingResult(
        source_group=source_group,
        metadata={
            "source_metric": source_metric,
            "p_value_correction_method": p_value_correction_method,
            "significance_alpha": significance_alpha,
            "roi_mode": roi_mode,
            "activity_zscore": activity_zscore,
            "activity_baseline_tmin_s": activity_baseline_tmin_s,
            "activity_baseline_tmax_s": activity_baseline_tmax_s,
        },
        activity_t_values=t_values,
        activity_p_values=p_values,
        activity_p_values_uncorrected=p_values_uncorrected,
        activity_significant_mask=significant_mask,
        metric_mean=metric_mean,
        metric_sem=metric_sem,
        epoch_activity_t=epoch_mean_t_values,
        epoch_activity_p=epoch_mean_p_values,
        epoch_activity_df=epoch_mean_df,
        epoch_mean_metric_mean=epoch_mean_metric_mean,
        epoch_mean_metric_sem=epoch_mean_metric_sem,
        time_axis_s=time_axis_s,
        region_names=region_names,
        condition_labels=condition_labels,
        roi_channel_counts=roi_channel_counts,
        roi_subject_counts=roi_subject_counts,
        contributions=contributions,
        condition_a_activity_mean=condition_a_activity_mean,
        condition_a_activity_sem=condition_a_activity_sem,
        condition_b_activity_mean=condition_b_activity_mean,
        condition_b_activity_sem=condition_b_activity_sem,
        condition_a_activity_contributions=condition_a_activity_contributions,
        condition_b_activity_contributions=condition_b_activity_contributions,
        contribution_labels=contribution_labels,
        source_metric=source_metric,
        p_value_correction_method=p_value_correction_method,
        significance_alpha=significance_alpha,
        roi_mode=roi_mode,
        atlas_name=atlas_name,
        source_subject_stats_files=source_subject_stats_files,
        source_electrodes_files=source_electrodes_files,
        excluded_rois=excluded_rois,
        cluster_p_values=cluster_p_values,
        cluster_best_cluster_windows_s=cluster_windows,
        cluster_null_distributions=cluster_null_distributions,
    )


def _read_excluded_rois_hdf5(fh: h5py.File) -> dict[str, str]:
    ex_region_ds = dataset_or_none(fh, "excluded_rois/region")
    ex_reason_ds = dataset_or_none(fh, "excluded_rois/reason")
    if ex_region_ds is None or ex_reason_ds is None:
        return {}
    ex_regions = decode_str_array(np.asarray(ex_region_ds[:], dtype=object))
    ex_reasons = decode_str_array(np.asarray(ex_reason_ds[:], dtype=object))
    return dict(zip(ex_regions, ex_reasons))


def _read_contributions_hdf5(fh: h5py.File) -> list[ROIChannelContribution]:
    if "contributions" not in fh:
        return []
    group = fh["contributions"]
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


def _read_activity_contributions_hdf5(
    fh: h5py.File,
    *,
    region_names: list[str],
    n_t: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[str]]]:
    if "activity_contributions" not in fh:
        return [], [], []
    out_a: list[np.ndarray] = []
    out_b: list[np.ndarray] = []
    out_labels: list[list[str]] = []
    group = fh["activity_contributions"]
    for roi_idx in range(len(region_names)):
        roi_key = str(roi_idx)
        if roi_key not in group:
            out_a.append(np.empty((0, n_t), dtype=np.float64))
            out_b.append(np.empty((0, n_t), dtype=np.float64))
            out_labels.append([])
            continue
        roi_group = group[roi_key]
        out_a.append(np.asarray(roi_group["condition_a"][:], dtype=np.float64))
        out_b.append(np.asarray(roi_group["condition_b"][:], dtype=np.float64))
        out_labels.append(
            decode_str_array(np.asarray(roi_group["labels"][:], dtype=object))
            if "labels" in roi_group
            else []
        )
    return out_a, out_b, out_labels


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


def _read_activity_contributions_mat(
    raw: object,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[str]]]:
    from gin_bids_py_analysis.processing.utils.matlab import mat_str_list

    if raw is None:
        return [], [], []

    condition_a: list[np.ndarray] = []
    condition_b: list[np.ndarray] = []
    labels: list[list[str]] = []
    a_raw = getattr(raw, "condition_a", None)
    b_raw = getattr(raw, "condition_b", None)
    labels_raw = getattr(raw, "labels", None)
    if a_raw is not None:
        for item in np.asarray(a_raw).ravel():
            condition_a.append(np.asarray(item, dtype=np.float64))
    if b_raw is not None:
        for item in np.asarray(b_raw).ravel():
            condition_b.append(np.asarray(item, dtype=np.float64))
    if labels_raw is not None:
        for item in np.asarray(labels_raw).ravel():
            labels.append(mat_str_list(item))
    return condition_a, condition_b, labels
