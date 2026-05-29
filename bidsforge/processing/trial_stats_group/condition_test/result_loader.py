"""Load a pre-computed ``ConditionTestGroupProcessingResult`` from disk."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.processing.utils.hdf5 import (
    dataset_or_none,
    decode_str_array,
    float_scalar,
    str_scalar,
)

from ..result import (
    GroupEpochStats,
    GroupEstimate,
    GroupEstimatePair,
    GroupTimecourseStats,
    IndexedConditionContributions,
    ROIChannelContribution,
)
from .result import ConditionTestEpochSummary, ConditionTestGroupProcessingResult


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
        _require_schema(fh, path.name)
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

        t_values = _read_2d("stats/signal_activity/t_values")
        p_values = _read_2d("stats/signal_activity/p_values", fill=1.0)
        p_values_uncorrected = _read_2d("stats/signal_activity/p_values_uncorrected", fill=1.0)
        significance_alpha = float_scalar(
            dataset_or_none(fh, "meta/significance_alpha"),
            default=0.05,
        )
        significant_mask_ds = dataset_or_none(fh, "stats/signal_activity/significant_mask")
        if significant_mask_ds is not None:
            significant_mask = np.asarray(significant_mask_ds[:], dtype=bool)
        else:
            significant_mask = np.isfinite(p_values) & (p_values < significance_alpha)

        _early_labels = decode_str_array(
            np.asarray(fh["meta"]["condition_labels"][:], dtype=object)
        )
        label_a = _early_labels[0] if len(_early_labels) >= 1 else "condition_a"
        label_b = _early_labels[1] if len(_early_labels) >= 2 else "condition_b"

        metric_mean = _read_2d("data/condition_difference/mean")
        metric_sem = _read_2d("data/condition_difference/sem")
        condition_a_activity_mean = _read_2d(f"data/signal_activity/{label_a}/mean")
        condition_a_activity_sem = _read_2d(f"data/signal_activity/{label_a}/sem")
        condition_b_activity_mean = _read_2d(f"data/signal_activity/{label_b}/mean")
        condition_b_activity_sem = _read_2d(f"data/signal_activity/{label_b}/sem")

        epoch_mean_t_values = _read_1d("data/summary_epoch/t_values")
        epoch_mean_p_values = _read_1d("data/summary_epoch/p_values", fill=1.0)
        epoch_mean_df = _read_1d("data/summary_epoch/df")
        epoch_mean_metric_mean = _read_1d("data/summary_epoch/condition_difference_mean")
        epoch_mean_metric_sem = _read_1d("data/summary_epoch/condition_difference_sem")

        roi_channel_counts = _read_1d_int("meta/roi_channel_counts")
        roi_subject_counts = _read_1d_int("meta/roi_subject_counts")

        primary_condition_metric = str_scalar(dataset_or_none(fh, "meta/primary_condition_metric"), default="mean_difference")
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
        cluster_windows: list[list[tuple[float, float]]] | None = None
        cluster_null_distributions: list[np.ndarray] | None = None
        if "stats/signal_activity/cluster" in fh:
            cluster_stats = fh["stats/signal_activity/cluster"]
            cluster_p_ds = dataset_or_none(cluster_stats, "p_values")
            if cluster_p_ds is not None:
                cluster_p_values = np.asarray(cluster_p_ds[:], dtype=np.float64)
                starts = np.asarray(cluster_stats["cluster_starts_s"][:], dtype=np.float64)
                ends = np.asarray(cluster_stats["cluster_ends_s"][:], dtype=np.float64)
                if starts.ndim == 1:
                    starts = starts[:, np.newaxis]
                    ends = ends[:, np.newaxis]
                cluster_windows = [
                    [
                        (float(s), float(e))
                        for s, e in zip(row_s, row_e)
                        if np.isfinite(s) and np.isfinite(e)
                    ]
                    for row_s, row_e in zip(starts, ends)
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
            "primary_condition_metric": primary_condition_metric,
            "p_value_correction_method": p_value_correction_method,
            "significance_alpha": significance_alpha,
            "roi_mode": roi_mode,
            "activity_zscore": activity_zscore,
            "activity_baseline_tmin_s": activity_baseline_tmin_s,
            "activity_baseline_tmax_s": activity_baseline_tmax_s,
        },
        signal_activity_stats=GroupTimecourseStats(
            t_values=t_values,
            p_values=p_values,
            p_values_uncorrected=p_values_uncorrected,
            significant_mask=significant_mask,
        ),
        condition_difference=GroupEstimate(mean=metric_mean, sem=metric_sem),
        signal_activity_epoch=GroupEpochStats(
            t=epoch_mean_t_values,
            p=epoch_mean_p_values,
            df=epoch_mean_df,
        ),
        summary_epoch=ConditionTestEpochSummary(
            t_values=epoch_mean_t_values,
            p_values=epoch_mean_p_values,
            df=epoch_mean_df,
            source_condition_difference=GroupEstimate(
                mean=epoch_mean_metric_mean,
                sem=epoch_mean_metric_sem,
            ),
        ),
        time_axis_s=time_axis_s,
        region_names=region_names,
        condition_labels=condition_labels,
        roi_channel_counts=roi_channel_counts,
        roi_subject_counts=roi_subject_counts,
        contributions=contributions,
        signal_activity=GroupEstimatePair(
            condition_a=GroupEstimate(
                mean=condition_a_activity_mean,
                sem=condition_a_activity_sem,
            ),
            condition_b=GroupEstimate(
                mean=condition_b_activity_mean,
                sem=condition_b_activity_sem,
            ),
        ),
        signal_activity_contributions=IndexedConditionContributions(
            condition_a=condition_a_activity_contributions,
            condition_b=condition_b_activity_contributions,
            labels=contribution_labels,
        ),
        primary_condition_metric=primary_condition_metric,
        p_value_correction_method=p_value_correction_method,
        significance_alpha=significance_alpha,
        roi_mode=roi_mode,
        atlas_name=atlas_name,
        source_subject_stats_files=source_subject_stats_files,
        source_electrodes_files=source_electrodes_files,
        excluded_rois=excluded_rois,
        cluster_p_values=cluster_p_values,
        cluster_windows_s=cluster_windows,
        cluster_null_distributions=cluster_null_distributions,
    )


def _load_from_matlab(path: Path) -> ConditionTestGroupProcessingResult:
    from bidsforge.processing.utils.matlab import (
        mat_float,
        mat_root,
        mat_str,
        mat_str_list,
    )
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    data = mat_root(mat, "condition_test_group")
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

    stats_root = data.stats
    stats = getattr(stats_root, "signal_activity", stats_root)
    data_root = getattr(data, "data", data)
    signal_activity = getattr(data_root, "signal_activity", None)
    condition_difference_data = getattr(data_root, "condition_difference", None)
    summary = getattr(data_root, "summary_epoch", getattr(data, "summary_epoch", None))
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

    _early_labels = mat_str_list(getattr(meta, "condition_labels", None))
    label_a = _early_labels[0] if len(_early_labels) >= 1 else "condition_a"
    label_b = _early_labels[1] if len(_early_labels) >= 2 else "condition_b"

    metric_mean = _mat_2d(condition_difference_data, "mean")
    metric_sem = _mat_2d(condition_difference_data, "sem")
    condition_a_activity = getattr(signal_activity, label_a, None)
    condition_b_activity = getattr(signal_activity, label_b, None)
    condition_a_activity_mean = _mat_2d(condition_a_activity, "mean")
    condition_a_activity_sem = _mat_2d(condition_a_activity, "sem")
    condition_b_activity_mean = _mat_2d(condition_b_activity, "mean")
    condition_b_activity_sem = _mat_2d(condition_b_activity, "sem")

    epoch_mean_t_values = _mat_1d(summary, "t_values")
    epoch_mean_p_values = _mat_1d(summary, "p_values", fill=1.0)
    epoch_mean_df = _mat_1d(summary, "df")
    epoch_mean_metric_mean = _mat_1d(summary, "condition_difference_mean")
    epoch_mean_metric_sem = _mat_1d(summary, "condition_difference_sem")

    primary_condition_metric = mat_str(
        getattr(meta, "primary_condition_metric", None),
        default="mean_difference",
    )
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
    contributions_root = getattr(data, "contributions", None)
    contributions = _read_contributions_mat(
        getattr(contributions_root, "summary", contributions_root)
    )
    (
        condition_a_activity_contributions,
        condition_b_activity_contributions,
        contribution_labels,
    ) = _read_activity_contributions_mat(
        getattr(contributions_root, "signal_activity", None)
    )

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
    cluster_windows: list[list[tuple[float, float]]] | None = None
    cluster_null_distributions: list[np.ndarray] | None = None
    cluster_stats = getattr(stats, "cluster", None)
    if cluster_stats is not None:
        raw = getattr(cluster_stats, "p_values", None)
        if raw is not None:
            cluster_p_values = np.asarray(raw, dtype=np.float64).ravel()
            _starts = np.asarray(
                getattr(cluster_stats, "cluster_starts_s", np.zeros((len(cluster_p_values), 0))),
                dtype=np.float64,
            )
            _ends = np.asarray(
                getattr(cluster_stats, "cluster_ends_s", np.zeros((len(cluster_p_values), 0))),
                dtype=np.float64,
            )
            if _starts.ndim == 1:
                _starts = _starts[:, np.newaxis]
                _ends = _ends[:, np.newaxis]
            cluster_windows = [
                [
                    (float(s), float(e))
                    for s, e in zip(row_s, row_e)
                    if np.isfinite(s) and np.isfinite(e)
                ]
                for row_s, row_e in zip(_starts, _ends)
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
            "primary_condition_metric": primary_condition_metric,
            "p_value_correction_method": p_value_correction_method,
            "significance_alpha": significance_alpha,
            "roi_mode": roi_mode,
            "activity_zscore": activity_zscore,
            "activity_baseline_tmin_s": activity_baseline_tmin_s,
            "activity_baseline_tmax_s": activity_baseline_tmax_s,
        },
        signal_activity_stats=GroupTimecourseStats(
            t_values=t_values,
            p_values=p_values,
            p_values_uncorrected=p_values_uncorrected,
            significant_mask=significant_mask,
        ),
        condition_difference=GroupEstimate(mean=metric_mean, sem=metric_sem),
        signal_activity_epoch=GroupEpochStats(
            t=epoch_mean_t_values,
            p=epoch_mean_p_values,
            df=epoch_mean_df,
        ),
        summary_epoch=ConditionTestEpochSummary(
            t_values=epoch_mean_t_values,
            p_values=epoch_mean_p_values,
            df=epoch_mean_df,
            source_condition_difference=GroupEstimate(
                mean=epoch_mean_metric_mean,
                sem=epoch_mean_metric_sem,
            ),
        ),
        time_axis_s=time_axis_s,
        region_names=region_names,
        condition_labels=condition_labels,
        roi_channel_counts=roi_channel_counts,
        roi_subject_counts=roi_subject_counts,
        contributions=contributions,
        signal_activity=GroupEstimatePair(
            condition_a=GroupEstimate(
                mean=condition_a_activity_mean,
                sem=condition_a_activity_sem,
            ),
            condition_b=GroupEstimate(
                mean=condition_b_activity_mean,
                sem=condition_b_activity_sem,
            ),
        ),
        signal_activity_contributions=IndexedConditionContributions(
            condition_a=condition_a_activity_contributions,
            condition_b=condition_b_activity_contributions,
            labels=contribution_labels,
        ),
        primary_condition_metric=primary_condition_metric,
        p_value_correction_method=p_value_correction_method,
        significance_alpha=significance_alpha,
        roi_mode=roi_mode,
        atlas_name=atlas_name,
        source_subject_stats_files=source_subject_stats_files,
        source_electrodes_files=source_electrodes_files,
        excluded_rois=excluded_rois,
        cluster_p_values=cluster_p_values,
        cluster_windows_s=cluster_windows,
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


def _read_activity_contributions_hdf5(
    fh: h5py.File,
    *,
    region_names: list[str],
    n_t: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[str]]]:
    if "contributions/signal_activity" not in fh:
        return [], [], []
    out_a: list[np.ndarray] = []
    out_b: list[np.ndarray] = []
    out_labels: list[list[str]] = []
    group = fh["contributions/signal_activity"]
    for roi_idx, roi_key in enumerate(region_names):
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
    from bidsforge.processing.utils.matlab import mat_str_list

    if raw is None:
        return {}
    return dict(
        zip(
            mat_str_list(getattr(raw, "region", None)),
            mat_str_list(getattr(raw, "reason", None)),
        )
    )


def _read_contributions_mat(raw: object) -> list[ROIChannelContribution]:
    from bidsforge.processing.utils.matlab import mat_str_list

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
    from bidsforge.processing.utils.matlab import mat_str_list

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


def _require_schema(fh: h5py.File, path_name: str) -> None:
    schema_version = str_scalar(dataset_or_none(fh, "meta/schema_version"), default="")
    if schema_version != "1.0":
        raise ValueError(
            f"{path_name}: unsupported trial_stats_group schema. "
            "schema_version='1.0' is required; regenerate outputs with the current writer."
        )

