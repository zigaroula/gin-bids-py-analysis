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

from ..result import ROIChannelContribution
from .result import RegressionGroupProcessingResult


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

        source_metric_t_values = _read_2d("source_metric/t_values")
        source_metric_p_values = _read_2d("source_metric/p_values", fill=1.0)
        source_metric_p_values_uncorrected = _read_2d(
            "source_metric/p_values_uncorrected",
            fill=1.0,
        )
        sig_metric_ds = dataset_or_none(fh, "source_metric/significant_mask")
        if sig_metric_ds is not None:
            source_metric_significant_mask = np.asarray(sig_metric_ds[:], dtype=bool)
        else:
            source_metric_significant_mask = (
                np.isfinite(source_metric_p_values)
                & (source_metric_p_values < significance_alpha)
            )
        condition_a_source_metric_mean = _read_2d("source_metric/condition_a_mean")
        condition_a_source_metric_sem = _read_2d("source_metric/condition_a_sem")
        condition_b_source_metric_mean = _read_2d("source_metric/condition_b_mean")
        condition_b_source_metric_sem = _read_2d("source_metric/condition_b_sem")
        epoch_source_metric_t = _read_1d("source_metric/epoch_summary/t")
        epoch_source_metric_p = _read_1d("source_metric/epoch_summary/p", fill=1.0)
        epoch_source_metric_df = _read_1d("source_metric/epoch_summary/df")

        activity_t_values = _read_2d("activity/t_values")
        activity_p_values = _read_2d("activity/p_values", fill=1.0)
        activity_p_values_uncorrected = _read_2d(
            "activity/p_values_uncorrected",
            fill=1.0,
        )
        sig_activity_ds = dataset_or_none(fh, "activity/significant_mask")
        if sig_activity_ds is not None:
            activity_significant_mask = np.asarray(sig_activity_ds[:], dtype=bool)
        else:
            activity_significant_mask = np.isfinite(activity_p_values) & (
                activity_p_values < significance_alpha
            )
        epoch_activity_t = _read_1d("activity/epoch_summary/t")
        epoch_activity_p = _read_1d("activity/epoch_summary/p", fill=1.0)
        epoch_activity_df = _read_1d("activity/epoch_summary/df")

        condition_a_activity_mean = _read_2d("means/condition_a_mean")
        condition_a_activity_sem = _read_2d("means/condition_a_sem")
        condition_b_activity_mean = _read_2d("means/condition_b_mean")
        condition_b_activity_sem = _read_2d("means/condition_b_sem")

        condition_a_r_value_mean = _read_2d("r_values/condition_a_mean")
        condition_a_r_value_sem = _read_2d("r_values/condition_a_sem")
        condition_b_r_value_mean = _read_2d("r_values/condition_b_mean")
        condition_b_r_value_sem = _read_2d("r_values/condition_b_sem")

        roi_channel_counts = _read_1d_int("meta/roi_channel_counts")
        roi_subject_counts = _read_1d_int("meta/roi_subject_counts")

        labels = decode_str_array(
            np.asarray(fh["meta"]["condition_labels"][:], dtype=object)
        )
        condition_labels: tuple[str, str] = (
            labels[0] if len(labels) >= 1 else "condition_a",
            labels[1] if len(labels) >= 2 else "condition_b",
        )
        source_metric = str_scalar(
            dataset_or_none(fh, "meta/source_metric"),
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
            "source_metric": source_metric,
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
            "trial_activity_summary_missing_response_policy": str_scalar(
                dataset_or_none(
                    fh,
                    "meta/trial_activity_summary_missing_response_policy",
                ),
                default="drop_trial",
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
            condition_a_activity_contributions,
            condition_b_activity_contributions,
            contribution_labels,
        ) = _read_indexed_condition_contributions_hdf5(
            fh,
            group_name="activity_contributions",
            region_names=region_names,
            n_t=n_t,
        )
        (
            condition_a_source_metric_contributions,
            condition_b_source_metric_contributions,
            _,
        ) = _read_indexed_condition_contributions_hdf5(
            fh,
            group_name="source_metric_contributions",
            region_names=region_names,
            n_t=n_t,
        )
        (
            condition_a_scatter_predictor,
            condition_a_scatter_activity,
            condition_b_scatter_predictor,
            condition_b_scatter_activity,
        ) = _read_scatter_hdf5(fh, region_names=region_names)

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

    source_group = BIDSFileGroup(primary=BIDSFile.from_path(path))
    return RegressionGroupProcessingResult(
        source_group=source_group,
        metadata=metadata,
        source_metric_t_values=source_metric_t_values,
        source_metric_p_values=source_metric_p_values,
        source_metric_p_values_uncorrected=source_metric_p_values_uncorrected,
        source_metric_significant_mask=source_metric_significant_mask,
        activity_t_values=activity_t_values,
        activity_p_values=activity_p_values,
        activity_p_values_uncorrected=activity_p_values_uncorrected,
        activity_significant_mask=activity_significant_mask,
        condition_a_source_metric_mean=condition_a_source_metric_mean,
        condition_a_source_metric_sem=condition_a_source_metric_sem,
        condition_b_source_metric_mean=condition_b_source_metric_mean,
        condition_b_source_metric_sem=condition_b_source_metric_sem,
        condition_a_activity_mean=condition_a_activity_mean,
        condition_a_activity_sem=condition_a_activity_sem,
        condition_b_activity_mean=condition_b_activity_mean,
        condition_b_activity_sem=condition_b_activity_sem,
        condition_a_r_value_mean=condition_a_r_value_mean,
        condition_a_r_value_sem=condition_a_r_value_sem,
        condition_b_r_value_mean=condition_b_r_value_mean,
        condition_b_r_value_sem=condition_b_r_value_sem,
        epoch_source_metric_t=epoch_source_metric_t,
        epoch_source_metric_p=epoch_source_metric_p,
        epoch_source_metric_df=epoch_source_metric_df,
        epoch_activity_t=epoch_activity_t,
        epoch_activity_p=epoch_activity_p,
        epoch_activity_df=epoch_activity_df,
        time_axis_s=time_axis_s,
        region_names=region_names,
        condition_labels=condition_labels,
        source_metric=source_metric,
        contrast_mode=contrast_mode,
        roi_channel_counts=roi_channel_counts,
        roi_subject_counts=roi_subject_counts,
        contributions=contributions,
        condition_a_source_metric_contributions=condition_a_source_metric_contributions,
        condition_b_source_metric_contributions=condition_b_source_metric_contributions,
        condition_a_activity_contributions=condition_a_activity_contributions,
        condition_b_activity_contributions=condition_b_activity_contributions,
        contribution_labels=contribution_labels,
        p_value_correction_method=p_value_correction_method,
        significance_alpha=significance_alpha,
        roi_mode=roi_mode,
        atlas_name=atlas_name,
        source_subject_stats_files=source_subject_stats_files,
        source_electrodes_files=source_electrodes_files,
        excluded_rois=excluded_rois,
        condition_a_scatter_predictor=condition_a_scatter_predictor,
        condition_a_scatter_activity=condition_a_scatter_activity,
        condition_b_scatter_predictor=condition_b_scatter_predictor,
        condition_b_scatter_activity=condition_b_scatter_activity,
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
    significance_alpha = mat_float(getattr(meta, "significance_alpha", None), default=0.05)

    source_metric = data.source_metric
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
    condition_a_source_metric_mean = _mat_2d(source_metric, "condition_a_mean")
    condition_a_source_metric_sem = _mat_2d(source_metric, "condition_a_sem")
    condition_b_source_metric_mean = _mat_2d(source_metric, "condition_b_mean")
    condition_b_source_metric_sem = _mat_2d(source_metric, "condition_b_sem")
    epoch_summary = getattr(source_metric, "epoch_summary", None)
    epoch_source_metric_t = _mat_1d(epoch_summary, "t")
    epoch_source_metric_p = _mat_1d(epoch_summary, "p", fill=1.0)
    epoch_source_metric_df = _mat_1d(epoch_summary, "df")

    activity = data.activity
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

    means = data.means
    condition_a_activity_mean = _mat_2d(means, "condition_a_mean")
    condition_a_activity_sem = _mat_2d(means, "condition_a_sem")
    condition_b_activity_mean = _mat_2d(means, "condition_b_mean")
    condition_b_activity_sem = _mat_2d(means, "condition_b_sem")

    r_values = getattr(data, "r_values", None)
    condition_a_r_value_mean = _mat_2d(r_values, "condition_a_mean")
    condition_a_r_value_sem = _mat_2d(r_values, "condition_a_sem")
    condition_b_r_value_mean = _mat_2d(r_values, "condition_b_mean")
    condition_b_r_value_sem = _mat_2d(r_values, "condition_b_sem")

    labels = mat_str_list(getattr(meta, "condition_labels", None))
    condition_labels: tuple[str, str] = (
        labels[0] if len(labels) >= 1 else "condition_a",
        labels[1] if len(labels) >= 2 else "condition_b",
    )
    source_metric_name = mat_str(getattr(meta, "source_metric", None), default="slope")
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
        "source_metric": source_metric_name,
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
        "trial_activity_summary_missing_response_policy": mat_str(
            getattr(meta, "trial_activity_summary_missing_response_policy", None),
            default="drop_trial",
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
    contributions = _read_contributions_mat(getattr(data, "contributions", None))
    (
        condition_a_activity_contributions,
        condition_b_activity_contributions,
        contribution_labels,
    ) = _read_indexed_condition_contributions_mat(
        getattr(data, "activity_contributions", None)
    )
    (
        condition_a_source_metric_contributions,
        condition_b_source_metric_contributions,
        _,
    ) = _read_indexed_condition_contributions_mat(
        getattr(data, "source_metric_contributions", None)
    )
    (
        condition_a_scatter_predictor,
        condition_a_scatter_activity,
        condition_b_scatter_predictor,
        condition_b_scatter_activity,
    ) = _read_scatter_mat(getattr(data, "scatter_data", None))

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

    source_group = BIDSFileGroup(primary=BIDSFile.from_path(path))
    return RegressionGroupProcessingResult(
        source_group=source_group,
        metadata=metadata,
        source_metric_t_values=source_metric_t_values,
        source_metric_p_values=source_metric_p_values,
        source_metric_p_values_uncorrected=source_metric_p_values_uncorrected,
        source_metric_significant_mask=source_metric_significant_mask,
        activity_t_values=activity_t_values,
        activity_p_values=activity_p_values,
        activity_p_values_uncorrected=activity_p_values_uncorrected,
        activity_significant_mask=activity_significant_mask,
        condition_a_source_metric_mean=condition_a_source_metric_mean,
        condition_a_source_metric_sem=condition_a_source_metric_sem,
        condition_b_source_metric_mean=condition_b_source_metric_mean,
        condition_b_source_metric_sem=condition_b_source_metric_sem,
        condition_a_activity_mean=condition_a_activity_mean,
        condition_a_activity_sem=condition_a_activity_sem,
        condition_b_activity_mean=condition_b_activity_mean,
        condition_b_activity_sem=condition_b_activity_sem,
        condition_a_r_value_mean=condition_a_r_value_mean,
        condition_a_r_value_sem=condition_a_r_value_sem,
        condition_b_r_value_mean=condition_b_r_value_mean,
        condition_b_r_value_sem=condition_b_r_value_sem,
        epoch_source_metric_t=epoch_source_metric_t,
        epoch_source_metric_p=epoch_source_metric_p,
        epoch_source_metric_df=epoch_source_metric_df,
        epoch_activity_t=epoch_activity_t,
        epoch_activity_p=epoch_activity_p,
        epoch_activity_df=epoch_activity_df,
        time_axis_s=time_axis_s,
        region_names=region_names,
        condition_labels=condition_labels,
        source_metric=source_metric_name,
        contrast_mode=contrast_mode,
        roi_channel_counts=roi_channel_counts,
        roi_subject_counts=roi_subject_counts,
        contributions=contributions,
        condition_a_source_metric_contributions=condition_a_source_metric_contributions,
        condition_b_source_metric_contributions=condition_b_source_metric_contributions,
        condition_a_activity_contributions=condition_a_activity_contributions,
        condition_b_activity_contributions=condition_b_activity_contributions,
        contribution_labels=contribution_labels,
        p_value_correction_method=p_value_correction_method,
        significance_alpha=significance_alpha,
        roi_mode=roi_mode,
        atlas_name=atlas_name,
        source_subject_stats_files=source_subject_stats_files,
        source_electrodes_files=source_electrodes_files,
        excluded_rois=excluded_rois,
        condition_a_scatter_predictor=condition_a_scatter_predictor,
        condition_a_scatter_activity=condition_a_scatter_activity,
        condition_b_scatter_predictor=condition_b_scatter_predictor,
        condition_b_scatter_activity=condition_b_scatter_activity,
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


def _read_indexed_condition_contributions_hdf5(
    fh: h5py.File,
    *,
    group_name: str,
    region_names: list[str],
    n_t: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[str]]]:
    if group_name not in fh:
        return [], [], []
    out_a: list[np.ndarray] = []
    out_b: list[np.ndarray] = []
    out_labels: list[list[str]] = []
    group = fh[group_name]
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


def _read_scatter_hdf5(
    fh: h5py.File,
    *,
    region_names: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    if "scatter_data" not in fh:
        return [], [], [], []
    out_pred_a: list[np.ndarray] = []
    out_act_a: list[np.ndarray] = []
    out_pred_b: list[np.ndarray] = []
    out_act_b: list[np.ndarray] = []
    group = fh["scatter_data"]
    for roi_idx in range(len(region_names)):
        roi_key = str(roi_idx)
        if roi_key not in group:
            out_pred_a.append(np.empty(0, dtype=np.float64))
            out_act_a.append(np.empty(0, dtype=np.float64))
            out_pred_b.append(np.empty(0, dtype=np.float64))
            out_act_b.append(np.empty(0, dtype=np.float64))
            continue
        roi_group = group[roi_key]
        out_pred_a.append(np.asarray(roi_group["condition_a_predictor"][:], dtype=np.float64))
        out_act_a.append(np.asarray(roi_group["condition_a_activity"][:], dtype=np.float64))
        out_pred_b.append(np.asarray(roi_group["condition_b_predictor"][:], dtype=np.float64))
        out_act_b.append(np.asarray(roi_group["condition_b_activity"][:], dtype=np.float64))
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
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[str]]]:
    from gin_bids_py_analysis.processing.utils.matlab import mat_str_list

    if raw is None:
        return [], [], []
    out_a: list[np.ndarray] = []
    out_b: list[np.ndarray] = []
    out_labels: list[list[str]] = []
    cond_a_raw = getattr(raw, "condition_a", None)
    cond_b_raw = getattr(raw, "condition_b", None)
    labels_raw = getattr(raw, "labels", None)
    if cond_a_raw is not None:
        for item in np.asarray(cond_a_raw).ravel():
            out_a.append(np.asarray(item, dtype=np.float64))
    if cond_b_raw is not None:
        for item in np.asarray(cond_b_raw).ravel():
            out_b.append(np.asarray(item, dtype=np.float64))
    if labels_raw is not None:
        for item in np.asarray(labels_raw).ravel():
            out_labels.append(mat_str_list(item))
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
        ("condition_a_activity", out_act_a),
        ("condition_b_predictor", out_pred_b),
        ("condition_b_activity", out_act_b),
    ):
        cell = getattr(raw, attr, None)
        if cell is None:
            continue
        for item in np.asarray(cell).ravel():
            target.append(np.asarray(item, dtype=np.float64).ravel())
    return out_pred_a, out_act_a, out_pred_b, out_act_b
