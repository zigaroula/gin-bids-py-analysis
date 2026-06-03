from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.processing.utils.group_stats import ROIChannelContribution
from bidsforge.processing.utils.hdf5 import dataset_or_none, decode_str_array, str_scalar
from bidsforge.processing.utils.matlab import mat_root, mat_str, mat_str_list

from ..result import (
    TFGroupEpochStats,
    TFGroupEstimate,
    TFGroupEstimatePair,
    TFGroupStats,
    TFIndexedConditionContributions,
)
from .result import TFRegressionGroupStats, TimeFrequencyRegressionGroupResult


def load_time_frequency_regression_group_result(
    path: Path | str,
) -> TimeFrequencyRegressionGroupResult:
    path = Path(path)
    if path.suffix.lower() == ".mat":
        return _load_matlab(path)
    return _load_hdf5(path)


def _load_hdf5(path: Path) -> TimeFrequencyRegressionGroupResult:
    with h5py.File(path, "r") as fh:
        _require_schema(fh, path.name, "time_frequency_regression_group")
        regions = decode_str_array(np.asarray(fh["axes/region"][:], dtype=object))
        labels = decode_str_array(np.asarray(fh["meta/condition_labels"][:], dtype=object))
        label_a, label_b = labels[0], labels[1]
        return TimeFrequencyRegressionGroupResult(
            source_group=BIDSFileGroup(primary=BIDSFile.from_path(path)),
            metadata=_load_meta(fh),
            frequency_hz=np.asarray(fh["axes/frequency_hz"][:], dtype=np.float64),
            time_axis_s=np.asarray(fh["axes/time_s"][:], dtype=np.float64),
            region_names=regions,
            condition_labels=(label_a, label_b),
            roi_channel_counts=np.asarray(fh["meta/roi_channel_counts"][:], dtype=np.int64),
            roi_subject_counts=np.asarray(fh["meta/roi_subject_counts"][:], dtype=np.int64),
            contributions=_load_summary_contributions(fh),
            source_subject_stats_files=_load_optional_str_list(
                fh,
                "provenance/source_subject_stats_files",
            ),
            source_electrodes_files=_load_optional_str_list(
                fh,
                "provenance/source_electrodes_files",
            ),
            excluded_rois=_load_excluded_rois(fh),
            p_value_correction_method=str_scalar(
                dataset_or_none(fh, "meta/p_value_correction_method"),
                default="none",
            ),
            significance_alpha=float(np.asarray(fh["meta/significance_alpha"][()]).item()),
            roi_mode=str_scalar(dataset_or_none(fh, "meta/roi_mode"), default="manual"),
            atlas_name=str_scalar(dataset_or_none(fh, "meta/atlas_name"), default="") or None,
            source_metric=TFGroupEstimatePair(
                condition_a=_load_estimate(
                    fh,
                    f"data/regression/source_metric/{label_a}",
                ),
                condition_b=_load_estimate(
                    fh,
                    f"data/regression/source_metric/{label_b}",
                ),
            ),
            signal_activity=TFGroupEstimatePair(
                condition_a=_load_estimate(fh, f"data/signal_activity/{label_a}"),
                condition_b=_load_estimate(fh, f"data/signal_activity/{label_b}"),
            ),
            r_value=TFGroupEstimatePair(
                condition_a=_load_estimate(fh, f"data/regression/r_value/{label_a}"),
                condition_b=_load_estimate(fh, f"data/regression/r_value/{label_b}"),
            ),
            stats=TFRegressionGroupStats(
                condition_contrast=_load_stats(
                    fh,
                    "stats/regression/condition_contrast",
                ),
                condition_a_vs_zero=_load_stats(
                    fh,
                    f"stats/regression/{label_a}_vs_zero",
                ),
                condition_b_vs_zero=_load_stats(
                    fh,
                    f"stats/regression/{label_b}_vs_zero",
                ),
                epoch_summary=_load_epoch(fh, "stats/regression/epoch_summary"),
            ),
            source_metric_contributions=_load_indexed_condition_contributions(
                fh,
                "contributions/regression/source_metric",
                regions,
                label_a,
                label_b,
            ),
            signal_activity_contributions=_load_indexed_condition_contributions(
                fh,
                "contributions/signal_activity",
                regions,
                label_a,
                label_b,
            ),
            primary_regression_metric=str_scalar(
                dataset_or_none(fh, "meta/primary_regression_metric"),
                default="t_values",
            ),
            contrast_mode=str_scalar(dataset_or_none(fh, "meta/contrast_mode"), default="paired"),
        )


def _load_matlab(path: Path) -> TimeFrequencyRegressionGroupResult:
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    root = mat_root(mat, "time_frequency_regression_group")
    labels = mat_str_list(root.meta.condition_labels)
    label_a, label_b = labels[0], labels[1]
    regions = mat_str_list(root.axes.region)
    n_rois = len(regions)
    n_freqs = int(np.asarray(root.axes.frequency_hz).size)
    n_times = int(np.asarray(root.axes.time_s).size)
    return TimeFrequencyRegressionGroupResult(
        source_group=BIDSFileGroup(primary=BIDSFile.from_path(path)),
        metadata={"analysis_type": mat_str(getattr(root.meta, "analysis_type", None))},
        frequency_hz=np.asarray(root.axes.frequency_hz, dtype=np.float64).ravel(),
        time_axis_s=np.asarray(root.axes.time_s, dtype=np.float64).ravel(),
        region_names=regions,
        condition_labels=(label_a, label_b),
        roi_channel_counts=np.asarray(root.meta.roi_channel_counts, dtype=np.int64).ravel(),
        roi_subject_counts=np.asarray(root.meta.roi_subject_counts, dtype=np.int64).ravel(),
        p_value_correction_method=mat_str(
            getattr(root.meta, "p_value_correction_method", None),
            default="none",
        ),
        significance_alpha=float(np.asarray(root.meta.significance_alpha).ravel()[0]),
        roi_mode=mat_str(getattr(root.meta, "roi_mode", None), default="manual"),
        atlas_name=mat_str(getattr(root.meta, "atlas_name", None), default="") or None,
        source_metric=TFGroupEstimatePair(
            condition_a=TFGroupEstimate(
                mean=_coerce_group_tf(
                    getattr(root.data.regression.source_metric, label_a).mean,
                    n_rois,
                    n_freqs,
                    n_times,
                ),
                sem=_coerce_group_tf(
                    getattr(root.data.regression.source_metric, label_a).sem,
                    n_rois,
                    n_freqs,
                    n_times,
                ),
            ),
            condition_b=TFGroupEstimate(
                mean=_coerce_group_tf(
                    getattr(root.data.regression.source_metric, label_b).mean,
                    n_rois,
                    n_freqs,
                    n_times,
                ),
                sem=_coerce_group_tf(
                    getattr(root.data.regression.source_metric, label_b).sem,
                    n_rois,
                    n_freqs,
                    n_times,
                ),
            ),
        ),
        signal_activity=TFGroupEstimatePair(
            condition_a=TFGroupEstimate(
                mean=_coerce_group_tf(getattr(root.data.signal_activity, label_a).mean, n_rois, n_freqs, n_times),
                sem=_coerce_group_tf(getattr(root.data.signal_activity, label_a).sem, n_rois, n_freqs, n_times),
            ),
            condition_b=TFGroupEstimate(
                mean=_coerce_group_tf(getattr(root.data.signal_activity, label_b).mean, n_rois, n_freqs, n_times),
                sem=_coerce_group_tf(getattr(root.data.signal_activity, label_b).sem, n_rois, n_freqs, n_times),
            ),
        ),
        r_value=TFGroupEstimatePair(
            condition_a=TFGroupEstimate(
                mean=_coerce_group_tf(getattr(root.data.regression.r_value, label_a).mean, n_rois, n_freqs, n_times),
                sem=_coerce_group_tf(getattr(root.data.regression.r_value, label_a).sem, n_rois, n_freqs, n_times),
            ),
            condition_b=TFGroupEstimate(
                mean=_coerce_group_tf(getattr(root.data.regression.r_value, label_b).mean, n_rois, n_freqs, n_times),
                sem=_coerce_group_tf(getattr(root.data.regression.r_value, label_b).sem, n_rois, n_freqs, n_times),
            ),
        ),
        stats=TFRegressionGroupStats(
            condition_contrast=TFGroupStats(
                t_values=_coerce_group_tf(
                    root.stats.regression.condition_contrast.t_values,
                    n_rois,
                    n_freqs,
                    n_times,
                ),
                p_values=_coerce_group_tf(
                    root.stats.regression.condition_contrast.p_values,
                    n_rois,
                    n_freqs,
                    n_times,
                ),
                p_values_uncorrected=_coerce_group_tf(
                    root.stats.regression.condition_contrast.p_values_uncorrected,
                    n_rois,
                    n_freqs,
                    n_times,
                ),
                significant_mask=_coerce_group_tf(
                    root.stats.regression.condition_contrast.significant_mask,
                    n_rois,
                    n_freqs,
                    n_times,
                    dtype=bool,
                ),
            ),
            epoch_summary=TFGroupEpochStats(
                t=np.asarray(root.stats.regression.epoch_summary.t, dtype=np.float64).ravel(),
                p=np.asarray(root.stats.regression.epoch_summary.p, dtype=np.float64).ravel(),
                df=np.asarray(root.stats.regression.epoch_summary.df, dtype=np.float64).ravel(),
                mean=np.asarray(root.stats.regression.epoch_summary.mean, dtype=np.float64).ravel(),
                sem=np.asarray(root.stats.regression.epoch_summary.sem, dtype=np.float64).ravel(),
            ),
        ),
        primary_regression_metric=mat_str(
            getattr(root.meta, "primary_regression_metric", None),
            default="t_values",
        ),
        contrast_mode=mat_str(getattr(root.meta, "contrast_mode", None), default="paired"),
    )


def _require_schema(fh: h5py.File, name: str, analysis_type: str) -> None:
    schema = str_scalar(dataset_or_none(fh, "meta/schema_name"), default="")
    actual = str_scalar(dataset_or_none(fh, "meta/analysis_type"), default="")
    if schema != "time_frequency_stats_group" or actual != analysis_type:
        raise ValueError(
            f"{name}: unsupported schema_name={schema!r}, analysis_type={actual!r}."
        )


def _load_meta(fh: h5py.File) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, node in fh["meta"].items():
        if isinstance(node, h5py.Dataset) and node.shape == ():
            value = node[()]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            out[key] = value
    return out


def _load_estimate(fh: h5py.File, key: str) -> TFGroupEstimate:
    return TFGroupEstimate(
        mean=np.asarray(fh[f"{key}/mean"][:], dtype=np.float64),
        sem=np.asarray(fh[f"{key}/sem"][:], dtype=np.float64),
    )


def _load_stats(fh: h5py.File, key: str) -> TFGroupStats:
    group = fh[key]
    cluster = group.get("cluster")
    return TFGroupStats(
        t_values=np.asarray(group["t_values"][:], dtype=np.float64),
        p_values=np.asarray(group["p_values"][:], dtype=np.float64),
        p_values_uncorrected=np.asarray(group["p_values_uncorrected"][:], dtype=np.float64),
        significant_mask=np.asarray(group["significant_mask"][:], dtype=bool),
        cluster_labels=(
            [np.asarray(item, dtype=np.int64) for item in cluster["labels"][:]]
            if cluster is not None and "labels" in cluster
            else None
        ),
        cluster_sums=(
            [np.asarray(item, dtype=np.float64) for item in cluster["sums"][:]]
            if cluster is not None and "sums" in cluster
            else None
        ),
        cluster_null_distributions=(
            [np.asarray(item, dtype=np.float64) for item in cluster["null_distributions"][:]]
            if cluster is not None and "null_distributions" in cluster
            else None
        ),
    )


def _load_epoch(fh: h5py.File, key: str) -> TFGroupEpochStats:
    return TFGroupEpochStats(
        t=np.asarray(fh[f"{key}/t"][:], dtype=np.float64),
        p=np.asarray(fh[f"{key}/p"][:], dtype=np.float64),
        df=np.asarray(fh[f"{key}/df"][:], dtype=np.float64),
        mean=np.asarray(fh[f"{key}/mean"][:], dtype=np.float64),
        sem=np.asarray(fh[f"{key}/sem"][:], dtype=np.float64),
    )


def _load_summary_contributions(fh: h5py.File) -> list[ROIChannelContribution]:
    if "contributions/summary" not in fh:
        return []
    group = fh["contributions/summary"]
    rois = decode_str_array(np.asarray(group["region"][:], dtype=object))
    subjects = decode_str_array(np.asarray(group["subject"][:], dtype=object))
    channels = decode_str_array(np.asarray(group["channel"][:], dtype=object))
    files = decode_str_array(np.asarray(group["source_stats_file"][:], dtype=object))
    return [
        ROIChannelContribution(roi=roi, subject=subject, channel=channel, source_stats_file=file)
        for roi, subject, channel, file in zip(rois, subjects, channels, files, strict=False)
    ]


def _load_indexed_condition_contributions(
    fh: h5py.File,
    key: str,
    regions: list[str],
    label_a: str,
    label_b: str,
) -> TFIndexedConditionContributions:
    if key not in fh:
        return TFIndexedConditionContributions()
    a: list[np.ndarray] = []
    b: list[np.ndarray] = []
    labels: list[list[str]] = []
    for roi in regions:
        if f"{key}/{roi}" not in fh:
            a.append(np.array([]))
            b.append(np.array([]))
            labels.append([])
            continue
        group = fh[f"{key}/{roi}"]
        a.append(np.asarray(group[label_a][:], dtype=np.float64))
        b.append(np.asarray(group[label_b][:], dtype=np.float64))
        labels.append(decode_str_array(np.asarray(group["labels"][:], dtype=object)))
    return TFIndexedConditionContributions(condition_a=a, condition_b=b, labels=labels)


def _load_excluded_rois(fh: h5py.File) -> dict[str, str]:
    if "excluded_rois" not in fh:
        return {}
    group = fh["excluded_rois"]
    regions = decode_str_array(np.asarray(group["region"][:], dtype=object))
    reasons = decode_str_array(np.asarray(group["reason"][:], dtype=object))
    return dict(zip(regions, reasons, strict=False))


def _load_optional_str_list(fh: h5py.File, key: str) -> list[str]:
    ds = dataset_or_none(fh, key)
    if ds is None:
        return []
    return decode_str_array(np.asarray(ds[:], dtype=object))


def _coerce_group_tf(
    value: object,
    n_rois: int,
    n_freqs: int,
    n_times: int,
    *,
    dtype: object = np.float64,
) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    expected = (n_rois, n_freqs, n_times)
    if arr.shape == expected:
        return arr
    if n_rois == 1 and arr.shape == (n_freqs, n_times):
        return arr.reshape(expected)
    if n_freqs == 1 and arr.shape == (n_rois, n_times):
        return arr.reshape(expected)
    if n_times == 1 and arr.shape == (n_rois, n_freqs):
        return arr.reshape(expected)
    if arr.size == n_rois * n_freqs * n_times:
        return arr.reshape(expected)
    return np.full(expected, np.nan, dtype=dtype)
