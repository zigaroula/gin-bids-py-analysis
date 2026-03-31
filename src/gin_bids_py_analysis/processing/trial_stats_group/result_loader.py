"""Load a pre-computed ``TrialStatsGroupProcessingResult`` from disk.

Supports the HDF5 (``.h5`` / ``.hdf5``) and MATLAB (``.mat``) formats written
by ``TrialStatsGroupProcessingWriter``.  The returned result can be fed directly
to the visualization layer without re-running the group processing pipeline.
"""

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
    int_scalar,
    str_scalar,
)

from .result import ROIChannelContribution, TrialStatsGroupProcessingResult


def load_trial_stats_group_result(
    path: Path | str,
) -> TrialStatsGroupProcessingResult:
    """Load a pre-computed ``TrialStatsGroupProcessingResult`` from *path*.

    Parameters
    ----------
    path:
        Path to an ``.h5``/``.hdf5`` or ``.mat`` group stats file written by
        ``TrialStatsGroupProcessingWriter``.

    Returns
    -------
    TrialStatsGroupProcessingResult
        A fully populated result object ready for visualization.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be interpreted as a valid group stats output.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Group stats file not found: {path}")
    ext = path.suffix.lower()
    if ext == ".mat":
        return _load_from_matlab(path)
    return _load_from_hdf5(path)


# ---------------------------------------------------------------------------
# HDF5 loader
# ---------------------------------------------------------------------------


def _load_from_hdf5(path: Path) -> TrialStatsGroupProcessingResult:
    with h5py.File(path, "r") as fh:
        # --- axes ---
        if "axes" not in fh or "region" not in fh["axes"]:
            raise ValueError(
                f"{path.name}: axes/region dataset is required for group stats."
            )
        region_names = decode_str_array(np.asarray(fh["axes"]["region"][:]))
        time_axis_s = np.asarray(fh["axes"]["time_s"][:], dtype=np.float64)
        n_rois = len(region_names)
        n_t = len(time_axis_s)
        _zeros = np.zeros((n_rois, n_t), dtype=np.float64)

        # --- stats arrays ---
        def _read_2d(key: str, fill: float = 0.0) -> np.ndarray:
            ds = dataset_or_none(fh, key)
            if ds is None:
                return np.full((n_rois, n_t), fill, dtype=np.float64)
            return np.asarray(ds[:], dtype=np.float64)

        t_values = _read_2d("stats/t_values")
        p_values = _read_2d("stats/p_values", fill=1.0)
        p_values_uncorrected = _read_2d("stats/p_values_uncorrected", fill=1.0)
        sig_ds = dataset_or_none(fh, "stats/significant_mask")
        significance_alpha = float_scalar(
            dataset_or_none(fh, "meta/significance_alpha"), default=0.05
        )
        if sig_ds is not None:
            _loaded_mask = np.asarray(sig_ds[:], dtype=bool)
            if _loaded_mask.ndim == 2 and _loaded_mask.shape == (n_rois, n_t):
                significant_mask = _loaded_mask
            else:
                significant_mask = np.isfinite(p_values) & (p_values < significance_alpha)
        else:
            significant_mask = np.isfinite(p_values) & (p_values < significance_alpha)

        # --- means ---
        metric_mean = _read_2d("means/metric_mean")
        condition_a_group_mean = _read_2d("means/condition_a_mean")
        condition_a_group_sem = _read_2d("means/condition_a_sem")
        condition_b_group_mean = _read_2d("means/condition_b_mean")
        condition_b_group_sem = _read_2d("means/condition_b_sem")

        # --- uncertainty ---
        metric_sem = _read_2d("uncertainty/metric_sem")

        # --- summary epoch ---
        def _read_1d(key: str, fill: float = 0.0) -> np.ndarray:
            ds = dataset_or_none(fh, key)
            if ds is None:
                return np.full(n_rois, fill, dtype=np.float64)
            return np.asarray(ds[:], dtype=np.float64).ravel()

        epoch_mean_t_values = _read_1d("summary_epoch/t_values")
        epoch_mean_p_values = _read_1d("summary_epoch/p_values", fill=1.0)
        epoch_mean_df = _read_1d("summary_epoch/df")
        epoch_mean_metric_mean = _read_1d("summary_epoch/metric_mean")
        epoch_mean_metric_sem = _read_1d("summary_epoch/metric_sem")

        rc_ds = dataset_or_none(fh, "summary_epoch/roi_channel_counts")
        roi_channel_counts = (
            np.asarray(rc_ds[:], dtype=np.int64).ravel()
            if rc_ds is not None
            else np.zeros(n_rois, dtype=np.int64)
        )
        rs_ds = dataset_or_none(fh, "summary_epoch/roi_subject_counts")
        roi_subject_counts = (
            np.asarray(rs_ds[:], dtype=np.int64).ravel()
            if rs_ds is not None
            else np.zeros(n_rois, dtype=np.int64)
        )

        # --- meta ---
        source_metric = str_scalar(dataset_or_none(fh, "meta/source_metric"), default="mean_difference")
        labels_ds = dataset_or_none(fh, "meta/condition_labels")
        if labels_ds is not None:
            labels = decode_str_array(np.asarray(labels_ds[:], dtype=object))
            condition_labels: tuple[str, str] = (
                labels[0] if len(labels) >= 1 else "condition_a",
                labels[1] if len(labels) >= 2 else "condition_b",
            )
        else:
            condition_labels = ("condition_a", "condition_b")
        p_value_correction_method = str_scalar(
            dataset_or_none(fh, "meta/p_value_correction_method"), default="none"
        )
        roi_mode = str_scalar(dataset_or_none(fh, "meta/roi_mode"), default="manual")
        atlas_name_raw = str_scalar(dataset_or_none(fh, "meta/atlas_name"), default="")
        atlas_name: str | None = atlas_name_raw.strip() or None

        # excluded ROIs
        ex_region_ds = dataset_or_none(fh, "meta/excluded_rois/region")
        ex_reason_ds = dataset_or_none(fh, "meta/excluded_rois/reason")
        if ex_region_ds is not None and ex_reason_ds is not None:
            ex_regions = decode_str_array(np.asarray(ex_region_ds[:], dtype=object))
            ex_reasons = decode_str_array(np.asarray(ex_reason_ds[:], dtype=object))
            excluded_rois: dict[str, str] = dict(zip(ex_regions, ex_reasons))
        else:
            excluded_rois = {}

        # --- contributions ---
        contributions: list[ROIChannelContribution] = []
        if "contributions" in fh:
            cg = fh["contributions"]
            if all(k in cg for k in ["region", "subject", "channel", "source_stats_file"]):
                reis = decode_str_array(np.asarray(cg["region"][:], dtype=object))
                subjs = decode_str_array(np.asarray(cg["subject"][:], dtype=object))
                chs = decode_str_array(np.asarray(cg["channel"][:], dtype=object))
                srcs = decode_str_array(np.asarray(cg["source_stats_file"][:], dtype=object))
                contributions = [
                    ROIChannelContribution(roi=r, subject=s, channel=c, source_stats_file=f)
                    for r, s, c, f in zip(reis, subjs, chs, srcs)
                ]

        # --- contribution epochs (optional) ---
        cond_a_contribs: list[np.ndarray] = []
        cond_b_contribs: list[np.ndarray] = []
        contrib_labels: list[list[str]] = []
        if "contribution_epochs" in fh:
            ce_grp = fh["contribution_epochs"]
            for roi_name in region_names:
                if roi_name in ce_grp:
                    roi_grp = ce_grp[roi_name]
                    ca = (
                        np.asarray(roi_grp["condition_a"][:], dtype=np.float64)
                        if "condition_a" in roi_grp
                        else np.empty((0, n_t), dtype=np.float64)
                    )
                    cb = (
                        np.asarray(roi_grp["condition_b"][:], dtype=np.float64)
                        if "condition_b" in roi_grp
                        else np.empty((0, n_t), dtype=np.float64)
                    )
                    lbl = (
                        decode_str_array(np.asarray(roi_grp["labels"][:], dtype=object))
                        if "labels" in roi_grp
                        else []
                    )
                    cond_a_contribs.append(ca)
                    cond_b_contribs.append(cb)
                    contrib_labels.append(lbl)

        # --- provenance ---
        if "provenance" in fh:
            prov = fh["provenance"]
            source_trial_stats_files = (
                decode_str_array(np.asarray(prov["source_trial_stats_files"][:], dtype=object))
                if "source_trial_stats_files" in prov
                else []
            )
            source_electrodes_files = (
                decode_str_array(np.asarray(prov["source_electrodes_files"][:], dtype=object))
                if "source_electrodes_files" in prov
                else []
            )
        else:
            source_trial_stats_files = []
            source_electrodes_files = []

        # --- cluster permutation results (optional) ---
        cluster_p_values: np.ndarray | None = None
        cluster_windows: list[tuple[float, float] | None] | None = None
        cluster_null_dists: list[np.ndarray] | None = None
        if "cluster_stats" in fh:
            cs_grp = fh["cluster_stats"]
            cp_ds = dataset_or_none(cs_grp, "p_values")
            if cp_ds is not None:
                cluster_p_values = np.asarray(cp_ds[:], dtype=np.float64)
                starts_ds = dataset_or_none(cs_grp, "best_cluster_start_s")
                ends_ds = dataset_or_none(cs_grp, "best_cluster_end_s")
                if starts_ds is not None and ends_ds is not None:
                    starts = np.asarray(starts_ds[:], dtype=np.float64)
                    ends = np.asarray(ends_ds[:], dtype=np.float64)
                    cluster_windows = [
                        (float(s), float(e)) if (np.isfinite(s) and np.isfinite(e)) else None
                        for s, e in zip(starts, ends)
                    ]
                null_ds = dataset_or_none(cs_grp, "null_distributions")
                if null_ds is not None:
                    null_matrix = np.asarray(null_ds[:], dtype=np.float32)
                    cluster_null_dists = [null_matrix[i] for i in range(null_matrix.shape[0])]

    source_group = BIDSFileGroup(primary=BIDSFile.from_path(path))
    return TrialStatsGroupProcessingResult(
        source_group=source_group,
        metadata={
            "source_metric": source_metric,
            "p_value_correction_method": p_value_correction_method,
            "significance_alpha": significance_alpha,
            "roi_mode": roi_mode,
        },
        t_values=t_values,
        p_values=p_values,
        p_values_uncorrected=p_values_uncorrected,
        significant_mask=significant_mask,
        metric_mean=metric_mean,
        metric_sem=metric_sem,
        time_axis_s=time_axis_s,
        region_names=region_names,
        source_metric=source_metric,
        condition_labels=condition_labels,
        p_value_correction_method=p_value_correction_method,
        significance_alpha=significance_alpha,
        roi_mode=roi_mode,
        atlas_name=atlas_name,
        epoch_mean_t_values=epoch_mean_t_values,
        epoch_mean_p_values=epoch_mean_p_values,
        epoch_mean_df=epoch_mean_df,
        epoch_mean_metric_mean=epoch_mean_metric_mean,
        epoch_mean_metric_sem=epoch_mean_metric_sem,
        roi_channel_counts=roi_channel_counts,
        roi_subject_counts=roi_subject_counts,
        contributions=contributions,
        condition_a_group_mean=condition_a_group_mean,
        condition_a_group_sem=condition_a_group_sem,
        condition_b_group_mean=condition_b_group_mean,
        condition_b_group_sem=condition_b_group_sem,
        condition_a_contributions=cond_a_contribs,
        condition_b_contributions=cond_b_contribs,
        contribution_labels=contrib_labels,
        source_trial_stats_files=source_trial_stats_files,
        source_electrodes_files=source_electrodes_files,
        excluded_rois=excluded_rois,
        cluster_p_values=cluster_p_values,
        cluster_best_cluster_windows_s=cluster_windows,
        cluster_null_distributions=cluster_null_dists,
    )


# ---------------------------------------------------------------------------
# MATLAB loader
# ---------------------------------------------------------------------------


def _load_from_matlab(path: Path) -> TrialStatsGroupProcessingResult:
    from gin_bids_py_analysis.processing.utils.matlab import (
        mat_float,
        mat_str,
        mat_str_list,
    )
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=False, struct_as_record=False)
    data = mat["data"]
    axes = data.axes
    stats = data.stats
    means = data.means
    uncertainty = getattr(data, "uncertainty", None)
    summary = getattr(data, "summary_epoch", None)
    meta = data.meta
    contribs_raw = getattr(data, "contributions", None)
    prov = getattr(data, "provenance", None)

    region_names = mat_str_list(getattr(axes, "region", None))
    if not region_names:
        raise ValueError(f"{path.name}: axes.region is required in .mat group stats file.")
    time_axis_s = np.asarray(axes.time_s, dtype=np.float64).ravel()
    n_rois = len(region_names)
    n_t = len(time_axis_s)
    _zeros = np.zeros((n_rois, n_t), dtype=np.float64)

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
    condition_a_group_mean = _mat_2d(means, "condition_a_mean")
    condition_a_group_sem = _mat_2d(means, "condition_a_sem")
    condition_b_group_mean = _mat_2d(means, "condition_b_mean")
    condition_b_group_sem = _mat_2d(means, "condition_b_sem")
    metric_sem = _mat_2d(uncertainty, "metric_sem")

    epoch_mean_t_values = _mat_1d(summary, "t_values")
    epoch_mean_p_values = _mat_1d(summary, "p_values", fill=1.0)
    epoch_mean_df = _mat_1d(summary, "df")
    epoch_mean_metric_mean = _mat_1d(summary, "metric_mean")
    epoch_mean_metric_sem = _mat_1d(summary, "metric_sem")
    roi_channel_counts = _mat_1d(summary, "roi_channel_counts", dtype=np.int64)
    roi_subject_counts = _mat_1d(summary, "roi_subject_counts", dtype=np.int64)

    source_metric = mat_str(getattr(meta, "source_metric", None), default="mean_difference")
    labels_raw = getattr(meta, "condition_labels", None)
    labels = mat_str_list(labels_raw)
    condition_labels: tuple[str, str] = (
        labels[0] if len(labels) >= 1 else "condition_a",
        labels[1] if len(labels) >= 2 else "condition_b",
    )
    p_value_correction_method = mat_str(
        getattr(meta, "p_value_correction_method", None), default="none"
    )
    roi_mode = mat_str(getattr(meta, "roi_mode", None), default="manual")
    atlas_name_raw = mat_str(getattr(meta, "atlas_name", None), default="")
    atlas_name: str | None = atlas_name_raw.strip() or None

    excluded_rois: dict[str, str] = {}
    ex_raw = getattr(meta, "excluded_rois", None)
    if ex_raw is not None:
        ex_regions = mat_str_list(getattr(ex_raw, "region", None))
        ex_reasons = mat_str_list(getattr(ex_raw, "reason", None))
        excluded_rois = dict(zip(ex_regions, ex_reasons))

    contributions: list[ROIChannelContribution] = []
    if contribs_raw is not None:
        reis = mat_str_list(getattr(contribs_raw, "region", None))
        subjs = mat_str_list(getattr(contribs_raw, "subject", None))
        chs = mat_str_list(getattr(contribs_raw, "channel", None))
        srcs = mat_str_list(getattr(contribs_raw, "source_stats_file", None))
        contributions = [
            ROIChannelContribution(roi=r, subject=s, channel=c, source_stats_file=f)
            for r, s, c, f in zip(reis, subjs, chs, srcs)
        ]

    source_trial_stats_files = (
        mat_str_list(getattr(prov, "source_trial_stats_files", None)) if prov is not None else []
    )
    source_electrodes_files = (
        mat_str_list(getattr(prov, "source_electrodes_files", None)) if prov is not None else []
    )

    source_group = BIDSFileGroup(primary=BIDSFile.from_path(path))
    return TrialStatsGroupProcessingResult(
        source_group=source_group,
        metadata={
            "source_metric": source_metric,
            "p_value_correction_method": p_value_correction_method,
            "significance_alpha": significance_alpha,
            "roi_mode": roi_mode,
        },
        t_values=t_values,
        p_values=p_values,
        p_values_uncorrected=p_values_uncorrected,
        significant_mask=significant_mask,
        metric_mean=metric_mean,
        metric_sem=metric_sem,
        time_axis_s=time_axis_s,
        region_names=region_names,
        source_metric=source_metric,
        condition_labels=condition_labels,
        p_value_correction_method=p_value_correction_method,
        significance_alpha=significance_alpha,
        roi_mode=roi_mode,
        atlas_name=atlas_name,
        epoch_mean_t_values=epoch_mean_t_values,
        epoch_mean_p_values=epoch_mean_p_values,
        epoch_mean_df=epoch_mean_df,
        epoch_mean_metric_mean=epoch_mean_metric_mean,
        epoch_mean_metric_sem=epoch_mean_metric_sem,
        roi_channel_counts=roi_channel_counts,
        roi_subject_counts=roi_subject_counts,
        contributions=contributions,
        condition_a_group_mean=condition_a_group_mean,
        condition_a_group_sem=condition_a_group_sem,
        condition_b_group_mean=condition_b_group_mean,
        condition_b_group_sem=condition_b_group_sem,
        condition_a_contributions=[],
        condition_b_contributions=[],
        contribution_labels=[],
        source_trial_stats_files=source_trial_stats_files,
        source_electrodes_files=source_electrodes_files,
        excluded_rois=excluded_rois,
        cluster_p_values=None,
        cluster_best_cluster_windows_s=None,
        cluster_null_distributions=None,
    )
