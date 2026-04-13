"""Load a pre-computed ``TrialSlopeStatsGroupProcessingResult`` from disk.

Supports the HDF5 (``.h5`` / ``.hdf5``) and MATLAB (``.mat``) formats written
by ``TrialSlopeStatsGroupProcessingWriter``.  The returned result can be fed directly
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

from .result import ROIChannelContribution, TrialSlopeStatsGroupProcessingResult


def load_trial_slope_stats_group_result(
    path: Path | str,
) -> TrialSlopeStatsGroupProcessingResult:
    """Load a pre-computed ``TrialSlopeStatsGroupProcessingResult`` from *path*.

    Parameters
    ----------
    path:
        Path to an ``.h5``/``.hdf5`` or ``.mat`` group slope-stats file written by
        ``TrialSlopeStatsGroupProcessingWriter``.

    Returns
    -------
    TrialSlopeStatsGroupProcessingResult
        A fully populated result object ready for visualization.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be interpreted as a valid group slope-stats output.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Group slope-stats file not found: {path}")
    ext = path.suffix.lower()
    if ext == ".mat":
        return _load_from_matlab(path)
    return _load_from_hdf5(path)


# ---------------------------------------------------------------------------
# HDF5 loader
# ---------------------------------------------------------------------------


def _load_from_hdf5(path: Path) -> TrialSlopeStatsGroupProcessingResult:
    with h5py.File(path, "r") as fh:
        # --- axes ---
        if "axes" not in fh or "region" not in fh["axes"]:
            raise ValueError(
                f"{path.name}: axes/region dataset is required for group slope-stats."
            )
        region_names = decode_str_array(np.asarray(fh["axes"]["region"][:]))
        time_axis_s = np.asarray(fh["axes"]["time_s"][:], dtype=np.float64)
        n_rois = len(region_names)
        n_t = len(time_axis_s)

        significance_alpha = float_scalar(
            dataset_or_none(fh, "meta/significance_alpha"), default=0.05
        )

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

        # --- regression (condition_a vs condition_b slope comparison) ---
        slope_t = _read_2d("regression/t_values")
        slope_p = _read_2d("regression/p_values", fill=1.0)
        slope_p_uncorr = _read_2d("regression/p_values_uncorrected", fill=1.0)
        slope_sig_ds = dataset_or_none(fh, "regression/significant_mask")
        if slope_sig_ds is not None:
            loaded = np.asarray(slope_sig_ds[:], dtype=bool)
            slope_sig = loaded if loaded.shape == (n_rois, n_t) else np.isfinite(slope_p) & (slope_p < significance_alpha)
        else:
            slope_sig = np.isfinite(slope_p) & (slope_p < significance_alpha)
        ca_slope_mean = _read_2d("regression/slope_mean_a")
        ca_slope_sem = _read_2d("regression/slope_sem_a")
        cb_slope_mean = _read_2d("regression/slope_mean_b")
        cb_slope_sem = _read_2d("regression/slope_sem_b")

        ep_slope_t = _read_1d("regression/epoch_summary/t")
        ep_slope_p = _read_1d("regression/epoch_summary/p", fill=1.0)
        ep_slope_df = _read_1d("regression/epoch_summary/df")

        # --- activity (condition_a vs condition_b activity comparison) ---
        activity_t = _read_2d("activity/t_values")
        activity_p = _read_2d("activity/p_values", fill=1.0)
        activity_p_uncorr = _read_2d("activity/p_values_uncorrected", fill=1.0)
        activity_sig_ds = dataset_or_none(fh, "activity/significant_mask")
        if activity_sig_ds is not None:
            loaded = np.asarray(activity_sig_ds[:], dtype=bool)
            activity_sig = loaded if loaded.shape == (n_rois, n_t) else np.isfinite(activity_p) & (activity_p < significance_alpha)
        else:
            activity_sig = np.isfinite(activity_p) & (activity_p < significance_alpha)

        ep_activity_t = _read_1d("activity/epoch_summary/t")
        ep_activity_p = _read_1d("activity/epoch_summary/p", fill=1.0)
        ep_activity_df = _read_1d("activity/epoch_summary/df")

        # --- means (activity) ---
        ca_act_mean = _read_2d("means/condition_a_mean")
        ca_act_sem = _read_2d("means/condition_a_sem")
        cb_act_mean = _read_2d("means/condition_b_mean")
        cb_act_sem = _read_2d("means/condition_b_sem")

        # --- r_values ---
        ca_r_mean = _read_2d("r_values/condition_a_mean")
        ca_r_sem = _read_2d("r_values/condition_a_sem")
        cb_r_mean = _read_2d("r_values/condition_b_mean")
        cb_r_sem = _read_2d("r_values/condition_b_sem")

        # --- roi counts ---
        roi_channel_counts = _read_1d_int("meta/roi_channel_counts")
        roi_subject_counts = _read_1d_int("meta/roi_subject_counts")

        # --- meta ---
        labels_ds = dataset_or_none(fh, "meta/condition_labels")
        if labels_ds is not None:
            labels = decode_str_array(np.asarray(labels_ds[:], dtype=object))
            condition_labels: tuple[str, str] = (
                labels[0] if len(labels) >= 1 else "condition_a",
                labels[1] if len(labels) >= 2 else "condition_b",
            )
        else:
            condition_labels = ("condition_a", "condition_b")
        source_metric = str_scalar(
            dataset_or_none(fh, "meta/source_metric"),
            default="slope",
        )
        contrast_mode = str_scalar(
            dataset_or_none(fh, "meta/contrast_mode"),
            default="paired",
        )
        p_value_correction_method = str_scalar(
            dataset_or_none(fh, "meta/p_value_correction_method"), default="none"
        )
        roi_mode = str_scalar(dataset_or_none(fh, "meta/roi_mode"), default="manual")
        atlas_name_raw = str_scalar(dataset_or_none(fh, "meta/atlas_name"), default="")
        atlas_name: str | None = atlas_name_raw.strip() or None

        metadata: dict = {
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
            "activity_zscore": _require_group_activity_zscore_hdf5(fh=fh, path=path),
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
            val = ds[()]
            if isinstance(val, (bytes, np.bytes_)):
                metadata[key] = val.decode("utf-8")
            elif isinstance(val, (np.floating, float)):
                metadata[key] = float(val)
            else:
                metadata[key] = int(val)

        # --- excluded ROIs ---
        ex_name_ds = dataset_or_none(fh, "excluded_rois/name")
        ex_reason_ds = dataset_or_none(fh, "excluded_rois/reason")
        if ex_name_ds is not None and ex_reason_ds is not None:
            ex_names = decode_str_array(np.asarray(ex_name_ds[:], dtype=object))
            ex_reasons = decode_str_array(np.asarray(ex_reason_ds[:], dtype=object))
            excluded_rois: dict[str, str] = dict(zip(ex_names, ex_reasons))
        else:
            excluded_rois = {}

        # --- contributions ---
        contributions: list[ROIChannelContribution] = []
        if "contributions" in fh:
            cg = fh["contributions"]
            if all(k in cg for k in ["roi", "subject", "channel", "source_stats_file"]):
                rois = decode_str_array(np.asarray(cg["roi"][:], dtype=object))
                subjs = decode_str_array(np.asarray(cg["subject"][:], dtype=object))
                chs = decode_str_array(np.asarray(cg["channel"][:], dtype=object))
                srcs = decode_str_array(np.asarray(cg["source_stats_file"][:], dtype=object))
                contributions = [
                    ROIChannelContribution(roi=r, subject=s, channel=c, source_stats_file=f)
                    for r, s, c, f in zip(rois, subjs, chs, srcs)
                ]

        # --- contribution samples (ragged, indexed by ROI index) ---
        slope_a_contribs: list[np.ndarray] = []
        slope_b_contribs: list[np.ndarray] = []
        activity_a_contribs: list[np.ndarray] = []
        activity_b_contribs: list[np.ndarray] = []
        contrib_labels: list[list[str]] = []
        if "contribution_samples" in fh:
            cs_grp = fh["contribution_samples"]
            for roi_idx in range(n_rois):
                roi_key = str(roi_idx)
                if roi_key in cs_grp:
                    roi_grp = cs_grp[roi_key]
                    sa = (
                        np.asarray(roi_grp["condition_a_slope"][:], dtype=np.float64)
                        if "condition_a_slope" in roi_grp
                        else np.empty((0, n_t), dtype=np.float64)
                    )
                    sb = (
                        np.asarray(roi_grp["condition_b_slope"][:], dtype=np.float64)
                        if "condition_b_slope" in roi_grp
                        else np.empty((0, n_t), dtype=np.float64)
                    )
                    aa = (
                        np.asarray(roi_grp["condition_a_activity"][:], dtype=np.float64)
                        if "condition_a_activity" in roi_grp
                        else np.empty((0, n_t), dtype=np.float64)
                    )
                    ab = (
                        np.asarray(roi_grp["condition_b_activity"][:], dtype=np.float64)
                        if "condition_b_activity" in roi_grp
                        else np.empty((0, n_t), dtype=np.float64)
                    )
                    lbl = (
                        decode_str_array(np.asarray(roi_grp["labels"][:], dtype=object))
                        if "labels" in roi_grp
                        else []
                    )
                    slope_a_contribs.append(sa)
                    slope_b_contribs.append(sb)
                    activity_a_contribs.append(aa)
                    activity_b_contribs.append(ab)
                    contrib_labels.append(lbl)

        # --- scatter data (per-ROI predictor vs epoch-mean-activity) ---
        scatter_pred_a: list[np.ndarray] = []
        scatter_act_a: list[np.ndarray] = []
        scatter_pred_b: list[np.ndarray] = []
        scatter_act_b: list[np.ndarray] = []
        if "scatter_data" in fh:
            sd_grp = fh["scatter_data"]
            for roi_idx in range(n_rois):
                roi_key = str(roi_idx)
                if roi_key in sd_grp:
                    rg = sd_grp[roi_key]
                    scatter_pred_a.append(
                        np.asarray(rg["condition_a_predictor"][:], dtype=np.float64)
                        if "condition_a_predictor" in rg
                        else np.empty(0, dtype=np.float64)
                    )
                    scatter_act_a.append(
                        np.asarray(rg["condition_a_activity"][:], dtype=np.float64)
                        if "condition_a_activity" in rg
                        else np.empty(0, dtype=np.float64)
                    )
                    scatter_pred_b.append(
                        np.asarray(rg["condition_b_predictor"][:], dtype=np.float64)
                        if "condition_b_predictor" in rg
                        else np.empty(0, dtype=np.float64)
                    )
                    scatter_act_b.append(
                        np.asarray(rg["condition_b_activity"][:], dtype=np.float64)
                        if "condition_b_activity" in rg
                        else np.empty(0, dtype=np.float64)
                    )

        # --- provenance ---
        source_trial_slope_stats_files: list[str] = []
        source_electrodes_files: list[str] = []
        if "provenance" in fh:
            prov = fh["provenance"]
            if "source_trial_slope_stats_files" in prov:
                source_trial_slope_stats_files = decode_str_array(
                    np.asarray(prov["source_trial_slope_stats_files"][:], dtype=object)
                )
            if "source_electrodes_files" in prov:
                source_electrodes_files = decode_str_array(
                    np.asarray(prov["source_electrodes_files"][:], dtype=object)
                )

    source_group = BIDSFileGroup(primary=BIDSFile.from_path(path))
    return TrialSlopeStatsGroupProcessingResult(
        source_group=source_group,
        metadata=metadata,
        slope_t_values=slope_t,
        slope_p_values=slope_p,
        slope_p_values_uncorrected=slope_p_uncorr,
        slope_significant_mask=slope_sig,
        activity_t_values=activity_t,
        activity_p_values=activity_p,
        activity_p_values_uncorrected=activity_p_uncorr,
        activity_significant_mask=activity_sig,
        condition_a_slope_mean=ca_slope_mean,
        condition_a_slope_sem=ca_slope_sem,
        condition_b_slope_mean=cb_slope_mean,
        condition_b_slope_sem=cb_slope_sem,
        epoch_slope_t=ep_slope_t,
        epoch_slope_p=ep_slope_p,
        epoch_slope_df=ep_slope_df,
        epoch_activity_t=ep_activity_t,
        epoch_activity_p=ep_activity_p,
        epoch_activity_df=ep_activity_df,
        condition_a_activity_mean=ca_act_mean,
        condition_a_activity_sem=ca_act_sem,
        condition_b_activity_mean=cb_act_mean,
        condition_b_activity_sem=cb_act_sem,
        condition_a_r_value_mean=ca_r_mean,
        condition_a_r_value_sem=ca_r_sem,
        condition_b_r_value_mean=cb_r_mean,
        condition_b_r_value_sem=cb_r_sem,
        time_axis_s=time_axis_s,
        region_names=region_names,
        condition_labels=condition_labels,
        source_metric=source_metric,
        contrast_mode=contrast_mode,
        roi_channel_counts=roi_channel_counts,
        roi_subject_counts=roi_subject_counts,
        contributions=contributions,
        condition_a_slope_contributions=slope_a_contribs,
        condition_b_slope_contributions=slope_b_contribs,
        condition_a_activity_contributions=activity_a_contribs,
        condition_b_activity_contributions=activity_b_contribs,
        contribution_labels=contrib_labels,
        p_value_correction_method=p_value_correction_method,
        significance_alpha=significance_alpha,
        roi_mode=roi_mode,
        atlas_name=atlas_name,
        source_trial_slope_stats_files=source_trial_slope_stats_files,
        source_electrodes_files=source_electrodes_files,
        excluded_rois=excluded_rois,
        condition_a_scatter_predictor=scatter_pred_a,
        condition_a_scatter_activity=scatter_act_a,
        condition_b_scatter_predictor=scatter_pred_b,
        condition_b_scatter_activity=scatter_act_b,
    )


# ---------------------------------------------------------------------------
# MATLAB loader
# ---------------------------------------------------------------------------


def _load_from_matlab(path: Path) -> TrialSlopeStatsGroupProcessingResult:
    from gin_bids_py_analysis.processing.utils.matlab import (
        mat_float,
        mat_str,
        mat_str_list,
    )
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=False, struct_as_record=False)
    data = mat["data"]
    axes = data.axes
    regression = data.regression
    activity_raw = getattr(data, "activity", None)
    means = data.means
    r_values = getattr(data, "r_values", None)
    meta = data.meta
    contribs_raw = getattr(data, "contributions", None)
    prov = getattr(data, "provenance", None)

    region_names = mat_str_list(getattr(axes, "region", None))
    if not region_names:
        raise ValueError(f"{path.name}: axes.region is required in .mat group slope-stats file.")
    time_axis_s = np.asarray(axes.time_s, dtype=np.float64).ravel()
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

    significance_alpha = mat_float(getattr(meta, "significance_alpha", None), default=0.05)

    # regression (condition_a vs condition_b slope comparison)
    ep_reg = getattr(regression, "epoch_summary", None) if regression is not None else None
    slope_t = _mat_2d(regression, "t_values")
    slope_p = _mat_2d(regression, "p_values", fill=1.0)
    slope_p_uncorr = _mat_2d(regression, "p_values_uncorrected", fill=1.0)
    sig_raw = getattr(regression, "significant_mask", None) if regression is not None else None
    slope_sig = (
        np.asarray(sig_raw, dtype=bool).reshape(n_rois, n_t)
        if sig_raw is not None
        else np.isfinite(slope_p) & (slope_p < significance_alpha)
    )
    ca_slope_mean = _mat_2d(regression, "slope_mean_a")
    ca_slope_sem = _mat_2d(regression, "slope_sem_a")
    cb_slope_mean = _mat_2d(regression, "slope_mean_b")
    cb_slope_sem = _mat_2d(regression, "slope_sem_b")
    ep_slope_t = _mat_1d(ep_reg, "t")
    ep_slope_p = _mat_1d(ep_reg, "p", fill=1.0)
    ep_slope_df = _mat_1d(ep_reg, "df")

    # activity (condition_a vs condition_b activity comparison)
    ep_act = getattr(activity_raw, "epoch_summary", None) if activity_raw is not None else None
    activity_t = _mat_2d(activity_raw, "t_values")
    activity_p = _mat_2d(activity_raw, "p_values", fill=1.0)
    activity_p_uncorr = _mat_2d(activity_raw, "p_values_uncorrected", fill=1.0)
    sig_raw_act = getattr(activity_raw, "significant_mask", None) if activity_raw is not None else None
    activity_sig = (
        np.asarray(sig_raw_act, dtype=bool).reshape(n_rois, n_t)
        if sig_raw_act is not None
        else np.isfinite(activity_p) & (activity_p < significance_alpha)
    )
    ep_activity_t = _mat_1d(ep_act, "t")
    ep_activity_p = _mat_1d(ep_act, "p", fill=1.0)
    ep_activity_df = _mat_1d(ep_act, "df")

    # means
    ca_act_mean = _mat_2d(means, "condition_a_mean")
    ca_act_sem = _mat_2d(means, "condition_a_sem")
    cb_act_mean = _mat_2d(means, "condition_b_mean")
    cb_act_sem = _mat_2d(means, "condition_b_sem")

    # r_values
    ca_r_mean = _mat_2d(r_values, "condition_a_mean")
    ca_r_sem = _mat_2d(r_values, "condition_a_sem")
    cb_r_mean = _mat_2d(r_values, "condition_b_mean")
    cb_r_sem = _mat_2d(r_values, "condition_b_sem")

    # roi counts
    roi_channel_counts = _mat_1d_int(meta, "roi_channel_counts")
    roi_subject_counts = _mat_1d_int(meta, "roi_subject_counts")

    # meta
    labels_raw = getattr(meta, "condition_labels", None)
    if labels_raw is not None:
        labels = mat_str_list(labels_raw)
        condition_labels: tuple[str, str] = (
            labels[0] if len(labels) >= 1 else "condition_a",
            labels[1] if len(labels) >= 2 else "condition_b",
        )
    else:
        condition_labels = ("condition_a", "condition_b")
    source_metric = mat_str(
        getattr(meta, "source_metric", None),
        default="slope",
    )
    contrast_mode = mat_str(
        getattr(meta, "contrast_mode", None),
        default="paired",
    )
    p_value_correction_method = mat_str(
        getattr(meta, "p_value_correction_method", None), default="none"
    )
    roi_mode = mat_str(getattr(meta, "roi_mode", None), default="manual")
    atlas_name_raw = mat_str(getattr(meta, "atlas_name", None), default="")
    atlas_name: str | None = atlas_name_raw.strip() or None

    metadata: dict = {
        "p_value_correction_method": p_value_correction_method,
        "significance_alpha": significance_alpha,
        "source_metric": source_metric,
        "contrast_mode": contrast_mode,
        "roi_mode": roi_mode,
        "atlas_name": atlas_name,
        "predictor": mat_str(getattr(meta, "predictor", None), default=""),
        "predictor_zscore": mat_str(getattr(meta, "predictor_zscore", None), default="none"),
        "predictor_transform_by_condition_json": mat_str(
            getattr(meta, "predictor_transform_by_condition_json", None),
            default="{}",
        ),
        "activity_zscore": _require_group_activity_zscore_mat(meta, path=path),
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

    # excluded ROIs
    excl_raw = getattr(meta, "excluded_rois", None)
    if excl_raw is not None:
        ex_names = mat_str_list(getattr(excl_raw, "name", None))
        ex_reasons = mat_str_list(getattr(excl_raw, "reason", None))
        excluded_rois: dict[str, str] = dict(zip(ex_names, ex_reasons))
    else:
        excluded_rois = {}

    # contributions
    contributions: list[ROIChannelContribution] = []
    if contribs_raw is not None:
        roi_raw = getattr(contribs_raw, "roi", None)
        subj_raw = getattr(contribs_raw, "subject", None)
        ch_raw = getattr(contribs_raw, "channel", None)
        src_raw = getattr(contribs_raw, "source_stats_file", None)
        if all(v is not None for v in [roi_raw, subj_raw, ch_raw, src_raw]):
            rois = mat_str_list(roi_raw)
            subjs = mat_str_list(subj_raw)
            chs = mat_str_list(ch_raw)
            srcs = mat_str_list(src_raw)
            contributions = [
                ROIChannelContribution(roi=r, subject=s, channel=c, source_stats_file=f)
                for r, s, c, f in zip(rois, subjs, chs, srcs)
            ]

    # contribution samples (cell arrays)
    slope_a_contribs: list[np.ndarray] = []
    slope_b_contribs: list[np.ndarray] = []
    activity_a_contribs: list[np.ndarray] = []
    activity_b_contribs: list[np.ndarray] = []
    contrib_labels: list[list[str]] = []
    cs_raw = getattr(data, "contribution_samples", None)
    if cs_raw is not None:
        sa_raw = getattr(cs_raw, "condition_a_slope", None)
        sb_raw = getattr(cs_raw, "condition_b_slope", None)
        aa_raw = getattr(cs_raw, "condition_a_activity", None)
        ab_raw = getattr(cs_raw, "condition_b_activity", None)
        lbl_raw = getattr(cs_raw, "labels", None)
        if sa_raw is not None:
            cell: np.ndarray = np.asarray(sa_raw).ravel()
            for i in range(min(n_rois, len(cell))):
                slope_a_contribs.append(np.asarray(cell[i], dtype=np.float64))
        if sb_raw is not None:
            cell = np.asarray(sb_raw).ravel()
            for i in range(min(n_rois, len(cell))):
                slope_b_contribs.append(np.asarray(cell[i], dtype=np.float64))
        if aa_raw is not None:
            cell = np.asarray(aa_raw).ravel()
            for i in range(min(n_rois, len(cell))):
                activity_a_contribs.append(np.asarray(cell[i], dtype=np.float64))
        if ab_raw is not None:
            cell = np.asarray(ab_raw).ravel()
            for i in range(min(n_rois, len(cell))):
                activity_b_contribs.append(np.asarray(cell[i], dtype=np.float64))
        if lbl_raw is not None:
            cell = np.asarray(lbl_raw).ravel()
            for i in range(min(n_rois, len(cell))):
                contrib_labels.append(mat_str_list(cell[i]))

    # scatter data (per-ROI scatter from condition_a/b predictor vs epoch means)
    scatter_pred_a_m: list[np.ndarray] = []
    scatter_act_a_m: list[np.ndarray] = []
    scatter_pred_b_m: list[np.ndarray] = []
    scatter_act_b_m: list[np.ndarray] = []
    scatter_raw = getattr(data, "scatter_data", None)
    if scatter_raw is not None:
        cap_raw = getattr(scatter_raw, "condition_a_predictor", None)
        caa_raw = getattr(scatter_raw, "condition_a_activity", None)
        cbp_raw = getattr(scatter_raw, "condition_b_predictor", None)
        cba_raw = getattr(scatter_raw, "condition_b_activity", None)
        if cap_raw is not None:
            cell: np.ndarray = np.asarray(cap_raw).ravel()
            for i in range(min(n_rois, len(cell))):
                scatter_pred_a_m.append(np.asarray(cell[i], dtype=np.float64).ravel())
        if caa_raw is not None:
            cell = np.asarray(caa_raw).ravel()
            for i in range(min(n_rois, len(cell))):
                scatter_act_a_m.append(np.asarray(cell[i], dtype=np.float64).ravel())
        if cbp_raw is not None:
            cell = np.asarray(cbp_raw).ravel()
            for i in range(min(n_rois, len(cell))):
                scatter_pred_b_m.append(np.asarray(cell[i], dtype=np.float64).ravel())
        if cba_raw is not None:
            cell = np.asarray(cba_raw).ravel()
            for i in range(min(n_rois, len(cell))):
                scatter_act_b_m.append(np.asarray(cell[i], dtype=np.float64).ravel())

    # provenance
    source_trial_slope_stats_files: list[str] = []
    source_electrodes_files: list[str] = []
    if prov is not None:
        src_raw_prov = getattr(prov, "source_trial_slope_stats_files", None)
        if src_raw_prov is not None:
            source_trial_slope_stats_files = mat_str_list(src_raw_prov)
        elec_raw = getattr(prov, "source_electrodes_files", None)
        if elec_raw is not None:
            source_electrodes_files = mat_str_list(elec_raw)

    source_group = BIDSFileGroup(primary=BIDSFile.from_path(path))
    return TrialSlopeStatsGroupProcessingResult(
        source_group=source_group,
        metadata=metadata,
        slope_t_values=slope_t,
        slope_p_values=slope_p,
        slope_p_values_uncorrected=slope_p_uncorr,
        slope_significant_mask=slope_sig,
        activity_t_values=activity_t,
        activity_p_values=activity_p,
        activity_p_values_uncorrected=activity_p_uncorr,
        activity_significant_mask=activity_sig,
        condition_a_slope_mean=ca_slope_mean,
        condition_a_slope_sem=ca_slope_sem,
        condition_b_slope_mean=cb_slope_mean,
        condition_b_slope_sem=cb_slope_sem,
        epoch_slope_t=ep_slope_t,
        epoch_slope_p=ep_slope_p,
        epoch_slope_df=ep_slope_df,
        epoch_activity_t=ep_activity_t,
        epoch_activity_p=ep_activity_p,
        epoch_activity_df=ep_activity_df,
        condition_a_activity_mean=ca_act_mean,
        condition_a_activity_sem=ca_act_sem,
        condition_b_activity_mean=cb_act_mean,
        condition_b_activity_sem=cb_act_sem,
        condition_a_r_value_mean=ca_r_mean,
        condition_a_r_value_sem=ca_r_sem,
        condition_b_r_value_mean=cb_r_mean,
        condition_b_r_value_sem=cb_r_sem,
        time_axis_s=time_axis_s,
        region_names=region_names,
        condition_labels=condition_labels,
        source_metric=source_metric,
        contrast_mode=contrast_mode,
        roi_channel_counts=roi_channel_counts,
        roi_subject_counts=roi_subject_counts,
        contributions=contributions,
        condition_a_slope_contributions=slope_a_contribs,
        condition_b_slope_contributions=slope_b_contribs,
        condition_a_activity_contributions=activity_a_contribs,
        condition_b_activity_contributions=activity_b_contribs,
        contribution_labels=contrib_labels,
        p_value_correction_method=p_value_correction_method,
        significance_alpha=significance_alpha,
        roi_mode=roi_mode,
        atlas_name=atlas_name,
        source_trial_slope_stats_files=source_trial_slope_stats_files,
        source_electrodes_files=source_electrodes_files,
        excluded_rois=excluded_rois,
        condition_a_scatter_predictor=scatter_pred_a_m,
        condition_a_scatter_activity=scatter_act_a_m,
        condition_b_scatter_predictor=scatter_pred_b_m,
        condition_b_scatter_activity=scatter_act_b_m,
    )


def _require_group_activity_zscore_hdf5(*, fh: h5py.File, path: Path) -> str:
    ds = dataset_or_none(fh, "meta/activity_zscore")
    if ds is None:
        raise ValueError(
            f"{path.name}: unsupported legacy trial_slope_stats_group schema; "
            "meta/activity_zscore is required."
        )
    return str_scalar(ds, default="none")


def _require_group_activity_zscore_mat(meta: object, path: Path) -> str:
    raw = getattr(meta, "activity_zscore", None)
    if raw is None:
        raise ValueError(
            f"{path.name}: unsupported legacy trial_slope_stats_group schema; "
            "meta.activity_zscore is required."
        )
    from gin_bids_py_analysis.processing.utils.matlab import mat_str

    return mat_str(raw, default="none")
