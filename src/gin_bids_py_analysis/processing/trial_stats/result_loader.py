"""Load a pre-computed ``TrialStatsProcessingResult`` from disk.

Supports the HDF5 (``.h5`` / ``.hdf5``) and MATLAB (``.mat``) formats written
by ``TrialStatsProcessingWriter``.  The returned result object contains the
same fields as one produced by ``TrialStatsProcessing.process_group()`` and can
be fed directly to the visualization layer without re-running the processing
pipeline.
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

from .result import TrialStatsProcessingResult


def load_trial_stats_result(path: Path | str) -> TrialStatsProcessingResult:
    """Load a pre-computed ``TrialStatsProcessingResult`` from *path*.

    Parameters
    ----------
    path:
        Path to an ``.h5``/``.hdf5`` or ``.mat`` trial-stats file written by
        ``TrialStatsProcessingWriter``.

    Returns
    -------
    TrialStatsProcessingResult
        A fully populated result object ready for visualization.  Fields that
        are optional in the file (e.g. epochs, permuted t-values) are set to
        empty arrays / ``None`` when absent.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be interpreted as a valid trial-stats output.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trial stats file not found: {path}")
    ext = path.suffix.lower()
    if ext == ".mat":
        return _load_from_matlab(path)
    return _load_from_hdf5(path)


# ---------------------------------------------------------------------------
# HDF5 loader
# ---------------------------------------------------------------------------


def _load_from_hdf5(path: Path) -> TrialStatsProcessingResult:
    with h5py.File(path, "r") as fh:
        # --- axes (channel / region names + time) ---
        analysis_level = str_scalar(
            dataset_or_none(fh, "meta/analysis_level"), default="channel"
        )
        axis_name = "region" if analysis_level == "roi" else "channel"
        if "axes" not in fh or axis_name not in fh["axes"]:
            raise ValueError(
                f"{path.name}: axes/{axis_name} dataset is required."
            )
        channel_names = decode_str_array(np.asarray(fh["axes"][axis_name][:]))
        time_axis_s = np.asarray(fh["axes"]["time_s"][:], dtype=np.float64)
        n_ch = len(channel_names)
        n_t = len(time_axis_s)
        _zeros = np.zeros((n_ch, n_t), dtype=np.float64)

        # --- condition labels ---
        labels_ds = dataset_or_none(fh, "meta/trial_count_labels")
        if labels_ds is not None:
            labels = decode_str_array(np.asarray(labels_ds[:], dtype=object))
            condition_a = labels[0] if len(labels) >= 1 else "condition_a"
            condition_b = labels[1] if len(labels) >= 2 else "condition_b"
        elif "means" in fh:
            mean_keys = [k for k in fh["means"].keys() if k != "difference"]
            condition_a = mean_keys[0] if len(mean_keys) >= 1 else "condition_a"
            condition_b = mean_keys[1] if len(mean_keys) >= 2 else "condition_b"
        else:
            condition_a, condition_b = "condition_a", "condition_b"

        # --- trial counts ---
        trial_counts_ds = dataset_or_none(fh, "meta/trial_counts")
        if trial_counts_ds is not None:
            tc = np.asarray(trial_counts_ds[:], dtype=np.int64).ravel()
            condition_a_trial_count = int(tc[0]) if len(tc) >= 1 else 0
            condition_b_trial_count = int(tc[1]) if len(tc) >= 2 else 0
        else:
            condition_a_trial_count = 0
            condition_b_trial_count = 0

        # --- statistical arrays ---
        def _read_stats(key: str, fill: float = 0.0) -> np.ndarray:
            ds = dataset_or_none(fh, key)
            if ds is None:
                return np.full((n_ch, n_t), fill, dtype=np.float64)
            return np.asarray(ds[:], dtype=np.float64)

        t_values = _read_stats("stats/t_values")
        p_values = _read_stats("stats/p_values", fill=1.0)
        p_values_uncorrected = _read_stats("stats/p_values_uncorrected", fill=1.0)
        sig_ds = dataset_or_none(fh, "stats/significant_mask")
        significance_alpha = float_scalar(
            dataset_or_none(fh, "meta/significance_alpha"), default=0.05
        )
        if sig_ds is not None:
            _loaded_mask = np.asarray(sig_ds[:], dtype=bool)
            if _loaded_mask.ndim == 2 and _loaded_mask.shape == (n_ch, n_t):
                significant_mask = _loaded_mask
            else:
                significant_mask = np.isfinite(p_values) & (
                    p_values < significance_alpha
                )
        else:
            significant_mask = np.isfinite(p_values) & (
                p_values < significance_alpha
            )
        perm_ds = dataset_or_none(fh, "stats/permuted_t_values")
        permuted_t_values: np.ndarray | None = (
            np.asarray(perm_ds[:], dtype=np.float32)
            if perm_ds is not None
            else None
        )
        ch_sig_ds = dataset_or_none(fh, "stats/channel_significant_mask")
        channel_significant_mask: np.ndarray | None = (
            np.asarray(ch_sig_ds[:], dtype=bool)
            if ch_sig_ds is not None
            else None
        )

        # --- means ---
        def _read_arr(key: str) -> np.ndarray:
            ds = dataset_or_none(fh, key)
            return np.asarray(ds[:], dtype=np.float64) if ds is not None else _zeros.copy()

        condition_a_mean = _read_arr(f"means/{condition_a}")
        condition_b_mean = _read_arr(f"means/{condition_b}")
        mean_difference = _read_arr("means/difference")

        # --- uncertainty ---
        condition_a_sem = _read_arr(f"uncertainty/{condition_a}_sem")
        condition_b_sem = _read_arr(f"uncertainty/{condition_b}_sem")
        difference_sem = _read_arr("uncertainty/difference_sem")
        difference_ci95_low = _read_arr("uncertainty/difference_ci95_low")
        difference_ci95_high = _read_arr("uncertainty/difference_ci95_high")

        # --- meta ---
        sfreq = float_scalar(
            dataset_or_none(fh, "meta/sampling_frequency_hz"), default=0.0
        )
        p_value_correction_method = str_scalar(
            dataset_or_none(fh, "meta/p_value_correction_method"), default="fdr_bh"
        )
        stats_valid_ds = dataset_or_none(fh, "meta/stats_valid")
        if stats_valid_ds is not None:
            stats_valid = bool(stats_valid_ds[()])
        else:
            stats_valid = bool(np.any(np.isfinite(t_values) & (t_values != 0)))
        atlas_name_raw = str_scalar(
            dataset_or_none(fh, "meta/atlas_name"), default=""
        )
        atlas_name: str | None = atlas_name_raw.strip() or None
        window_ms = float_scalar(dataset_or_none(fh, "meta/window_ms"), default=0.0)
        n_bins = int_scalar(dataset_or_none(fh, "meta/n_bins"), default=0)
        activity_zscore_ds = dataset_or_none(fh, "meta/activity_zscore")
        if activity_zscore_ds is None:
            raise ValueError(
                f"{path.name}: unsupported legacy trial_stats schema; "
                "meta/activity_zscore is required."
            )
        activity_zscore = str_scalar(activity_zscore_ds, default="none")
        activity_baseline_tmin_s = float_scalar(
            dataset_or_none(fh, "meta/activity_baseline_tmin_s"),
            default=-0.2,
        )
        activity_baseline_tmax_s = float_scalar(
            dataset_or_none(fh, "meta/activity_baseline_tmax_s"),
            default=0.0,
        )
        n_perms = int_scalar(dataset_or_none(fh, "meta/n_permutations"), default=0)

        # --- epochs (optional, present only when include_epochs=True) ---
        if "epochs" in fh:
            eg = fh["epochs"]
            condition_a_epochs = (
                np.asarray(eg["condition_a"][:], dtype=np.float64)
                if "condition_a" in eg
                else np.array([])
            )
            condition_b_epochs = (
                np.asarray(eg["condition_b"][:], dtype=np.float64)
                if "condition_b" in eg
                else np.array([])
            )
        else:
            condition_a_epochs = np.array([])
            condition_b_epochs = np.array([])

        # --- provenance (optional block, only written when include_epochs=True) ---
        if "provenance" in fh:
            prov = fh["provenance"]
            source_ieeg_files = (
                decode_str_array(
                    np.asarray(prov["source_ieeg_files"][:], dtype=object)
                )
                if "source_ieeg_files" in prov
                else []
            )
            source_electrodes_files = (
                decode_str_array(
                    np.asarray(prov["source_electrodes_files"][:], dtype=object)
                )
                if "source_electrodes_files" in prov
                else []
            )
        else:
            source_ieeg_files = []
            source_electrodes_files = []

    source_group = BIDSFileGroup(primary=BIDSFile.from_path(path))
    return TrialStatsProcessingResult(
        source_group=source_group,
        metadata={
            "activity_zscore": activity_zscore,
            "activity_baseline_tmin_s": activity_baseline_tmin_s,
            "activity_baseline_tmax_s": activity_baseline_tmax_s,
            "n_permutations": (
                permuted_t_values.shape[0]
                if permuted_t_values is not None
                else n_perms
            ),
        },
        t_values=t_values,
        p_values=p_values,
        p_values_uncorrected=p_values_uncorrected,
        significant_mask=significant_mask,
        condition_a_mean=condition_a_mean,
        condition_b_mean=condition_b_mean,
        mean_difference=mean_difference,
        condition_a_sem=condition_a_sem,
        condition_b_sem=condition_b_sem,
        difference_sem=difference_sem,
        difference_ci95_low=difference_ci95_low,
        difference_ci95_high=difference_ci95_high,
        time_axis_s=time_axis_s,
        channel_names=channel_names,
        condition_a=condition_a,
        condition_b=condition_b,
        condition_a_trial_count=condition_a_trial_count,
        condition_b_trial_count=condition_b_trial_count,
        sfreq=sfreq,
        analysis_level=analysis_level,
        atlas_name=atlas_name,
        window_ms=window_ms,
        n_bins=n_bins,
        activity_zscore=activity_zscore,
        activity_baseline_tmin_s=activity_baseline_tmin_s,
        activity_baseline_tmax_s=activity_baseline_tmax_s,
        p_value_correction_method=p_value_correction_method,
        significance_alpha=significance_alpha,
        stats_valid=stats_valid,
        condition_a_epochs=condition_a_epochs,
        condition_b_epochs=condition_b_epochs,
        permuted_t_values=permuted_t_values,
        channel_significant_mask=channel_significant_mask,
        source_ieeg_files=source_ieeg_files,
        source_electrodes_files=source_electrodes_files,
    )


# ---------------------------------------------------------------------------
# MATLAB loader
# ---------------------------------------------------------------------------


def _load_from_matlab(path: Path) -> TrialStatsProcessingResult:
    from gin_bids_py_analysis.processing.utils.matlab import (
        mat_float,
        mat_int,
        mat_str,
        mat_str_list,
        matlab_safe_name,
    )
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=False, struct_as_record=False)
    data = mat["data"]
    meta = data.meta
    axes = data.axes
    stats = data.stats
    means = data.means
    uncertainty = getattr(data, "uncertainty", None)
    prov = getattr(data, "provenance", None)

    analysis_level = mat_str(getattr(meta, "analysis_level", None), default="channel")
    axis_attr = "region" if analysis_level == "roi" else "channel"
    channel_names = mat_str_list(getattr(axes, axis_attr, None))
    if not channel_names:
        raise ValueError(
            f"{path.name}: axes.{axis_attr} is required in .mat trial-stats file."
        )
    time_axis_s = np.asarray(axes.time_s, dtype=np.float64).ravel()
    n_ch = len(channel_names)
    n_t = len(time_axis_s)
    _zeros = np.zeros((n_ch, n_t), dtype=np.float64)

    # --- condition labels ---
    labels_raw = getattr(meta, "trial_count_labels", None)
    if labels_raw is not None:
        labels = mat_str_list(labels_raw)
        condition_a = labels[0] if len(labels) >= 1 else "condition_a"
        condition_b = labels[1] if len(labels) >= 2 else "condition_b"
    else:
        condition_a, condition_b = "condition_a", "condition_b"

    # --- trial counts ---
    trial_counts_raw = getattr(meta, "trial_counts", None)
    if trial_counts_raw is not None:
        tc = np.asarray(trial_counts_raw, dtype=np.int64).ravel()
        condition_a_trial_count = int(tc[0]) if len(tc) >= 1 else 0
        condition_b_trial_count = int(tc[1]) if len(tc) >= 2 else 0
    else:
        condition_a_trial_count = 0
        condition_b_trial_count = 0

    # --- stats ---
    def _mat_arr(obj: object, attr: str) -> np.ndarray:
        raw = getattr(obj, attr, None) if obj is not None else None
        if raw is None:
            return _zeros.copy()
        return np.asarray(raw, dtype=np.float64).reshape(n_ch, n_t)

    t_values = _mat_arr(stats, "t_values")
    p_values = _mat_arr(stats, "p_values")
    p_values_uc_raw = getattr(stats, "p_values_uncorrected", None)
    p_values_uncorrected = (
        np.asarray(p_values_uc_raw, dtype=np.float64).reshape(n_ch, n_t)
        if p_values_uc_raw is not None
        else p_values.copy()
    )
    sig_raw = getattr(stats, "significant_mask", None)
    significance_alpha = mat_float(
        getattr(meta, "significance_alpha", None), default=0.05
    )
    if sig_raw is not None:
        significant_mask = np.asarray(sig_raw, dtype=bool).reshape(n_ch, n_t)
    else:
        significant_mask = np.isfinite(p_values) & (p_values < significance_alpha)

    # --- means ---
    safe_a = matlab_safe_name(condition_a)
    safe_b = matlab_safe_name(condition_b)
    condition_a_mean = _mat_arr(means, safe_a)
    condition_b_mean = _mat_arr(means, safe_b)
    mean_difference = _mat_arr(means, "difference")

    # --- uncertainty ---
    condition_a_sem = _mat_arr(uncertainty, f"{safe_a}_sem")
    condition_b_sem = _mat_arr(uncertainty, f"{safe_b}_sem")
    difference_sem = _mat_arr(uncertainty, "difference_sem")
    difference_ci95_low = _mat_arr(uncertainty, "difference_ci95_low")
    difference_ci95_high = _mat_arr(uncertainty, "difference_ci95_high")

    # --- meta ---
    sfreq = mat_float(getattr(meta, "sampling_frequency_hz", None), default=0.0)
    p_value_correction_method = mat_str(
        getattr(meta, "p_value_correction_method", None), default="fdr_bh"
    )
    stats_valid_raw = getattr(meta, "stats_valid", None)
    if stats_valid_raw is not None:
        stats_valid = bool(np.asarray(stats_valid_raw).ravel()[0])
    else:
        stats_valid = bool(np.any(np.isfinite(t_values) & (t_values != 0)))
    atlas_name_raw = mat_str(getattr(meta, "atlas_name", None), default="")
    atlas_name: str | None = atlas_name_raw.strip() or None
    window_ms = mat_float(getattr(meta, "window_ms", None), default=0.0)
    n_bins = mat_int(getattr(meta, "n_bins", None), default=0)
    activity_zscore_raw = getattr(meta, "activity_zscore", None)
    if activity_zscore_raw is None:
        raise ValueError(
            f"{path.name}: unsupported legacy trial_stats schema; "
            "meta.activity_zscore is required."
        )
    activity_zscore = mat_str(activity_zscore_raw, default="none")
    activity_baseline_tmin_s = mat_float(
        getattr(meta, "activity_baseline_tmin_s", None),
        default=-0.2,
    )
    activity_baseline_tmax_s = mat_float(
        getattr(meta, "activity_baseline_tmax_s", None),
        default=0.0,
    )

    # --- provenance ---
    source_ieeg_files = (
        mat_str_list(getattr(prov, "source_ieeg_files", None)) if prov is not None else []
    )
    source_electrodes_files = (
        mat_str_list(getattr(prov, "source_electrodes_files", None))
        if prov is not None
        else []
    )

    source_group = BIDSFileGroup(primary=BIDSFile.from_path(path))
    return TrialStatsProcessingResult(
        source_group=source_group,
        metadata={
            "activity_zscore": activity_zscore,
            "activity_baseline_tmin_s": activity_baseline_tmin_s,
            "activity_baseline_tmax_s": activity_baseline_tmax_s,
        },
        t_values=t_values,
        p_values=p_values,
        p_values_uncorrected=p_values_uncorrected,
        significant_mask=significant_mask,
        condition_a_mean=condition_a_mean,
        condition_b_mean=condition_b_mean,
        mean_difference=mean_difference,
        condition_a_sem=condition_a_sem,
        condition_b_sem=condition_b_sem,
        difference_sem=difference_sem,
        difference_ci95_low=difference_ci95_low,
        difference_ci95_high=difference_ci95_high,
        time_axis_s=time_axis_s,
        channel_names=channel_names,
        condition_a=condition_a,
        condition_b=condition_b,
        condition_a_trial_count=condition_a_trial_count,
        condition_b_trial_count=condition_b_trial_count,
        sfreq=sfreq,
        analysis_level=analysis_level,
        atlas_name=atlas_name,
        window_ms=window_ms,
        n_bins=n_bins,
        activity_zscore=activity_zscore,
        activity_baseline_tmin_s=activity_baseline_tmin_s,
        activity_baseline_tmax_s=activity_baseline_tmax_s,
        p_value_correction_method=p_value_correction_method,
        significance_alpha=significance_alpha,
        stats_valid=stats_valid,
        condition_a_epochs=np.array([]),
        condition_b_epochs=np.array([]),
        permuted_t_values=None,
        source_ieeg_files=source_ieeg_files,
        source_electrodes_files=source_electrodes_files,
    )
