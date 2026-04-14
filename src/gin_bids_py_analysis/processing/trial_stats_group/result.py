from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gin_bids_py_analysis.processing.base import BaseProcessingResult
from gin_bids_py_analysis.processing.utils.group_stats import ROIChannelContribution

__all__ = ["BaseTrialStatsGroupProcessingResult", "ROIChannelContribution"]


@dataclass
class BaseTrialStatsGroupProcessingResult(BaseProcessingResult):
    """Shared result surface for group-level trial-statistics pipelines."""

    # --- Axes ---
    time_axis_s: np.ndarray = field(default_factory=lambda: np.array([]))
    """Time axis for the epoch window, shape (n_times,), in seconds."""
    region_names: list[str] = field(default_factory=list)
    """Ordered list of ROI names corresponding to the first axis of all 2-D arrays."""
    condition_labels: tuple[str, str] = ("condition_a", "condition_b")
    """Human-readable labels for condition A and condition B."""

    # --- ROI composition ---
    roi_channel_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    """Number of contributing channels per ROI, shape (n_rois,), dtype int64."""
    roi_subject_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    """Number of contributing subjects per ROI, shape (n_rois,), dtype int64."""
    contributions: list[ROIChannelContribution] = field(default_factory=list)
    """Flat list of every (roi, subject, channel) triple that contributed to the group stats."""

    # --- Activity means per condition ---
    condition_a_activity_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    """Mean broadband activity for condition A, shape (n_rois, n_times)."""
    condition_a_activity_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    """SEM of broadband activity for condition A, shape (n_rois, n_times)."""
    condition_b_activity_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    """Mean broadband activity for condition B, shape (n_rois, n_times)."""
    condition_b_activity_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    """SEM of broadband activity for condition B, shape (n_rois, n_times)."""

    # --- Per-channel activity time-courses used to build group statistics ---
    condition_a_activity_contributions: list = field(default_factory=list)
    """Per-ROI list of stacked per-channel activity arrays for condition A.
    Element i has shape (n_channels_in_roi_i, n_times)."""
    condition_b_activity_contributions: list = field(default_factory=list)
    """Per-ROI list of stacked per-channel activity arrays for condition B.
    Element i has shape (n_channels_in_roi_i, n_times)."""
    contribution_labels: list = field(default_factory=list)
    """Per-ROI list of string labels ("subject/channel") matching contribution rows."""

    # --- Group-level activity statistics (t-test / one-sample, corrected) ---
    activity_t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    """t-statistics for the activity contrast, shape (n_rois, n_times)."""
    activity_p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    """Corrected p-values for the activity contrast, shape (n_rois, n_times)."""
    activity_p_values_uncorrected: np.ndarray = field(default_factory=lambda: np.array([]))
    """Raw (uncorrected) p-values for the activity contrast, shape (n_rois, n_times)."""
    activity_significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    """Boolean significance mask for the activity contrast, shape (n_rois, n_times)."""

    # --- Epoch-level summary statistics (scalar per ROI, across the full epoch window) ---
    epoch_activity_t: np.ndarray = field(default_factory=lambda: np.array([]))
    """t-statistic computed on the epoch-mean activity (one value per ROI), shape (n_rois,)."""
    epoch_activity_p: np.ndarray = field(default_factory=lambda: np.array([]))
    """p-value for the epoch-mean activity test, shape (n_rois,)."""
    epoch_activity_df: np.ndarray = field(default_factory=lambda: np.array([]))
    """Degrees of freedom for the epoch-mean activity test, shape (n_rois,)."""

    # --- Configuration and provenance ---
    source_metric: str = ""
    """Name of the scalar metric extracted from subject-level stats files (e.g. 'mean_difference', 'slope')."""
    p_value_correction_method: str = "none"
    """Multiple-comparison correction applied to p-values ('none', 'fdr_bh', 'bonferroni', 'cluster_permutation')."""
    significance_alpha: float = 0.05
    """Significance threshold used to build the significant_mask fields."""
    roi_mode: str = "manual"
    """How ROIs were defined: 'manual' (explicit channel lists) or 'atlas' (electrode TSV column)."""
    atlas_name: str | None = None
    """Name of the atlas column in _electrodes.tsv used when roi_mode='atlas'; None otherwise."""
    source_subject_stats_files: list[str] = field(default_factory=list)
    """Absolute paths of the subject-level stats files that were aggregated."""
    source_electrodes_files: list[str] = field(default_factory=list)
    """Absolute paths of the _electrodes.tsv files consulted for atlas or channel metadata."""
    excluded_rois: dict[str, str] = field(default_factory=dict)
    """ROIs that were dropped before statistics, keyed by ROI name; value is the exclusion reason."""
