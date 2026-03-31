from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gin_bids_py_analysis.processing.base import BaseProcessingResult


@dataclass(frozen=True)
class ROIChannelContribution:
    """One channel contribution used for one ROI in the group-level analysis."""

    roi: str
    subject: str
    channel: str
    source_stats_file: str


@dataclass
class TrialStatsGroupProcessingResult(BaseProcessingResult):
    """Structured outputs for group-level ROI statistics."""

    t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values_uncorrected: np.ndarray = field(default_factory=lambda: np.array([]))
    significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    metric_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    metric_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    time_axis_s: np.ndarray = field(default_factory=lambda: np.array([]))
    region_names: list[str] = field(default_factory=list)
    source_metric: str = "mean_difference"
    condition_labels: tuple[str, str] = ("condition_a", "condition_b")
    p_value_correction_method: str = "none"
    significance_alpha: float = 0.05
    roi_mode: str = "manual"
    atlas_name: str | None = None

    epoch_mean_t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    epoch_mean_p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    epoch_mean_df: np.ndarray = field(default_factory=lambda: np.array([]))
    epoch_mean_metric_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    epoch_mean_metric_sem: np.ndarray = field(default_factory=lambda: np.array([]))

    roi_channel_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    roi_subject_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    contributions: list[ROIChannelContribution] = field(default_factory=list)

    condition_a_group_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_group_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_group_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_group_sem: np.ndarray = field(default_factory=lambda: np.array([]))

    condition_a_contributions: list = field(default_factory=list)
    """Per-ROI list of condition-A contribution arrays.

    ``condition_a_contributions[roi_idx]`` is a 2-D array of shape
    ``(n_contributions, n_times)`` where each row is one (subject, channel)
    sample used to compute the group mean for that ROI.
    """
    condition_b_contributions: list = field(default_factory=list)
    """Per-ROI list of condition-B contribution arrays.  Same structure as
    ``condition_a_contributions``.
    """
    contribution_labels: list = field(default_factory=list)
    """Per-ROI list of human-readable row labels.

    ``contribution_labels[roi_idx]`` is a list of strings, one per row of the
    corresponding ``condition_a/b_contributions`` array, formatted as
    ``"subject/channel"``.
    """

    source_trial_stats_files: list[str] = field(default_factory=list)
    source_electrodes_files: list[str] = field(default_factory=list)
    excluded_rois: dict[str, str] = field(default_factory=dict)

    # --- Cluster-permutation results (populated only when method='cluster_permutation') ---
    cluster_p_values: np.ndarray | None = None
    """Cluster-permutation p-values, one per ROI.  Shape ``(n_rois,)``.

    ``None`` when ``p_value_correction_method != 'cluster_permutation'``.
    The p-value for ROI *i* is the fraction of null-distribution values that
    exceed ``|best_cluster_t_sum[i]|``. Set to ``1.0`` for ROIs with no detected
    cluster.
    """
    cluster_best_cluster_windows_s: list[tuple[float, float] | None] | None = None
    """Time window (onset_s, offset_s) of the best cluster per ROI.

    ``None`` at the list level when method != 'cluster_permutation'.
    Individual entries are ``None`` for ROIs where no temporal cluster was found.
    """
    cluster_null_distributions: list[np.ndarray] | None = None
    """Null distributions of maximum cluster t-sums, one per ROI.

    Each entry has shape ``(n_group_perm,)``.  Useful for diagnostics or
    custom threshold selection.  ``None`` when method != 'cluster_permutation'.
    """
