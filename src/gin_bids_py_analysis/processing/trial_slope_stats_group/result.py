from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gin_bids_py_analysis.processing.base import BaseProcessingResult
from gin_bids_py_analysis.processing.utils.group_stats import ROIChannelContribution

__all__ = ["TrialSlopeStatsGroupProcessingResult"]


@dataclass
class TrialSlopeStatsGroupProcessingResult(BaseProcessingResult):
    """Structured outputs for group-level ROI statistics on trial-slope-stats data."""

    # --- Per-condition slope one-sample tests vs 0 (n_rois, n_times) ---
    condition_a_slope_t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_slope_p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_slope_p_values_uncorrected: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_slope_significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))

    condition_b_slope_t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_slope_p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_slope_p_values_uncorrected: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_slope_significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))

    # --- Group mean / SEM of slopes per condition (n_rois, n_times) ---
    condition_a_slope_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_slope_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_slope_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_slope_sem: np.ndarray = field(default_factory=lambda: np.array([]))

    # --- Group mean / SEM of mean activity per condition (n_rois, n_times) ---
    condition_a_activity_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_activity_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_activity_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_activity_sem: np.ndarray = field(default_factory=lambda: np.array([]))

    # --- Group mean / SEM of r_value per condition (n_rois, n_times) ---
    condition_a_r_value_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_r_value_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_r_value_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_r_value_sem: np.ndarray = field(default_factory=lambda: np.array([]))

    # --- Per-condition epoch-level summaries (n_rois,) ---
    condition_a_epoch_slope_t: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_epoch_slope_p: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_epoch_slope_df: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_epoch_slope_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_epoch_slope_sem: np.ndarray = field(default_factory=lambda: np.array([]))

    condition_b_epoch_slope_t: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_epoch_slope_p: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_epoch_slope_df: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_epoch_slope_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_epoch_slope_sem: np.ndarray = field(default_factory=lambda: np.array([]))

    # --- Axes and identifiers ---
    time_axis_s: np.ndarray = field(default_factory=lambda: np.array([]))
    region_names: list[str] = field(default_factory=list)
    condition_labels: tuple[str, str] = ("condition_a", "condition_b")

    # --- ROI / contribution metadata ---
    roi_channel_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    roi_subject_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    contributions: list[ROIChannelContribution] = field(default_factory=list)

    condition_a_slope_contributions: list = field(default_factory=list)
    """Per-ROI list of condition-A slope contribution arrays.

    ``condition_a_slope_contributions[roi_idx]`` is a 2-D array of shape
    ``(n_contributions, n_times)`` where each row is one (subject, channel)
    slope timecourse used to compute the group statistics for that ROI.
    """
    condition_b_slope_contributions: list = field(default_factory=list)
    """Per-ROI list of condition-B slope contribution arrays. Same structure as
    ``condition_a_slope_contributions``.
    """
    condition_a_activity_contributions: list = field(default_factory=list)
    """Per-ROI list of condition-A mean-activity contribution arrays.  Same shape."""
    condition_b_activity_contributions: list = field(default_factory=list)
    """Per-ROI list of condition-B mean-activity contribution arrays.  Same shape."""
    contribution_labels: list = field(default_factory=list)
    """Per-ROI list of human-readable row labels formatted as ``"subject/channel"``."""

    # --- Analysis parameters stored in the result for provenance ---
    p_value_correction_method: str = "none"
    significance_alpha: float = 0.05
    roi_mode: str = "manual"
    atlas_name: str | None = None

    # --- Provenance ---
    source_trial_slope_stats_files: list[str] = field(default_factory=list)
    source_electrodes_files: list[str] = field(default_factory=list)
    excluded_rois: dict[str, str] = field(default_factory=dict)
