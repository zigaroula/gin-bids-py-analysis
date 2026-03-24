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

    source_trial_stats_files: list[str] = field(default_factory=list)
    source_electrodes_files: list[str] = field(default_factory=list)
    excluded_rois: dict[str, str] = field(default_factory=dict)
