from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..result import BaseTrialStatsGroupProcessingResult

__all__ = ["RegressionGroupProcessingResult"]


@dataclass
class RegressionGroupProcessingResult(BaseTrialStatsGroupProcessingResult):
    """Structured outputs for group-level ROI statistics on regression data."""

    # --- Group-level statistics on the regression source metric (e.g. slope) ---
    source_metric_t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    """t-statistics for the source-metric contrast, shape (n_rois, n_times)."""
    source_metric_p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    """Corrected p-values for the source-metric contrast, shape (n_rois, n_times)."""
    source_metric_p_values_uncorrected: np.ndarray = field(default_factory=lambda: np.array([]))
    """Raw (uncorrected) p-values for the source-metric contrast, shape (n_rois, n_times)."""
    source_metric_significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    """Boolean significance mask for the source-metric contrast, shape (n_rois, n_times)."""

    # --- Source-metric means per condition ---
    condition_a_source_metric_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    """Mean source metric (e.g. slope) for condition A, shape (n_rois, n_times)."""
    condition_a_source_metric_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    """SEM of source metric for condition A, shape (n_rois, n_times)."""
    condition_b_source_metric_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    """Mean source metric for condition B, shape (n_rois, n_times)."""
    condition_b_source_metric_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    """SEM of source metric for condition B, shape (n_rois, n_times)."""

    # --- Correlation coefficient (r) summaries per condition ---
    condition_a_r_value_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    """Mean Pearson r between predictor and activity for condition A, shape (n_rois, n_times)."""
    condition_a_r_value_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    """SEM of Pearson r for condition A, shape (n_rois, n_times)."""
    condition_b_r_value_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    """Mean Pearson r for condition B, shape (n_rois, n_times)."""
    condition_b_r_value_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    """SEM of Pearson r for condition B, shape (n_rois, n_times)."""

    # --- Epoch-level summary for the source metric ---
    epoch_source_metric_t: np.ndarray = field(default_factory=lambda: np.array([]))
    """t-statistic on the epoch-averaged source metric per ROI, shape (n_rois,)."""
    epoch_source_metric_p: np.ndarray = field(default_factory=lambda: np.array([]))
    """p-value for the epoch-averaged source-metric test, shape (n_rois,)."""
    epoch_source_metric_df: np.ndarray = field(default_factory=lambda: np.array([]))
    """Degrees of freedom for the epoch-averaged source-metric test, shape (n_rois,)."""

    # --- Per-channel source-metric time-courses (for contribution plots) ---
    condition_a_source_metric_contributions: list = field(default_factory=list)
    """Per-ROI list of stacked per-channel source-metric arrays for condition A.
    Element i has shape (n_channels_in_roi_i, n_times)."""
    condition_b_source_metric_contributions: list = field(default_factory=list)
    """Per-ROI list of stacked per-channel source-metric arrays for condition B.
    Element i has shape (n_channels_in_roi_i, n_times)."""

    # --- Scatter data (predictor vs activity per channel per trial, for scatter plots) ---
    condition_a_scatter_predictor: list = field(default_factory=list)
    """Per-ROI list of 1-D predictor value arrays (one per contributing channel) for condition A."""
    condition_a_scatter_activity: list = field(default_factory=list)
    """Per-ROI list of 1-D activity value arrays (one per contributing channel) for condition A."""
    condition_b_scatter_predictor: list = field(default_factory=list)
    """Per-ROI list of 1-D predictor value arrays for condition B."""
    condition_b_scatter_activity: list = field(default_factory=list)
    """Per-ROI list of 1-D activity value arrays for condition B."""

    # --- Regression-specific configuration ---
    contrast_mode: str = "paired"
    """How conditions are contrasted: 'paired' (within-subject) or 'independent' (between-subject)."""

    # --- Cluster permutation stats (only populated when p_value_correction_method='cluster_permutation') ---
    cluster_p_values: np.ndarray | None = None
    """Cluster-based permutation p-values, one per ROI, shape (n_rois,). None when not computed."""
    cluster_best_cluster_windows_s: list[tuple[float, float] | None] | None = None
    """Best cluster window [t_start_s, t_end_s] per ROI. None entries mean no cluster was found."""
    cluster_null_distributions: list[np.ndarray] | None = None
    """Per-ROI null distributions of max-cluster t-sum statistics. Each array has shape (n_permutations,)."""
