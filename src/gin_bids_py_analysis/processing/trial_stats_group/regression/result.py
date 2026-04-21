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

    # --- Per-condition one-sample t-test vs 0 on the averaged slope ---
    condition_a_source_metric_vs_zero_t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    """One-sample t-test (vs 0) t-statistics for condition A slope mean, shape (n_rois, n_times)."""
    condition_a_source_metric_vs_zero_p_values_uncorrected: np.ndarray = field(default_factory=lambda: np.array([]))
    """Raw (uncorrected) p-values for condition A slope vs 0, shape (n_rois, n_times)."""
    condition_a_source_metric_vs_zero_p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    """Corrected p-values for condition A slope vs 0, shape (n_rois, n_times)."""
    condition_a_source_metric_vs_zero_significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    """Boolean significance mask for condition A slope vs 0, shape (n_rois, n_times)."""
    condition_a_source_metric_vs_zero_cluster_p_values: np.ndarray | None = None
    """Cluster permutation p-values for condition A vs 0 (best cluster), one per ROI, shape (n_rois,). None when not computed."""
    condition_a_source_metric_vs_zero_cluster_windows_s: list[list[tuple[float, float]]] | None = None
    """Significant cluster windows per ROI for condition A vs 0. Each inner list holds (start_s, end_s)
    tuples for up to n_clusters_to_keep clusters that passed p < significance_alpha."""
    condition_a_source_metric_vs_zero_cluster_null_distributions: list[np.ndarray] | None = None
    """Per-ROI null distributions for condition A vs 0. Each array has shape (n_permutations,)."""
    condition_b_source_metric_vs_zero_t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    """One-sample t-test (vs 0) t-statistics for condition B slope mean, shape (n_rois, n_times)."""
    condition_b_source_metric_vs_zero_p_values_uncorrected: np.ndarray = field(default_factory=lambda: np.array([]))
    """Raw (uncorrected) p-values for condition B slope vs 0, shape (n_rois, n_times)."""
    condition_b_source_metric_vs_zero_p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    """Corrected p-values for condition B slope vs 0, shape (n_rois, n_times)."""
    condition_b_source_metric_vs_zero_significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    """Boolean significance mask for condition B slope vs 0, shape (n_rois, n_times)."""
    condition_b_source_metric_vs_zero_cluster_p_values: np.ndarray | None = None
    """Cluster permutation p-values for condition B vs 0 (best cluster), one per ROI, shape (n_rois,). None when not computed."""
    condition_b_source_metric_vs_zero_cluster_windows_s: list[list[tuple[float, float]]] | None = None
    """Significant cluster windows per ROI for condition B vs 0. Each inner list holds (start_s, end_s)
    tuples for up to n_clusters_to_keep clusters that passed p < significance_alpha."""
    condition_b_source_metric_vs_zero_cluster_null_distributions: list[np.ndarray] | None = None
    """Per-ROI null distributions for condition B vs 0. Each array has shape (n_permutations,)."""

    # --- Cluster permutation stats (only populated when p_value_correction_method='cluster_permutation') ---
    cluster_p_values: np.ndarray | None = None
    """Cluster-based permutation p-values (best cluster), one per ROI, shape (n_rois,). None when not computed."""
    cluster_windows_s: list[list[tuple[float, float]]] | None = None
    """Significant cluster windows per ROI for the slope contrast. Each inner list holds (start_s, end_s)
    tuples for up to n_clusters_to_keep clusters that passed p < significance_alpha."""
    cluster_null_distributions: list[np.ndarray] | None = None
    """Per-ROI null distributions of max-cluster t-sum statistics. Each array has shape (n_permutations,)."""
    manual_roi_missing_channels: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    """Manual ROI channel assignments absent from the available subject result files."""
