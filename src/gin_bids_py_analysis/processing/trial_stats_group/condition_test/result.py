from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..result import BaseTrialStatsGroupProcessingResult

__all__ = ["ConditionTestGroupProcessingResult"]


@dataclass
class ConditionTestGroupProcessingResult(BaseTrialStatsGroupProcessingResult):
    """Structured outputs for group-level condition-test ROI statistics."""

    # --- Source-metric statistics (the compared quantity, e.g. mean_difference or t_values) ---
    metric_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    """Mean of the source metric across channels/subjects per ROI, shape (n_rois, n_times)."""
    metric_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    """SEM of the source metric across channels/subjects per ROI, shape (n_rois, n_times)."""

    # --- Epoch-level summary for the source metric ---
    epoch_mean_metric_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    """Mean of the epoch-averaged source metric per ROI (scalar summary), shape (n_rois,)."""
    epoch_mean_metric_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    """SEM of the epoch-averaged source metric per ROI, shape (n_rois,)."""

    # --- Optional cluster-permutation results (only present when p_value_correction_method='cluster_permutation') ---
    cluster_p_values: np.ndarray | None = None
    """Cluster-level p-value per ROI, shape (n_rois,); None when cluster permutation was not used."""
    cluster_best_cluster_windows_s: list[tuple[float, float] | None] | None = None
    """Time window (start_s, end_s) of the most significant cluster per ROI; None entry if no cluster found."""
    cluster_null_distributions: list[np.ndarray] | None = None
    """Per-ROI permutation null distributions of max-cluster-mass statistic, each shape (n_permutations,)."""
