from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..result import BaseTrialStatsGroupProcessingResult

__all__ = ["ConditionTestGroupProcessingResult"]


@dataclass
class ConditionTestGroupProcessingResult(BaseTrialStatsGroupProcessingResult):
    """Structured outputs for group-level condition-test ROI statistics."""

    t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values_uncorrected: np.ndarray = field(default_factory=lambda: np.array([]))
    significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    metric_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    metric_sem: np.ndarray = field(default_factory=lambda: np.array([]))

    epoch_mean_t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    epoch_mean_p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    epoch_mean_df: np.ndarray = field(default_factory=lambda: np.array([]))
    epoch_mean_metric_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    epoch_mean_metric_sem: np.ndarray = field(default_factory=lambda: np.array([]))

    condition_a_group_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_group_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_group_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_group_sem: np.ndarray = field(default_factory=lambda: np.array([]))

    condition_a_contributions: list = field(default_factory=list)
    condition_b_contributions: list = field(default_factory=list)
    contribution_labels: list = field(default_factory=list)

    source_condition_test_files: list[str] = field(default_factory=list)
    source_electrodes_files: list[str] = field(default_factory=list)

    cluster_p_values: np.ndarray | None = None
    cluster_best_cluster_windows_s: list[tuple[float, float] | None] | None = None
    cluster_null_distributions: list[np.ndarray] | None = None
