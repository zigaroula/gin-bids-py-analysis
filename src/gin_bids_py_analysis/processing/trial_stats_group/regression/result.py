from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..result import BaseTrialStatsGroupProcessingResult

__all__ = ["RegressionGroupProcessingResult"]


@dataclass
class RegressionGroupProcessingResult(BaseTrialStatsGroupProcessingResult):
    """Structured outputs for group-level ROI statistics on regression data."""

    source_metric_t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    source_metric_p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    source_metric_p_values_uncorrected: np.ndarray = field(default_factory=lambda: np.array([]))
    source_metric_significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))

    activity_t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    activity_p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    activity_p_values_uncorrected: np.ndarray = field(default_factory=lambda: np.array([]))
    activity_significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))

    condition_a_source_metric_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_source_metric_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_source_metric_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_source_metric_sem: np.ndarray = field(default_factory=lambda: np.array([]))

    condition_a_r_value_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_r_value_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_r_value_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_r_value_sem: np.ndarray = field(default_factory=lambda: np.array([]))

    epoch_source_metric_t: np.ndarray = field(default_factory=lambda: np.array([]))
    epoch_source_metric_p: np.ndarray = field(default_factory=lambda: np.array([]))
    epoch_source_metric_df: np.ndarray = field(default_factory=lambda: np.array([]))

    epoch_activity_t: np.ndarray = field(default_factory=lambda: np.array([]))
    epoch_activity_p: np.ndarray = field(default_factory=lambda: np.array([]))
    epoch_activity_df: np.ndarray = field(default_factory=lambda: np.array([]))

    condition_a_source_metric_contributions: list = field(default_factory=list)
    condition_b_source_metric_contributions: list = field(default_factory=list)

    condition_a_scatter_predictor: list = field(default_factory=list)
    condition_a_scatter_activity: list = field(default_factory=list)
    condition_b_scatter_predictor: list = field(default_factory=list)
    condition_b_scatter_activity: list = field(default_factory=list)

    contrast_mode: str = "paired"
