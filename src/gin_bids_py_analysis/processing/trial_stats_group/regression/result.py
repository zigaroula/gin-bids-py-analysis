from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..result import BaseTrialStatsGroupProcessingResult

__all__ = ["RegressionGroupProcessingResult"]


@dataclass
class RegressionGroupProcessingResult(BaseTrialStatsGroupProcessingResult):
    """Structured outputs for group-level ROI statistics on regression data."""

    slope_t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    slope_p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    slope_p_values_uncorrected: np.ndarray = field(default_factory=lambda: np.array([]))
    slope_significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))

    activity_t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    activity_p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    activity_p_values_uncorrected: np.ndarray = field(default_factory=lambda: np.array([]))
    activity_significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))

    condition_a_slope_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_slope_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_slope_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_slope_sem: np.ndarray = field(default_factory=lambda: np.array([]))

    condition_a_activity_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_activity_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_activity_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_activity_sem: np.ndarray = field(default_factory=lambda: np.array([]))

    condition_a_r_value_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_r_value_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_r_value_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_r_value_sem: np.ndarray = field(default_factory=lambda: np.array([]))

    epoch_slope_t: np.ndarray = field(default_factory=lambda: np.array([]))
    epoch_slope_p: np.ndarray = field(default_factory=lambda: np.array([]))
    epoch_slope_df: np.ndarray = field(default_factory=lambda: np.array([]))

    epoch_activity_t: np.ndarray = field(default_factory=lambda: np.array([]))
    epoch_activity_p: np.ndarray = field(default_factory=lambda: np.array([]))
    epoch_activity_df: np.ndarray = field(default_factory=lambda: np.array([]))

    condition_a_slope_contributions: list = field(default_factory=list)
    condition_b_slope_contributions: list = field(default_factory=list)
    condition_a_activity_contributions: list = field(default_factory=list)
    condition_b_activity_contributions: list = field(default_factory=list)
    contribution_labels: list = field(default_factory=list)

    condition_a_scatter_predictor: list = field(default_factory=list)
    condition_a_scatter_activity: list = field(default_factory=list)
    condition_b_scatter_predictor: list = field(default_factory=list)
    condition_b_scatter_activity: list = field(default_factory=list)

    contrast_mode: str = "paired"
    source_regression_files: list[str] = field(default_factory=list)
    source_electrodes_files: list[str] = field(default_factory=list)
