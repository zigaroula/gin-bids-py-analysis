from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..result import BaseTrialStatsProcessingResult


@dataclass
class RegressionProcessingResult(BaseTrialStatsProcessingResult):
    """Structured outputs for subject-level trial slope regression statistics."""

    condition_a_slope: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_intercept: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_r_value: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_p_value: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_p_value_corrected: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_slope: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_intercept: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_r_value: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_p_value: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_p_value_corrected: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_trials_used: int = 0
    condition_b_trials_used: int = 0
    condition_a_predictor_raw_values: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_predictor_raw_values: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_predictor_transformed_values: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_predictor_transformed_values: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_predictor_values: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_predictor_values: np.ndarray = field(default_factory=lambda: np.array([]))
    analysis_type: str = "slope_regression"
    excluded_channels: dict[str, str] = field(default_factory=dict)
    excluded_trial_channel_pairs: dict[str, list[int]] = field(default_factory=dict)
    predictor: str = "predictor_value"
    predictor_zscore: str = "none"
    predictor_transform_by_condition: dict[str, dict[str, float]] = field(default_factory=dict)
    condition_a_stats_valid: bool = False
    condition_b_stats_valid: bool = False
    condition_a_epoch_means: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_epoch_means: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_permuted_slopes: np.ndarray | None = None
    """Permuted slope maps for condition A, shape ``(n_perm, n_channels, n_times)``, float32.
    None when ``n_permutations == 0`` or ``condition_a_stats_valid`` is False."""
    condition_b_permuted_slopes: np.ndarray | None = None
    """Permuted slope maps for condition B, shape ``(n_perm, n_channels, n_times)``, float32.
    None when ``n_permutations == 0`` or ``condition_b_stats_valid`` is False."""
