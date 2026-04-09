from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gin_bids_py_analysis.processing.base import BaseProcessingResult
from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial


@dataclass
class TrialSlopeStatsProcessingResult(BaseProcessingResult):
    """Structured outputs for subject-level trial slope statistics."""

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

    condition_a_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_sem: np.ndarray = field(default_factory=lambda: np.array([]))

    time_axis_s: np.ndarray = field(default_factory=lambda: np.array([]))
    channel_names: list[str] = field(default_factory=list)
    condition_a: str = "condition_a"
    condition_b: str = "condition_b"
    condition_a_trial_count: int = 0
    condition_b_trial_count: int = 0
    condition_a_trials_used: int = 0
    condition_b_trials_used: int = 0
    sfreq: float = 0.0

    condition_a_predictor_raw_values: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_predictor_raw_values: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_predictor_transformed_values: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    condition_b_predictor_transformed_values: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    condition_a_predictor_values: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_predictor_values: np.ndarray = field(default_factory=lambda: np.array([]))

    resolved_trials: list[ResolvedTrial] = field(default_factory=list)
    source_ieeg_files: list[str] = field(default_factory=list)
    source_table_files: list[str] = field(default_factory=list)
    source_electrodes_files: list[str] = field(default_factory=list)

    analysis_level: str = "channel"
    analysis_type: str = "slope_regression"
    atlas_name: str | None = None
    atlas_regions: list[str] = field(default_factory=list)
    region_channels: dict[str, list[str]] = field(default_factory=dict)
    window_ms: float = 0.0
    n_bins: int = 0
    activity_zscore: str = "none"
    activity_baseline_tmin_s: float = -0.2
    activity_baseline_tmax_s: float = 0.0

    predictor: str = "predictor_value"
    predictor_zscore: str = "none"
    predictor_transform_by_condition: dict[str, dict[str, float]] = field(default_factory=dict)
    p_value_correction_method: str = "fdr_bh"
    significance_alpha: float = 0.05

    condition_a_stats_valid: bool = False
    condition_b_stats_valid: bool = False
    stats_valid: bool = False

    condition_a_epochs: np.ndarray = field(default_factory=lambda: np.array([]))
    """Individual trial epochs for condition A (n_trials, n_channels, n_times)."""

    condition_b_epochs: np.ndarray = field(default_factory=lambda: np.array([]))
    """Individual trial epochs for condition B (n_trials, n_channels, n_times)."""

    condition_a_epoch_means: np.ndarray = field(default_factory=lambda: np.array([]))
    """Per-trial mean activity over the epoch time window, condition A.
    Shape ``(n_channels, n_trials_a)``.  Always computed; does not require
    ``include_epochs=True`` in the writer params.
    """

    condition_b_epoch_means: np.ndarray = field(default_factory=lambda: np.array([]))
    """Per-trial mean activity over the epoch time window, condition B.
    Shape ``(n_channels, n_trials_b)``.
    """


