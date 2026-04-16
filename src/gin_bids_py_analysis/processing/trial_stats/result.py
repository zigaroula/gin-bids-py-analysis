from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gin_bids_py_analysis.processing.base import BaseProcessingResult
from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial


@dataclass
class BaseTrialStatsProcessingResult(BaseProcessingResult):
    """Shared subject-level result fields for trial statistics pipelines."""

    time_axis_s: np.ndarray = field(default_factory=lambda: np.array([]))
    channel_names: list[str] = field(default_factory=list)
    condition_a: str = "condition_a"
    condition_b: str = "condition_b"
    condition_a_trial_count: int = 0
    condition_b_trial_count: int = 0
    condition_a_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    sfreq: float = 0.0
    resolved_trials: list[ResolvedTrial] = field(default_factory=list)
    source_ieeg_files: list[str] = field(default_factory=list)
    source_table_files: list[str] = field(default_factory=list)
    source_electrodes_files: list[str] = field(default_factory=list)
    analysis_level: str = "channel"
    atlas_name: str | None = None
    atlas_regions: list[str] = field(default_factory=list)
    region_channels: dict[str, list[str]] = field(default_factory=dict)
    window_ms: float = 0.0
    n_bins: int = 0
    activity_zscore: str = "none"
    activity_baseline_tmin_s: float = -0.2
    activity_baseline_tmax_s: float = 0.0
    activity_baseline_scope: str = "global"
    activity_baseline_remove_outlier_trial_means: bool = False
    p_value_correction_method: str = "fdr_bh"
    significance_alpha: float = 0.05
    trial_activity_summary_kind: str = "epoch_mean"
    trial_activity_summary_missing_response_policy: str = "clamp_to_epoch"
    trial_activity_summary_source: dict[str, str] = field(default_factory=dict)
    trial_activity_summary_label: str = "Epoch mean activity"
    epoch_cleaning_audit: dict[str, object] = field(default_factory=dict)
    stats_valid: bool = False
    condition_a_epochs: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_epochs: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_trial_activity_summary_values: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    condition_b_trial_activity_summary_values: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
