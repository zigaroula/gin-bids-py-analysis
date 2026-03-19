from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gin_bids_py_analysis.processing.base import BaseProcessingResult

from .resolver import ResolvedTrial


@dataclass
class TrialStatsProcessingResult(BaseProcessingResult):
    """Structured outputs for subject-level trial statistics."""

    t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values_uncorrected: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_difference: np.ndarray = field(default_factory=lambda: np.array([]))
    significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    time_axis_s: np.ndarray = field(default_factory=lambda: np.array([]))
    channel_names: list[str] = field(default_factory=list)
    condition_a: str = "accepted"
    condition_b: str = "rejected"
    condition_a_trial_count: int = 0
    condition_b_trial_count: int = 0
    sfreq: float = 0.0
    resolved_trials: list[ResolvedTrial] = field(default_factory=list)
    source_ieeg_files: list[str] = field(default_factory=list)
    source_table_files: list[str] = field(default_factory=list)
    source_electrodes_files: list[str] = field(default_factory=list)
    analysis_level: str = "channel"
    atlas_name: str | None = None
    atlas_regions: list[str] = field(default_factory=list)
    window_ms: float = 0.0
    n_bins: int = 0
    p_value_correction_method: str = "fdr_bh"
    significance_alpha: float = 0.05
    stats_valid: bool = False
