from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gin_bids_py_analysis.processing.base import BaseProcessingResult

from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial


@dataclass
class TrialStatsProcessingResult(BaseProcessingResult):
    """Structured outputs for subject-level trial statistics."""

    t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values_uncorrected: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_difference: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    difference_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    difference_ci95_low: np.ndarray = field(default_factory=lambda: np.array([]))
    difference_ci95_high: np.ndarray = field(default_factory=lambda: np.array([]))
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
    region_channels: dict[str, list[str]] = field(default_factory=dict)
    window_ms: float = 0.0
    n_bins: int = 0
    activity_scaling: str = "none"
    activity_baseline_tmin_s: float = -0.2
    activity_baseline_tmax_s: float = 0.0
    p_value_correction_method: str = "fdr_bh"
    significance_alpha: float = 0.05
    stats_valid: bool = False
    condition_a_epochs: np.ndarray = field(default_factory=lambda: np.array([]))
    """Individual trial epochs for condition A.  Shape: ``(n_trials, n_channels, n_times)``."""
    condition_b_epochs: np.ndarray = field(default_factory=lambda: np.array([]))
    """Individual trial epochs for condition B.  Shape: ``(n_trials, n_channels, n_times)``."""
    permuted_t_values: np.ndarray | None = None
    """Permuted t-values forming the null distribution.

    Shape: ``(n_perm, n_channels, n_times)``, dtype ``float32``.
    ``None`` when ``n_permutations == 0`` (permutation tests disabled).
    Stored in the output HDF5 file under ``/stats/permuted_t_values``.
    """
    channel_significant_mask: np.ndarray | None = None
    """Per-channel significance flag.

    Shape: ``(n_channels,)``, dtype ``bool``.
    ``None`` when ``channel_significance_mode='none'`` (default).
    Derived by :meth:`TrialStatsProcessing.process_group` according to
    ``TrialStatsParams.channel_significance_mode``.
    """
