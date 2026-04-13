from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..result import BaseTrialStatsProcessingResult


@dataclass
class ConditionTestProcessingResult(BaseTrialStatsProcessingResult):
    """Structured outputs for subject-level condition-test statistics."""

    t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values_uncorrected: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_difference: np.ndarray = field(default_factory=lambda: np.array([]))
    difference_sem: np.ndarray = field(default_factory=lambda: np.array([]))
    difference_ci95_low: np.ndarray = field(default_factory=lambda: np.array([]))
    difference_ci95_high: np.ndarray = field(default_factory=lambda: np.array([]))
    significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    permuted_t_values: np.ndarray | None = None
    channel_significant_mask: np.ndarray | None = None
