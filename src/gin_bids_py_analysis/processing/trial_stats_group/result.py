from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gin_bids_py_analysis.processing.base import BaseProcessingResult
from gin_bids_py_analysis.processing.utils.group_stats import ROIChannelContribution

__all__ = ["BaseTrialStatsGroupProcessingResult", "ROIChannelContribution"]


@dataclass
class BaseTrialStatsGroupProcessingResult(BaseProcessingResult):
    """Shared result surface for group-level trial-statistics pipelines."""

    time_axis_s: np.ndarray = field(default_factory=lambda: np.array([]))
    region_names: list[str] = field(default_factory=list)
    condition_labels: tuple[str, str] = ("condition_a", "condition_b")
    roi_channel_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    roi_subject_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    contributions: list[ROIChannelContribution] = field(default_factory=list)
    source_metric: str = ""
    p_value_correction_method: str = "none"
    significance_alpha: float = 0.05
    roi_mode: str = "manual"
    atlas_name: str | None = None
    excluded_rois: dict[str, str] = field(default_factory=dict)
