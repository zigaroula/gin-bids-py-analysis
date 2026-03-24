"""Group-level ROI statistics built from trial_stats channel-level outputs."""

__version__ = "0.1.0"

from .params import TrialStatsGroupParams, TrialStatsGroupWriterParams
from .processor import (
    TrialStatsGroupProcessing,
    build_trial_stats_compatible_groups,
)
from .result import ROIChannelContribution, TrialStatsGroupProcessingResult
from .writer import TrialStatsGroupProcessingWriter

__all__ = [
    "ROIChannelContribution",
    "TrialStatsGroupParams",
    "TrialStatsGroupProcessing",
    "TrialStatsGroupProcessingResult",
    "TrialStatsGroupProcessingWriter",
    "TrialStatsGroupWriterParams",
    "build_trial_stats_compatible_groups",
]
