"""Subject-level trial slope statistics on iEEG recordings."""

__version__ = "0.1.0"

from .params import TrialSlopeStatsParams, TrialSlopeStatsWriterParams
from .processor import TrialSlopeStatsProcessing
from .result import TrialSlopeStatsProcessingResult
from .result_loader import load_trial_slope_stats_result
from .writer import TrialSlopeStatsProcessingWriter

__all__ = [
    "TrialSlopeStatsParams",
    "TrialSlopeStatsProcessing",
    "TrialSlopeStatsProcessingResult",
    "TrialSlopeStatsWriterParams",
    "TrialSlopeStatsProcessingWriter",
    "load_trial_slope_stats_result",
]
