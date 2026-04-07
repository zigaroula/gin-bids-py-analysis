"""Subject-level trial statistics on iEEG recordings."""

__version__ = "0.1.0"

from .params import TrialStatsParams, TrialStatsWriterParams
from .processor import TrialStatsProcessing
from .result import TrialStatsProcessingResult
from .writer import TrialStatsProcessingWriter

__all__ = [
    "TrialStatsParams",
    "TrialStatsProcessing",
    "TrialStatsProcessingResult",
    "TrialStatsProcessingWriter",
    "TrialStatsWriterParams",
]
