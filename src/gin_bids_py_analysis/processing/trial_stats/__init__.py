"""Subject-level trial statistics on Hilbert derivatives."""

__version__ = "0.1.0"

from .params import TrialStatsParams, TrialStatsWriterParams
from .processor import TrialStatsProcessing
from .resolver import (
    ResolvedTrial,
    TableTrialLabelResolver,
    TrialLabelResolver,
)
from .result import TrialStatsProcessingResult
from .writer import TrialStatsProcessingWriter

__all__ = [
    "ResolvedTrial",
    "TableTrialLabelResolver",
    "TrialLabelResolver",
    "TrialStatsParams",
    "TrialStatsProcessing",
    "TrialStatsProcessingResult",
    "TrialStatsProcessingWriter",
    "TrialStatsWriterParams",
]
