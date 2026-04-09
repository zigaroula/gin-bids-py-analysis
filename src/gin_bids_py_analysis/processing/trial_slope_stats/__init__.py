"""Subject-level trial slope statistics on iEEG recordings."""

__version__ = "0.1.0"

from .params import (
    TrialActivitySummaryAnnotationEventSource,
    TrialActivitySummaryConfig,
    TrialActivitySummaryTableColumnSource,
    TrialSlopeStatsParams,
    TrialSlopeStatsWriterParams,
)
from .processor import TrialSlopeStatsProcessing
from .result import TrialSlopeStatsProcessingResult
from .result_loader import load_trial_slope_stats_result
from .writer import TrialSlopeStatsProcessingWriter
from ..utils.condition_rules import ConditionDefinition, ConditionExpr
from ..utils.trial_resolver import TableTrialResolver, TrialResolver

__all__ = [
    "TrialResolver",
    "TableTrialResolver",
    "ConditionExpr",
    "ConditionDefinition",
    "TrialActivitySummaryAnnotationEventSource",
    "TrialActivitySummaryConfig",
    "TrialActivitySummaryTableColumnSource",
    "TrialSlopeStatsParams",
    "TrialSlopeStatsProcessing",
    "TrialSlopeStatsProcessingResult",
    "TrialSlopeStatsWriterParams",
    "TrialSlopeStatsProcessingWriter",
    "load_trial_slope_stats_result",
]
