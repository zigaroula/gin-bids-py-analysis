"""Subject-level trial statistics on iEEG recordings."""

__version__ = "0.1.0"

from .params import TrialStatsParams, TrialStatsWriterParams
from .processor import TrialStatsProcessing
from .result import TrialStatsProcessingResult
from .writer import TrialStatsProcessingWriter
from ..utils.condition_rules import ConditionDefinition, ConditionExpr
from ..utils.trial_resolver import TableTrialResolver, TrialResolver

__all__ = [
    "TrialResolver",
    "TableTrialResolver",
    "ConditionExpr",
    "ConditionDefinition",
    "TrialStatsParams",
    "TrialStatsProcessing",
    "TrialStatsProcessingResult",
    "TrialStatsProcessingWriter",
    "TrialStatsWriterParams",
]
