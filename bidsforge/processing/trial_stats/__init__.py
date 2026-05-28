"""Shared trial-statistics domain package."""

from .params import (
    BaseTrialStatsParams,
    BaseTrialStatsWriterParams,
    EpochCleaningConfig,
    TrialActivitySummaryAnnotationEventSource,
    TrialActivitySummaryConfig,
    TrialActivitySummaryTableColumnSource,
)
from .processor import BaseTrialStatsProcessing
from .result import (
    BaseTrialStatsProcessingResult,
    ConditionEpochs,
    ConditionSignalActivity,
    ConditionTrialSummaryValues,
    SignalActivityEstimate,
)
from .writer import BaseTrialStatsProcessingWriter
from .condition_test import (
    ConditionContrast,
    ConditionTestParams,
    ConditionTestProcessing,
    ConditionTestProcessingResult,
    ConditionTestProcessingWriter,
    ConditionTestWriterParams,
    DifferenceEstimate,
    load_condition_test_result,
)
from .regression import (
    PredictorAffineTransform,
    RegressionParams,
    RegressionProcessing,
    RegressionProcessingResult,
    RegressionProcessingWriter,
    RegressionWriterParams,
    load_regression_result,
)
from ..utils.condition_rules import ConditionDefinition, ConditionExpr
from ..utils.trial_resolver import TableTrialResolver, TrialResolver

__all__ = [
    "TrialResolver",
    "TableTrialResolver",
    "ConditionExpr",
    "ConditionDefinition",
    "BaseTrialStatsParams",
    "BaseTrialStatsWriterParams",
    "BaseTrialStatsProcessingResult",
    "ConditionEpochs",
    "ConditionSignalActivity",
    "ConditionTrialSummaryValues",
    "SignalActivityEstimate",
    "BaseTrialStatsProcessing",
    "BaseTrialStatsProcessingWriter",
    "ConditionContrast",
    "ConditionTestParams",
    "ConditionTestProcessing",
    "ConditionTestProcessingResult",
    "ConditionTestProcessingWriter",
    "ConditionTestWriterParams",
    "DifferenceEstimate",
    "load_condition_test_result",
    "PredictorAffineTransform",
    "TrialActivitySummaryTableColumnSource",
    "TrialActivitySummaryAnnotationEventSource",
    "TrialActivitySummaryConfig",
    "EpochCleaningConfig",
    "RegressionParams",
    "RegressionProcessing",
    "RegressionProcessingResult",
    "RegressionProcessingWriter",
    "RegressionWriterParams",
    "load_regression_result",
]
