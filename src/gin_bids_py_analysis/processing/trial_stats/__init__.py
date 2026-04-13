"""Shared trial-statistics domain package."""

from .params import BaseTrialStatsParams, BaseTrialStatsWriterParams
from .processor import BaseTrialStatsProcessing
from .result import BaseTrialStatsProcessingResult
from .writer import BaseTrialStatsProcessingWriter
from .condition_test import (
    ConditionTestParams,
    ConditionTestProcessing,
    ConditionTestProcessingResult,
    ConditionTestProcessingWriter,
    ConditionTestWriterParams,
    load_condition_test_result,
)
from .regression import (
    EpochCleaningConfig,
    PredictorAffineTransform,
    RegressionParams,
    RegressionProcessing,
    RegressionProcessingResult,
    RegressionProcessingWriter,
    RegressionWriterParams,
    TrialActivitySummaryAnnotationEventSource,
    TrialActivitySummaryConfig,
    TrialActivitySummaryTableColumnSource,
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
    "BaseTrialStatsProcessing",
    "BaseTrialStatsProcessingWriter",
    "ConditionTestParams",
    "ConditionTestProcessing",
    "ConditionTestProcessingResult",
    "ConditionTestProcessingWriter",
    "ConditionTestWriterParams",
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
