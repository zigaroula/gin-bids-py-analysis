"""Regression subject-level trial statistics pipeline."""

from ..params import (
    EpochCleaningConfig,
    TrialActivitySummaryAnnotationEventSource,
    TrialActivitySummaryConfig,
    TrialActivitySummaryTableColumnSource,
)
from .params import (
    PredictorAffineTransform,
    RegressionParams,
    RegressionWriterParams,
)
from .processor import RegressionProcessing
from .result import (
    ConditionPredictorValues,
    ConditionRegressionStats,
    RegressionPredictor,
    RegressionProcessingResult,
    RegressionStats,
)
from .result_loader import load_regression_result
from .writer import RegressionProcessingWriter

__all__ = [
    "PredictorAffineTransform",
    "TrialActivitySummaryTableColumnSource",
    "TrialActivitySummaryAnnotationEventSource",
    "TrialActivitySummaryConfig",
    "EpochCleaningConfig",
    "RegressionParams",
    "RegressionProcessing",
    "ConditionPredictorValues",
    "ConditionRegressionStats",
    "RegressionPredictor",
    "RegressionProcessingResult",
    "RegressionStats",
    "RegressionWriterParams",
    "RegressionProcessingWriter",
    "load_regression_result",
]
