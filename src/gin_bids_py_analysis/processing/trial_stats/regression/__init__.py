"""Regression subject-level trial statistics pipeline."""

from .params import (
    EpochCleaningConfig,
    PredictorAffineTransform,
    RegressionParams,
    RegressionWriterParams,
    TrialActivitySummaryAnnotationEventSource,
    TrialActivitySummaryConfig,
    TrialActivitySummaryTableColumnSource,
)
from .processor import RegressionProcessing
from .result import RegressionProcessingResult
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
    "RegressionProcessingResult",
    "RegressionWriterParams",
    "RegressionProcessingWriter",
    "load_regression_result",
]
