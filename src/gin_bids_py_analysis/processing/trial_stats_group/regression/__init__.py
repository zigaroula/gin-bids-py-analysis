"""Group-level ROI statistics built from subject-level regression outputs."""

from .params import RegressionGroupParams, RegressionGroupWriterParams
from .processor import RegressionGroupProcessing, build_regression_compatible_groups
from .result import (
    RegressionGroupProcessingResult,
    RegressionMetricStats,
    ScatterData,
    VsZeroStatsPair,
)
from .result_loader import load_regression_group_result
from .writer import RegressionGroupProcessingWriter

__all__ = [
    "RegressionGroupParams",
    "RegressionGroupProcessing",
    "RegressionGroupProcessingResult",
    "RegressionMetricStats",
    "ScatterData",
    "VsZeroStatsPair",
    "RegressionGroupProcessingWriter",
    "RegressionGroupWriterParams",
    "build_regression_compatible_groups",
    "load_regression_group_result",
]

