"""Shared group-level trial-statistics domain package."""

from .params import BaseTrialStatsGroupParams, BaseTrialStatsGroupWriterParams
from .processor import (
    BaseTrialStatsGroupContributionRecord,
    BaseTrialStatsGroupProcessing,
    BaseTrialStatsGroupSnapshot,
    BaseTrialStatsGroupSnapshotSignature,
)
from .result import BaseTrialStatsGroupProcessingResult
from .writer import BaseTrialStatsGroupProcessingWriter
from .condition_test import (
    ConditionTestGroupParams,
    ConditionTestGroupProcessing,
    ConditionTestGroupProcessingResult,
    ConditionTestGroupProcessingWriter,
    ConditionTestGroupWriterParams,
    build_condition_test_compatible_groups,
    load_condition_test_group_result,
)
from .regression import (
    RegressionGroupParams,
    RegressionGroupProcessing,
    RegressionGroupProcessingResult,
    RegressionGroupProcessingWriter,
    RegressionGroupWriterParams,
    build_regression_compatible_groups,
    load_regression_group_result,
)
from gin_bids_py_analysis.processing.utils.group_stats import ROIChannelContribution

__all__ = [
    "ROIChannelContribution",
    "BaseTrialStatsGroupParams",
    "BaseTrialStatsGroupWriterParams",
    "BaseTrialStatsGroupSnapshotSignature",
    "BaseTrialStatsGroupSnapshot",
    "BaseTrialStatsGroupContributionRecord",
    "BaseTrialStatsGroupProcessingResult",
    "BaseTrialStatsGroupProcessing",
    "BaseTrialStatsGroupProcessingWriter",
    "ConditionTestGroupParams",
    "ConditionTestGroupProcessing",
    "ConditionTestGroupProcessingResult",
    "ConditionTestGroupProcessingWriter",
    "ConditionTestGroupWriterParams",
    "build_condition_test_compatible_groups",
    "load_condition_test_group_result",
    "RegressionGroupParams",
    "RegressionGroupProcessing",
    "RegressionGroupProcessingResult",
    "RegressionGroupProcessingWriter",
    "RegressionGroupWriterParams",
    "build_regression_compatible_groups",
    "load_regression_group_result",
]
