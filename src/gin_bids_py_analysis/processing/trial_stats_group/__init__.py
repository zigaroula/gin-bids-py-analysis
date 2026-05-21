"""Shared group-level trial-statistics domain package."""

from .params import BaseTrialStatsGroupParams, BaseTrialStatsGroupWriterParams
from .processor import (
    BaseTrialStatsGroupContributionRecord,
    BaseTrialStatsGroupProcessing,
    BaseTrialStatsGroupSnapshot,
    BaseTrialStatsGroupSnapshotSignature,
)
from .result import (
    BaseTrialStatsGroupProcessingResult,
    GroupEpochStats,
    GroupEstimate,
    GroupEstimatePair,
    GroupTimecourseStats,
    IndexedConditionContributions,
)
from .writer import BaseTrialStatsGroupProcessingWriter
from .condition_test import (
    ConditionTestEpochSummary,
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
    RegressionSourceMetricStats,
    ScatterData,
    VsZeroStatsPair,
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
    "GroupEpochStats",
    "GroupEstimate",
    "GroupEstimatePair",
    "GroupTimecourseStats",
    "IndexedConditionContributions",
    "BaseTrialStatsGroupProcessing",
    "BaseTrialStatsGroupProcessingWriter",
    "ConditionTestEpochSummary",
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
    "RegressionSourceMetricStats",
    "ScatterData",
    "VsZeroStatsPair",
    "build_regression_compatible_groups",
    "load_regression_group_result",
]
