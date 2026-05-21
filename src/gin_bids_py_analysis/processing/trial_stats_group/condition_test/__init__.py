"""Group-level ROI statistics built from subject-level condition_test outputs."""

from .params import ConditionTestGroupParams, ConditionTestGroupWriterParams
from .processor import (
    ConditionTestGroupProcessing,
    build_condition_test_compatible_groups,
)
from .result import ConditionTestEpochSummary, ConditionTestGroupProcessingResult
from .result_loader import load_condition_test_group_result
from .writer import ConditionTestGroupProcessingWriter

__all__ = [
    "ConditionTestGroupParams",
    "ConditionTestEpochSummary",
    "ConditionTestGroupProcessing",
    "ConditionTestGroupProcessingResult",
    "ConditionTestGroupProcessingWriter",
    "ConditionTestGroupWriterParams",
    "build_condition_test_compatible_groups",
    "load_condition_test_group_result",
]
