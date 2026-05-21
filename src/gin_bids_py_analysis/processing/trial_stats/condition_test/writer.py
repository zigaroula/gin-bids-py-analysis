from __future__ import annotations

from ..writer import BaseTrialStatsProcessingWriter
from .result import ConditionTestProcessingResult


class ConditionTestProcessingWriter(BaseTrialStatsProcessingWriter):
    """Write condition-test statistics plus a companion TSV trial audit table."""

    def _pipeline_name(self) -> str:
        return "conditiontest"

    def _validate_result(self, result: ConditionTestProcessingResult) -> None:
        if not isinstance(result, ConditionTestProcessingResult):
            raise TypeError(
                f"Expected ConditionTestProcessingResult, got {type(result).__name__!r}"
            )
