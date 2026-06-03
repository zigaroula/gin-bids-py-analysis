from __future__ import annotations

from ..writer import BaseTimeFrequencyStatsWriter


class TimeFrequencyConditionTestWriter(BaseTimeFrequencyStatsWriter):
    def _pipeline_name(self) -> str:
        return "time_frequency_condition_test"

