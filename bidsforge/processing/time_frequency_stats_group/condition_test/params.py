from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..params import BaseTimeFrequencyStatsGroupParams, BaseTimeFrequencyStatsGroupWriterParams


class TimeFrequencyConditionTestGroupParams(BaseTimeFrequencyStatsGroupParams):
    primary_condition_metric: Literal[
        "t_values",
        "mean_difference",
        "condition_a_mean",
        "condition_b_mean",
    ] = Field(default="t_values")


class TimeFrequencyConditionTestGroupWriterParams(BaseTimeFrequencyStatsGroupWriterParams):
    pipeline_label: str = "time_frequency_condition_test_group"
    output_description: str = "tfconditiontestgroup"
