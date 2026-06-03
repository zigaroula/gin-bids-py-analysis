from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..params import BaseTimeFrequencyStatsParams, BaseTimeFrequencyStatsWriterParams


class TimeFrequencyConditionTestParams(BaseTimeFrequencyStatsParams):
    """Parameters for condition A vs B statistics on stored TFR derivatives."""

    equal_var: bool = Field(
        default=False,
        description="Forwarded to scipy.stats.ttest_ind; False selects Welch's t-test.",
    )
    compute_grand_average: bool = Field(
        default=False,
        description="Compute all-kept-trial mean/std and one-sample t-test vs zero.",
    )


class TimeFrequencyConditionTestWriterParams(BaseTimeFrequencyStatsWriterParams):
    """Writer configuration for TF condition-test outputs."""

    pipeline_label: str = "time_frequency_condition_test"
    output_description: str = "tfconditiontest"
    output_format: Literal["hdf5", "matlab"] = "hdf5"

