from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ..params import BaseTimeFrequencyStatsGroupParams, BaseTimeFrequencyStatsGroupWriterParams


class TimeFrequencyRegressionGroupParams(BaseTimeFrequencyStatsGroupParams):
    primary_regression_metric: Literal["t_values", "slope", "r_value"] = Field(default="t_values")
    contrast_mode: Literal["paired"] = "paired"

    @model_validator(mode="after")
    def _validate_cluster_metric(self) -> "TimeFrequencyRegressionGroupParams":
        if (
            self.p_value_correction_method == "cluster_permutation"
            and self.cluster_permutation_method == "custom"
            and self.primary_regression_metric != "slope"
        ):
            raise ValueError(
                "cluster_permutation/custom for TF regression requires "
                "primary_regression_metric='slope' because subject files store permuted_slopes."
            )
        return self


class TimeFrequencyRegressionGroupWriterParams(BaseTimeFrequencyStatsGroupWriterParams):
    pipeline_label: str = "time_frequency_regression_group"
    output_description: str = "tfregressiongroup"
