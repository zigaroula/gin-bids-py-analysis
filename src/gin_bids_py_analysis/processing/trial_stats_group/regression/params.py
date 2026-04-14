from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..params import BaseTrialStatsGroupParams, BaseTrialStatsGroupWriterParams


class RegressionGroupParams(BaseTrialStatsGroupParams):
    """Parameters for group-level ROI statistics on regression outputs."""

    source_metric: Literal["slope", "r_value"] = Field(
        default="slope",
        description=(
            "Channel-level metric read from each subject regression file and used in "
            "the group contrast."
        ),
    )
    contrast_mode: Literal["paired", "unpaired"] = Field(
        default="paired",
        description=(
            "Contrast mode used to compare condition_a and condition_b at group level."
        ),
    )
    p_value_correction_method: Literal["none", "fdr_bh", "bonferroni"] = Field(
        default="none",
        description=(
            "Multiple-comparisons correction applied across ROI x time tests, "
            "independently for each condition's slope and activity contrast."
        ),
    )


class RegressionGroupWriterParams(BaseTrialStatsGroupWriterParams):
    """Writer configuration for group-level regression outputs."""

    pipeline_label: str = "regression_group"
    output_description: str = "regressiongroup"
