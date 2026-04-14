from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

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
    p_value_correction_method: Literal["none", "fdr_bh", "bonferroni", "cluster_permutation"] = Field(
        default="none",
        description=(
            "Multiple-comparisons correction applied across ROI x time tests, "
            "independently for each condition's slope and activity contrast. "
            "Use 'cluster_permutation' for cluster-based permutation testing "
            "(requires contrast_mode='paired' and subject files computed with n_permutations > 0)."
        ),
    )

    @model_validator(mode="after")
    def _validate_cluster_permutation_compat(self) -> "RegressionGroupParams":
        if (
            self.p_value_correction_method == "cluster_permutation"
            and self.contrast_mode == "unpaired"
        ):
            raise ValueError(
                "cluster_permutation is only supported with contrast_mode='paired'. "
                "Unpaired cluster permutation is not currently implemented."
            )
        return self


class RegressionGroupWriterParams(BaseTrialStatsGroupWriterParams):
    """Writer configuration for group-level regression outputs."""

    pipeline_label: str = "regression_group"
    output_description: str = "regressiongroup"
