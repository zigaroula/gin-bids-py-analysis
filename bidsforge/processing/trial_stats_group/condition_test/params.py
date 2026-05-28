from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from ..params import BaseTrialStatsGroupParams, BaseTrialStatsGroupWriterParams


class ConditionTestGroupParams(BaseTrialStatsGroupParams):
    """Parameters for group-level ROI one-sample statistics on condition_test outputs."""

    primary_condition_metric: Literal[
        "mean_difference",
        "t_values",
        "condition_a_mean",
        "condition_b_mean",
    ] = Field(
        default="mean_difference",
        description=(
            "Channel-level metric read from each subject condition_test file and "
            "tested against 0."
        ),
    )
    p_value_correction_method: Literal[
        "none",
        "fdr_bh",
        "bonferroni",
        "cluster_permutation",
    ] = Field(
        default="none",
        description=(
            "Multiple-comparisons correction across ROI x time tests. "
            "'cluster_permutation' builds a null distribution of maximum temporal-cluster "
            "t-sums by drawing from per-channel permuted t-values stored in each source "
            "condition_test file."
        ),
    )



class ConditionTestGroupWriterParams(BaseTrialStatsGroupWriterParams):
    """Writer configuration for group-level condition_test outputs."""

    pipeline_label: str = "condition_test_group"
    output_description: str = "conditiontestgroup"
