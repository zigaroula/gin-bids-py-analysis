from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from ..params import BaseTrialStatsGroupParams, BaseTrialStatsGroupWriterParams


class ConditionTestGroupParams(BaseTrialStatsGroupParams):
    """Parameters for group-level ROI one-sample statistics on condition_test outputs."""

    source_metric: Literal[
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
    n_group_permutations: int = Field(
        default=10000,
        ge=1,
        description=(
            "Number of group-level null iterations when method='cluster_permutation'. "
            "Ignored for all other correction methods."
        ),
    )
    cluster_threshold_alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description=(
            "Significance threshold applied to the group one-sample t-test within each "
            "null iteration to detect temporal clusters. Used only when "
            "p_value_correction_method='cluster_permutation'."
        ),
    )
    permutation_seed: int | None = Field(
        default=None,
        description=(
            "Seed for the NumPy random generator used when building the cluster null "
            "distribution. None selects a non-reproducible seed."
        ),
    )
    cluster_permutation_method: Literal["custom", "mne"] = Field(
        default="custom",
        description=(
            "Strategy for building the group-level cluster null distribution when "
            "p_value_correction_method='cluster_permutation'. 'custom' samples from "
            "stored per-channel permuted t-value pools. 'mne' runs "
            "mne.stats.permutation_cluster_1samp_test directly on contribution timecourses."
        ),
    )

    @field_validator("cluster_permutation_method", mode="before")
    @classmethod
    def _validate_cluster_method_rename(cls, value: object) -> object:
        if value in {"hierarchical", "sign_flip"}:
            raise ValueError(
                "cluster_permutation_method values 'hierarchical' and 'sign_flip' were "
                "renamed to 'custom' and 'mne'."
            )
        return value


class ConditionTestGroupWriterParams(BaseTrialStatsGroupWriterParams):
    """Writer configuration for group-level condition_test outputs."""

    pipeline_label: str = "condition_test_group"
    output_description: str = "conditiontestgroup"
