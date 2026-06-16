from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from ..params import BaseTrialStatsParams, BaseTrialStatsWriterParams

_VALID_ACTIVITY_ZSCORE = frozenset({"none", "baseline"})
_VALID_PVALUE_METHODS = frozenset({"none", "fdr_bh", "bonferroni"})


def _normalize_choice(value: object) -> str:
    return str(value).strip().lower()


class ConditionTestParams(BaseTrialStatsParams):
    """Parameters for subject-level condition-test statistics on iEEG data."""

    equal_var: bool = Field(
        default=False,
        description="Forwarded to scipy.stats.ttest_ind; False selects Welch's t-test.",
    )
    channel_significance_mode: Literal["none", "single_bin", "duration"] = Field(
        default="none",
        description=(
            "Method used to derive a per-channel significance flag "
            "(``channel_significant_mask``, shape ``(n_channels,)``)."
        ),
    )
    channel_significance_duration_threshold_ms: float = Field(
        default=100.0,
        gt=0.0,
        description=(
            "Minimum total significant duration (in milliseconds) required to flag a channel "
            "as significant in 'duration' mode."
        ),
    )
    difference_ci95_n_bootstraps: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of bootstrap resamples used to estimate 95% confidence intervals "
            "for mean(condition_a) - mean(condition_b). 0 uses a fast parametric "
            "CI consistent with the selected t-test variance model."
        ),
    )

    @field_validator("activity_zscore")
    @classmethod
    def _validate_activity_zscore(cls, value: str) -> str:
        cleaned = _normalize_choice(value)
        if cleaned not in _VALID_ACTIVITY_ZSCORE:
            raise ValueError(
                "activity_zscore must be one of 'none' or 'baseline' for condition-test stats."
            )
        return cleaned

    @field_validator("p_value_correction_method")
    @classmethod
    def _validate_pvalue_correction_method(cls, value: str) -> str:
        cleaned = _normalize_choice(value)
        if cleaned not in _VALID_PVALUE_METHODS:
            raise ValueError(
                "p_value_correction_method must be one of "
                "'none', 'fdr_bh', or 'bonferroni'."
            )
        return cleaned

    @model_validator(mode="after")
    def _validate_condition_test(self) -> "ConditionTestParams":
        return self


class ConditionTestWriterParams(BaseTrialStatsWriterParams):
    """Writer configuration for condition-test outputs."""

    pipeline_label: str = "condition_test"
    output_description: str = "conditiontest"
