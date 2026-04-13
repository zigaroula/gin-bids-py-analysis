from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..params import (
    BaseTrialStatsParams,
    BaseTrialStatsWriterParams,
    TrialActivitySummaryAnnotationEventSource,
    TrialActivitySummaryConfig,
    TrialActivitySummaryTableColumnSource,
)

_VALID_ACTIVITY_ZSCORE = frozenset({"none", "baseline", "across_trials"})
_VALID_PVALUE_METHODS = frozenset({"none", "fdr_bh", "bonferroni"})


def _normalize_choice(value: object) -> str:
    return str(value).strip().lower()


class PredictorAffineTransform(BaseModel):
    """Affine transform applied to predictor values for one condition."""

    scale: float = Field(default=1.0)
    offset: float = Field(default=0.0)

    @field_validator("scale", "offset")
    @classmethod
    def _validate_finite(cls, value: float) -> float:
        if not float("-inf") < float(value) < float("inf"):
            raise ValueError("Predictor transform values must be finite.")
        return float(value)


class EpochCleaningConfig(BaseModel):
    """Configuration for epoch-level and channel-level quality control."""

    reject_trials_by_epoch_mean: bool = Field(
        default=False,
        description=(
            "Level A: NaN-mask (trial, channel) pairs where the trial epoch mean "
            "deviates more than epoch_mean_threshold_factor × std from the "
            "channel's across-trial mean."
        ),
    )
    epoch_mean_threshold_factor: float = Field(
        default=3.0,
        gt=0.0,
        description="Outlier threshold (in std) for Level A mean-based trial rejection.",
    )
    reject_trials_by_epoch_max: bool = Field(
        default=False,
        description=(
            "Level A: NaN-mask (trial, channel) pairs where the trial epoch maximum "
            "deviates more than epoch_max_threshold_factor × std from the "
            "channel's across-trial mean of maxima."
        ),
    )
    epoch_max_threshold_factor: float = Field(
        default=3.0,
        gt=0.0,
        description="Outlier threshold (in std) for Level A max-based trial rejection.",
    )
    reject_by_trial_mean_spread: bool = Field(
        default=False,
        description=(
            "Level B: exclude channels whose across-trial standard deviation of "
            "per-trial epoch means is an outlier in the channel population."
        ),
    )
    trial_mean_spread_threshold: float = Field(
        default=1.0,
        gt=0.0,
        description="Outlier threshold (in std) for Level B mean-spread channel rejection.",
    )
    reject_by_trial_max_spread: bool = Field(
        default=False,
        description=(
            "Level B: exclude channels whose across-trial standard deviation of "
            "per-trial epoch maxima is an outlier in the channel population."
        ),
    )
    trial_max_spread_threshold: float = Field(
        default=1.0,
        gt=0.0,
        description="Outlier threshold (in std) for Level B max-spread channel rejection.",
    )
    max_nan_trial_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Level B: exclude channels where the fraction of NaN trials meets or "
            "exceeds this threshold. None disables the check."
        ),
    )


class RegressionParams(BaseTrialStatsParams):
    """Parameters for subject-level slope regression on epoched iEEG data."""

    min_trials_per_condition: int = Field(
        default=3,
        ge=3,
        description=(
            "Minimum number of kept trials required in each condition to report non-NaN "
            "regression statistics."
        ),
    )
    predictor: str = Field(
        default="predictor_value",
        description=(
            "Resolved-trial key containing the continuous predictor value used in "
            "the per-condition regression."
        ),
    )
    predictor_transform_by_condition: dict[str, PredictorAffineTransform] = Field(
        default_factory=dict,
        description=(
            "Optional affine transform applied to the predictor by trial condition. "
            "Each entry is shaped as {condition_label: {scale: float, offset: float}}."
        ),
    )
    predictor_zscore: Literal["none", "condition", "global"] = Field(
        default="none",
        description=(
            "Predictor z-score mode applied within each condition after the affine transform."
        ),
    )
    epoch_cleaning: EpochCleaningConfig = Field(
        default_factory=EpochCleaningConfig,
        description=(
            "Epoch-level and channel-level quality control applied after epoch stacking "
            "and before activity z-scoring."
        ),
    )

    @field_validator("activity_zscore")
    @classmethod
    def _validate_activity_zscore(cls, value: str) -> str:
        cleaned = _normalize_choice(value)
        if cleaned not in _VALID_ACTIVITY_ZSCORE:
            raise ValueError(
                "activity_zscore must be one of 'none', 'baseline', or 'across_trials' "
                "for regression stats."
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

    @field_validator("predictor_transform_by_condition", mode="before")
    @classmethod
    def _coerce_predictor_transform_by_condition(
        cls,
        value: object,
    ) -> dict[str, dict[str, float]]:
        if value in (None, ""):
            return {}
        if not isinstance(value, dict):
            raise ValueError(
                "predictor_transform_by_condition must be a mapping shaped as "
                "{condition_label: {scale: float, offset: float}}."
            )
        cleaned: dict[str, dict[str, float]] = {}
        for raw_condition, raw_transform in value.items():
            condition = str(raw_condition).strip()
            if not condition:
                continue
            if raw_transform is None:
                cleaned[condition] = {}
                continue
            if not isinstance(raw_transform, dict):
                raise ValueError(
                    "Each predictor transform must be a mapping with optional "
                    "'scale' and 'offset' keys."
                )
            transform_dict: dict[str, float] = {}
            if "scale" in raw_transform:
                transform_dict["scale"] = float(raw_transform["scale"])
            if "offset" in raw_transform:
                transform_dict["offset"] = float(raw_transform["offset"])
            cleaned[condition] = transform_dict
        return cleaned

    @model_validator(mode="after")
    def _validate_regression(self) -> "RegressionParams":
        predictor_key = self.predictor.strip()
        if not predictor_key:
            raise ValueError("predictor must be a non-empty string.")
        self.predictor = predictor_key
        self.predictor_transform_by_condition = {
            str(condition).strip(): transform
            for condition, transform in self.predictor_transform_by_condition.items()
            if str(condition).strip()
        }
        return self


class RegressionWriterParams(BaseTrialStatsWriterParams):
    """Writer configuration for regression outputs."""

    pipeline_label: str = "regression"
    output_description: str = "regression"
