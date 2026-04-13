from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..params import BaseTrialStatsParams, BaseTrialStatsWriterParams

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


class TrialActivitySummaryTableColumnSource(BaseModel):
    """Resolve response timing from a trial metadata column."""

    source: Literal["table_column"] = Field(default="table_column")
    column: str = Field(
        description="Resolved-trial metadata column containing response timing."
    )
    units: Literal["s", "ms"] = Field(
        default="s",
        description="Units used by the response timing column.",
    )

    @field_validator("column", mode="before")
    @classmethod
    def _validate_column(cls, value: object) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("trial_activity_summary.response.column must be non-empty.")
        return cleaned


class TrialActivitySummaryAnnotationEventSource(BaseModel):
    """Resolve response timing from annotations in the raw recording."""

    source: Literal["annotation_event_code"] = Field(default="annotation_event_code")
    event_code: str = Field(
        description="Annotation event code identifying the response event."
    )
    occurrence: Literal["first_after_anchor"] = Field(
        default="first_after_anchor",
        description=(
            "Response event selection policy. Only 'first_after_anchor' is "
            "currently supported."
        ),
    )

    @field_validator("event_code", mode="before")
    @classmethod
    def _validate_event_code(cls, value: object) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError(
                "trial_activity_summary.response.event_code must be non-empty."
            )
        return cleaned


TrialActivitySummaryResponseSource = Annotated[
    TrialActivitySummaryTableColumnSource | TrialActivitySummaryAnnotationEventSource,
    Field(discriminator="source"),
]


class TrialActivitySummaryConfig(BaseModel):
    """Configuration for the per-trial activity summary used by scatter plots."""

    kind: Literal["epoch_mean", "anchor_to_response_mean"] = Field(
        default="epoch_mean",
        description=(
            "Summary computed for each trial and feature. 'epoch_mean' averages the "
            "whole epoched window. 'anchor_to_response_mean' averages from t=0 to "
            "a response boundary resolved from either a trial metadata column or an "
            "annotation event code."
        ),
    )
    missing_response_policy: Literal["clamp_to_epoch", "drop_trial"] = Field(
        default="clamp_to_epoch",
        description=(
            "Policy applied when the resolved response boundary falls outside the "
            "epoched window or cannot produce a valid averaging interval."
        ),
    )
    response: TrialActivitySummaryResponseSource | None = Field(
        default=None,
        description=(
            "Response-boundary source used when kind='anchor_to_response_mean'."
        ),
    )

    @model_validator(mode="after")
    def _validate_response(self) -> "TrialActivitySummaryConfig":
        if self.kind == "anchor_to_response_mean" and self.response is None:
            raise ValueError(
                "trial_activity_summary.response must be provided when "
                "kind='anchor_to_response_mean'."
            )
        return self


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
    trial_activity_summary: TrialActivitySummaryConfig = Field(
        default_factory=TrialActivitySummaryConfig,
        description=(
            "Per-trial activity summary stored in the result and consumed by the "
            "channel-level scatter plot."
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

    @field_validator("trial_activity_summary", mode="before")
    @classmethod
    def _coerce_trial_activity_summary(
        cls,
        value: object,
    ) -> TrialActivitySummaryConfig | dict[str, object]:
        if value in (None, ""):
            return {}
        if isinstance(value, TrialActivitySummaryConfig):
            return value
        if not isinstance(value, dict):
            raise ValueError(
                "trial_activity_summary must be a mapping shaped like "
                "{'kind': 'epoch_mean' | 'anchor_to_response_mean', ...}."
            )
        return value

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
