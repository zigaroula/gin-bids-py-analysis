from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from bidsforge.processing.base import BaseProcessingParams, BaseWriterParams
from bidsforge.processing.utils.field_names import sanitize_field_name, validate_field_name


PowerMode = Literal["stored", "raw", "baseline_corrected"]
TimeSelectionMode = Literal["strict", "inclusive"]


class BaseTimeFrequencyStatsParams(BaseProcessingParams):
    """Shared parameters for subject-level statistics on time-frequency derivatives."""

    anchor_event_codes: list[str] = Field(
        default_factory=list,
        description=(
            "Event codes used to map stored TFR trials back to behavior rows. "
            "When empty, one synthetic anchor is created per stored TFR trial and "
            "table rows are matched by order."
        ),
    )
    time_window_s: tuple[float, float] | None = Field(
        default=None,
        description="Optional time window applied to the stored TFR time axis.",
    )
    time_selection: TimeSelectionMode = Field(
        default="strict",
        description=(
            "'strict' keeps t > min and t < max. "
            "'inclusive' keeps t >= min and t <= max."
        ),
    )
    power_mode: PowerMode = Field(
        default="stored",
        description=(
            "Which TFR values to analyze: stored power_db, reconstructed raw dB, "
            "or baseline-corrected dB."
        ),
    )
    baseline_grand_average: bool = Field(
        default=False,
        description=(
            "Subtract each trial/channel/frequency time mean before statistics, "
            "using the average across the selected time axis."
        ),
    )
    condition_a: str = Field(default="condition_a")
    condition_b: str = Field(default="condition_b")
    min_trials_per_condition: int = Field(default=2, ge=2)
    p_value_correction_method: Literal["none", "fdr_bh", "bonferroni"] = Field(
        default="fdr_bh"
    )
    significance_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    n_permutations: int = Field(default=0, ge=0)
    permutation_seed: int | None = None

    @field_validator("anchor_event_codes", mode="before")
    @classmethod
    def _coerce_anchor_event_codes(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (str, int)):
            return [str(value)]
        return [str(item) for item in value]

    @model_validator(mode="after")
    def _validate_common(self) -> "BaseTimeFrequencyStatsParams":
        original_a, original_b = self.condition_a, self.condition_b
        self.condition_a = sanitize_field_name(original_a)
        validate_field_name(original_a, self.condition_a, context="condition_a")
        self.condition_b = sanitize_field_name(original_b)
        validate_field_name(original_b, self.condition_b, context="condition_b")
        if self.condition_a == self.condition_b:
            raise ValueError("condition_a and condition_b must be different.")
        if self.time_window_s is not None:
            start, stop = self.time_window_s
            if stop <= start:
                raise ValueError("time_window_s must be ordered as (start, stop).")
        return self


class BaseTimeFrequencyStatsWriterParams(BaseWriterParams):
    """Shared writer parameters for subject-level TF statistics outputs."""

    output_modality: str = "ieeg"
    output_suffix: str = "stats"
    output_format: Literal["hdf5", "matlab"] = "hdf5"
    include_epochs: bool = Field(
        default=False,
        description="When True, write condition-specific flattened TF epochs.",
    )

