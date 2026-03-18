from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from gin_bids_py_analysis.processing.base import BaseProcessingParams, BaseWriterParams


class TrialStatsParams(BaseProcessingParams):
    """Parameters for subject-level trial statistics on ieeg data."""

    anchor_event_codes: list[str] = Field(
        description="Event codes used to define the trial anchor in ieeg annotations."
    )
    tmin_s: float = Field(description="Epoch start relative to the anchor event, in seconds.")
    tmax_s: float = Field(description="Epoch end relative to the anchor event, in seconds.")
    condition_a: str = Field(
        default="condition_a",
        description="Canonical label for the first trial condition.",
    )
    condition_b: str = Field(
        default="condition_b",
        description="Canonical label for the second trial condition.",
    )
    min_trials_per_condition: int = Field(
        default=2,
        ge=2,
        description="Minimum number of kept trials required in each condition to report non-NaN statistics.",
    )
    drop_partial_epochs: bool = Field(
        default=True,
        description="When True, exclude epochs that would extend outside the recording bounds.",
    )
    equal_var: bool = Field(
        default=False,
        description="Forwarded to scipy.stats.ttest_ind; False selects Welch's t-test.",
    )
    p_value_correction_method: Literal["none", "fdr_bh", "bonferroni"] = Field(
        default="fdr_bh",
        description=(
            "Multiple-comparisons correction for p-values across all channel x time tests. "
            "Use 'none' to disable correction."
        ),
    )
    significance_alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description=(
            "Significance threshold applied to (possibly corrected) p-values when building "
            "the significance mask."
        ),
    )

    @field_validator("anchor_event_codes", mode="before")
    @classmethod
    def _coerce_anchor_codes(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (str, int)):
            return [str(value)]
        return [str(item) for item in value]

    @model_validator(mode="after")
    def _check_window_and_labels(self) -> "TrialStatsParams":
        if not self.anchor_event_codes:
            raise ValueError("anchor_event_codes must contain at least one event code.")
        if self.tmax_s <= self.tmin_s:
            raise ValueError("tmax_s must be greater than tmin_s.")
        if self.condition_a == self.condition_b:
            raise ValueError("condition_a and condition_b must be different.")
        return self


class TrialStatsWriterParams(BaseWriterParams):
    """Writer configuration for trial statistics outputs."""

    pipeline_label: str = "trial_stats"
    output_modality: str = "ieeg"
    output_suffix: str = "stats"
    output_description: str = "trialstats"
    output_format: Literal["hdf5"] = Field(
        default="hdf5",
        description="The trial-stats writer always emits an HDF5 file plus a TSV companion trial table.",
    )
