from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from gin_bids_py_analysis.processing.base import BaseProcessingParams, BaseWriterParams


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
    """Configuration for the per-trial activity summary saved with results."""

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


class BaseTrialStatsParams(BaseProcessingParams):
    """Shared parameter surface for subject-level trial statistics pipelines."""

    anchor_event_codes: list[str] = Field(
        description="Event codes used to define trial anchors in iEEG annotations."
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
        description=(
            "Minimum number of kept trials required in each condition to report "
            "non-NaN subject-level statistics."
        ),
    )
    drop_partial_epochs: bool = Field(
        default=True,
        description="When True, exclude epochs that would extend outside recording bounds.",
    )
    p_value_correction_method: str = Field(
        default="fdr_bh",
        description="Multiple-comparisons correction applied across feature x time tests.",
    )
    significance_alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Significance threshold applied to corrected p-values.",
    )
    atlas_name: str | None = Field(
        default=None,
        description=(
            "Column name in *_electrodes.tsv used to group channels into ROI regions. "
            "When unset, statistics are computed channel-by-channel."
        ),
    )
    atlas_regions: list[str] = Field(
        default_factory=list,
        description="Optional subset of atlas regions to include when atlas_name is set.",
    )
    window_ms: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Temporal bin size in milliseconds. 0 disables window binning; >0 averages "
            "non-overlapping bins before subject-level statistics."
        ),
    )
    n_bins: int = Field(
        default=0,
        ge=-1,
        description=(
            "Number of contiguous non-overlapping temporal bins. 0 or -1 disables "
            "count-based binning. Cannot be combined with window_ms > 0."
        ),
    )
    activity_zscore: str = Field(
        default="none",
        description="Optional activity z-score mode applied before subject-level statistics.",
    )
    activity_baseline_tmin_s: float = Field(
        default=-0.2,
        description=(
            "Baseline window start in seconds, relative to the anchor event. "
            "Used only when activity_zscore='baseline'."
        ),
    )
    activity_baseline_tmax_s: float = Field(
        default=0.0,
        description=(
            "Baseline window end in seconds, relative to the anchor event. "
            "Used only when activity_zscore='baseline'."
        ),
    )
    activity_baseline_scope: Literal["trial", "condition", "global"] = Field(
        default="global",
        description=(
            "Scope used to build the baseline reference when activity_zscore='baseline'."
        ),
    )
    activity_baseline_remove_outlier_trial_means: bool = Field(
        default=False,
        description=(
            "When True and activity_zscore='baseline' with scope 'condition' or 'global', "
            "remove outlier baseline trial-means before estimating the baseline reference."
        ),
    )
    experiment_start_event_code: str | None = Field(
        default=None,
        description=(
            "Event code marking the experiment start. The first occurrence defines the "
            "left boundary for anchor filtering."
        ),
    )
    experiment_end_event_code: str | None = Field(
        default=None,
        description=(
            "Event code marking the experiment end. The last occurrence defines the "
            "right boundary for anchor filtering."
        ),
    )
    trial_activity_summary: TrialActivitySummaryConfig = Field(
        default_factory=TrialActivitySummaryConfig,
        description=(
            "Per-trial activity summary computed from the kept epoched data and saved "
            "in the subject-level result."
        ),
    )
    n_permutations: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of condition-label permutations to compute and store in the output. "
            "0 disables permutation tests entirely."
        ),
    )
    permutation_seed: int | None = Field(
        default=None,
        description=(
            "Seed for the NumPy random generator used during permutation testing. "
            "None selects a non-reproducible random seed."
        ),
    )

    @field_validator("experiment_start_event_code", "experiment_end_event_code", mode="before")
    @classmethod
    def _coerce_boundary_codes(cls, value: object) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned if cleaned else None

    @field_validator("anchor_event_codes", mode="before")
    @classmethod
    def _coerce_anchor_codes(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (str, int)):
            return [str(value)]
        return [str(item) for item in value]

    @field_validator("atlas_regions", mode="before")
    @classmethod
    def _coerce_atlas_regions(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            cleaned = value.strip()
            return [cleaned] if cleaned else []
        return [str(item).strip() for item in value if str(item).strip()]

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
    def _validate_common(self) -> "BaseTrialStatsParams":
        if not self.anchor_event_codes:
            raise ValueError("anchor_event_codes must contain at least one event code.")
        if self.tmax_s <= self.tmin_s:
            raise ValueError("tmax_s must be greater than tmin_s.")
        if self.condition_a == self.condition_b:
            raise ValueError("condition_a and condition_b must be different.")
        if self.atlas_name is not None:
            self.atlas_name = self.atlas_name.strip() or None
        if self.atlas_regions and not self.atlas_name:
            raise ValueError("atlas_regions requires atlas_name to be set.")
        if self.n_bins < 0:
            self.n_bins = 0
        if self.window_ms > 0 and self.n_bins > 0:
            raise ValueError("window_ms and n_bins are mutually exclusive; define only one.")
        if self.activity_baseline_tmax_s <= self.activity_baseline_tmin_s:
            raise ValueError(
                "activity_baseline_tmax_s must be greater than activity_baseline_tmin_s."
            )
        if self.activity_zscore == "baseline":
            if self.activity_baseline_tmin_s < self.tmin_s:
                raise ValueError(
                    "activity_baseline_tmin_s must be within the epoch window when "
                    "activity_zscore='baseline'."
                )
            if self.activity_baseline_tmax_s > self.tmax_s:
                raise ValueError(
                    "activity_baseline_tmax_s must be within the epoch window when "
                    "activity_zscore='baseline'."
                )
        return self


class BaseTrialStatsWriterParams(BaseWriterParams):
    """Shared writer configuration for subject-level trial statistics outputs."""

    output_modality: str = "ieeg"
    output_suffix: str = "stats"
    output_format: Literal["hdf5", "matlab"] = Field(
        default="hdf5",
        description=(
            "Output format for subject-level trial statistics results. "
            "'hdf5' writes an HDF5 file (.h5); 'matlab' writes a MATLAB file (.mat). "
            "A TSV companion trial table is written in both cases."
        ),
    )
    include_epochs: bool = Field(
        default=False,
        description=(
            "When True, individual per-trial epoch arrays are written to the output file "
            "in addition to summary statistics."
        ),
    )
