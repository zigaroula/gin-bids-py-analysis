from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from gin_bids_py_analysis.processing.base import BaseProcessingParams, BaseWriterParams


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


class TrialSlopeStatsParams(BaseProcessingParams):
    """Parameters for subject-level slope regression on trial-epoched iEEG data."""

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
        default=3,
        ge=3,
        description=(
            "Minimum number of kept trials required in each condition to report non-NaN "
            "regression statistics."
        ),
    )
    drop_partial_epochs: bool = Field(
        default=True,
        description="When True, exclude epochs that would extend outside recording bounds.",
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
    predictor_scaling: Literal["none", "zscore_within_condition"] = Field(
        default="none",
        description=(
            "Predictor scaling mode applied within each condition after the affine "
            "transform. 'none' keeps transformed values; 'zscore_within_condition' "
            "z-scores the predictor across kept trials within each condition."
        ),
    )
    p_value_correction_method: Literal["none", "fdr_bh", "bonferroni"] = Field(
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
            "When unset, regression is computed channel-by-channel."
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
            "non-overlapping bins before regression."
        ),
    )
    n_bins: int = Field(
        default=0,
        ge=-1,
        description=(
            "Number of contiguous non-overlapping temporal bins. 0 or -1 disables count-based "
            "binning. Cannot be combined with window_ms > 0."
        ),
    )
    activity_scaling: Literal["none", "zscore_by_baseline"] = Field(
        default="none",
        description=(
            "Optional scaling applied to the input activity before per-condition regression. "
            "'none' keeps the original units. "
            "'zscore_by_baseline' z-scores activity relative to a pooled baseline window, "
            "independently for each channel/ROI."
        ),
    )
    activity_baseline_tmin_s: float = Field(
        default=-0.2,
        description=(
            "Baseline window start in seconds, relative to the anchor event. "
            "Used only when activity_scaling='zscore_by_baseline'."
        ),
    )
    activity_baseline_tmax_s: float = Field(
        default=0.0,
        description=(
            "Baseline window end in seconds, relative to the anchor event. "
            "Used only when activity_scaling='zscore_by_baseline'."
        ),
    )
    experiment_start_event_code: str | None = Field(
        default=None,
        description=(
            "Event code marking the experiment start. "
            "The FIRST occurrence of this code in the recording defines the left boundary. "
            "Anchor events with onset_s <= that boundary are excluded. "
            "When absent or not found in the recording no filtering is applied. "
            "The epoch window may still extend before the boundary."
        ),
    )
    experiment_end_event_code: str | None = Field(
        default=None,
        description=(
            "Event code marking the experiment end. "
            "The LAST occurrence of this code in the recording defines the right boundary. "
            "Anchor events with onset_s >= that boundary are excluded. "
            "When absent or not found in the recording no filtering is applied. "
            "The epoch window may still extend after the boundary."
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
    def _validate(self) -> "TrialSlopeStatsParams":
        if not self.anchor_event_codes:
            raise ValueError("anchor_event_codes must contain at least one event code.")
        if self.tmax_s <= self.tmin_s:
            raise ValueError("tmax_s must be greater than tmin_s.")
        if self.condition_a == self.condition_b:
            raise ValueError("condition_a and condition_b must be different.")
        predictor_key = self.predictor.strip()
        if not predictor_key:
            raise ValueError("predictor must be a non-empty string.")
        self.predictor = predictor_key
        self.predictor_transform_by_condition = {
            str(condition).strip(): transform
            for condition, transform in self.predictor_transform_by_condition.items()
            if str(condition).strip()
        }
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
        if self.activity_scaling == "zscore_by_baseline":
            if self.activity_baseline_tmin_s < self.tmin_s:
                raise ValueError(
                    "activity_baseline_tmin_s must be within the epoch window when "
                    "activity_scaling='zscore_by_baseline'."
                )
            if self.activity_baseline_tmax_s > self.tmax_s:
                raise ValueError(
                    "activity_baseline_tmax_s must be within the epoch window when "
                    "activity_scaling='zscore_by_baseline'."
                )
        return self


class TrialSlopeStatsWriterParams(BaseWriterParams):
    """Writer configuration for trial slope statistics outputs."""

    pipeline_label: str = "trial_slope_stats"
    output_modality: str = "ieeg"
    output_suffix: str = "stats"
    output_description: str = "trialslopestats"
    output_format: Literal["hdf5", "matlab"] = Field(
        default="hdf5",
        description=(
            "Output format for trial-slope results. 'hdf5' writes an HDF5 file (.h5); "
            "'matlab' writes a MATLAB file (.mat). A TSV companion trial table is written "
            "in both cases."
        ),
    )
    include_epochs: bool = Field(
        default=False,
        description=(
            "When True, individual per-trial epoch arrays are written to the output file in "
            "addition to summary and regression statistics."
        ),
    )


