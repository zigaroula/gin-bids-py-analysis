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
    p_value_correction_method: Literal["none", "fdr_bh", "bonferroni", "permutation"] = Field(
        default="fdr_bh",
        description=(
            "Multiple-comparisons correction for p-values across all channel x time tests. "
            "'fdr_bh' and 'bonferroni' use standard parametric approaches. "
            "'permutation' computes pointwise p-values from the null t-distribution built by "
            "randomly shuffling condition labels (requires n_permutations > 0). "
            "Use 'none' to disable correction."
        ),
    )
    n_permutations: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of condition-label permutations to compute and store in the output. "
            "0 disables permutation tests entirely. "
            "A positive value generates a null t-value distribution for use as a standalone "
            "correction method ('permutation') and/or as input to the group-level cluster "
            "permutation test in trial_stats_group."
        ),
    )
    permutation_seed: int | None = Field(
        default=None,
        description=(
            "Seed for the NumPy random generator used during permutation testing. "
            "None selects a non-reproducible random seed."
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
    atlas_name: str | None = Field(
        default=None,
        description=(
            "Column name in *_electrodes.tsv used to group channels into ROI regions. "
            "When unset, statistics are computed channel-by-channel."
        ),
    )
    atlas_regions: list[str] = Field(
        default_factory=list,
        description=(
            "Optional subset of atlas region labels to include. "
            "Used only when atlas_name is set."
        ),
    )
    window_ms: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Temporal bin size in milliseconds. "
            "0 disables window-based binning; >0 averages non-overlapping bins before stats. "
            "If binning would leave a trailing 1-sample tail (common with inclusive epoch bounds), "
            "that sample is merged into the previous bin. "
            "Cannot be combined with n_bins > 0."
        ),
    )
    n_bins: int = Field(
        default=0,
        ge=-1,
        description=(
            "Number of contiguous non-overlapping temporal bins. "
            "0 or -1 disables count-based binning (sample-by-sample unless window_ms is used). "
            "Cannot be combined with window_ms > 0."
        ),
    )
    channel_significance_mode: Literal["none", "single_bin", "duration"] = Field(
        default="none",
        description=(
            "Method used to derive a per-channel significance flag "
            "(``channel_significant_mask``, shape ``(n_channels,)``). "
            "'none' disables per-channel significance (default, fully backward-compatible). "
            "'single_bin' collapses each epoch to its temporal mean and runs a t-test across "
            "trials, equivalent to running the pipeline with n_bins=1; "
            "the same p_value_correction_method is applied across channels. "
            "'duration' sums the duration of all significant bins in "
            "``significant_mask`` and flags a channel if that total meets "
            "``channel_significance_duration_threshold_ms``."
        ),
    )
    channel_significance_duration_threshold_ms: float = Field(
        default=100.0,
        gt=0.0,
        description=(
            "Minimum total significant duration (in milliseconds) required to flag a channel "
            "as significant in 'duration' mode. Each significant bin contributes its full bin "
            "duration to the channel total. "
            "Only used when channel_significance_mode='duration'."
        ),
    )
    activity_scaling: Literal["none", "zscore_by_baseline"] = Field(
        default="none",
        description=(
            "Optional scaling applied to the input activity before subject-level statistics. "
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

    @model_validator(mode="after")
    def _check_window_and_labels(self) -> "TrialStatsParams":
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
            raise ValueError(
                "window_ms and n_bins are mutually exclusive; define only one."
            )
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
        if self.p_value_correction_method == "permutation" and self.n_permutations == 0:
            raise ValueError(
                "p_value_correction_method='permutation' requires n_permutations > 0."
            )
        return self


class TrialStatsWriterParams(BaseWriterParams):
    """Writer configuration for trial statistics outputs."""

    pipeline_label: str = "trial_stats"
    output_modality: str = "ieeg"
    output_suffix: str = "stats"
    output_description: str = "trialstats"
    output_format: Literal["hdf5", "matlab"] = Field(
        default="hdf5",
        description=(
            "Output format for trial-stats results. "
            "'hdf5' writes an HDF5 file (.h5); 'matlab' writes a MATLAB file (.mat). "
            "A TSV companion trial table is written in both cases."
        ),
    )
    include_epochs: bool = Field(
        default=False,
        description=(
            "When True, the individual per-trial epoch arrays (condition_a_epochs and "
            "condition_b_epochs) are written to the output file in addition to the "
            "per-channel means and statistics. This can significantly increase file size."
        ),
    )
