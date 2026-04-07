from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from gin_bids_py_analysis.bids.helpers import normalize_subject_value
from gin_bids_py_analysis.processing.base import BaseProcessingParams, BaseWriterParams


class TrialSlopeStatsGroupParams(BaseProcessingParams):
    """Parameters for group-level ROI statistics on trial_slope_stats outputs."""

    p_value_correction_method: Literal["none", "fdr_bh", "bonferroni"] = Field(
        default="none",
        description=(
            "Multiple-comparisons correction applied across ROI x time tests, "
            "independently for each condition's slope."
        ),
    )
    significance_alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Threshold used to build significance masks from corrected p-values.",
    )
    roi_mode: Literal["atlas", "manual"] = Field(
        description="ROI definition mode. Exactly one mode is supported per run.",
    )
    atlas_name: str | None = Field(
        default=None,
        description="Electrodes-table column carrying ROI labels when roi_mode='atlas'.",
    )
    manual_region_channels: dict[str, dict[str, list[str]]] = Field(
        default_factory=dict,
        description=(
            "Manual ROI mapping for roi_mode='manual'. Expected shape: "
            "{roi_name: {subject_id: [channel_name, ...]}}."
        ),
    )
    min_channels_per_roi: int = Field(
        default=1,
        ge=1,
        description="Minimum pooled channel count required to keep a ROI in outputs.",
    )
    min_subjects_per_roi: int = Field(
        default=1,
        ge=1,
        description="Minimum unique subject count required to keep a ROI in outputs.",
    )

    @field_validator("manual_region_channels", mode="before")
    @classmethod
    def _coerce_manual_region_channels(
        cls,
        value: object,
    ) -> dict[str, dict[str, list[str]]]:
        if value in (None, ""):
            return {}
        if not isinstance(value, dict):
            raise ValueError(
                "manual_region_channels must be a mapping shaped as "
                "{roi: {subject: [channel, ...]}}."
            )

        cleaned: dict[str, dict[str, list[str]]] = {}
        for raw_roi, raw_subject_map in value.items():
            roi = str(raw_roi).strip()
            if not roi:
                continue
            if not isinstance(raw_subject_map, dict):
                raise ValueError(
                    f"manual_region_channels[{raw_roi!r}] must be a mapping of subjects."
                )

            subject_map: dict[str, list[str]] = {}
            for raw_subject, raw_channels in raw_subject_map.items():
                subject = normalize_subject_value(str(raw_subject))
                if not subject:
                    continue
                if raw_channels is None:
                    channels: list[str] = []
                elif isinstance(raw_channels, str):
                    channels = [raw_channels.strip()] if raw_channels.strip() else []
                else:
                    channels = [str(item).strip() for item in raw_channels if str(item).strip()]

                unique_channels: list[str] = []
                seen_channels: set[str] = set()
                for channel in channels:
                    key = channel.casefold()
                    if key in seen_channels:
                        continue
                    seen_channels.add(key)
                    unique_channels.append(channel)
                if unique_channels:
                    subject_map[subject] = unique_channels
            if subject_map:
                cleaned[roi] = subject_map
        return cleaned

    @model_validator(mode="after")
    def _validate_roi_mode(self) -> "TrialSlopeStatsGroupParams":
        if self.atlas_name is not None:
            self.atlas_name = self.atlas_name.strip() or None

        has_manual = bool(self.manual_region_channels)
        if self.roi_mode == "atlas":
            if self.atlas_name is None:
                raise ValueError("atlas_name is required when roi_mode='atlas'.")
            if has_manual:
                raise ValueError(
                    "manual_region_channels cannot be set when roi_mode='atlas'."
                )
            return self

        if self.atlas_name is not None:
            raise ValueError("atlas_name cannot be set when roi_mode='manual'.")
        if not has_manual:
            raise ValueError(
                "manual_region_channels is required when roi_mode='manual'."
            )
        return self


class TrialSlopeStatsGroupWriterParams(BaseWriterParams):
    """Writer configuration for group-level trial-slope-stats outputs."""

    pipeline_label: str = "trial_slope_stats_group"
    output_modality: str = "ieeg"
    output_suffix: str = "stats"
    output_description: str = "trialslopestatsgroup"
    output_format: Literal["hdf5", "matlab"] = Field(
        default="hdf5",
        description=(
            "Output format for group-level trial-slope-stats results. "
            "'hdf5' writes an HDF5 file (.h5); 'matlab' writes a MATLAB file (.mat)."
        ),
    )
