from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from bidsforge.bids.helpers import normalize_subject_value
from bidsforge.processing.base import BaseProcessingParams, BaseWriterParams
from bidsforge.processing.utils.field_names import sanitize_field_name, validate_field_name


class BaseTimeFrequencyStatsGroupParams(BaseProcessingParams):
    significance_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    roi_mode: Literal["atlas", "manual"]
    atlas_name: str | None = None
    manual_region_channels: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    min_channels_per_roi: int = Field(default=1, ge=1)
    min_subjects_per_roi: int = Field(default=1, ge=1)
    p_value_correction_method: Literal[
        "none",
        "fdr_bh",
        "bonferroni",
        "cluster_permutation",
    ] = "none"
    cluster_permutation_method: Literal["custom", "sign_flip"] = "custom"
    cluster_threshold_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    cluster_percentile_alpha: float = Field(default=0.005, gt=0.0, lt=0.5)
    n_group_permutations: int = Field(default=10000, ge=1)
    permutation_seed: int | None = None

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
                "manual_region_channels must be shaped as {roi: {subject: [channel, ...]}}."
            )
        cleaned: dict[str, dict[str, list[str]]] = {}
        for raw_roi, raw_subjects in value.items():
            roi = sanitize_field_name(str(raw_roi).strip())
            validate_field_name(str(raw_roi).strip(), roi, context="ROI name")
            if not isinstance(raw_subjects, dict):
                raise ValueError(f"manual_region_channels[{raw_roi!r}] must map subjects.")
            subject_map: dict[str, list[str]] = {}
            for raw_subject, raw_channels in raw_subjects.items():
                subject = normalize_subject_value(str(raw_subject))
                if isinstance(raw_channels, str):
                    channels = [raw_channels.strip()] if raw_channels.strip() else []
                else:
                    channels = [str(ch).strip() for ch in (raw_channels or []) if str(ch).strip()]
                unique: list[str] = []
                seen: set[str] = set()
                for channel in channels:
                    key = channel.casefold()
                    if key not in seen:
                        seen.add(key)
                        unique.append(channel)
                if unique:
                    subject_map[subject] = unique
            if subject_map:
                if roi in cleaned:
                    raise ValueError(f"Duplicate sanitized ROI name {roi!r}.")
                cleaned[roi] = subject_map
        return cleaned

    @model_validator(mode="after")
    def _validate_roi_mode(self) -> "BaseTimeFrequencyStatsGroupParams":
        if self.atlas_name is not None:
            self.atlas_name = self.atlas_name.strip() or None
        if self.roi_mode == "atlas":
            if not self.atlas_name:
                raise ValueError("atlas_name is required when roi_mode='atlas'.")
            if self.manual_region_channels:
                raise ValueError("manual_region_channels cannot be set with roi_mode='atlas'.")
            return self
        if self.atlas_name is not None:
            raise ValueError("atlas_name cannot be set when roi_mode='manual'.")
        if not self.manual_region_channels:
            raise ValueError("manual_region_channels is required when roi_mode='manual'.")
        return self


class BaseTimeFrequencyStatsGroupWriterParams(BaseWriterParams):
    pipeline_label: str
    output_modality: str = "ieeg"
    output_suffix: str = "stats"
    output_description: str
    output_format: Literal["hdf5", "matlab"] = "hdf5"
