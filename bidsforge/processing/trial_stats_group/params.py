from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from bidsforge.bids.helpers import normalize_subject_value
from bidsforge.processing.base import BaseProcessingParams, BaseWriterParams
from bidsforge.processing.utils.field_names import sanitize_field_name, validate_field_name


def sanitize_roi_name(name: str) -> str:
    """Sanitize a ROI name for use as an HDF5 group key or MATLAB struct field name."""
    return sanitize_field_name(name)


def validate_roi_name(original: str, sanitized: str) -> None:
    """Validate a sanitized ROI name; raises ValueError if invalid."""
    validate_field_name(original, sanitized, context="ROI name")


class BaseTrialStatsGroupParams(BaseProcessingParams):
    """Shared parameter surface for group-level trial-statistics pipelines."""

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
    n_group_permutations: int = Field(
        default=10000,
        ge=1,
        description=(
            "Number of group-level null iterations when "
            "p_value_correction_method='cluster_permutation'. "
            "Ignored for all other correction methods."
        ),
    )
    cluster_threshold_alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description=(
            "Significance threshold applied to the group one-sample t-test within each "
            "null iteration to detect temporal clusters. Used only when "
            "p_value_correction_method='cluster_permutation'."
        ),
    )
    permutation_seed: int | None = Field(
        default=None,
        description=(
            "Seed for the NumPy random generator used when building the cluster null "
            "distribution. None selects a non-reproducible seed."
        ),
    )
    cluster_permutation_method: Literal["custom", "sign_flip"] = Field(
        default="custom",
        description=(
            "Strategy for building the group-level cluster null distribution when "
            "p_value_correction_method='cluster_permutation'. "
            "'custom' draws from pre-computed per-channel permuted value pools produced "
            "by within-subject label-shuffling (requires subject-level files computed "
            "with n_permutations > 0). "
            "'sign_flip' stacks each channel's observed timecourse into a "
            "(n_channels, n_times) matrix and applies a sign-flip permutation test "
            "(MNE one-sample test). No pre-computed permutations required."
        ),
    )
    n_clusters_to_keep: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of largest clusters (by |t-sum|) to retain per ROI when "
            "p_value_correction_method='cluster_permutation'. Each retained cluster "
            "is individually tested against the null distribution; only clusters with "
            "p < significance_alpha contribute time points to the significance mask. "
            "The cluster-level p-value stored in the result always corresponds to the "
            "largest (first) candidate cluster."
        ),
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
            roi = sanitize_roi_name(str(raw_roi).strip())
            if not roi or set(roi) == {"_"}:
                continue
            validate_roi_name(str(raw_roi).strip(), roi)
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
                if roi in cleaned:
                    raise ValueError(
                        f"ROI name {raw_roi!r} maps to {roi!r} after sanitization, "
                        "but that name is already used by another ROI. "
                        "Please use unique names."
                    )
                cleaned[roi] = subject_map
        return cleaned

    @model_validator(mode="after")
    def _validate_roi_mode(self) -> "BaseTrialStatsGroupParams":
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


class BaseTrialStatsGroupWriterParams(BaseWriterParams):
    """Shared writer configuration for group-level trial-statistics pipelines."""

    pipeline_label: str
    output_modality: str = "ieeg"
    output_suffix: str = "stats"
    output_description: str
    output_format: Literal["hdf5", "matlab"] = Field(
        default="hdf5",
        description=(
            "Output format for group-level trial-statistics results. "
            "'hdf5' writes an HDF5 file (.h5); 'matlab' writes a MATLAB file (.mat)."
        ),
    )
