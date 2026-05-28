"""Shared processor helpers for group-level trial statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.bids.helpers import normalize_subject_value
from gin_bids_py_analysis.bids.matching import find_best_entity_match
from gin_bids_py_analysis.processing.base import BaseProcessing
from gin_bids_py_analysis.processing.utils.channels import normalize_channel_name
from gin_bids_py_analysis.processing.utils.tables import select_column

from gin_bids_py_analysis.processing.utils.field_names import sanitize_field_name, validate_field_name

from .compatibility import ContributionT, InputT
from .params import BaseTrialStatsGroupParams


@dataclass(frozen=True)
class BaseTrialStatsGroupContributionRecord:
    """Common ROI contribution metadata shared by group pipelines."""

    roi: str
    subject: str
    channel: str
    source_stats_file: str


def collect_manual_roi_records(
    *,
    inputs: Sequence[InputT],
    manual_region_channels: dict[str, dict[str, list[str]]],
    create_record: Callable[[str, str, InputT, int], ContributionT],
) -> dict[str, list[ContributionT]]:
    """Resolve manual ROI channel lists into contribution records."""

    roi_records: dict[str, list[ContributionT]] = {
        roi: [] for roi in manual_region_channels
    }
    for item in inputs:
        subject_key = normalize_subject_value(str(item.subject).strip())
        for roi, subject_map in manual_region_channels.items():
            channels = subject_map.get(subject_key, [])
            for channel in channels:
                idx = item.channel_index_by_norm.get(normalize_channel_name(channel))
                if idx is None:
                    continue
                roi_records[roi].append(create_record(roi, subject_key, item, idx))
    return roi_records


def find_missing_manual_roi_channels(
    *,
    inputs: Sequence[InputT],
    manual_region_channels: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, list[str]]]:
    """Return manual ROI channels that are absent from the available inputs."""

    available_channels_by_subject: dict[str, set[str]] = {}
    for item in inputs:
        subject_key = normalize_subject_value(str(item.subject).strip())
        subject_channels = available_channels_by_subject.setdefault(subject_key, set())
        subject_channels.update(item.channel_index_by_norm.keys())

    missing: dict[str, dict[str, list[str]]] = {}
    for roi, subject_map in manual_region_channels.items():
        roi_missing: dict[str, list[str]] = {}
        for raw_subject, channels in subject_map.items():
            subject_key = normalize_subject_value(str(raw_subject).strip())
            available_channels = available_channels_by_subject.get(subject_key, set())
            missing_channels: list[str] = []
            seen_channels: set[str] = set()
            for channel in channels:
                normalized_channel = normalize_channel_name(channel)
                if normalized_channel in seen_channels:
                    continue
                seen_channels.add(normalized_channel)
                if normalized_channel not in available_channels:
                    missing_channels.append(channel)
            if missing_channels:
                roi_missing[subject_key] = missing_channels
        if roi_missing:
            missing[roi] = roi_missing
    return missing


def format_manual_roi_missing_channels_message(
    missing_manual_roi_channels: dict[str, dict[str, list[str]]],
) -> str:
    """Format a compact human-readable warning for missing manual ROI channels."""

    if not missing_manual_roi_channels:
        return ""

    parts: list[str] = []
    for roi, subject_map in missing_manual_roi_channels.items():
        for subject, channels in subject_map.items():
            joined_channels = ", ".join(channels)
            parts.append(f"{roi}/{subject}: {joined_channels}")
    return "Missing manual channels: " + "; ".join(parts)


def collect_atlas_roi_records(
    *,
    inputs: Sequence[InputT],
    atlas_name: str,
    create_record: Callable[[str, str, InputT, int], ContributionT],
) -> tuple[dict[str, list[ContributionT]], set[str]]:
    """Resolve atlas-based ROI groupings into contribution records."""

    roi_records: dict[str, list[ContributionT]] = {}
    used_electrode_paths: set[str] = set()

    for item in inputs:
        region_to_channels, used_paths = resolve_input_atlas_regions(
            item=item,
            atlas_name=atlas_name,
        )
        used_electrode_paths.update(used_paths)
        for roi, channels in region_to_channels.items():
            records = roi_records.setdefault(roi, [])
            for channel in channels:
                idx = item.channel_index_by_norm.get(normalize_channel_name(channel))
                if idx is None:
                    continue
                records.append(create_record(roi, item.subject, item, idx))

    sanitized_records: dict[str, list[ContributionT]] = {}
    for roi_name, records in roi_records.items():
        sanitized = sanitize_field_name(roi_name)
        validate_field_name(roi_name, sanitized, context="ROI name")
        if sanitized in sanitized_records:
            raise ValueError(
                f"Atlas ROI name {roi_name!r} maps to {sanitized!r} after sanitization, "
                "but that name is already used by another ROI. "
                "Please use unique names."
            )
        sanitized_records[sanitized] = records
    return sanitized_records, used_electrode_paths


def resolve_input_atlas_regions(
    *,
    item: InputT,
    atlas_name: str,
) -> tuple[dict[str, list[str]], list[str]]:
    """Resolve atlas labels for a loaded subject input."""

    if not item.source_ieeg_files:
        raise ValueError(
            f"{item.stats_file.path.name}: provenance/source_ieeg_files is required for atlas mode."
        )
    if not item.source_electrodes_files:
        raise ValueError(
            f"{item.stats_file.path.name}: provenance/source_electrodes_files is required for atlas mode."
        )

    electrode_files = [
        BIDSFile.from_path(Path(path))
        for path in item.source_electrodes_files
        if Path(path).exists()
    ]
    if not electrode_files:
        raise ValueError(
            f"{item.stats_file.path.name}: no existing electrodes file found in provenance."
        )

    assigned_region_by_channel: dict[str, str] = {}
    display_channel_by_norm: dict[str, str] = {}
    used_electrodes: set[str] = set()

    for source_ieeg_path in item.source_ieeg_files:
        ieeg_path = Path(source_ieeg_path)
        ieeg_file = BIDSFile.from_path(ieeg_path)
        matched_electrodes = find_best_entity_match(
            ieeg_file,
            electrode_files,
            ambiguity_label="electrodes table",
            ambiguity_hint="Disambiguate entities in source_electrodes_files.",
        )
        if matched_electrodes is None:
            raise ValueError(
                f"{item.stats_file.path.name}: no matching electrodes file found for {ieeg_path.name}."
            )
        used_electrodes.add(str(matched_electrodes.path))

        with matched_electrodes.ensure_loaded() as rows:
            if not rows:
                continue
            columns = list(rows[0].keys())
            channel_col = select_column(columns, preferred=["name", "channel", "label"])
            atlas_col = select_column(columns, preferred=[atlas_name])
            if channel_col is None:
                raise ValueError(
                    f"Electrodes table {matched_electrodes.path.name} must contain a channel column."
                )
            if atlas_col is None:
                raise ValueError(
                    f"Electrodes table {matched_electrodes.path.name} has no column matching "
                    f"atlas_name={atlas_name!r}."
                )

            per_file_map: dict[str, str] = {}
            for row in rows:
                channel = (row.get(channel_col) or "").strip()
                region = (row.get(atlas_col) or "").strip()
                if not channel or not region or is_na_like_region_label(region):
                    continue
                channel_key = normalize_channel_name(channel)
                previous_region = per_file_map.get(channel_key)
                if previous_region is not None and previous_region != region:
                    raise ValueError(
                        f"Channel {channel!r} has multiple atlas labels in "
                        f"{matched_electrodes.path.name}."
                    )
                per_file_map[channel_key] = region

        for channel in item.channel_names:
            channel_key = normalize_channel_name(channel)
            region = per_file_map.get(channel_key)
            if region is None:
                continue
            previous = assigned_region_by_channel.get(channel_key)
            if previous is not None and previous != region:
                raise ValueError(
                    f"{item.stats_file.path.name}: channel {channel!r} mapped to multiple ROI "
                    f"labels ({previous!r} and {region!r}) across electrodes files."
                )
            assigned_region_by_channel[channel_key] = region
            display_channel_by_norm[channel_key] = channel

    region_to_channels: dict[str, list[str]] = {}
    for channel in item.channel_names:
        channel_key = normalize_channel_name(channel)
        region = assigned_region_by_channel.get(channel_key)
        if region is None:
            continue
        region_to_channels.setdefault(region, []).append(display_channel_by_norm[channel_key])

    if not region_to_channels:
        raise ValueError(
            f"{item.stats_file.path.name}: atlas mode produced no channel-to-ROI mapping."
        )
    return region_to_channels, sorted(used_electrodes)


def is_na_like_region_label(label: str) -> bool:
    """Return True for placeholder atlas labels that should be ignored."""

    normalized = "".join(ch for ch in label.strip().casefold() if ch.isalnum())
    return normalized in {"na", "notavailable", "notapplicable"}


class BaseTrialStatsGroupProcessing(BaseProcessing):
    """Shared lightweight base for group-level trial-statistics processors."""

    def __init__(self, params: BaseTrialStatsGroupParams) -> None:
        self.params = params

    @staticmethod
    def sorted_group_files(group: BIDSFileGroup) -> list[BIDSFile]:
        return sorted(group.all_files, key=lambda item: str(item.path))

    @staticmethod
    def stack_rows(rows: list[np.ndarray], n_times: int) -> np.ndarray:
        if rows:
            return np.stack(rows, axis=0).astype(np.float64)
        return np.empty((0, n_times), dtype=np.float64)

    @staticmethod
    def array_1d(values: list[float]) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)
