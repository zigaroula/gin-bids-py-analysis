from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, TypeVar

import numpy as np

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.processing.base import BaseProcessing
from bidsforge.processing.utils.channels import normalize_channel_name
from bidsforge.processing.utils.group_stats import ROIChannelContribution
from bidsforge.processing.utils.tables import select_column
from bidsforge.processing.utils.field_names import sanitize_field_name, validate_field_name

from .compatibility import TFSubjectStatsInput
from .params import BaseTimeFrequencyStatsGroupParams


ContributionT = TypeVar("ContributionT")
InputT = TypeVar("InputT", bound=TFSubjectStatsInput)


@dataclass(frozen=True)
class BaseTFGroupContributionRecord:
    roi: str
    subject: str
    channel: str
    source_stats_file: str

    def as_summary(self) -> ROIChannelContribution:
        return ROIChannelContribution(
            roi=self.roi,
            subject=self.subject,
            channel=self.channel,
            source_stats_file=self.source_stats_file,
        )


class BaseTimeFrequencyStatsGroupProcessing(BaseProcessing):
    def __init__(self, params: BaseTimeFrequencyStatsGroupParams) -> None:
        self.params = params

    @staticmethod
    def sorted_group_files(group: BIDSFileGroup) -> list[BIDSFile]:
        return sorted(group.all_files, key=lambda item: str(item.path))

    @staticmethod
    def stack_maps(values: list[np.ndarray], n_freqs: int, n_times: int) -> np.ndarray:
        if values:
            return np.stack(values, axis=0).astype(np.float64)
        return np.empty((0, n_freqs, n_times), dtype=np.float64)

    @staticmethod
    def array_1d(values: list[float]) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)


def collect_manual_roi_records(
    *,
    inputs: Sequence[InputT],
    manual_region_channels: dict[str, dict[str, list[str]]],
    create_record: Callable[[str, str, InputT, int], ContributionT],
) -> dict[str, list[ContributionT]]:
    out: dict[str, list[ContributionT]] = {roi: [] for roi in manual_region_channels}
    for item in inputs:
        for roi, subject_map in manual_region_channels.items():
            for channel in subject_map.get(item.subject, []):
                idx = item.channel_index_by_norm.get(normalize_channel_name(channel))
                if idx is not None:
                    out[roi].append(create_record(roi, item.subject, item, idx))
    return out


def collect_atlas_roi_records(
    *,
    inputs: Sequence[InputT],
    atlas_name: str,
    create_record: Callable[[str, str, InputT, int], ContributionT],
) -> tuple[dict[str, list[ContributionT]], set[str]]:
    roi_records: dict[str, list[ContributionT]] = {}
    used_electrodes: set[str] = set()
    for item in inputs:
        regions, used = resolve_input_atlas_regions(item=item, atlas_name=atlas_name)
        used_electrodes.update(used)
        for roi, channels in regions.items():
            sanitized = sanitize_field_name(roi)
            validate_field_name(roi, sanitized, context="ROI name")
            records = roi_records.setdefault(sanitized, [])
            for channel in channels:
                idx = item.channel_index_by_norm.get(normalize_channel_name(channel))
                if idx is not None:
                    records.append(create_record(sanitized, item.subject, item, idx))
    return roi_records, used_electrodes


def resolve_input_atlas_regions(
    *,
    item: TFSubjectStatsInput,
    atlas_name: str,
) -> tuple[dict[str, list[str]], list[str]]:
    electrode_paths = [Path(path) for path in item.source_electrodes_files if Path(path).exists()]
    if not electrode_paths:
        raise ValueError(
            f"{item.stats_file.path.name}: atlas mode requires provenance/source_electrodes_files."
        )
    channel_to_region: dict[str, str] = {}
    display_channel: dict[str, str] = {}
    used: list[str] = []
    for path in electrode_paths:
        file = BIDSFile.from_path(path)
        used.append(str(path))
        with file.ensure_loaded() as rows:
            if not rows:
                continue
            columns = list(rows[0].keys())
            channel_col = select_column(columns, preferred=["name", "channel", "label"])
            atlas_col = select_column(columns, preferred=[atlas_name])
            if channel_col is None or atlas_col is None:
                continue
            for row in rows:
                channel = str(row.get(channel_col) or "").strip()
                region = str(row.get(atlas_col) or "").strip()
                if not channel or not region or _is_na_like_region(region):
                    continue
                key = normalize_channel_name(channel)
                if key in item.channel_index_by_norm:
                    channel_to_region[key] = region
                    display_channel[key] = item.channel_names[item.channel_index_by_norm[key]]
    region_to_channels: dict[str, list[str]] = {}
    for key, region in channel_to_region.items():
        region_to_channels.setdefault(region, []).append(display_channel[key])
    if not region_to_channels:
        raise ValueError(f"{item.stats_file.path.name}: atlas mode produced no ROI channels.")
    return region_to_channels, used


def check_roi_exclusion(
    records: Sequence[object],
    *,
    min_channels_per_roi: int,
    min_subjects_per_roi: int,
) -> str | None:
    if not records:
        return "no_channels"
    n_channels = len(records)
    n_subjects = len({str(getattr(record, "subject")) for record in records})
    if n_channels < min_channels_per_roi:
        return f"insufficient_channels:{n_channels}<{min_channels_per_roi}"
    if n_subjects < min_subjects_per_roi:
        return f"insufficient_subjects:{n_subjects}<{min_subjects_per_roi}"
    return None


def _is_na_like_region(label: str) -> bool:
    normalized = "".join(ch for ch in label.casefold() if ch.isalnum())
    return normalized in {"na", "notavailable", "notapplicable"}
