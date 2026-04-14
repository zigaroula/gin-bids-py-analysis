"""Shared processor helpers for group-level trial statistics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

import h5py
import numpy as np

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.bids.helpers import normalize_subject_value
from gin_bids_py_analysis.bids.matching import find_best_entity_match
from gin_bids_py_analysis.processing.base import BaseProcessing
from gin_bids_py_analysis.processing.utils.channels import normalize_channel_name
from gin_bids_py_analysis.processing.utils.hdf5 import dataset_or_none, decode_str_array
from gin_bids_py_analysis.processing.utils.matlab import mat_str_list
from gin_bids_py_analysis.processing.utils.tables import select_column

from .params import BaseTrialStatsGroupParams


@dataclass(frozen=True)
class BaseTrialStatsGroupSnapshotSignature:
    """Common compatibility signature for one subject-level input snapshot."""

    task: str
    source_desc: str
    condition_labels: tuple[str, str]
    time_axis_hash: str
    time_axis_len: int
    binning_mode: str
    window_ms: float
    n_bins: int
    effective_n_bins: int
    analysis_level: str

    @property
    def base_key(self) -> tuple[Any, ...]:
        return (
            self.task,
            self.source_desc,
            self.condition_labels,
            self.time_axis_hash,
            self.time_axis_len,
            self.binning_mode,
            self.window_ms,
            self.n_bins,
            self.effective_n_bins,
            self.analysis_level,
        )

    @property
    def key(self) -> tuple[Any, ...]:
        return self.base_key


@dataclass(frozen=True)
class BaseTrialStatsGroupSnapshot:
    """Common snapshot metadata for one loaded subject-level result."""

    stats_file: BIDSFile
    subject: str
    task: str
    source_desc: str
    condition_labels: tuple[str, str]
    channel_names: list[str]
    channel_index_by_norm: dict[str, int]
    time_axis_s: np.ndarray
    analysis_level: str
    binning_mode: str
    window_ms: float
    n_bins: int
    effective_n_bins: int
    source_ieeg_files: list[str]
    source_electrodes_files: list[str]
    signature: BaseTrialStatsGroupSnapshotSignature


@dataclass(frozen=True)
class BaseRawTrialStatsData:
    """Common fields shared by all format-agnostic subject-level raw stats containers."""

    analysis_level: str
    channels: list[str]
    time_axis_s: np.ndarray
    condition_labels: tuple[str, str]
    binning_mode: str
    window_ms: float
    n_bins: int
    effective_n_bins: int
    activity_zscore: str
    activity_baseline_tmin_s: float
    activity_baseline_tmax_s: float
    source_ieeg_files: list[str]
    source_electrodes_files: list[str]


@dataclass(frozen=True)
class BaseTrialStatsGroupContributionRecord:
    """Common ROI contribution metadata shared by group pipelines."""

    roi: str
    subject: str
    channel: str
    source_stats_file: str


SignatureT = TypeVar("SignatureT", bound=BaseTrialStatsGroupSnapshotSignature)
SnapshotT = TypeVar("SnapshotT", bound=BaseTrialStatsGroupSnapshot)
ContributionT = TypeVar("ContributionT", bound=BaseTrialStatsGroupContributionRecord)


def hash_time_axis(time_axis_s: np.ndarray) -> str:
    """Return a stable content hash for a time axis."""

    return hashlib.sha1(np.asarray(time_axis_s, dtype=np.float64).tobytes()).hexdigest()


def build_compatible_groups(
    stats_files: Sequence[BIDSFile],
    *,
    read_signature: Callable[[BIDSFile], SignatureT],
) -> list[BIDSFileGroup]:
    """Group subject-level stats files by processing compatibility."""

    if not stats_files:
        return []

    groups: dict[tuple[Any, ...], list[BIDSFile]] = {}
    for file in sorted(stats_files, key=lambda item: str(item.path)):
        signature = read_signature(file)
        groups.setdefault(signature.key, []).append(file)

    out_groups: list[BIDSFileGroup] = []
    for key in sorted(groups, key=str):
        files = groups[key]
        out_groups.append(BIDSFileGroup(primary=files[0], secondaries=files[1:]))
    return out_groups


def validate_group_compatibility(
    snapshots: Sequence[SnapshotT],
    *,
    empty_message: str,
    non_channel_message: str,
    incompatible_message: str,
) -> None:
    """Validate that a set of loaded subject snapshots can be processed together."""

    if not snapshots:
        raise ValueError(empty_message)
    if any(snapshot.analysis_level != "channel" for snapshot in snapshots):
        non_channel = [
            snapshot.stats_file.path.name
            for snapshot in snapshots
            if snapshot.analysis_level != "channel"
        ]
        raise ValueError(f"{non_channel_message} Found non-channel files: {', '.join(non_channel)}.")

    reference = snapshots[0].signature.key
    for snapshot in snapshots[1:]:
        if snapshot.signature.key != reference:
            raise ValueError(incompatible_message)


def collect_manual_roi_records(
    *,
    snapshots: Sequence[SnapshotT],
    manual_region_channels: dict[str, dict[str, list[str]]],
    create_record: Callable[[str, str, SnapshotT, int], ContributionT],
) -> dict[str, list[ContributionT]]:
    """Resolve manual ROI channel lists into contribution records."""

    roi_records: dict[str, list[ContributionT]] = {
        roi: [] for roi in manual_region_channels
    }
    for snapshot in snapshots:
        subject_key = normalize_subject_value(str(snapshot.subject).strip())
        for roi, subject_map in manual_region_channels.items():
            channels = subject_map.get(subject_key, [])
            for channel in channels:
                idx = snapshot.channel_index_by_norm.get(normalize_channel_name(channel))
                if idx is None:
                    continue
                roi_records[roi].append(create_record(roi, subject_key, snapshot, idx))
    return roi_records


def collect_atlas_roi_records(
    *,
    snapshots: Sequence[SnapshotT],
    atlas_name: str,
    create_record: Callable[[str, str, SnapshotT, int], ContributionT],
) -> tuple[dict[str, list[ContributionT]], set[str]]:
    """Resolve atlas-based ROI groupings into contribution records."""

    roi_records: dict[str, list[ContributionT]] = {}
    used_electrode_paths: set[str] = set()

    for snapshot in snapshots:
        region_to_channels, used_paths = resolve_snapshot_atlas_regions(
            snapshot=snapshot,
            atlas_name=atlas_name,
        )
        used_electrode_paths.update(used_paths)
        for roi, channels in region_to_channels.items():
            records = roi_records.setdefault(roi, [])
            for channel in channels:
                idx = snapshot.channel_index_by_norm.get(normalize_channel_name(channel))
                if idx is None:
                    continue
                records.append(create_record(roi, snapshot.subject, snapshot, idx))
    return roi_records, used_electrode_paths


def resolve_snapshot_atlas_regions(
    *,
    snapshot: SnapshotT,
    atlas_name: str,
) -> tuple[dict[str, list[str]], list[str]]:
    """Resolve atlas labels for a loaded subject snapshot."""

    if not snapshot.source_ieeg_files:
        raise ValueError(
            f"{snapshot.stats_file.path.name}: provenance/source_ieeg_files is required for atlas mode."
        )
    if not snapshot.source_electrodes_files:
        raise ValueError(
            f"{snapshot.stats_file.path.name}: provenance/source_electrodes_files is required for atlas mode."
        )

    electrode_files = [
        BIDSFile.from_path(Path(path))
        for path in snapshot.source_electrodes_files
        if Path(path).exists()
    ]
    if not electrode_files:
        raise ValueError(
            f"{snapshot.stats_file.path.name}: no existing electrodes file found in provenance."
        )

    assigned_region_by_channel: dict[str, str] = {}
    display_channel_by_norm: dict[str, str] = {}
    used_electrodes: set[str] = set()

    for source_ieeg_path in snapshot.source_ieeg_files:
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
                f"{snapshot.stats_file.path.name}: no matching electrodes file found for {ieeg_path.name}."
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

        for channel in snapshot.channel_names:
            channel_key = normalize_channel_name(channel)
            region = per_file_map.get(channel_key)
            if region is None:
                continue
            previous = assigned_region_by_channel.get(channel_key)
            if previous is not None and previous != region:
                raise ValueError(
                    f"{snapshot.stats_file.path.name}: channel {channel!r} mapped to multiple ROI "
                    f"labels ({previous!r} and {region!r}) across electrodes files."
                )
            assigned_region_by_channel[channel_key] = region
            display_channel_by_norm[channel_key] = channel

    region_to_channels: dict[str, list[str]] = {}
    for channel in snapshot.channel_names:
        channel_key = normalize_channel_name(channel)
        region = assigned_region_by_channel.get(channel_key)
        if region is None:
            continue
        region_to_channels.setdefault(region, []).append(display_channel_by_norm[channel_key])

    if not region_to_channels:
        raise ValueError(
            f"{snapshot.stats_file.path.name}: atlas mode produced no channel-to-ROI mapping."
        )
    return region_to_channels, sorted(used_electrodes)


def read_condition_labels_hdf5(fh: h5py.File) -> tuple[str, str]:
    """Read condition labels from a subject-level HDF5 result."""

    labels_dataset = dataset_or_none(fh, "meta/trial_count_labels")
    if labels_dataset is not None:
        labels = decode_str_array(np.asarray(labels_dataset[:], dtype=object))
        if len(labels) >= 2:
            return labels[0], labels[1]

    if "means" in fh:
        mean_keys = list(fh["means"].keys())
        if "difference" in mean_keys:
            mean_keys = [key for key in mean_keys if key != "difference"]
        if len(mean_keys) >= 2:
            return mean_keys[0], mean_keys[1]
    return "condition_a", "condition_b"


def mat_condition_labels(meta: Any) -> tuple[str, str]:
    """Read condition labels from a subject-level MATLAB result struct."""

    labels = mat_str_list(getattr(meta, "trial_count_labels", None))
    if len(labels) >= 2:
        return labels[0], labels[1]
    return "condition_a", "condition_b"


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
    def build_output_entities(task: str) -> dict[str, str]:
        entities: dict[str, str] = {"subject": "group"}
        if task:
            entities["task"] = task
        return entities

    @staticmethod
    def stack_rows(rows: list[np.ndarray], n_times: int) -> np.ndarray:
        if rows:
            return np.stack(rows, axis=0).astype(np.float64)
        return np.empty((0, n_times), dtype=np.float64)

    @staticmethod
    def array_1d(values: list[float]) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)
