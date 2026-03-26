from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.bids.helpers import (
    normalize_subject_value,
)
from gin_bids_py_analysis.bids.matching import find_best_entity_match
from gin_bids_py_analysis.processing.base import BaseProcessing
from gin_bids_py_analysis.processing.utils.channels import normalize_channel_name
from gin_bids_py_analysis.processing.utils.hdf5 import (
    coerce_feature_time,
    dataset_or_none,
    decode_str_array,
    float_scalar,
    int_scalar,
    str_scalar,
)
from gin_bids_py_analysis.processing.utils.matlab import (
    mat_float,
    mat_int,
    mat_str,
    mat_str_list,
    matlab_safe_name,
)
from gin_bids_py_analysis.processing.utils.tables import select_column

from .params import TrialStatsGroupParams
from .result import ROIChannelContribution, TrialStatsGroupProcessingResult
from .stats import (
    compute_one_sample_epoch_summary,
    compute_one_sample_timecourse,
    correct_p_values,
)


@dataclass(frozen=True)
class _RawTrialStatsData:
    """Format-agnostic in-memory representation of one trial-stats file."""

    analysis_level: str
    channels: list[str]
    time_axis_s: np.ndarray
    metric_values: np.ndarray  # shape (n_channels, n_times)
    condition_labels: tuple[str, str]
    binning_mode: str
    window_ms: float
    n_bins: int
    effective_n_bins: int
    source_ieeg_files: list[str]
    source_electrodes_files: list[str]


@dataclass(frozen=True)
class _SnapshotSignature:
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
    def key(self) -> tuple[Any, ...]:
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


@dataclass(frozen=True)
class _TrialStatsSnapshot:
    stats_file: BIDSFile
    subject: str
    task: str
    source_desc: str
    condition_labels: tuple[str, str]
    channel_names: list[str]
    channel_index_by_norm: dict[str, int]
    time_axis_s: np.ndarray
    metric_values: np.ndarray
    analysis_level: str
    binning_mode: str
    window_ms: float
    n_bins: int
    effective_n_bins: int
    source_ieeg_files: list[str]
    source_electrodes_files: list[str]
    signature: _SnapshotSignature


@dataclass(frozen=True)
class _ContributionRecord:
    roi: str
    subject: str
    channel: str
    source_stats_file: str
    values: np.ndarray


def build_trial_stats_compatible_groups(
    stats_files: Sequence[BIDSFile],
    *,
    source_metric: str,
) -> list[BIDSFileGroup]:
    """Group trial-stats files by compatibility for one group-level execution pass."""
    if not stats_files:
        return []

    groups: dict[tuple[Any, ...], list[BIDSFile]] = {}
    for file in sorted(stats_files, key=lambda item: str(item.path)):
        signature = _read_snapshot_signature(file, source_metric=source_metric)
        groups.setdefault(signature.key, []).append(file)

    out_groups: list[BIDSFileGroup] = []
    for key in sorted(groups, key=str):
        files = groups[key]
        out_groups.append(BIDSFileGroup(primary=files[0], secondaries=files[1:]))
    return out_groups


class TrialStatsGroupProcessing(BaseProcessing):
    """Compute group-level ROI one-sample tests from channel-level trial-stats files."""

    def __init__(self, params: TrialStatsGroupParams) -> None:
        self.params = params

    def process_group(
        self,
        group: BIDSFileGroup,
        progress_tracking_position: int = 0,
    ) -> TrialStatsGroupProcessingResult:
        del progress_tracking_position
        files = sorted(group.all_files, key=lambda item: str(item.path))
        if not files:
            raise ValueError("TrialStatsGroupProcessing requires at least one *_stats.h5 file.")

        snapshots = [
            _load_trial_stats_snapshot(file, source_metric=self.params.source_metric)
            for file in files
        ]
        _validate_group_compatibility(snapshots)

        first = snapshots[0]
        n_times = int(len(first.time_axis_s))
        excluded_rois: dict[str, str] = {}

        if self.params.roi_mode == "manual":
            roi_records = _collect_manual_roi_records(
                snapshots=snapshots,
                manual_region_channels=self.params.manual_region_channels,
            )
            used_electrode_paths: set[str] = set()
        else:
            assert self.params.atlas_name is not None
            roi_records, used_electrode_paths = _collect_atlas_roi_records(
                snapshots=snapshots,
                atlas_name=self.params.atlas_name,
            )

        region_names: list[str] = []
        rows_t: list[np.ndarray] = []
        rows_p_uncorrected: list[np.ndarray] = []
        rows_mean: list[np.ndarray] = []
        rows_sem: list[np.ndarray] = []
        roi_channel_counts: list[int] = []
        roi_subject_counts: list[int] = []
        summary_t: list[float] = []
        summary_p: list[float] = []
        summary_df: list[float] = []
        summary_mean: list[float] = []
        summary_sem: list[float] = []
        contributions_out: list[ROIChannelContribution] = []

        for roi, records in roi_records.items():
            if not records:
                excluded_rois[roi] = "no_channels"
                continue
            subject_count = len({record.subject for record in records})
            channel_count = len(records)
            if channel_count < self.params.min_channels_per_roi:
                excluded_rois[roi] = (
                    f"insufficient_channels:{channel_count}<{self.params.min_channels_per_roi}"
                )
                continue
            if subject_count < self.params.min_subjects_per_roi:
                excluded_rois[roi] = (
                    f"insufficient_subjects:{subject_count}<{self.params.min_subjects_per_roi}"
                )
                continue

            samples = np.stack([record.values for record in records], axis=0).astype(np.float64)
            t_values, p_values_raw, mean_values, sem_values = compute_one_sample_timecourse(samples)
            t_summary, p_summary, df_summary, mean_summary, sem_summary = (
                compute_one_sample_epoch_summary(samples)
            )

            region_names.append(roi)
            rows_t.append(t_values)
            rows_p_uncorrected.append(p_values_raw)
            rows_mean.append(mean_values)
            rows_sem.append(sem_values)
            roi_channel_counts.append(channel_count)
            roi_subject_counts.append(subject_count)
            summary_t.append(t_summary)
            summary_p.append(p_summary)
            summary_df.append(df_summary)
            summary_mean.append(mean_summary)
            summary_sem.append(sem_summary)
            contributions_out.extend(
                [
                    ROIChannelContribution(
                        roi=record.roi,
                        subject=record.subject,
                        channel=record.channel,
                        source_stats_file=record.source_stats_file,
                    )
                    for record in records
                ]
            )

        if rows_p_uncorrected:
            p_values_uncorrected = np.stack(rows_p_uncorrected, axis=0).astype(np.float64)
            t_values = np.stack(rows_t, axis=0).astype(np.float64)
            metric_mean = np.stack(rows_mean, axis=0).astype(np.float64)
            metric_sem = np.stack(rows_sem, axis=0).astype(np.float64)
        else:
            t_values = np.empty((0, n_times), dtype=np.float64)
            p_values_uncorrected = np.empty((0, n_times), dtype=np.float64)
            metric_mean = np.empty((0, n_times), dtype=np.float64)
            metric_sem = np.empty((0, n_times), dtype=np.float64)

        p_values = correct_p_values(
            p_values_uncorrected,
            method=self.params.p_value_correction_method,
        )
        significant_mask = np.isfinite(p_values) & (p_values < self.params.significance_alpha)

        output_entities = {
            "subject": "group",
        }
        if first.task:
            output_entities["task"] = first.task

        return TrialStatsGroupProcessingResult(
            source_group=group,
            output_entities=output_entities,
            metadata={
                "source_metric": self.params.source_metric,
                "roi_mode": self.params.roi_mode,
                "atlas_name": self.params.atlas_name,
                "min_channels_per_roi": self.params.min_channels_per_roi,
                "min_subjects_per_roi": self.params.min_subjects_per_roi,
                "p_value_correction_method": self.params.p_value_correction_method,
                "significance_alpha": self.params.significance_alpha,
                "condition_labels": list(first.condition_labels),
                "task": first.task,
                "source_desc": first.source_desc,
                "binning_mode": first.binning_mode,
                "window_ms": first.window_ms,
                "n_bins": first.n_bins,
                "effective_n_bins": first.effective_n_bins,
                "excluded_rois": dict(excluded_rois),
            },
            t_values=t_values,
            p_values=p_values,
            p_values_uncorrected=p_values_uncorrected,
            significant_mask=significant_mask,
            metric_mean=metric_mean,
            metric_sem=metric_sem,
            time_axis_s=first.time_axis_s.astype(np.float64),
            region_names=region_names,
            source_metric=self.params.source_metric,
            condition_labels=first.condition_labels,
            p_value_correction_method=self.params.p_value_correction_method,
            significance_alpha=self.params.significance_alpha,
            roi_mode=self.params.roi_mode,
            atlas_name=self.params.atlas_name,
            epoch_mean_t_values=np.asarray(summary_t, dtype=np.float64),
            epoch_mean_p_values=np.asarray(summary_p, dtype=np.float64),
            epoch_mean_df=np.asarray(summary_df, dtype=np.float64),
            epoch_mean_metric_mean=np.asarray(summary_mean, dtype=np.float64),
            epoch_mean_metric_sem=np.asarray(summary_sem, dtype=np.float64),
            roi_channel_counts=np.asarray(roi_channel_counts, dtype=np.int64),
            roi_subject_counts=np.asarray(roi_subject_counts, dtype=np.int64),
            contributions=contributions_out,
            source_trial_stats_files=[str(snapshot.stats_file.path) for snapshot in snapshots],
            source_electrodes_files=sorted(used_electrode_paths),
            excluded_rois=excluded_rois,
        )


def _collect_manual_roi_records(
    *,
    snapshots: Sequence[_TrialStatsSnapshot],
    manual_region_channels: dict[str, dict[str, list[str]]],
) -> dict[str, list[_ContributionRecord]]:
    roi_records: dict[str, list[_ContributionRecord]] = {
        roi: [] for roi in manual_region_channels
    }
    for snapshot in snapshots:
        raw_subject = str(snapshot.subject).strip()
        subject_key = normalize_subject_value(raw_subject)
        for roi, subject_map in manual_region_channels.items():
            channels = subject_map.get(subject_key, [])
            for channel in channels:
                idx = snapshot.channel_index_by_norm.get(normalize_channel_name(channel))
                if idx is None:
                    continue
                roi_records[roi].append(
                    _ContributionRecord(
                        roi=roi,
                        subject=subject_key,
                        channel=snapshot.channel_names[idx],
                        source_stats_file=str(snapshot.stats_file.path),
                        values=np.asarray(snapshot.metric_values[idx, :], dtype=np.float64),
                    )
                )
    return roi_records


def _collect_atlas_roi_records(
    *,
    snapshots: Sequence[_TrialStatsSnapshot],
    atlas_name: str,
) -> tuple[dict[str, list[_ContributionRecord]], set[str]]:
    roi_records: dict[str, list[_ContributionRecord]] = {}
    used_electrode_paths: set[str] = set()

    for snapshot in snapshots:
        region_to_channels, used_paths = _resolve_snapshot_atlas_regions(
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
                records.append(
                    _ContributionRecord(
                        roi=roi,
                        subject=snapshot.subject,
                        channel=snapshot.channel_names[idx],
                        source_stats_file=str(snapshot.stats_file.path),
                        values=np.asarray(snapshot.metric_values[idx, :], dtype=np.float64),
                    )
                )
    return roi_records, used_electrode_paths


def _resolve_snapshot_atlas_regions(
    *,
    snapshot: _TrialStatsSnapshot,
    atlas_name: str,
) -> tuple[dict[str, list[str]], list[str]]:
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
        if not ieeg_path.exists():
            continue
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
                    f"Electrodes table {matched_electrodes.path.name} has no column matching atlas_name={atlas_name!r}."
                )

            per_file_map: dict[str, str] = {}
            for row in rows:
                channel = (row.get(channel_col) or "").strip()
                region = (row.get(atlas_col) or "").strip()
                if not channel or not region or _is_na_like_region_label(region):
                    continue
                channel_key = normalize_channel_name(channel)
                previous_region = per_file_map.get(channel_key)
                if previous_region is not None and previous_region != region:
                    raise ValueError(
                        f"Channel {channel!r} has multiple atlas labels in {matched_electrodes.path.name}."
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
                    f"{snapshot.stats_file.path.name}: channel {channel!r} mapped to multiple ROI labels "
                    f"({previous!r} and {region!r}) across electrodes files."
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


def _validate_group_compatibility(snapshots: Sequence[_TrialStatsSnapshot]) -> None:
    if not snapshots:
        raise ValueError("At least one trial-stats snapshot is required.")
    if any(snapshot.analysis_level != "channel" for snapshot in snapshots):
        non_channel = [snapshot.stats_file.path.name for snapshot in snapshots if snapshot.analysis_level != "channel"]
        joined = ", ".join(non_channel)
        raise ValueError(
            "trial_stats_group requires channel-level trial_stats inputs. "
            f"Found non-channel files: {joined}."
        )

    reference = snapshots[0].signature.key
    for snapshot in snapshots[1:]:
        if snapshot.signature.key != reference:
            raise ValueError(
                "Incompatible trial-stats inputs in one processing group. "
                "Use build_trial_stats_compatible_groups(...) to split heterogeneous files."
            )


def _build_signature(
    *,
    stats_file: BIDSFile,
    condition_labels: tuple[str, str],
    time_axis_s: np.ndarray,
    analysis_level: str,
    binning_mode: str,
    window_ms: float,
    n_bins: int,
    effective_n_bins: int,
) -> _SnapshotSignature:
    time_hash = hashlib.sha1(np.asarray(time_axis_s, dtype=np.float64).tobytes()).hexdigest()
    return _SnapshotSignature(
        task=str(stats_file.get("task") or ""),
        source_desc=str(stats_file.get("desc") or ""),
        condition_labels=condition_labels,
        time_axis_hash=time_hash,
        time_axis_len=int(len(time_axis_s)),
        binning_mode=str(binning_mode or "none"),
        window_ms=float(window_ms),
        n_bins=int(n_bins),
        effective_n_bins=int(effective_n_bins),
        analysis_level=str(analysis_level or "channel"),
    )


def _read_snapshot_signature(
    stats_file: BIDSFile,
    *,
    source_metric: str,
) -> _SnapshotSignature:
    raw = _load_raw_trial_stats(stats_file, source_metric=source_metric)
    return _build_signature(
        stats_file=stats_file,
        condition_labels=raw.condition_labels,
        time_axis_s=raw.time_axis_s,
        analysis_level=raw.analysis_level,
        binning_mode=raw.binning_mode,
        window_ms=raw.window_ms,
        n_bins=raw.n_bins,
        effective_n_bins=raw.effective_n_bins,
    )


def _load_trial_stats_snapshot(
    stats_file: BIDSFile,
    *,
    source_metric: str,
) -> _TrialStatsSnapshot:
    raw = _load_raw_trial_stats(stats_file, source_metric=source_metric)
    raw_subject = str(stats_file.get("subject") or stats_file.get("sub") or "").strip()
    subject = normalize_subject_value(raw_subject)

    channel_index_by_norm: dict[str, int] = {}
    for idx, name in enumerate(raw.channels):
        key = normalize_channel_name(name)
        channel_index_by_norm.setdefault(key, idx)

    signature = _build_signature(
        stats_file=stats_file,
        condition_labels=raw.condition_labels,
        time_axis_s=raw.time_axis_s,
        analysis_level=raw.analysis_level,
        binning_mode=raw.binning_mode,
        window_ms=raw.window_ms,
        n_bins=raw.n_bins,
        effective_n_bins=raw.effective_n_bins,
    )
    return _TrialStatsSnapshot(
        stats_file=stats_file,
        subject=subject,
        task=str(stats_file.get("task") or ""),
        source_desc=str(stats_file.get("desc") or ""),
        condition_labels=raw.condition_labels,
        channel_names=raw.channels,
        channel_index_by_norm=channel_index_by_norm,
        time_axis_s=raw.time_axis_s,
        metric_values=raw.metric_values,
        analysis_level=raw.analysis_level,
        binning_mode=raw.binning_mode,
        window_ms=raw.window_ms,
        n_bins=raw.n_bins,
        effective_n_bins=raw.effective_n_bins,
        source_ieeg_files=raw.source_ieeg_files,
        source_electrodes_files=raw.source_electrodes_files,
        signature=signature,
    )


def _load_raw_trial_stats(
    stats_file: BIDSFile,
    *,
    source_metric: str,
) -> _RawTrialStatsData:
    """Dispatch to the appropriate format-specific loader based on file extension."""
    extension = (stats_file.extension or "").lower()
    if extension == ".mat":
        return _load_raw_from_matlab(stats_file, source_metric=source_metric)
    return _load_raw_from_hdf5(stats_file, source_metric=source_metric)


def _load_raw_from_hdf5(
    stats_file: BIDSFile,
    *,
    source_metric: str,
) -> _RawTrialStatsData:
    with stats_file.ensure_loaded() as fh:
        analysis_level = str_scalar(dataset_or_none(fh, "meta/analysis_level"), default="channel")
        axis_name = "channel" if analysis_level == "channel" else "region"
        if axis_name not in fh["axes"]:
            raise ValueError(
                f"{stats_file.path.name}: axes/{axis_name} dataset is required."
            )
        channels = decode_str_array(np.asarray(fh["axes"][axis_name][:]))
        time_axis_s = np.asarray(fh["axes"]["time_s"][:], dtype=np.float64)
        condition_labels = _read_condition_labels_hdf5(fh)
        metric_values = _load_metric_matrix_hdf5(
            fh,
            source_metric=source_metric,
            condition_labels=condition_labels,
            n_channels=len(channels),
            n_times=len(time_axis_s),
        )
        binning_mode = str_scalar(dataset_or_none(fh, "meta/binning_mode"), default="none")
        window_ms = float_scalar(dataset_or_none(fh, "meta/window_ms"), default=0.0)
        n_bins = int_scalar(dataset_or_none(fh, "meta/n_bins"), default=0)
        effective_n_bins = int_scalar(
            dataset_or_none(fh, "meta/effective_n_bins"),
            default=len(time_axis_s),
        )
        source_ieeg_files = decode_str_array(
            np.asarray(fh["provenance"]["source_ieeg_files"][:], dtype=object)
        ) if "provenance" in fh and "source_ieeg_files" in fh["provenance"] else []
        source_electrodes_files = decode_str_array(
            np.asarray(fh["provenance"]["source_electrodes_files"][:], dtype=object)
        ) if "provenance" in fh and "source_electrodes_files" in fh["provenance"] else []

    return _RawTrialStatsData(
        analysis_level=analysis_level,
        channels=channels,
        time_axis_s=time_axis_s,
        metric_values=metric_values,
        condition_labels=condition_labels,
        binning_mode=binning_mode,
        window_ms=window_ms,
        n_bins=n_bins,
        effective_n_bins=effective_n_bins,
        source_ieeg_files=source_ieeg_files,
        source_electrodes_files=source_electrodes_files,
    )


def _read_condition_labels_hdf5(fh: h5py.File) -> tuple[str, str]:
    labels_dataset = dataset_or_none(fh, "meta/trial_count_labels")
    if labels_dataset is not None:
        labels = decode_str_array(np.asarray(labels_dataset[:], dtype=object))
        if len(labels) >= 2:
            return labels[0], labels[1]

    means_group = fh["means"]
    mean_keys = [key for key in means_group.keys() if key != "difference"]
    if len(mean_keys) >= 2:
        return mean_keys[0], mean_keys[1]
    return "condition_a", "condition_b"


def _load_metric_matrix_hdf5(
    fh: h5py.File,
    *,
    source_metric: str,
    condition_labels: tuple[str, str],
    n_channels: int,
    n_times: int,
) -> np.ndarray:
    if source_metric == "mean_difference":
        dataset = dataset_or_none(fh, "means/difference")
        if dataset is None:
            raise ValueError(f"{fh.filename}: means/difference dataset is required.")
        raw = np.asarray(dataset[:], dtype=np.float64)
        return coerce_feature_time(raw, n_features=n_channels, n_times=n_times)

    if source_metric == "t_values":
        dataset = dataset_or_none(fh, "stats/t_values")
        if dataset is None:
            raise ValueError(f"{fh.filename}: stats/t_values dataset is required.")
        raw = np.asarray(dataset[:], dtype=np.float64)
        return coerce_feature_time(raw, n_features=n_channels, n_times=n_times)

    metric_name = condition_labels[0] if source_metric == "condition_a_mean" else condition_labels[1]
    dataset = dataset_or_none(fh, f"means/{metric_name}")
    if dataset is None:
        raise ValueError(
            f"{fh.filename}: means/{metric_name!r} dataset is required "
            f"for source_metric={source_metric!r}."
        )
    raw = np.asarray(dataset[:], dtype=np.float64)
    return coerce_feature_time(raw, n_features=n_channels, n_times=n_times)


def _load_raw_from_matlab(
    stats_file: BIDSFile,
    *,
    source_metric: str,
) -> _RawTrialStatsData:
    with stats_file.ensure_loaded() as mat:
        data = mat["data"]
        meta = data.meta
        axes = data.axes
        prov = getattr(data, "provenance", None)

        analysis_level = mat_str(meta.analysis_level, default="channel")
        axis_attr = "channel" if analysis_level == "channel" else "region"
        channels = mat_str_list(getattr(axes, axis_attr, None))
        if not channels:
            raise ValueError(
                f"{stats_file.path.name}: axes.{axis_attr} array is required in .mat file."
            )

        time_axis_s = np.asarray(axes.time_s, dtype=np.float64).ravel()
        condition_labels = _mat_condition_labels(meta)
        metric_values = _load_metric_matrix_mat(
            data,
            source_metric=source_metric,
            condition_labels=condition_labels,
            n_channels=len(channels),
            n_times=len(time_axis_s),
            filename=stats_file.path.name,
        )
        binning_mode = mat_str(getattr(meta, "binning_mode", None), default="none")
        window_ms = mat_float(getattr(meta, "window_ms", None), default=0.0)
        n_bins = mat_int(getattr(meta, "n_bins", None), default=0)
        effective_n_bins = mat_int(
            getattr(meta, "effective_n_bins", None), default=len(time_axis_s)
        )
        source_ieeg_files: list[str] = []
        source_electrodes_files: list[str] = []
        if prov is not None:
            source_ieeg_files = mat_str_list(getattr(prov, "source_ieeg_files", None))
            source_electrodes_files = mat_str_list(getattr(prov, "source_electrodes_files", None))

    return _RawTrialStatsData(
        analysis_level=analysis_level,
        channels=channels,
        time_axis_s=time_axis_s,
        metric_values=metric_values,
        condition_labels=condition_labels,
        binning_mode=binning_mode,
        window_ms=window_ms,
        n_bins=n_bins,
        effective_n_bins=effective_n_bins,
        source_ieeg_files=source_ieeg_files,
        source_electrodes_files=source_electrodes_files,
    )


def _mat_condition_labels(meta: Any) -> tuple[str, str]:
    labels = mat_str_list(getattr(meta, "trial_count_labels", None))
    if len(labels) >= 2:
        return labels[0], labels[1]
    return "condition_a", "condition_b"


def _load_metric_matrix_mat(
    data: Any,
    *,
    source_metric: str,
    condition_labels: tuple[str, str],
    n_channels: int,
    n_times: int,
    filename: str,
) -> np.ndarray:
    means = data.means
    stats = data.stats

    if source_metric == "mean_difference":
        raw = np.asarray(means.difference, dtype=np.float64)
        return coerce_feature_time(raw, n_features=n_channels, n_times=n_times)

    if source_metric == "t_values":
        raw = np.asarray(stats.t_values, dtype=np.float64)
        return coerce_feature_time(raw, n_features=n_channels, n_times=n_times)

    safe_a = matlab_safe_name(condition_labels[0])
    safe_b = matlab_safe_name(condition_labels[1])
    attr_name = safe_a if source_metric == "condition_a_mean" else safe_b
    metric_array = getattr(means, attr_name, None)
    if metric_array is None:
        raise ValueError(
            f"{filename}: means.{attr_name!r} field is required "
            f"for source_metric={source_metric!r}."
        )
    raw = np.asarray(metric_array, dtype=np.float64)
    return coerce_feature_time(raw, n_features=n_channels, n_times=n_times)


def _is_na_like_region_label(label: str) -> bool:
    normalized = "".join(ch for ch in label.strip().casefold() if ch.isalnum())
    return normalized in {"na", "notavailable", "notapplicable"}
