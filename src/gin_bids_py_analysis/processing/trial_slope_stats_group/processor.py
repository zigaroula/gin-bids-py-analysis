from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.bids.helpers import normalize_subject_value
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

from gin_bids_py_analysis.processing.utils.group_stats import (
    compute_condition_group_stats,
    compute_one_sample_epoch_summary,
    compute_one_sample_timecourse,
)
from gin_bids_py_analysis.processing.utils.statistics import correct_p_values

from .params import TrialSlopeStatsGroupParams
from .result import ROIChannelContribution, TrialSlopeStatsGroupProcessingResult


# ---------------------------------------------------------------------------
# Internal data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _RawSlopeStatsData:
    """Format-agnostic in-memory representation of one trial_slope_stats file."""

    analysis_level: str
    channels: list[str]
    time_axis_s: np.ndarray
    condition_labels: tuple[str, str]
    condition_a_slope: np.ndarray       # (n_channels, n_times)
    condition_b_slope: np.ndarray       # (n_channels, n_times)
    condition_a_mean: np.ndarray        # (n_channels, n_times)
    condition_b_mean: np.ndarray        # (n_channels, n_times)
    condition_a_r_value: np.ndarray     # (n_channels, n_times)
    condition_b_r_value: np.ndarray     # (n_channels, n_times)
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
class _SlopeStatsSnapshot:
    stats_file: BIDSFile
    subject: str
    task: str
    source_desc: str
    condition_labels: tuple[str, str]
    channel_names: list[str]
    channel_index_by_norm: dict[str, int]
    time_axis_s: np.ndarray
    condition_a_slope: np.ndarray
    condition_b_slope: np.ndarray
    condition_a_mean: np.ndarray
    condition_b_mean: np.ndarray
    condition_a_r_value: np.ndarray
    condition_b_r_value: np.ndarray
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
    slope_a_values: np.ndarray      # (n_times,)
    slope_b_values: np.ndarray      # (n_times,)
    mean_a_values: np.ndarray       # (n_times,)
    mean_b_values: np.ndarray       # (n_times,)
    r_value_a_values: np.ndarray    # (n_times,)
    r_value_b_values: np.ndarray    # (n_times,)


# ---------------------------------------------------------------------------
# Public grouping helper
# ---------------------------------------------------------------------------

def build_trial_slope_stats_compatible_groups(
    stats_files: Sequence[BIDSFile],
) -> list[BIDSFileGroup]:
    """Group trial_slope_stats files by compatibility for one group-level execution pass."""
    if not stats_files:
        return []

    groups: dict[tuple[Any, ...], list[BIDSFile]] = {}
    for file in sorted(stats_files, key=lambda item: str(item.path)):
        signature = _read_snapshot_signature(file)
        groups.setdefault(signature.key, []).append(file)

    out_groups: list[BIDSFileGroup] = []
    for key in sorted(groups, key=str):
        files = groups[key]
        out_groups.append(BIDSFileGroup(primary=files[0], secondaries=files[1:]))
    return out_groups


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class TrialSlopeStatsGroupProcessing(BaseProcessing):
    """Compute group-level ROI regression-slope tests from trial_slope_stats files."""

    def __init__(self, params: TrialSlopeStatsGroupParams) -> None:
        self.params = params

    def process_group(
        self,
        group: BIDSFileGroup,
        progress_tracking_position: int = 0,
    ) -> TrialSlopeStatsGroupProcessingResult:
        del progress_tracking_position
        files = sorted(group.all_files, key=lambda item: str(item.path))
        if not files:
            raise ValueError(
                "TrialSlopeStatsGroupProcessing requires at least one trial_slope_stats file."
            )

        snapshots = [_load_slope_stats_snapshot(file) for file in files]
        _validate_group_compatibility(snapshots)

        first = snapshots[0]
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

        method = self.params.p_value_correction_method

        region_names: list[str] = []
        # Per-condition slope timecourse statistics
        rows_t_a: list[np.ndarray] = []
        rows_p_uncorr_a: list[np.ndarray] = []
        rows_slope_mean_a: list[np.ndarray] = []
        rows_slope_sem_a: list[np.ndarray] = []
        rows_t_b: list[np.ndarray] = []
        rows_p_uncorr_b: list[np.ndarray] = []
        rows_slope_mean_b: list[np.ndarray] = []
        rows_slope_sem_b: list[np.ndarray] = []
        # Activity and r_value group means
        rows_activity_mean_a: list[np.ndarray] = []
        rows_activity_sem_a: list[np.ndarray] = []
        rows_activity_mean_b: list[np.ndarray] = []
        rows_activity_sem_b: list[np.ndarray] = []
        rows_r_value_mean_a: list[np.ndarray] = []
        rows_r_value_sem_a: list[np.ndarray] = []
        rows_r_value_mean_b: list[np.ndarray] = []
        rows_r_value_sem_b: list[np.ndarray] = []
        # Epoch-level summaries
        epoch_t_a: list[float] = []
        epoch_p_a: list[float] = []
        epoch_df_a: list[float] = []
        epoch_mean_a: list[float] = []
        epoch_sem_a: list[float] = []
        epoch_t_b: list[float] = []
        epoch_p_b: list[float] = []
        epoch_df_b: list[float] = []
        epoch_mean_b: list[float] = []
        epoch_sem_b: list[float] = []
        # Contribution metadata
        roi_channel_counts: list[int] = []
        roi_subject_counts: list[int] = []
        contributions_out: list[ROIChannelContribution] = []
        slope_a_contribution_samples: list[np.ndarray] = []
        slope_b_contribution_samples: list[np.ndarray] = []
        activity_a_contribution_samples: list[np.ndarray] = []
        activity_b_contribution_samples: list[np.ndarray] = []
        contribution_label_rows: list[list[str]] = []

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

            samples_slope_a = np.stack(
                [r.slope_a_values for r in records], axis=0
            ).astype(np.float64)
            samples_slope_b = np.stack(
                [r.slope_b_values for r in records], axis=0
            ).astype(np.float64)
            samples_mean_a = np.stack(
                [r.mean_a_values for r in records], axis=0
            ).astype(np.float64)
            samples_mean_b = np.stack(
                [r.mean_b_values for r in records], axis=0
            ).astype(np.float64)
            samples_r_a = np.stack(
                [r.r_value_a_values for r in records], axis=0
            ).astype(np.float64)
            samples_r_b = np.stack(
                [r.r_value_b_values for r in records], axis=0
            ).astype(np.float64)

            t_a, p_a_raw, slope_mean_a, slope_sem_a = compute_one_sample_timecourse(samples_slope_a)
            t_b, p_b_raw, slope_mean_b, slope_sem_b = compute_one_sample_timecourse(samples_slope_b)
            ep_t_a, ep_p_a, ep_df_a, ep_mean_a, ep_sem_a = compute_one_sample_epoch_summary(samples_slope_a)
            ep_t_b, ep_p_b, ep_df_b, ep_mean_b, ep_sem_b = compute_one_sample_epoch_summary(samples_slope_b)
            act_mean_a, act_sem_a = compute_condition_group_stats(samples_mean_a)
            act_mean_b, act_sem_b = compute_condition_group_stats(samples_mean_b)
            r_mean_a, r_sem_a = compute_condition_group_stats(samples_r_a)
            r_mean_b, r_sem_b = compute_condition_group_stats(samples_r_b)

            region_names.append(roi)
            rows_t_a.append(t_a)
            rows_p_uncorr_a.append(p_a_raw)
            rows_slope_mean_a.append(slope_mean_a)
            rows_slope_sem_a.append(slope_sem_a)
            rows_t_b.append(t_b)
            rows_p_uncorr_b.append(p_b_raw)
            rows_slope_mean_b.append(slope_mean_b)
            rows_slope_sem_b.append(slope_sem_b)
            rows_activity_mean_a.append(act_mean_a)
            rows_activity_sem_a.append(act_sem_a)
            rows_activity_mean_b.append(act_mean_b)
            rows_activity_sem_b.append(act_sem_b)
            rows_r_value_mean_a.append(r_mean_a)
            rows_r_value_sem_a.append(r_sem_a)
            rows_r_value_mean_b.append(r_mean_b)
            rows_r_value_sem_b.append(r_sem_b)
            epoch_t_a.append(ep_t_a)
            epoch_p_a.append(ep_p_a)
            epoch_df_a.append(ep_df_a)
            epoch_mean_a.append(ep_mean_a)
            epoch_sem_a.append(ep_sem_a)
            epoch_t_b.append(ep_t_b)
            epoch_p_b.append(ep_p_b)
            epoch_df_b.append(ep_df_b)
            epoch_mean_b.append(ep_mean_b)
            epoch_sem_b.append(ep_sem_b)
            roi_channel_counts.append(channel_count)
            roi_subject_counts.append(subject_count)
            slope_a_contribution_samples.append(samples_slope_a)
            slope_b_contribution_samples.append(samples_slope_b)
            activity_a_contribution_samples.append(samples_mean_a)
            activity_b_contribution_samples.append(samples_mean_b)
            contribution_label_rows.append([f"{r.subject}/{r.channel}" for r in records])
            contributions_out.extend(
                ROIChannelContribution(
                    roi=record.roi,
                    subject=record.subject,
                    channel=record.channel,
                    source_stats_file=record.source_stats_file,
                )
                for record in records
            )

        n_rois = len(region_names)
        n_times = int(len(first.time_axis_s))

        def _stack_rows(rows: list[np.ndarray]) -> np.ndarray:
            if rows:
                return np.stack(rows, axis=0).astype(np.float64)
            return np.empty((0, n_times), dtype=np.float64)

        def _arr_1d(values: list[float]) -> np.ndarray:
            return np.array(values, dtype=np.float64)

        t_values_a = _stack_rows(rows_t_a)
        p_values_uncorr_a = _stack_rows(rows_p_uncorr_a)
        t_values_b = _stack_rows(rows_t_b)
        p_values_uncorr_b = _stack_rows(rows_p_uncorr_b)

        # Apply p-value correction separately per condition across ROI x time
        p_values_a = _apply_correction_2d(p_values_uncorr_a, method=method)
        p_values_b = _apply_correction_2d(p_values_uncorr_b, method=method)

        alpha = self.params.significance_alpha
        sig_mask_a = np.isfinite(p_values_a) & (p_values_a < alpha)
        sig_mask_b = np.isfinite(p_values_b) & (p_values_b < alpha)

        result = TrialSlopeStatsGroupProcessingResult(
            source_group=BIDSFileGroup(primary=files[0], secondaries=files[1:]),
            metadata={
                "p_value_correction_method": method,
                "significance_alpha": alpha,
                "roi_mode": self.params.roi_mode,
                "atlas_name": self.params.atlas_name,
                "binning_mode": first.binning_mode,
                "window_ms": first.window_ms,
                "n_bins": first.n_bins,
                "effective_n_bins": first.effective_n_bins,
            },
            output_entities={"subject": "group"},
            condition_a_slope_t_values=t_values_a,
            condition_a_slope_p_values=p_values_a,
            condition_a_slope_p_values_uncorrected=p_values_uncorr_a,
            condition_a_slope_significant_mask=sig_mask_a,
            condition_b_slope_t_values=t_values_b,
            condition_b_slope_p_values=p_values_b,
            condition_b_slope_p_values_uncorrected=p_values_uncorr_b,
            condition_b_slope_significant_mask=sig_mask_b,
            condition_a_slope_mean=_stack_rows(rows_slope_mean_a),
            condition_a_slope_sem=_stack_rows(rows_slope_sem_a),
            condition_b_slope_mean=_stack_rows(rows_slope_mean_b),
            condition_b_slope_sem=_stack_rows(rows_slope_sem_b),
            condition_a_activity_mean=_stack_rows(rows_activity_mean_a),
            condition_a_activity_sem=_stack_rows(rows_activity_sem_a),
            condition_b_activity_mean=_stack_rows(rows_activity_mean_b),
            condition_b_activity_sem=_stack_rows(rows_activity_sem_b),
            condition_a_r_value_mean=_stack_rows(rows_r_value_mean_a),
            condition_a_r_value_sem=_stack_rows(rows_r_value_sem_a),
            condition_b_r_value_mean=_stack_rows(rows_r_value_mean_b),
            condition_b_r_value_sem=_stack_rows(rows_r_value_sem_b),
            condition_a_epoch_slope_t=_arr_1d(epoch_t_a),
            condition_a_epoch_slope_p=_arr_1d(epoch_p_a),
            condition_a_epoch_slope_df=_arr_1d(epoch_df_a),
            condition_a_epoch_slope_mean=_arr_1d(epoch_mean_a),
            condition_a_epoch_slope_sem=_arr_1d(epoch_sem_a),
            condition_b_epoch_slope_t=_arr_1d(epoch_t_b),
            condition_b_epoch_slope_p=_arr_1d(epoch_p_b),
            condition_b_epoch_slope_df=_arr_1d(epoch_df_b),
            condition_b_epoch_slope_mean=_arr_1d(epoch_mean_b),
            condition_b_epoch_slope_sem=_arr_1d(epoch_sem_b),
            time_axis_s=first.time_axis_s.copy(),
            region_names=region_names,
            condition_labels=first.condition_labels,
            roi_channel_counts=np.array(roi_channel_counts, dtype=np.int64),
            roi_subject_counts=np.array(roi_subject_counts, dtype=np.int64),
            contributions=contributions_out,
            condition_a_slope_contributions=slope_a_contribution_samples,
            condition_b_slope_contributions=slope_b_contribution_samples,
            condition_a_activity_contributions=activity_a_contribution_samples,
            condition_b_activity_contributions=activity_b_contribution_samples,
            contribution_labels=contribution_label_rows,
            p_value_correction_method=method,
            significance_alpha=alpha,
            roi_mode=self.params.roi_mode,
            atlas_name=self.params.atlas_name,
            source_trial_slope_stats_files=[str(s.stats_file.path) for s in snapshots],
            source_electrodes_files=sorted(used_electrode_paths),
            excluded_rois=excluded_rois,
        )
        return result


# ---------------------------------------------------------------------------
# P-value correction helper
# ---------------------------------------------------------------------------

def _apply_correction_2d(
    p_values: np.ndarray,
    *,
    method: str,
) -> np.ndarray:
    """Apply correction to a (n_rois, n_times) matrix by flattening, correcting, reshaping."""
    if method == "none" or p_values.size == 0:
        return p_values.copy()
    flat = p_values.ravel()
    corrected_flat = correct_p_values(flat, method=method)  # type: ignore[arg-type]
    return corrected_flat.reshape(p_values.shape)


# ---------------------------------------------------------------------------
# ROI record collection
# ---------------------------------------------------------------------------

def _collect_manual_roi_records(
    *,
    snapshots: Sequence[_SlopeStatsSnapshot],
    manual_region_channels: dict[str, dict[str, list[str]]],
) -> dict[str, list[_ContributionRecord]]:
    roi_records: dict[str, list[_ContributionRecord]] = {
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
                roi_records[roi].append(
                    _ContributionRecord(
                        roi=roi,
                        subject=subject_key,
                        channel=snapshot.channel_names[idx],
                        source_stats_file=str(snapshot.stats_file.path),
                        slope_a_values=np.asarray(snapshot.condition_a_slope[idx, :], dtype=np.float64),
                        slope_b_values=np.asarray(snapshot.condition_b_slope[idx, :], dtype=np.float64),
                        mean_a_values=np.asarray(snapshot.condition_a_mean[idx, :], dtype=np.float64),
                        mean_b_values=np.asarray(snapshot.condition_b_mean[idx, :], dtype=np.float64),
                        r_value_a_values=np.asarray(snapshot.condition_a_r_value[idx, :], dtype=np.float64),
                        r_value_b_values=np.asarray(snapshot.condition_b_r_value[idx, :], dtype=np.float64),
                    )
                )
    return roi_records


def _collect_atlas_roi_records(
    *,
    snapshots: Sequence[_SlopeStatsSnapshot],
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
                        slope_a_values=np.asarray(snapshot.condition_a_slope[idx, :], dtype=np.float64),
                        slope_b_values=np.asarray(snapshot.condition_b_slope[idx, :], dtype=np.float64),
                        mean_a_values=np.asarray(snapshot.condition_a_mean[idx, :], dtype=np.float64),
                        mean_b_values=np.asarray(snapshot.condition_b_mean[idx, :], dtype=np.float64),
                        r_value_a_values=np.asarray(snapshot.condition_a_r_value[idx, :], dtype=np.float64),
                        r_value_b_values=np.asarray(snapshot.condition_b_r_value[idx, :], dtype=np.float64),
                    )
                )
    return roi_records, used_electrode_paths


def _resolve_snapshot_atlas_regions(
    *,
    snapshot: _SlopeStatsSnapshot,
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
                if not channel or not region or _is_na_like_region_label(region):
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


# ---------------------------------------------------------------------------
# Compatibility validation
# ---------------------------------------------------------------------------

def _validate_group_compatibility(snapshots: Sequence[_SlopeStatsSnapshot]) -> None:
    if not snapshots:
        raise ValueError("At least one trial_slope_stats snapshot is required.")
    if any(snapshot.analysis_level != "channel" for snapshot in snapshots):
        non_channel = [
            snapshot.stats_file.path.name
            for snapshot in snapshots
            if snapshot.analysis_level != "channel"
        ]
        raise ValueError(
            "trial_slope_stats_group requires channel-level inputs. "
            f"Found non-channel files: {', '.join(non_channel)}."
        )

    reference = snapshots[0].signature.key
    for snapshot in snapshots[1:]:
        if snapshot.signature.key != reference:
            raise ValueError(
                "Incompatible trial_slope_stats inputs in one processing group. "
                "Use build_trial_slope_stats_compatible_groups() to split heterogeneous files."
            )


# ---------------------------------------------------------------------------
# Snapshot I/O helpers
# ---------------------------------------------------------------------------

def _read_snapshot_signature(stats_file: BIDSFile) -> _SnapshotSignature:
    raw = _load_raw_slope_stats(stats_file)
    return _build_signature(stats_file=stats_file, raw=raw)


def _load_slope_stats_snapshot(stats_file: BIDSFile) -> _SlopeStatsSnapshot:
    raw = _load_raw_slope_stats(stats_file)
    raw_subject = str(stats_file.get("subject") or stats_file.get("sub") or "").strip()
    subject = normalize_subject_value(raw_subject)

    channel_index_by_norm: dict[str, int] = {}
    for idx, name in enumerate(raw.channels):
        key = normalize_channel_name(name)
        channel_index_by_norm.setdefault(key, idx)

    signature = _build_signature(stats_file=stats_file, raw=raw)
    return _SlopeStatsSnapshot(
        stats_file=stats_file,
        subject=subject,
        task=str(stats_file.get("task") or ""),
        source_desc=str(stats_file.get("desc") or ""),
        condition_labels=raw.condition_labels,
        channel_names=raw.channels,
        channel_index_by_norm=channel_index_by_norm,
        time_axis_s=raw.time_axis_s,
        condition_a_slope=raw.condition_a_slope,
        condition_b_slope=raw.condition_b_slope,
        condition_a_mean=raw.condition_a_mean,
        condition_b_mean=raw.condition_b_mean,
        condition_a_r_value=raw.condition_a_r_value,
        condition_b_r_value=raw.condition_b_r_value,
        analysis_level=raw.analysis_level,
        binning_mode=raw.binning_mode,
        window_ms=raw.window_ms,
        n_bins=raw.n_bins,
        effective_n_bins=raw.effective_n_bins,
        source_ieeg_files=raw.source_ieeg_files,
        source_electrodes_files=raw.source_electrodes_files,
        signature=signature,
    )


def _build_signature(
    *,
    stats_file: BIDSFile,
    raw: _RawSlopeStatsData,
) -> _SnapshotSignature:
    time_hash = hashlib.sha1(
        np.asarray(raw.time_axis_s, dtype=np.float64).tobytes()
    ).hexdigest()
    return _SnapshotSignature(
        task=str(stats_file.get("task") or ""),
        source_desc=str(stats_file.get("desc") or ""),
        condition_labels=raw.condition_labels,
        time_axis_hash=time_hash,
        time_axis_len=int(len(raw.time_axis_s)),
        binning_mode=str(raw.binning_mode or "none"),
        window_ms=float(raw.window_ms),
        n_bins=int(raw.n_bins),
        effective_n_bins=int(raw.effective_n_bins),
        analysis_level=str(raw.analysis_level or "channel"),
    )


def _load_raw_slope_stats(stats_file: BIDSFile) -> _RawSlopeStatsData:
    extension = (stats_file.extension or "").lower()
    if extension == ".mat":
        return _load_raw_from_matlab(stats_file)
    return _load_raw_from_hdf5(stats_file)


def _load_raw_from_hdf5(stats_file: BIDSFile) -> _RawSlopeStatsData:
    with stats_file.ensure_loaded() as fh:
        analysis_type = str_scalar(dataset_or_none(fh, "meta/analysis_type"), default="")
        if analysis_type and analysis_type != "slope_regression":
            raise ValueError(
                f"{stats_file.path.name}: not a trial_slope_stats file "
                f"(meta/analysis_type={analysis_type!r})."
            )

        analysis_level = str_scalar(dataset_or_none(fh, "meta/analysis_level"), default="channel")
        axis_name = "channel" if analysis_level == "channel" else "region"
        if "axes" not in fh or axis_name not in fh["axes"]:
            raise ValueError(
                f"{stats_file.path.name}: axes/{axis_name} dataset is required."
            )
        channels = decode_str_array(np.asarray(fh["axes"][axis_name][:]))
        time_axis_s = np.asarray(fh["axes"]["time_s"][:], dtype=np.float64)
        condition_labels = _read_condition_labels_hdf5(fh)

        n_ch = len(channels)
        n_t = len(time_axis_s)
        _empty = np.full((n_ch, n_t), np.nan, dtype=np.float64)

        def _read_2d(path_key: str) -> np.ndarray:
            ds = dataset_or_none(fh, path_key)
            if ds is None:
                return _empty.copy()
            return coerce_feature_time(
                np.asarray(ds[:], dtype=np.float64), n_features=n_ch, n_times=n_t
            )

        condition_a_slope = _read_2d("regression/condition_a/slope")
        condition_b_slope = _read_2d("regression/condition_b/slope")
        condition_a_r_value = _read_2d("regression/condition_a/r_value")
        condition_b_r_value = _read_2d("regression/condition_b/r_value")

        # Mean activity is stored under means/{condition_name}
        cond_a_name = condition_labels[0]
        cond_b_name = condition_labels[1]
        condition_a_mean = _read_2d(f"means/{cond_a_name}")
        condition_b_mean = _read_2d(f"means/{cond_b_name}")

        binning_mode = str_scalar(dataset_or_none(fh, "meta/binning_mode"), default="none")
        window_ms = float_scalar(dataset_or_none(fh, "meta/window_ms"), default=0.0)
        n_bins = int_scalar(dataset_or_none(fh, "meta/n_bins"), default=0)
        effective_n_bins = int_scalar(
            dataset_or_none(fh, "meta/effective_n_bins"), default=n_t
        )

        source_ieeg_files: list[str] = []
        source_electrodes_files: list[str] = []
        if "provenance" in fh:
            prov = fh["provenance"]
            if "source_ieeg_files" in prov:
                source_ieeg_files = decode_str_array(
                    np.asarray(prov["source_ieeg_files"][:], dtype=object)
                )
            if "source_electrodes_files" in prov:
                source_electrodes_files = decode_str_array(
                    np.asarray(prov["source_electrodes_files"][:], dtype=object)
                )

    return _RawSlopeStatsData(
        analysis_level=analysis_level,
        channels=channels,
        time_axis_s=time_axis_s,
        condition_labels=condition_labels,
        condition_a_slope=condition_a_slope,
        condition_b_slope=condition_b_slope,
        condition_a_mean=condition_a_mean,
        condition_b_mean=condition_b_mean,
        condition_a_r_value=condition_a_r_value,
        condition_b_r_value=condition_b_r_value,
        binning_mode=binning_mode,
        window_ms=window_ms,
        n_bins=n_bins,
        effective_n_bins=effective_n_bins,
        source_ieeg_files=source_ieeg_files,
        source_electrodes_files=source_electrodes_files,
    )


def _load_raw_from_matlab(stats_file: BIDSFile) -> _RawSlopeStatsData:
    with stats_file.ensure_loaded() as mat:
        data = mat["data"]
        meta = data.meta
        axes = data.axes
        prov = getattr(data, "provenance", None)

        analysis_level = mat_str(getattr(meta, "analysis_level", None), default="channel")
        axis_attr = "channel" if analysis_level == "channel" else "region"
        channels = mat_str_list(getattr(axes, axis_attr, None))
        if not channels:
            raise ValueError(
                f"{stats_file.path.name}: axes.{axis_attr} array is required in .mat file."
            )

        time_axis_s = np.asarray(axes.time_s, dtype=np.float64).ravel()
        condition_labels = _mat_condition_labels(meta)
        n_ch = len(channels)
        n_t = len(time_axis_s)
        _empty = np.full((n_ch, n_t), np.nan, dtype=np.float64)

        def _read_mat_2d(obj: Any, attr: str) -> np.ndarray:
            arr = getattr(obj, attr, None)
            if arr is None:
                return _empty.copy()
            return coerce_feature_time(
                np.asarray(arr, dtype=np.float64), n_features=n_ch, n_times=n_t
            )

        regression = getattr(data, "regression", None)
        cond_a_reg = getattr(regression, "condition_a", None) if regression is not None else None
        cond_b_reg = getattr(regression, "condition_b", None) if regression is not None else None

        condition_a_slope = _read_mat_2d(cond_a_reg, "slope") if cond_a_reg is not None else _empty.copy()
        condition_b_slope = _read_mat_2d(cond_b_reg, "slope") if cond_b_reg is not None else _empty.copy()
        condition_a_r_value = _read_mat_2d(cond_a_reg, "r_value") if cond_a_reg is not None else _empty.copy()
        condition_b_r_value = _read_mat_2d(cond_b_reg, "r_value") if cond_b_reg is not None else _empty.copy()

        means = getattr(data, "means", None)
        safe_a = matlab_safe_name(condition_labels[0])
        safe_b = matlab_safe_name(condition_labels[1])
        condition_a_mean = _read_mat_2d(means, safe_a) if means is not None else _empty.copy()
        condition_b_mean = _read_mat_2d(means, safe_b) if means is not None else _empty.copy()

        binning_mode = mat_str(getattr(meta, "binning_mode", None), default="none")
        window_ms = mat_float(getattr(meta, "window_ms", None), default=0.0)
        n_bins = mat_int(getattr(meta, "n_bins", None), default=0)
        effective_n_bins = mat_int(getattr(meta, "effective_n_bins", None), default=n_t)

        source_ieeg_files: list[str] = []
        source_electrodes_files: list[str] = []
        if prov is not None:
            source_ieeg_files = mat_str_list(getattr(prov, "source_ieeg_files", None))
            source_electrodes_files = mat_str_list(getattr(prov, "source_electrodes_files", None))

    return _RawSlopeStatsData(
        analysis_level=analysis_level,
        channels=channels,
        time_axis_s=time_axis_s,
        condition_labels=condition_labels,
        condition_a_slope=condition_a_slope,
        condition_b_slope=condition_b_slope,
        condition_a_mean=condition_a_mean,
        condition_b_mean=condition_b_mean,
        condition_a_r_value=condition_a_r_value,
        condition_b_r_value=condition_b_r_value,
        binning_mode=binning_mode,
        window_ms=window_ms,
        n_bins=n_bins,
        effective_n_bins=effective_n_bins,
        source_ieeg_files=source_ieeg_files,
        source_electrodes_files=source_electrodes_files,
    )


def _read_condition_labels_hdf5(fh: h5py.File) -> tuple[str, str]:
    labels_ds = dataset_or_none(fh, "meta/trial_count_labels")
    if labels_ds is not None:
        labels = decode_str_array(np.asarray(labels_ds[:], dtype=object))
        if len(labels) >= 2:
            return labels[0], labels[1]
    if "means" in fh:
        mean_keys = [key for key in fh["means"].keys()]
        if len(mean_keys) >= 2:
            return mean_keys[0], mean_keys[1]
    return "condition_a", "condition_b"


def _mat_condition_labels(meta: Any) -> tuple[str, str]:
    labels = mat_str_list(getattr(meta, "trial_count_labels", None))
    if len(labels) >= 2:
        return labels[0], labels[1]
    return "condition_a", "condition_b"


def _is_na_like_region_label(label: str) -> bool:
    normalized = "".join(ch for ch in label.strip().casefold() if ch.isalnum())
    return normalized in {"na", "notavailable", "notapplicable"}
