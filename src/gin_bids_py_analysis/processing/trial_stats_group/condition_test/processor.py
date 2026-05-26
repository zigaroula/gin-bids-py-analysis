from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import h5py
import numpy as np
from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.bids.helpers import (
    normalize_subject_value,
)
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

from gin_bids_py_analysis.processing.utils.group_stats import (
    compute_condition_group_stats,
    compute_one_sample_epoch_summary,
    compute_one_sample_timecourse,
)

from ..processor import (
    BaseRawTrialStatsData,
    BaseTrialStatsGroupContributionRecord,
    BaseTrialStatsGroupProcessing,
    BaseTrialStatsGroupSnapshot,
    BaseTrialStatsGroupSnapshotSignature,
    build_compatible_groups,
    collect_atlas_roi_records,
    collect_manual_roi_records,
    hash_time_axis,
    validate_group_compatibility,
)
from ..result import (
    GroupEpochStats,
    GroupEstimate,
    GroupEstimatePair,
    GroupTimecourseStats,
    IndexedConditionContributions,
    ROIChannelContribution,
)
from .params import ConditionTestGroupParams
from .result import ConditionTestEpochSummary, ConditionTestGroupProcessingResult
from gin_bids_py_analysis.processing.utils.cluster_permutation import (
    compute_cluster_null_distribution,
    compute_mne_cluster_permutation,
    compute_cluster_permutation_pvalue,
    find_temporal_clusters,
)
from gin_bids_py_analysis.processing.utils.statistics import correct_p_values


@dataclass(frozen=True)
class _RawConditionTestStatsData(BaseRawTrialStatsData):
    """Format-agnostic in-memory representation of one subject condition_test file."""

    metric_values: np.ndarray  # shape (n_channels, n_times)
    condition_a_mean_values: np.ndarray  # shape (n_channels, n_times)
    condition_b_mean_values: np.ndarray  # shape (n_channels, n_times)
    permuted_t_values: np.ndarray | None  # shape (n_perm, n_channels, n_times) or None


@dataclass(frozen=True)
class _SnapshotSignature(BaseTrialStatsGroupSnapshotSignature):
    task: str
    source_desc: str
    condition_labels: tuple[str, str]
    time_axis_hash: str
    time_axis_len: int
    binning_mode: str
    window_ms: float
    n_bins: int
    effective_n_bins: int
    activity_zscore: str
    activity_baseline_tmin_s: float
    activity_baseline_tmax_s: float
    analysis_level: str

    @property
    def key(self) -> tuple[Any, ...]:
        return self.base_key + (
            self.activity_zscore,
            self.activity_baseline_tmin_s,
            self.activity_baseline_tmax_s,
        )


@dataclass(frozen=True)
class _ConditionTestStatsSnapshot(BaseTrialStatsGroupSnapshot):
    raw: _RawConditionTestStatsData


@dataclass(frozen=True)
class _ContributionRecord(BaseTrialStatsGroupContributionRecord):
    roi: str
    subject: str
    channel: str
    source_stats_file: str
    values: np.ndarray
    condition_a_values: np.ndarray  # shape (n_times,)
    condition_b_values: np.ndarray  # shape (n_times,)
    permuted_t_values: np.ndarray | None  # shape (n_perm, n_times) or None


def build_condition_test_compatible_groups(
    stats_files: Sequence[BIDSFile],
    *,
    primary_condition_metric: str,
) -> list[BIDSFileGroup]:
    """Group subject-level condition_test files by compatibility."""
    return build_compatible_groups(
        stats_files,
        read_signature=lambda file: _read_snapshot_signature(
            file,
            source_metric=primary_condition_metric,
        ),
    )


class ConditionTestGroupProcessing(BaseTrialStatsGroupProcessing):
    """Compute group-level ROI one-sample tests from subject-level condition_test files."""

    def __init__(self, params: ConditionTestGroupParams) -> None:
        self.params = params

    def process_group(
        self,
        group: BIDSFileGroup,
        progress_tracking_position: int = 0,
    ) -> ConditionTestGroupProcessingResult:
        del progress_tracking_position
        files = self.sorted_group_files(group)
        if not files:
            raise ValueError(
                "ConditionTestGroupProcessing requires at least one condition_test stats file."
            )

        snapshots = [
            _load_trial_stats_snapshot(
                file,
                source_metric=self.params.primary_condition_metric,
            )
            for file in files
        ]
        _validate_group_compatibility(snapshots)

        method = self.params.p_value_correction_method
        rng: np.random.Generator | None = None
        if method == "cluster_permutation":
            if self.params.cluster_permutation_method == "custom":
                for snap in snapshots:
                    if snap.raw.permuted_t_values is None:
                        raise ValueError(
                            f"cluster_permutation with cluster_permutation_method='custom' "
                            f"requires permuted_t_values in all source files, "
                            f"but {snap.stats_file.path.name} has none. "
                            "Re-run the subject-level condition_test analysis "
                            "with n_permutations > 0."
                        )
            rng = np.random.default_rng(self.params.permutation_seed)

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
        cluster_perm_t_collection: list[list[np.ndarray] | None] = []
        cluster_observed_collection: list[np.ndarray | None] = []
        rows_mean: list[np.ndarray] = []
        rows_sem: list[np.ndarray] = []
        rows_cond_a_mean: list[np.ndarray] = []
        rows_cond_a_sem: list[np.ndarray] = []
        rows_cond_b_mean: list[np.ndarray] = []
        rows_cond_b_sem: list[np.ndarray] = []
        roi_channel_counts: list[int] = []
        roi_subject_counts: list[int] = []
        summary_t: list[float] = []
        summary_p: list[float] = []
        summary_df: list[float] = []
        summary_mean: list[float] = []
        summary_sem: list[float] = []
        contributions_out: list[ROIChannelContribution] = []
        cond_a_contribution_samples: list[np.ndarray] = []
        cond_b_contribution_samples: list[np.ndarray] = []
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

            samples = np.stack([record.values for record in records], axis=0).astype(np.float64)
            t_values, p_values_raw, mean_values, sem_values = compute_one_sample_timecourse(samples)
            t_summary, p_summary, df_summary, mean_summary, sem_summary = (
                compute_one_sample_epoch_summary(samples)
            )

            samples_a = np.stack([record.condition_a_values for record in records], axis=0).astype(np.float64)
            samples_b = np.stack([record.condition_b_values for record in records], axis=0).astype(np.float64)
            cond_a_mean, cond_a_sem = compute_condition_group_stats(samples_a)
            cond_b_mean, cond_b_sem = compute_condition_group_stats(samples_b)

            if method == "cluster_permutation":
                if self.params.cluster_permutation_method == "custom":
                    perm_t_list: list[np.ndarray] | None = [
                        np.asarray(r.permuted_t_values, dtype=np.float64)
                        for r in records
                        if r.permuted_t_values is not None
                    ]
                    observed_samples: np.ndarray | None = None
                else:
                    perm_t_list = None
                    observed_samples = np.asarray(samples, dtype=np.float64)
            else:
                perm_t_list = None
                observed_samples = None
            cluster_perm_t_collection.append(perm_t_list)
            cluster_observed_collection.append(observed_samples)

            region_names.append(roi)
            rows_t.append(t_values)
            rows_p_uncorrected.append(p_values_raw)
            rows_mean.append(mean_values)
            rows_sem.append(sem_values)
            rows_cond_a_mean.append(cond_a_mean)
            rows_cond_a_sem.append(cond_a_sem)
            rows_cond_b_mean.append(cond_b_mean)
            rows_cond_b_sem.append(cond_b_sem)
            cond_a_contribution_samples.append(samples_a)
            cond_b_contribution_samples.append(samples_b)
            contribution_label_rows.append([f"{r.subject}/{r.channel}" for r in records])
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
            t_values = self.stack_rows(rows_t, n_times)
            metric_mean = self.stack_rows(rows_mean, n_times)
            metric_sem = self.stack_rows(rows_sem, n_times)
            condition_a_activity_mean = self.stack_rows(rows_cond_a_mean, n_times)
            condition_a_activity_sem = self.stack_rows(rows_cond_a_sem, n_times)
            condition_b_activity_mean = self.stack_rows(rows_cond_b_mean, n_times)
            condition_b_activity_sem = self.stack_rows(rows_cond_b_sem, n_times)
        else:
            t_values = np.empty((0, n_times), dtype=np.float64)
            p_values_uncorrected = np.empty((0, n_times), dtype=np.float64)
            metric_mean = np.empty((0, n_times), dtype=np.float64)
            metric_sem = np.empty((0, n_times), dtype=np.float64)
            condition_a_activity_mean = np.empty((0, n_times), dtype=np.float64)
            condition_a_activity_sem = np.empty((0, n_times), dtype=np.float64)
            condition_b_activity_mean = np.empty((0, n_times), dtype=np.float64)
            condition_b_activity_sem = np.empty((0, n_times), dtype=np.float64)

        p_values = correct_p_values(
            p_values_uncorrected,
            method=method,
        )

        cluster_p_values_out: np.ndarray | None = None
        cluster_windows_out: list[list[tuple[float, float]]] | None = None
        cluster_null_dists_out: list[np.ndarray] | None = None
        if method == "cluster_permutation" and rows_p_uncorrected and rng is not None:
            n_keep = self.params.n_clusters_to_keep
            cluster_p_values_list: list[float] = []
            cluster_windows_list: list[list[tuple[float, float]]] = []
            cluster_null_dists_list: list[np.ndarray] = []
            for roi_idx in range(len(region_names)):
                roi_t = rows_t[roi_idx]
                roi_p_raw = rows_p_uncorrected[roi_idx]
                h_mask = roi_p_raw < self.params.cluster_threshold_alpha
                observed_clusters = find_temporal_clusters(h_mask, roi_t)
                perm_t_roi = cluster_perm_t_collection[roi_idx]
                if not perm_t_roi:
                    if self.params.cluster_permutation_method == "mne":
                        obs_samples = cluster_observed_collection[roi_idx]
                        if obs_samples is None:
                            cluster_p_values_list.append(1.0)
                            cluster_windows_list.append([])
                            cluster_null_dists_list.append(np.zeros(0, dtype=np.float64))
                            continue
                        seed = int(rng.integers(0, np.iinfo(np.int32).max))
                        best_p, top_windows_idx, top_p_vals, null = compute_mne_cluster_permutation(
                            obs_samples,
                            cluster_threshold_alpha=self.params.cluster_threshold_alpha,
                            n_group_perm=self.params.n_group_permutations,
                            seed=seed,
                            n_clusters_to_keep=n_keep,
                        )
                        roi_windows: list[tuple[float, float]] = []
                        for win_idx, p in zip(top_windows_idx, top_p_vals):
                            if p < self.params.significance_alpha:
                                roi_windows.append((
                                    float(first.time_axis_s[win_idx[0]]),
                                    float(first.time_axis_s[win_idx[1]]),
                                ))
                        cluster_p_values_list.append(best_p)
                        cluster_windows_list.append(roi_windows)
                        cluster_null_dists_list.append(null)
                    else:
                        cluster_p_values_list.append(1.0)
                        cluster_windows_list.append([])
                        cluster_null_dists_list.append(np.zeros(0, dtype=np.float64))
                else:
                    null = compute_cluster_null_distribution(
                        perm_t_roi,
                        cluster_threshold_alpha=self.params.cluster_threshold_alpha,
                        n_group_perm=self.params.n_group_permutations,
                        rng=rng,
                    )
                    top_candidates = observed_clusters[:n_keep]
                    roi_windows = []
                    best_p = 1.0
                    for i, (start, end, tsum) in enumerate(top_candidates):
                        p = compute_cluster_permutation_pvalue(tsum, null)
                        if i == 0:
                            best_p = p
                        if p < self.params.significance_alpha:
                            roi_windows.append((
                                float(first.time_axis_s[start]),
                                float(first.time_axis_s[end]),
                            ))
                    cluster_p_values_list.append(best_p)
                    cluster_windows_list.append(roi_windows)
                    cluster_null_dists_list.append(null)
            cluster_p_values_out = np.asarray(cluster_p_values_list, dtype=np.float64)
            cluster_windows_out = cluster_windows_list
            cluster_null_dists_out = cluster_null_dists_list

        if method == "cluster_permutation" and cluster_windows_out is not None:
            significant_mask = np.zeros_like(p_values, dtype=bool)
            for roi_idx, roi_windows in enumerate(cluster_windows_out):
                for t_start_s, t_end_s in roi_windows:
                    in_window = (
                        (first.time_axis_s >= t_start_s)
                        & (first.time_axis_s <= t_end_s)
                    )
                    significant_mask[roi_idx, in_window] = True
        else:
            significant_mask = np.isfinite(p_values) & (p_values < self.params.significance_alpha)

        return ConditionTestGroupProcessingResult(
            source_group=group,
            metadata={
                "primary_condition_metric": self.params.primary_condition_metric,
                "roi_mode": self.params.roi_mode,
                "atlas_name": self.params.atlas_name,
                "min_channels_per_roi": self.params.min_channels_per_roi,
                "min_subjects_per_roi": self.params.min_subjects_per_roi,
                "p_value_correction_method": method,
                "significance_alpha": self.params.significance_alpha,
                "n_group_permutations": self.params.n_group_permutations,
                "cluster_threshold_alpha": self.params.cluster_threshold_alpha,
                "cluster_permutation_method": self.params.cluster_permutation_method,
                "condition_labels": list(first.condition_labels),
                "task": first.task,
                "source_desc": first.source_desc,
                "binning_mode": first.binning_mode,
                "window_ms": first.window_ms,
                "n_bins": first.n_bins,
                "effective_n_bins": first.effective_n_bins,
                "activity_zscore": first.raw.activity_zscore,
                "activity_baseline_tmin_s": first.raw.activity_baseline_tmin_s,
                "activity_baseline_tmax_s": first.raw.activity_baseline_tmax_s,
                "excluded_rois": dict(excluded_rois),
            },
            signal_activity_stats=GroupTimecourseStats(
                t_values=t_values,
                p_values=p_values,
                p_values_uncorrected=p_values_uncorrected,
                significant_mask=significant_mask,
            ),
            condition_difference=GroupEstimate(mean=metric_mean, sem=metric_sem),
            time_axis_s=first.time_axis_s.astype(np.float64),
            region_names=region_names,
            primary_condition_metric=self.params.primary_condition_metric,
            condition_labels=first.condition_labels,
            p_value_correction_method=self.params.p_value_correction_method,
            significance_alpha=self.params.significance_alpha,
            roi_mode=self.params.roi_mode,
            atlas_name=self.params.atlas_name,
            signal_activity_epoch=GroupEpochStats(
                t=self.array_1d(summary_t),
                p=self.array_1d(summary_p),
                df=self.array_1d(summary_df),
            ),
            summary_epoch=ConditionTestEpochSummary(
                t_values=self.array_1d(summary_t),
                p_values=self.array_1d(summary_p),
                df=self.array_1d(summary_df),
                condition_difference=GroupEstimate(
                    mean=self.array_1d(summary_mean),
                    sem=self.array_1d(summary_sem),
                ),
            ),
            roi_channel_counts=np.asarray(roi_channel_counts, dtype=np.int64),
            roi_subject_counts=np.asarray(roi_subject_counts, dtype=np.int64),
            contributions=contributions_out,
            signal_activity=GroupEstimatePair(
                condition_a=GroupEstimate(
                    mean=condition_a_activity_mean,
                    sem=condition_a_activity_sem,
                ),
                condition_b=GroupEstimate(
                    mean=condition_b_activity_mean,
                    sem=condition_b_activity_sem,
                ),
            ),
            signal_activity_contributions=IndexedConditionContributions(
                condition_a=cond_a_contribution_samples,
                condition_b=cond_b_contribution_samples,
                labels=contribution_label_rows,
            ),
            source_subject_stats_files=[str(snapshot.stats_file.path) for snapshot in snapshots],
            source_electrodes_files=sorted(used_electrode_paths),
            excluded_rois=excluded_rois,
            cluster_p_values=cluster_p_values_out,
            cluster_windows_s=cluster_windows_out,
            cluster_null_distributions=cluster_null_dists_out,
        )


def _collect_manual_roi_records(
    *,
    snapshots: Sequence[_ConditionTestStatsSnapshot],
    manual_region_channels: dict[str, dict[str, list[str]]],
) -> dict[str, list[_ContributionRecord]]:
    return collect_manual_roi_records(
        snapshots=snapshots,
        manual_region_channels=manual_region_channels,
        create_record=_create_contribution_record,
    )


def _collect_atlas_roi_records(
    *,
    snapshots: Sequence[_ConditionTestStatsSnapshot],
    atlas_name: str,
) -> tuple[dict[str, list[_ContributionRecord]], set[str]]:
    return collect_atlas_roi_records(
        snapshots=snapshots,
        atlas_name=atlas_name,
        create_record=_create_contribution_record,
    )


def _create_contribution_record(
    roi: str,
    subject: str,
    snapshot: _ConditionTestStatsSnapshot,
    idx: int,
) -> _ContributionRecord:
    return _ContributionRecord(
        roi=roi,
        subject=subject,
        channel=snapshot.channel_names[idx],
        source_stats_file=str(snapshot.stats_file.path),
        values=np.asarray(snapshot.raw.metric_values[idx, :], dtype=np.float64),
        condition_a_values=np.asarray(snapshot.raw.condition_a_mean_values[idx, :], dtype=np.float64),
        condition_b_values=np.asarray(snapshot.raw.condition_b_mean_values[idx, :], dtype=np.float64),
        permuted_t_values=(
            np.asarray(snapshot.raw.permuted_t_values[:, idx, :], dtype=np.float32)
            if snapshot.raw.permuted_t_values is not None
            else None
        ),
    )


def _validate_group_compatibility(snapshots: Sequence[_ConditionTestStatsSnapshot]) -> None:
    validate_group_compatibility(
        snapshots,
        empty_message="At least one condition_test snapshot is required.",
        non_channel_message=(
            "condition_test_group requires channel-level condition_test inputs."
        ),
        incompatible_message=(
            "Incompatible condition_test inputs in one processing group. "
            "Use build_condition_test_compatible_groups(...) to split heterogeneous files."
        ),
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
    activity_zscore: str,
    activity_baseline_tmin_s: float,
    activity_baseline_tmax_s: float,
) -> _SnapshotSignature:
    return _SnapshotSignature(
        task=str(stats_file.get("task") or ""),
        source_desc=str(stats_file.get("desc") or ""),
        condition_labels=condition_labels,
        time_axis_hash=hash_time_axis(time_axis_s),
        time_axis_len=int(len(time_axis_s)),
        binning_mode=str(binning_mode or "none"),
        window_ms=float(window_ms),
        n_bins=int(n_bins),
        effective_n_bins=int(effective_n_bins),
        activity_zscore=str(activity_zscore or "none"),
        activity_baseline_tmin_s=float(activity_baseline_tmin_s),
        activity_baseline_tmax_s=float(activity_baseline_tmax_s),
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
        activity_zscore=raw.activity_zscore,
        activity_baseline_tmin_s=raw.activity_baseline_tmin_s,
        activity_baseline_tmax_s=raw.activity_baseline_tmax_s,
    )


def _load_trial_stats_snapshot(
    stats_file: BIDSFile,
    *,
    source_metric: str,
) -> _ConditionTestStatsSnapshot:
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
        activity_zscore=raw.activity_zscore,
        activity_baseline_tmin_s=raw.activity_baseline_tmin_s,
        activity_baseline_tmax_s=raw.activity_baseline_tmax_s,
    )
    return _ConditionTestStatsSnapshot(
        stats_file=stats_file,
        subject=subject,
        task=str(stats_file.get("task") or ""),
        source_desc=str(stats_file.get("desc") or ""),
        condition_labels=raw.condition_labels,
        channel_names=raw.channels,
        channel_index_by_norm=channel_index_by_norm,
        time_axis_s=raw.time_axis_s,
        analysis_level=raw.analysis_level,
        binning_mode=raw.binning_mode,
        window_ms=raw.window_ms,
        n_bins=raw.n_bins,
        effective_n_bins=raw.effective_n_bins,
        source_ieeg_files=raw.source_ieeg_files,
        source_electrodes_files=raw.source_electrodes_files,
        signature=signature,
        raw=raw,
    )


def _load_raw_trial_stats(
    stats_file: BIDSFile,
    *,
    source_metric: str,
) -> _RawConditionTestStatsData:
    """Dispatch to the appropriate format-specific loader based on file extension."""
    extension = (stats_file.extension or "").lower()
    if extension == ".mat":
        return _load_raw_from_matlab(stats_file, source_metric=source_metric)
    return _load_raw_from_hdf5(stats_file, source_metric=source_metric)


def _load_raw_from_hdf5(
    stats_file: BIDSFile,
    *,
    source_metric: str,
) -> _RawConditionTestStatsData:
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
        n_ch = len(channels)
        n_t = len(time_axis_s)
        _empty_cond = np.full((n_ch, n_t), np.nan, dtype=np.float64)
        raw_a_ds = dataset_or_none(fh, "data/signal_activity/condition_a/mean")
        condition_a_mean_values = (
            coerce_feature_time(
                np.asarray(raw_a_ds[:], dtype=np.float64),
                n_features=n_ch,
                n_times=n_t,
            )
            if raw_a_ds is not None
            else _empty_cond.copy()
        )
        raw_b_ds = dataset_or_none(fh, "data/signal_activity/condition_b/mean")
        condition_b_mean_values = (
            coerce_feature_time(
                np.asarray(raw_b_ds[:], dtype=np.float64),
                n_features=n_ch,
                n_times=n_t,
            )
            if raw_b_ds is not None
            else _empty_cond.copy()
        )
        binning_mode = str_scalar(dataset_or_none(fh, "meta/binning_mode"), default="none")
        window_ms = float_scalar(dataset_or_none(fh, "meta/window_ms"), default=0.0)
        n_bins = int_scalar(dataset_or_none(fh, "meta/n_bins"), default=0)
        effective_n_bins = int_scalar(
            dataset_or_none(fh, "meta/effective_n_bins"),
            default=len(time_axis_s),
        )
        activity_zscore_ds = dataset_or_none(fh, "meta/activity_zscore")
        if activity_zscore_ds is None:
            raise ValueError(
                f"{stats_file.path.name}: unsupported legacy condition_test_group input; "
                "meta/activity_zscore is required."
            )
        activity_zscore = str_scalar(activity_zscore_ds, default="none")
        activity_baseline_tmin_s = float_scalar(
            dataset_or_none(fh, "meta/activity_baseline_tmin_s"),
            default=-0.2,
        )
        activity_baseline_tmax_s = float_scalar(
            dataset_or_none(fh, "meta/activity_baseline_tmax_s"),
            default=0.0,
        )
        source_ieeg_files = decode_str_array(
            np.asarray(fh["provenance"]["source_ieeg_files"][:], dtype=object)
        ) if "provenance" in fh and "source_ieeg_files" in fh["provenance"] else []
        source_electrodes_files = decode_str_array(
            np.asarray(fh["provenance"]["source_electrodes_files"][:], dtype=object)
        ) if "provenance" in fh and "source_electrodes_files" in fh["provenance"] else []
        perm_ds = dataset_or_none(fh, "stats/condition_contrast/permuted_t_values")
        permuted_t_values: np.ndarray | None = (
            np.asarray(perm_ds[:], dtype=np.float32) if perm_ds is not None else None
        )

    return _RawConditionTestStatsData(
        analysis_level=analysis_level,
        channels=channels,
        time_axis_s=time_axis_s,
        metric_values=metric_values,
        condition_a_mean_values=condition_a_mean_values,
        condition_b_mean_values=condition_b_mean_values,
        condition_labels=condition_labels,
        binning_mode=binning_mode,
        window_ms=window_ms,
        n_bins=n_bins,
        effective_n_bins=effective_n_bins,
        activity_zscore=activity_zscore,
        activity_baseline_tmin_s=activity_baseline_tmin_s,
        activity_baseline_tmax_s=activity_baseline_tmax_s,
        source_ieeg_files=source_ieeg_files,
        source_electrodes_files=source_electrodes_files,
        permuted_t_values=permuted_t_values,
    )


def _read_condition_labels_hdf5(fh: h5py.File) -> tuple[str, str]:
    labels_dataset = dataset_or_none(fh, "meta/condition_labels")
    if labels_dataset is not None:
        labels = decode_str_array(np.asarray(labels_dataset[:], dtype=object))
        if len(labels) >= 2:
            return labels[0], labels[1]
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
        dataset = dataset_or_none(fh, "data/signal_activity/difference/mean")
        if dataset is None:
            raise ValueError(f"{fh.filename}: data/signal_activity/difference/mean dataset is required.")
        raw = np.asarray(dataset[:], dtype=np.float64)
        return coerce_feature_time(raw, n_features=n_channels, n_times=n_times)

    if source_metric == "t_values":
        dataset = dataset_or_none(fh, "stats/condition_contrast/t_values")
        if dataset is None:
            raise ValueError(f"{fh.filename}: stats/condition_contrast/t_values dataset is required.")
        raw = np.asarray(dataset[:], dtype=np.float64)
        return coerce_feature_time(raw, n_features=n_channels, n_times=n_times)

    metric_name = "condition_a" if source_metric == "condition_a_mean" else "condition_b"
    dataset = dataset_or_none(fh, f"data/signal_activity/{metric_name}/mean")
    if dataset is None:
        raise ValueError(
            f"{fh.filename}: data/signal_activity/{metric_name}/mean dataset is required "
            f"for source_metric={source_metric!r}."
        )
    raw = np.asarray(dataset[:], dtype=np.float64)
    return coerce_feature_time(raw, n_features=n_channels, n_times=n_times)


def _load_raw_from_matlab(
    stats_file: BIDSFile,
    *,
    source_metric: str,
) -> _RawConditionTestStatsData:
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
        n_ch = len(channels)
        n_t = len(time_axis_s)
        _empty_cond = np.full((n_ch, n_t), np.nan, dtype=np.float64)
        means = data.means
        safe_a = matlab_safe_name(condition_labels[0])
        safe_b = matlab_safe_name(condition_labels[1])
        cond_a_array = getattr(means, safe_a, None)
        if cond_a_array is not None:
            condition_a_mean_values = coerce_feature_time(
                np.asarray(cond_a_array, dtype=np.float64), n_features=n_ch, n_times=n_t
            )
        else:
            condition_a_mean_values = _empty_cond.copy()
        cond_b_array = getattr(means, safe_b, None)
        if cond_b_array is not None:
            condition_b_mean_values = coerce_feature_time(
                np.asarray(cond_b_array, dtype=np.float64), n_features=n_ch, n_times=n_t
            )
        else:
            condition_b_mean_values = _empty_cond.copy()
        binning_mode = mat_str(getattr(meta, "binning_mode", None), default="none")
        window_ms = mat_float(getattr(meta, "window_ms", None), default=0.0)
        n_bins = mat_int(getattr(meta, "n_bins", None), default=0)
        effective_n_bins = mat_int(
            getattr(meta, "effective_n_bins", None), default=len(time_axis_s)
        )
        activity_zscore_raw = getattr(meta, "activity_zscore", None)
        if activity_zscore_raw is None:
            raise ValueError(
                f"{stats_file.path.name}: unsupported legacy condition_test_group input; "
                "meta.activity_zscore is required."
            )
        activity_zscore = mat_str(activity_zscore_raw, default="none")
        activity_baseline_tmin_s = mat_float(
            getattr(meta, "activity_baseline_tmin_s", None),
            default=-0.2,
        )
        activity_baseline_tmax_s = mat_float(
            getattr(meta, "activity_baseline_tmax_s", None),
            default=0.0,
        )
        source_ieeg_files: list[str] = []
        source_electrodes_files: list[str] = []
        if prov is not None:
            source_ieeg_files = mat_str_list(getattr(prov, "source_ieeg_files", None))
            source_electrodes_files = mat_str_list(getattr(prov, "source_electrodes_files", None))

    return _RawConditionTestStatsData(
        analysis_level=analysis_level,
        channels=channels,
        time_axis_s=time_axis_s,
        metric_values=metric_values,
        condition_a_mean_values=condition_a_mean_values,
        condition_b_mean_values=condition_b_mean_values,
        condition_labels=condition_labels,
        binning_mode=binning_mode,
        window_ms=window_ms,
        n_bins=n_bins,
        effective_n_bins=effective_n_bins,
        activity_zscore=activity_zscore,
        activity_baseline_tmin_s=activity_baseline_tmin_s,
        activity_baseline_tmax_s=activity_baseline_tmax_s,
        source_ieeg_files=source_ieeg_files,
        source_electrodes_files=source_electrodes_files,
        permuted_t_values=None,
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





