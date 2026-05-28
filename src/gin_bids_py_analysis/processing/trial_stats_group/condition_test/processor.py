from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup

from gin_bids_py_analysis.processing.utils.group_stats import (
    compute_condition_group_stats,
    compute_one_sample_epoch_summary,
    compute_one_sample_timecourse,
)
from gin_bids_py_analysis.processing.trial_stats.condition_test import (
    ConditionTestProcessingResult,
    load_condition_test_result,
)

from ..compatibility import (
    SubjectStatsInput,
    SubjectStatsSignature,
    build_compatible_groups,
    build_subject_stats_input,
    load_result_via_public_loader,
    validate_group_compatibility,
)
from ..processor import (
    BaseTrialStatsGroupContributionRecord,
    BaseTrialStatsGroupProcessing,
    collect_atlas_roi_records,
    collect_manual_roi_records,
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
class _ConditionTestStatsInput(SubjectStatsInput):
    result: ConditionTestProcessingResult
    source_metric: str


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
        read_signature=lambda file: _read_input_signature(
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

        inputs = [
            _load_trial_stats_input(
                file,
                source_metric=self.params.primary_condition_metric,
            )
            for file in files
        ]
        _validate_group_compatibility(inputs)

        method = self.params.p_value_correction_method
        rng: np.random.Generator | None = None
        if method == "cluster_permutation":
            if self.params.cluster_permutation_method == "custom":
                for item in inputs:
                    if item.result.contrast.permuted_t_values is None:
                        raise ValueError(
                            f"cluster_permutation with cluster_permutation_method='custom' "
                            f"requires permuted_t_values in all source files, "
                            f"but {item.stats_file.path.name} has none. "
                            "Re-run the subject-level condition_test analysis "
                            "with n_permutations > 0."
                        )
            rng = np.random.default_rng(self.params.permutation_seed)

        first = inputs[0]
        n_times = int(len(first.time_axis_s))
        excluded_rois: dict[str, str] = {}

        if self.params.roi_mode == "manual":
            roi_records = _collect_manual_roi_records(
                inputs=inputs,
                manual_region_channels=self.params.manual_region_channels,
            )
            used_electrode_paths: set[str] = set()
        else:
            assert self.params.atlas_name is not None
            roi_records, used_electrode_paths = _collect_atlas_roi_records(
                inputs=inputs,
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
                else:  # sign_flip
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
                    if self.params.cluster_permutation_method == "sign_flip":
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
                "activity_zscore": first.result.activity_zscore,
                "activity_baseline_tmin_s": first.result.activity_baseline_tmin_s,
                "activity_baseline_tmax_s": first.result.activity_baseline_tmax_s,
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
            source_subject_stats_files=[str(item.stats_file.path) for item in inputs],
            source_electrodes_files=sorted(used_electrode_paths),
            excluded_rois=excluded_rois,
            cluster_p_values=cluster_p_values_out,
            cluster_windows_s=cluster_windows_out,
            cluster_null_distributions=cluster_null_dists_out,
        )


def _collect_manual_roi_records(
    *,
    inputs: Sequence[_ConditionTestStatsInput],
    manual_region_channels: dict[str, dict[str, list[str]]],
) -> dict[str, list[_ContributionRecord]]:
    return collect_manual_roi_records(
        inputs=inputs,
        manual_region_channels=manual_region_channels,
        create_record=_create_contribution_record,
    )


def _collect_atlas_roi_records(
    *,
    inputs: Sequence[_ConditionTestStatsInput],
    atlas_name: str,
) -> tuple[dict[str, list[_ContributionRecord]], set[str]]:
    return collect_atlas_roi_records(
        inputs=inputs,
        atlas_name=atlas_name,
        create_record=_create_contribution_record,
    )


def _create_contribution_record(
    roi: str,
    subject: str,
    item: _ConditionTestStatsInput,
    idx: int,
) -> _ContributionRecord:
    result = item.result
    return _ContributionRecord(
        roi=roi,
        subject=subject,
        channel=item.channel_names[idx],
        source_stats_file=str(item.stats_file.path),
        values=np.asarray(
            _condition_metric_values(result, item.source_metric)[idx, :],
            dtype=np.float64,
        ),
        condition_a_values=np.asarray(result.signal_activity.condition_a.mean[idx, :], dtype=np.float64),
        condition_b_values=np.asarray(result.signal_activity.condition_b.mean[idx, :], dtype=np.float64),
        permuted_t_values=(
            np.asarray(result.contrast.permuted_t_values[:, idx, :], dtype=np.float32)
            if result.contrast.permuted_t_values is not None
            else None
        ),
    )


def _validate_group_compatibility(inputs: Sequence[_ConditionTestStatsInput]) -> None:
    validate_group_compatibility(
        inputs,
        empty_message="At least one condition_test input is required.",
        non_channel_message=(
            "condition_test_group requires channel-level condition_test inputs."
        ),
        incompatible_message=(
            "Incompatible condition_test inputs in one processing group. "
            "Use build_condition_test_compatible_groups(...) to split heterogeneous files."
        ),
    )


def _read_input_signature(
    stats_file: BIDSFile,
    *,
    source_metric: str,
) -> SubjectStatsSignature:
    item = _load_trial_stats_input(stats_file, source_metric=source_metric)
    return item.signature


def _load_trial_stats_input(
    stats_file: BIDSFile,
    *,
    source_metric: str,
) -> _ConditionTestStatsInput:
    result = load_result_via_public_loader(stats_file, load_condition_test_result)
    _validate_condition_metric_available(
        result,
        source_metric=source_metric,
        filename=stats_file.path.name,
    )
    base_input = build_subject_stats_input(
        stats_file,
        result,
        extra_key_parts={"primary_condition_metric": source_metric},
    )
    return _ConditionTestStatsInput(
        stats_file=base_input.stats_file,
        result=result,
        subject=base_input.subject,
        task=base_input.task,
        source_desc=base_input.source_desc,
        condition_labels=base_input.condition_labels,
        channel_names=base_input.channel_names,
        channel_index_by_norm=base_input.channel_index_by_norm,
        time_axis_s=base_input.time_axis_s,
        analysis_level=base_input.analysis_level,
        binning_mode=base_input.binning_mode,
        window_ms=base_input.window_ms,
        n_bins=base_input.n_bins,
        effective_n_bins=base_input.effective_n_bins,
        source_ieeg_files=base_input.source_ieeg_files,
        source_electrodes_files=base_input.source_electrodes_files,
        signature=base_input.signature,
        source_metric=source_metric,
    )


def _condition_metric_values(
    result: ConditionTestProcessingResult,
    source_metric: str,
) -> np.ndarray:
    if source_metric == "mean_difference":
        return np.asarray(result.difference.mean, dtype=np.float64)
    if source_metric == "t_values":
        return np.asarray(result.contrast.t_values, dtype=np.float64)
    if source_metric == "condition_a_mean":
        return np.asarray(result.signal_activity.condition_a.mean, dtype=np.float64)
    if source_metric == "condition_b_mean":
        return np.asarray(result.signal_activity.condition_b.mean, dtype=np.float64)
    raise ValueError(f"Unsupported primary_condition_metric={source_metric!r}.")


def _validate_condition_metric_available(
    result: ConditionTestProcessingResult,
    *,
    source_metric: str,
    filename: str,
) -> None:
    available = result.metadata.get("available_condition_metrics")
    if not available:
        available = result.available_condition_metrics()
    if source_metric not in set(str(item) for item in available):
        raise ValueError(
            f"primary_condition_metric={source_metric!r} is not available in "
            f"condition_test file: {filename}."
        )
