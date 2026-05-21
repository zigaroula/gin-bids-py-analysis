from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

import h5py
import numpy as np

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.bids.helpers import normalize_subject_value
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
    compute_one_sample_timecourse,
    compute_paired_epoch_summary,
    compute_paired_timecourse,
    compute_two_sample_epoch_summary,
    compute_two_sample_timecourse,
)
from gin_bids_py_analysis.processing.utils.statistics import correct_p_values

from gin_bids_py_analysis.processing.utils.cluster_permutation import (
    compute_cluster_null_distribution,
    compute_cluster_null_distribution_paired,
    compute_cluster_permutation_pvalue,
    compute_mne_cluster_permutation,
    find_temporal_clusters,
)
from gin_bids_py_analysis.processing.trial_stats.params import (
    normalize_trial_activity_summary_missing_response_policy,
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
    find_missing_manual_roi_channels,
    format_manual_roi_missing_channels_message,
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
from .params import RegressionGroupParams
from .result import (
    RegressionGroupProcessingResult,
    RegressionSourceMetricStats,
    ScatterData,
    VsZeroStatsPair,
)


# ---------------------------------------------------------------------------
# Internal data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _RawRegressionStatsData(BaseRawTrialStatsData):
    """Format-agnostic in-memory representation of one subject regression file."""

    available_metrics: frozenset[str]
    condition_a_slope: np.ndarray       # (n_channels, n_times)
    condition_b_slope: np.ndarray       # (n_channels, n_times)
    condition_a_mean: np.ndarray        # (n_channels, n_times)
    condition_b_mean: np.ndarray        # (n_channels, n_times)
    condition_a_r_value: np.ndarray     # (n_channels, n_times)
    condition_b_r_value: np.ndarray     # (n_channels, n_times)
    condition_a_predictor_raw_values: np.ndarray  # (n_trials_a,)
    condition_b_predictor_raw_values: np.ndarray  # (n_trials_b,)
    condition_a_predictor_transformed_values: np.ndarray  # (n_trials_a,)
    condition_b_predictor_transformed_values: np.ndarray  # (n_trials_b,)
    condition_a_predictor_values: np.ndarray  # (n_trials_a,)
    condition_b_predictor_values: np.ndarray  # (n_trials_b,)
    condition_a_trial_activity_summary_values: np.ndarray  # (n_channels, n_trials_a) or empty
    condition_b_trial_activity_summary_values: np.ndarray  # (n_channels, n_trials_b) or empty
    condition_a_permuted_slopes: np.ndarray | None  # (n_perm, n_channels, n_times) float32 or None
    condition_b_permuted_slopes: np.ndarray | None  # (n_perm, n_channels, n_times) float32 or None
    predictor: str
    predictor_zscore: str
    predictor_transform_by_condition: dict[str, dict[str, float]]
    trial_activity_summary_kind: str
    trial_activity_summary_missing_response_policy: str
    trial_activity_summary_source: dict[str, str]
    trial_activity_summary_label: str


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
    predictor: str
    predictor_zscore: str
    predictor_transform_by_condition_json: str
    activity_zscore: str
    activity_baseline_tmin_s: float
    activity_baseline_tmax_s: float
    trial_activity_summary_kind: str
    trial_activity_summary_missing_response_policy: str
    trial_activity_summary_source_json: str
    trial_activity_summary_label: str
    analysis_level: str

    @property
    def key(self) -> tuple[Any, ...]:
        return self.base_key + (
            self.predictor,
            self.predictor_zscore,
            self.predictor_transform_by_condition_json,
            self.activity_zscore,
            self.activity_baseline_tmin_s,
            self.activity_baseline_tmax_s,
            self.trial_activity_summary_kind,
            self.trial_activity_summary_missing_response_policy,
            self.trial_activity_summary_source_json,
            self.trial_activity_summary_label,
            self.analysis_level,
        )


@dataclass(frozen=True)
class _RegressionStatsSnapshot(BaseTrialStatsGroupSnapshot):
    raw: _RawRegressionStatsData


@dataclass(frozen=True)
class _ContributionRecord(BaseTrialStatsGroupContributionRecord):
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
    predictor_a_raw_values: np.ndarray  # (n_trials_a,)
    predictor_b_raw_values: np.ndarray  # (n_trials_b,)
    predictor_a_transformed_values: np.ndarray  # (n_trials_a,)
    predictor_b_transformed_values: np.ndarray  # (n_trials_b,)
    predictor_a_values: np.ndarray  # (n_trials_a,)
    predictor_b_values: np.ndarray  # (n_trials_b,)
    scatter_activity_a: np.ndarray  # (n_trials_a,) trial activity summary per trial for this channel
    scatter_activity_b: np.ndarray  # (n_trials_b,) trial activity summary per trial for this channel
    perm_slope_a_values: np.ndarray | None  # (n_perm, n_times) float32 or None
    perm_slope_b_values: np.ndarray | None  # (n_perm, n_times) float32 or None


# ---------------------------------------------------------------------------
# Public grouping helper
# ---------------------------------------------------------------------------

def build_regression_compatible_groups(
    stats_files: Sequence[BIDSFile],
) -> list[BIDSFileGroup]:
    """Group subject-level regression files by compatibility."""
    return build_compatible_groups(stats_files, read_signature=_read_snapshot_signature)


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class RegressionGroupProcessing(BaseTrialStatsGroupProcessing):
    """Compute group-level ROI regression contrasts from subject-level regression files."""

    def __init__(self, params: RegressionGroupParams) -> None:
        self.params = params

    def process_group(
        self,
        group: BIDSFileGroup,
        progress_tracking_position: int = 0,
    ) -> RegressionGroupProcessingResult:
        del progress_tracking_position
        files = self.sorted_group_files(group)
        if not files:
            raise ValueError(
                "RegressionGroupProcessing requires at least one regression stats file."
            )

        snapshots = [_load_slope_stats_snapshot(file) for file in files]
        _validate_group_compatibility(snapshots)
        _validate_source_metric_availability(
            snapshots,
            source_metric=self.params.source_metric,
        )

        first = snapshots[0]
        excluded_rois: dict[str, str] = {}
        missing_manual_channels: dict[str, dict[str, list[str]]] = {}

        if self.params.roi_mode == "manual":
            missing_manual_channels = find_missing_manual_roi_channels(
                snapshots=snapshots,
                manual_region_channels=self.params.manual_region_channels,
            )
            missing_message = format_manual_roi_missing_channels_message(
                missing_manual_channels
            )
            if missing_message:
                print(missing_message)
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
        rng = (
            np.random.default_rng(self.params.permutation_seed)
            if method == "cluster_permutation"
            else None
        )

        region_names: list[str] = []
        # Two-sample slope comparison timecourse statistics
        rows_slope_t: list[np.ndarray] = []
        rows_slope_p_uncorr: list[np.ndarray] = []
        # Per-ROI permuted slope lists for cluster permutation (custom method)
        perm_slope_a_collection: list[list[np.ndarray] | None] = []
        perm_slope_b_collection: list[list[np.ndarray] | None] = []
        # Per-ROI observed samples for cluster permutation (mne method)
        cluster_observed_collection: list[np.ndarray | None] = []
        rows_slope_mean_a: list[np.ndarray] = []
        rows_slope_sem_a: list[np.ndarray] = []
        rows_slope_mean_b: list[np.ndarray] = []
        rows_slope_sem_b: list[np.ndarray] = []
        # Two-sample activity comparison timecourse statistics
        rows_activity_t: list[np.ndarray] = []
        rows_activity_p_uncorr: list[np.ndarray] = []
        # Activity and r_value group means
        rows_activity_mean_a: list[np.ndarray] = []
        rows_activity_sem_a: list[np.ndarray] = []
        rows_activity_mean_b: list[np.ndarray] = []
        rows_activity_sem_b: list[np.ndarray] = []
        rows_r_value_mean_a: list[np.ndarray] = []
        rows_r_value_sem_a: list[np.ndarray] = []
        rows_r_value_mean_b: list[np.ndarray] = []
        rows_r_value_sem_b: list[np.ndarray] = []
        # Epoch-level two-sample summaries
        epoch_slope_t: list[float] = []
        epoch_slope_p: list[float] = []
        epoch_slope_df: list[float] = []
        epoch_activity_t: list[float] = []
        epoch_activity_p: list[float] = []
        epoch_activity_df: list[float] = []
        # Contribution metadata
        roi_channel_counts: list[int] = []
        roi_subject_counts: list[int] = []
        contributions_out: list[ROIChannelContribution] = []
        slope_a_contribution_samples: list[np.ndarray] = []
        slope_b_contribution_samples: list[np.ndarray] = []
        activity_a_contribution_samples: list[np.ndarray] = []
        activity_b_contribution_samples: list[np.ndarray] = []
        contribution_label_rows: list[list[str]] = []
        # Per-condition one-sample t-test vs 0 for the slope mean
        rows_vs_zero_t_a: list[np.ndarray] = []
        rows_vs_zero_p_a: list[np.ndarray] = []
        rows_vs_zero_t_b: list[np.ndarray] = []
        rows_vs_zero_p_b: list[np.ndarray] = []
        # Scatter data: per-ROI concatenated (predictor_value, epoch_mean) pairs
        scatter_a_predictor: list[np.ndarray] = []
        scatter_a_activity: list[np.ndarray] = []
        scatter_b_predictor: list[np.ndarray] = []
        scatter_b_activity: list[np.ndarray] = []

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

            samples_metric_a = np.stack(
                [
                    _metric_values_for_record(
                        r,
                        source_metric=self.params.source_metric,
                        condition="a",
                    )
                    for r in records
                ],
                axis=0,
            ).astype(np.float64)
            samples_metric_b = np.stack(
                [
                    _metric_values_for_record(
                        r,
                        source_metric=self.params.source_metric,
                        condition="b",
                    )
                    for r in records
                ],
                axis=0,
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

            if self.params.contrast_mode == "paired":
                t_slope, p_slope_raw = compute_paired_timecourse(samples_metric_a, samples_metric_b)
                ep_slope_t, ep_slope_p, ep_slope_df = compute_paired_epoch_summary(
                    samples_metric_a,
                    samples_metric_b,
                )
                t_activity, p_activity_raw = compute_paired_timecourse(samples_mean_a, samples_mean_b)
                ep_act_t, ep_act_p, ep_act_df = compute_paired_epoch_summary(
                    samples_mean_a,
                    samples_mean_b,
                )
            else:
                t_slope, p_slope_raw = compute_two_sample_timecourse(samples_metric_a, samples_metric_b)
                ep_slope_t, ep_slope_p, ep_slope_df = compute_two_sample_epoch_summary(
                    samples_metric_a,
                    samples_metric_b,
                )
                t_activity, p_activity_raw = compute_two_sample_timecourse(samples_mean_a, samples_mean_b)
                ep_act_t, ep_act_p, ep_act_df = compute_two_sample_epoch_summary(
                    samples_mean_a,
                    samples_mean_b,
                )

            if method == "cluster_permutation":
                if self.params.cluster_permutation_method == "custom":
                    perm_a_list: list[np.ndarray] | None = [
                        np.asarray(r.perm_slope_a_values, dtype=np.float64)
                        for r in records
                        if r.perm_slope_a_values is not None
                    ] or None
                    perm_b_list: list[np.ndarray] | None = [
                        np.asarray(r.perm_slope_b_values, dtype=np.float64)
                        for r in records
                        if r.perm_slope_b_values is not None
                    ] or None
                    observed_s: np.ndarray | None = None
                else:
                    perm_a_list = None
                    perm_b_list = None
                    observed_s = samples_metric_a - samples_metric_b
            else:
                perm_a_list = None
                perm_b_list = None
                observed_s = None
            perm_slope_a_collection.append(perm_a_list)
            perm_slope_b_collection.append(perm_b_list)
            cluster_observed_collection.append(observed_s)
            slope_mean_a, slope_sem_a = compute_condition_group_stats(samples_metric_a)
            slope_mean_b, slope_sem_b = compute_condition_group_stats(samples_metric_b)

            vz_t_a, vz_p_a, _, _ = compute_one_sample_timecourse(samples_metric_a)
            vz_t_b, vz_p_b, _, _ = compute_one_sample_timecourse(samples_metric_b)
            rows_vs_zero_t_a.append(vz_t_a)
            rows_vs_zero_p_a.append(vz_p_a)
            rows_vs_zero_t_b.append(vz_t_b)
            rows_vs_zero_p_b.append(vz_p_b)
            act_mean_a, act_sem_a = compute_condition_group_stats(samples_mean_a)
            act_mean_b, act_sem_b = compute_condition_group_stats(samples_mean_b)
            r_mean_a, r_sem_a = compute_condition_group_stats(samples_r_a)
            r_mean_b, r_sem_b = compute_condition_group_stats(samples_r_b)

            region_names.append(roi)
            rows_slope_t.append(t_slope)
            rows_slope_p_uncorr.append(p_slope_raw)
            rows_slope_mean_a.append(slope_mean_a)
            rows_slope_sem_a.append(slope_sem_a)
            rows_slope_mean_b.append(slope_mean_b)
            rows_slope_sem_b.append(slope_sem_b)
            rows_activity_t.append(t_activity)
            rows_activity_p_uncorr.append(p_activity_raw)
            rows_activity_mean_a.append(act_mean_a)
            rows_activity_sem_a.append(act_sem_a)
            rows_activity_mean_b.append(act_mean_b)
            rows_activity_sem_b.append(act_sem_b)
            rows_r_value_mean_a.append(r_mean_a)
            rows_r_value_sem_a.append(r_sem_a)
            rows_r_value_mean_b.append(r_mean_b)
            rows_r_value_sem_b.append(r_sem_b)
            epoch_slope_t.append(ep_slope_t)
            epoch_slope_p.append(ep_slope_p)
            epoch_slope_df.append(ep_slope_df)
            epoch_activity_t.append(ep_act_t)
            epoch_activity_p.append(ep_act_p)
            epoch_activity_df.append(ep_act_df)
            roi_channel_counts.append(channel_count)
            roi_subject_counts.append(subject_count)
            slope_a_contribution_samples.append(samples_metric_a)
            slope_b_contribution_samples.append(samples_metric_b)
            activity_a_contribution_samples.append(samples_mean_a)
            activity_b_contribution_samples.append(samples_mean_b)
            contribution_label_rows.append([f"{r.subject}/{r.channel}" for r in records])

            # Scatter: concatenate per-trial (predictor, activity-summary) pairs across records.
            valid_pred_a: list[np.ndarray] = []
            valid_act_a: list[np.ndarray] = []
            valid_pred_b: list[np.ndarray] = []
            valid_act_b: list[np.ndarray] = []
            for record in records:
                pred_a = np.asarray(record.predictor_a_values, dtype=np.float64).ravel()
                act_a = np.asarray(record.scatter_activity_a, dtype=np.float64).ravel()
                if pred_a.size > 0 and act_a.size > 0 and pred_a.size == act_a.size:
                    valid_pred_a.append(pred_a)
                    valid_act_a.append(act_a)

                pred_b = np.asarray(record.predictor_b_values, dtype=np.float64).ravel()
                act_b = np.asarray(record.scatter_activity_b, dtype=np.float64).ravel()
                if pred_b.size > 0 and act_b.size > 0 and pred_b.size == act_b.size:
                    valid_pred_b.append(pred_b)
                    valid_act_b.append(act_b)

            scatter_a_predictor.append(
                np.concatenate(valid_pred_a) if valid_pred_a else np.empty(0, dtype=np.float64)
            )
            scatter_a_activity.append(
                np.concatenate(valid_act_a) if valid_act_a else np.empty(0, dtype=np.float64)
            )
            scatter_b_predictor.append(
                np.concatenate(valid_pred_b) if valid_pred_b else np.empty(0, dtype=np.float64)
            )
            scatter_b_activity.append(
                np.concatenate(valid_act_b) if valid_act_b else np.empty(0, dtype=np.float64)
            )
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

        source_metric_t_values = self.stack_rows(rows_slope_t, n_times)
        source_metric_p_values_uncorr = self.stack_rows(rows_slope_p_uncorr, n_times)
        t_values_activity = self.stack_rows(rows_activity_t, n_times)
        p_values_activity_uncorr = self.stack_rows(rows_activity_p_uncorr, n_times)

        # Per-condition vs-zero statistics
        vs_zero_t_a = self.stack_rows(rows_vs_zero_t_a, n_times)
        vs_zero_p_raw_a = self.stack_rows(rows_vs_zero_p_a, n_times)
        vs_zero_t_b = self.stack_rows(rows_vs_zero_t_b, n_times)
        vs_zero_p_raw_b = self.stack_rows(rows_vs_zero_p_b, n_times)
        # For cluster permutation, skip multi-comparison correction here;
        # the cluster pass below produces the final significant mask.
        vz_correction = method if method != "cluster_permutation" else "none"
        vs_zero_p_a = _apply_correction_2d(vs_zero_p_raw_a, method=vz_correction)
        vs_zero_p_b = _apply_correction_2d(vs_zero_p_raw_b, method=vz_correction)

        # Apply p-value correction separately for the selected source metric and activity.
        source_metric_p_values = _apply_correction_2d(
            source_metric_p_values_uncorr,
            method=method,
        )
        p_values_activity = _apply_correction_2d(p_values_activity_uncorr, method=method)

        alpha = self.params.significance_alpha

        vs_zero_sig_a = np.isfinite(vs_zero_p_a) & (vs_zero_p_a < alpha)
        vs_zero_sig_b = np.isfinite(vs_zero_p_b) & (vs_zero_p_b < alpha)

        # --- Cluster permutation pass ---
        cluster_p_values_out: np.ndarray | None = None
        cluster_windows_out: list[list[tuple[float, float]]] | None = None
        cluster_null_dists_out: list[np.ndarray] | None = None
        if method == "cluster_permutation" and rows_slope_p_uncorr and rng is not None:
            n_keep = self.params.n_clusters_to_keep
            cluster_p_values_list: list[float] = []
            cluster_windows_list: list[list[tuple[float, float]]] = []
            cluster_null_dists_list: list[np.ndarray] = []
            for roi_idx in range(len(region_names)):
                roi_t = rows_slope_t[roi_idx]
                roi_p_raw = rows_slope_p_uncorr[roi_idx]
                h_mask = roi_p_raw < self.params.cluster_threshold_alpha
                observed_clusters = find_temporal_clusters(h_mask, roi_t)
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
                        if p < alpha:
                            roi_windows.append((
                                float(first.time_axis_s[win_idx[0]]),
                                float(first.time_axis_s[win_idx[1]]),
                            ))
                    cluster_p_values_list.append(best_p)
                    cluster_windows_list.append(roi_windows)
                    cluster_null_dists_list.append(null)
                else:
                    perm_a_roi = perm_slope_a_collection[roi_idx]
                    perm_b_roi = perm_slope_b_collection[roi_idx]
                    if not perm_a_roi or not perm_b_roi:
                        cluster_p_values_list.append(1.0)
                        cluster_windows_list.append([])
                        cluster_null_dists_list.append(np.zeros(0, dtype=np.float64))
                        continue
                    null = compute_cluster_null_distribution_paired(
                        perm_a_roi,
                        perm_b_roi,
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
                        if p < alpha:
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
            source_metric_significant_mask = np.zeros(
                (len(region_names), n_times), dtype=bool
            )
            for roi_idx, roi_windows in enumerate(cluster_windows_out):
                for t_start_s, t_end_s in roi_windows:
                    in_window = (
                        (first.time_axis_s >= t_start_s)
                        & (first.time_axis_s <= t_end_s)
                    )
                    source_metric_significant_mask[roi_idx, in_window] = True
        else:
            source_metric_significant_mask = (
                np.isfinite(source_metric_p_values)
                & (source_metric_p_values < alpha)
            )
        sig_mask_activity = np.isfinite(p_values_activity) & (p_values_activity < alpha)

        # --- Per-condition vs-zero cluster permutation pass ---
        vs_zero_cluster_p_a_out: np.ndarray | None = None
        vs_zero_cluster_windows_a_out: list[list[tuple[float, float]]] | None = None
        vs_zero_cluster_null_dists_a_out: list[np.ndarray] | None = None
        vs_zero_cluster_p_b_out: np.ndarray | None = None
        vs_zero_cluster_windows_b_out: list[list[tuple[float, float]]] | None = None
        vs_zero_cluster_null_dists_b_out: list[np.ndarray] | None = None
        if method == "cluster_permutation" and rows_vs_zero_t_a and rng is not None:
            n_keep = self.params.n_clusters_to_keep
            vz_cp_a: list[float] = []
            vz_cw_a: list[list[tuple[float, float]]] = []
            vz_cn_a: list[np.ndarray] = []
            vz_cp_b: list[float] = []
            vz_cw_b: list[list[tuple[float, float]]] = []
            vz_cn_b: list[np.ndarray] = []
            for roi_idx in range(len(region_names)):
                # Condition A vs zero
                roi_t_a = rows_vs_zero_t_a[roi_idx]
                roi_p_a = rows_vs_zero_p_a[roi_idx]
                h_mask_a = roi_p_a < self.params.cluster_threshold_alpha
                obs_clusters_a = find_temporal_clusters(h_mask_a, roi_t_a)
                # Condition B vs zero
                roi_t_b = rows_vs_zero_t_b[roi_idx]
                roi_p_b = rows_vs_zero_p_b[roi_idx]
                h_mask_b = roi_p_b < self.params.cluster_threshold_alpha
                obs_clusters_b = find_temporal_clusters(h_mask_b, roi_t_b)

                if self.params.cluster_permutation_method == "mne":
                    obs_a = slope_a_contribution_samples[roi_idx]
                    obs_b = slope_b_contribution_samples[roi_idx]
                    for obs, obs_clusters, cp_list, cw_list, cn_list in [
                        (obs_a, obs_clusters_a, vz_cp_a, vz_cw_a, vz_cn_a),
                        (obs_b, obs_clusters_b, vz_cp_b, vz_cw_b, vz_cn_b),
                    ]:
                        if obs is None or len(obs) < 2:
                            cp_list.append(1.0)
                            cw_list.append([])
                            cn_list.append(np.zeros(0, dtype=np.float64))
                            continue
                        seed = int(rng.integers(0, np.iinfo(np.int32).max))
                        best_p, top_windows_idx, top_p_vals, null = compute_mne_cluster_permutation(
                            obs,
                            cluster_threshold_alpha=self.params.cluster_threshold_alpha,
                            n_group_perm=self.params.n_group_permutations,
                            seed=seed,
                            n_clusters_to_keep=n_keep,
                        )
                        roi_vz_windows: list[tuple[float, float]] = []
                        for win_idx, p in zip(top_windows_idx, top_p_vals):
                            if p < alpha:
                                roi_vz_windows.append((
                                    float(first.time_axis_s[win_idx[0]]),
                                    float(first.time_axis_s[win_idx[1]]),
                                ))
                        cp_list.append(best_p)
                        cw_list.append(roi_vz_windows)
                        cn_list.append(null)
                else:
                    # Custom method: use per-channel permuted slope pools
                    for perm_list, obs_clusters, cp_list, cw_list, cn_list in [
                        (perm_slope_a_collection[roi_idx], obs_clusters_a, vz_cp_a, vz_cw_a, vz_cn_a),
                        (perm_slope_b_collection[roi_idx], obs_clusters_b, vz_cp_b, vz_cw_b, vz_cn_b),
                    ]:
                        if not perm_list:
                            cp_list.append(1.0)
                            cw_list.append([])
                            cn_list.append(np.zeros(0, dtype=np.float64))
                            continue
                        null = compute_cluster_null_distribution(
                            perm_list,
                            cluster_threshold_alpha=self.params.cluster_threshold_alpha,
                            n_group_perm=self.params.n_group_permutations,
                            rng=rng,
                        )
                        top_candidates = obs_clusters[:n_keep]
                        roi_vz_windows = []
                        best_p = 1.0
                        for i, (start, end, tsum) in enumerate(top_candidates):
                            p = compute_cluster_permutation_pvalue(tsum, null)
                            if i == 0:
                                best_p = p
                            if p < alpha:
                                roi_vz_windows.append((
                                    float(first.time_axis_s[start]),
                                    float(first.time_axis_s[end]),
                                ))
                        cp_list.append(best_p)
                        cw_list.append(roi_vz_windows)
                        cn_list.append(null)

            vs_zero_cluster_p_a_out = np.asarray(vz_cp_a, dtype=np.float64)
            vs_zero_cluster_windows_a_out = vz_cw_a
            vs_zero_cluster_null_dists_a_out = vz_cn_a
            vs_zero_cluster_p_b_out = np.asarray(vz_cp_b, dtype=np.float64)
            vs_zero_cluster_windows_b_out = vz_cw_b
            vs_zero_cluster_null_dists_b_out = vz_cn_b

        # Apply cluster-based significant masks for vs-zero tests.
        if method == "cluster_permutation" and vs_zero_cluster_windows_a_out is not None:
            vs_zero_sig_a = np.zeros((len(region_names), n_times), dtype=bool)
            for roi_idx, roi_windows in enumerate(vs_zero_cluster_windows_a_out):
                for t_start_s, t_end_s in roi_windows:
                    in_window = (
                        (first.time_axis_s >= t_start_s)
                        & (first.time_axis_s <= t_end_s)
                    )
                    vs_zero_sig_a[roi_idx, in_window] = True
        if method == "cluster_permutation" and vs_zero_cluster_windows_b_out is not None:
            vs_zero_sig_b = np.zeros((len(region_names), n_times), dtype=bool)
            for roi_idx, roi_windows in enumerate(vs_zero_cluster_windows_b_out):
                for t_start_s, t_end_s in roi_windows:
                    in_window = (
                        (first.time_axis_s >= t_start_s)
                        & (first.time_axis_s <= t_end_s)
                    )
                    vs_zero_sig_b[roi_idx, in_window] = True

        result = RegressionGroupProcessingResult(
            source_group=BIDSFileGroup(primary=files[0], secondaries=files[1:]),
            metadata={
                "p_value_correction_method": method,
                "significance_alpha": alpha,
                "source_metric": self.params.source_metric,
                "contrast_mode": self.params.contrast_mode,
                "roi_mode": self.params.roi_mode,
                "atlas_name": self.params.atlas_name,
                "binning_mode": first.binning_mode,
                "window_ms": first.window_ms,
                "n_bins": first.n_bins,
                "effective_n_bins": first.effective_n_bins,
                "predictor": first.raw.predictor,
                "predictor_zscore": first.raw.predictor_zscore,
                "predictor_transform_by_condition_json": json.dumps(
                    first.raw.predictor_transform_by_condition,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "activity_zscore": first.raw.activity_zscore,
                "activity_baseline_tmin_s": first.raw.activity_baseline_tmin_s,
                "activity_baseline_tmax_s": first.raw.activity_baseline_tmax_s,
                "trial_activity_summary_kind": first.raw.trial_activity_summary_kind,
                "trial_activity_summary_missing_response_policy": (
                    first.raw.trial_activity_summary_missing_response_policy
                ),
                "trial_activity_summary_source_json": json.dumps(
                    first.raw.trial_activity_summary_source,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "trial_activity_summary_label": first.raw.trial_activity_summary_label,
                "scatter_aggregation": "trial_pool",
            },
            source_metric_stats=RegressionSourceMetricStats(
                contrast=GroupTimecourseStats(
                    t_values=source_metric_t_values,
                    p_values=source_metric_p_values,
                    p_values_uncorrected=source_metric_p_values_uncorr,
                    significant_mask=source_metric_significant_mask,
                ),
                epoch_summary=GroupEpochStats(
                    t=self.array_1d(epoch_slope_t),
                    p=self.array_1d(epoch_slope_p),
                    df=self.array_1d(epoch_slope_df),
                ),
                vs_zero=VsZeroStatsPair(
                    condition_a=GroupTimecourseStats(
                        t_values=vs_zero_t_a,
                        p_values=vs_zero_p_a,
                        p_values_uncorrected=vs_zero_p_raw_a,
                        significant_mask=vs_zero_sig_a,
                    ),
                    condition_b=GroupTimecourseStats(
                        t_values=vs_zero_t_b,
                        p_values=vs_zero_p_b,
                        p_values_uncorrected=vs_zero_p_raw_b,
                        significant_mask=vs_zero_sig_b,
                    ),
                    condition_a_cluster_p_values=vs_zero_cluster_p_a_out,
                    condition_a_cluster_windows_s=vs_zero_cluster_windows_a_out,
                    condition_a_cluster_null_distributions=vs_zero_cluster_null_dists_a_out,
                    condition_b_cluster_p_values=vs_zero_cluster_p_b_out,
                    condition_b_cluster_windows_s=vs_zero_cluster_windows_b_out,
                    condition_b_cluster_null_distributions=vs_zero_cluster_null_dists_b_out,
                ),
            ),
            activity_stats=GroupTimecourseStats(
                t_values=t_values_activity,
                p_values=p_values_activity,
                p_values_uncorrected=p_values_activity_uncorr,
                significant_mask=sig_mask_activity,
            ),
            source_metric_data=GroupEstimatePair(
                condition_a=GroupEstimate(
                    mean=self.stack_rows(rows_slope_mean_a, n_times),
                    sem=self.stack_rows(rows_slope_sem_a, n_times),
                ),
                condition_b=GroupEstimate(
                    mean=self.stack_rows(rows_slope_mean_b, n_times),
                    sem=self.stack_rows(rows_slope_sem_b, n_times),
                ),
            ),
            activity=GroupEstimatePair(
                condition_a=GroupEstimate(
                    mean=self.stack_rows(rows_activity_mean_a, n_times),
                    sem=self.stack_rows(rows_activity_sem_a, n_times),
                ),
                condition_b=GroupEstimate(
                    mean=self.stack_rows(rows_activity_mean_b, n_times),
                    sem=self.stack_rows(rows_activity_sem_b, n_times),
                ),
            ),
            r_values=GroupEstimatePair(
                condition_a=GroupEstimate(
                    mean=self.stack_rows(rows_r_value_mean_a, n_times),
                    sem=self.stack_rows(rows_r_value_sem_a, n_times),
                ),
                condition_b=GroupEstimate(
                    mean=self.stack_rows(rows_r_value_mean_b, n_times),
                    sem=self.stack_rows(rows_r_value_sem_b, n_times),
                ),
            ),
            activity_epoch=GroupEpochStats(
                t=self.array_1d(epoch_activity_t),
                p=self.array_1d(epoch_activity_p),
                df=self.array_1d(epoch_activity_df),
            ),
            time_axis_s=first.time_axis_s.copy(),
            region_names=region_names,
            condition_labels=first.condition_labels,
            roi_channel_counts=np.array(roi_channel_counts, dtype=np.int64),
            roi_subject_counts=np.array(roi_subject_counts, dtype=np.int64),
            contributions=contributions_out,
            source_metric_contributions=IndexedConditionContributions(
                condition_a=slope_a_contribution_samples,
                condition_b=slope_b_contribution_samples,
                labels=contribution_label_rows,
            ),
            activity_contributions=IndexedConditionContributions(
                condition_a=activity_a_contribution_samples,
                condition_b=activity_b_contribution_samples,
                labels=contribution_label_rows,
            ),
            scatter=ScatterData(
                condition_a_predictor=scatter_a_predictor,
                condition_a_activity=scatter_a_activity,
                condition_b_predictor=scatter_b_predictor,
                condition_b_activity=scatter_b_activity,
            ),
            source_metric=self.params.source_metric,
            contrast_mode=self.params.contrast_mode,
            p_value_correction_method=method,
            significance_alpha=alpha,
            roi_mode=self.params.roi_mode,
            atlas_name=self.params.atlas_name,
            source_subject_stats_files=[str(s.stats_file.path) for s in snapshots],
            source_electrodes_files=sorted(used_electrode_paths),
            excluded_rois=excluded_rois,
            cluster_p_values=cluster_p_values_out,
            cluster_windows_s=cluster_windows_out,
            cluster_null_distributions=cluster_null_dists_out,
            manual_roi_missing_channels=missing_manual_channels,
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


def _metric_values_for_record(
    record: _ContributionRecord,
    *,
    source_metric: str,
    condition: str,
) -> np.ndarray:
    if condition not in {"a", "b"}:
        raise ValueError(f"Unsupported condition key {condition!r}.")

    suffix = "a" if condition == "a" else "b"
    if source_metric == "slope":
        return np.asarray(getattr(record, f"slope_{suffix}_values"), dtype=np.float64)
    if source_metric == "r_value":
        return np.asarray(getattr(record, f"r_value_{suffix}_values"), dtype=np.float64)
    raise ValueError(f"Unsupported source_metric={source_metric!r}.")


# ---------------------------------------------------------------------------
# ROI record collection
# ---------------------------------------------------------------------------

def _collect_manual_roi_records(
    *,
    snapshots: Sequence[_RegressionStatsSnapshot],
    manual_region_channels: dict[str, dict[str, list[str]]],
) -> dict[str, list[_ContributionRecord]]:
    return collect_manual_roi_records(
        snapshots=snapshots,
        manual_region_channels=manual_region_channels,
        create_record=_create_contribution_record,
    )


def _collect_atlas_roi_records(
    *,
    snapshots: Sequence[_RegressionStatsSnapshot],
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
    snapshot: _RegressionStatsSnapshot,
    idx: int,
) -> _ContributionRecord:
    return _ContributionRecord(
        roi=roi,
        subject=subject,
        channel=snapshot.channel_names[idx],
        source_stats_file=str(snapshot.stats_file.path),
        slope_a_values=np.asarray(snapshot.raw.condition_a_slope[idx, :], dtype=np.float64),
        slope_b_values=np.asarray(snapshot.raw.condition_b_slope[idx, :], dtype=np.float64),
        mean_a_values=np.asarray(snapshot.raw.condition_a_mean[idx, :], dtype=np.float64),
        mean_b_values=np.asarray(snapshot.raw.condition_b_mean[idx, :], dtype=np.float64),
        r_value_a_values=np.asarray(snapshot.raw.condition_a_r_value[idx, :], dtype=np.float64),
        r_value_b_values=np.asarray(snapshot.raw.condition_b_r_value[idx, :], dtype=np.float64),
        predictor_a_raw_values=np.asarray(snapshot.raw.condition_a_predictor_raw_values, dtype=np.float64),
        predictor_b_raw_values=np.asarray(snapshot.raw.condition_b_predictor_raw_values, dtype=np.float64),
        predictor_a_transformed_values=np.asarray(
            snapshot.raw.condition_a_predictor_transformed_values,
            dtype=np.float64,
        ),
        predictor_b_transformed_values=np.asarray(
            snapshot.raw.condition_b_predictor_transformed_values,
            dtype=np.float64,
        ),
        predictor_a_values=np.asarray(snapshot.raw.condition_a_predictor_values, dtype=np.float64),
        predictor_b_values=np.asarray(snapshot.raw.condition_b_predictor_values, dtype=np.float64),
        scatter_activity_a=(
            np.asarray(
                snapshot.raw.condition_a_trial_activity_summary_values[idx, :],
                dtype=np.float64,
            )
            if snapshot.raw.condition_a_trial_activity_summary_values.ndim == 2
            and snapshot.raw.condition_a_trial_activity_summary_values.shape[0] > idx
            else np.empty(0, dtype=np.float64)
        ),
        scatter_activity_b=(
            np.asarray(
                snapshot.raw.condition_b_trial_activity_summary_values[idx, :],
                dtype=np.float64,
            )
            if snapshot.raw.condition_b_trial_activity_summary_values.ndim == 2
            and snapshot.raw.condition_b_trial_activity_summary_values.shape[0] > idx
            else np.empty(0, dtype=np.float64)
        ),
        perm_slope_a_values=(
            np.asarray(snapshot.raw.condition_a_permuted_slopes[:, idx, :], dtype=np.float32)
            if snapshot.raw.condition_a_permuted_slopes is not None
            else None
        ),
        perm_slope_b_values=(
            np.asarray(snapshot.raw.condition_b_permuted_slopes[:, idx, :], dtype=np.float32)
            if snapshot.raw.condition_b_permuted_slopes is not None
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Compatibility validation
# ---------------------------------------------------------------------------

def _validate_group_compatibility(snapshots: Sequence[_RegressionStatsSnapshot]) -> None:
    validate_group_compatibility(
        snapshots,
        empty_message="At least one regression snapshot is required.",
        non_channel_message="regression_group requires channel-level regression inputs.",
        incompatible_message=(
            "Incompatible regression inputs in one processing group. "
            "Use build_regression_compatible_groups() to split heterogeneous files."
        ),
    )


def _validate_source_metric_availability(
    snapshots: Sequence[_RegressionStatsSnapshot],
    *,
    source_metric: str,
) -> None:
    missing = [
        snapshot.stats_file.path.name
        for snapshot in snapshots
        if source_metric not in snapshot.raw.available_metrics
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"source_metric={source_metric!r} is not available in these regression "
            f"files: {joined}."
        )


# ---------------------------------------------------------------------------
# Snapshot I/O helpers
# ---------------------------------------------------------------------------

def _read_snapshot_signature(stats_file: BIDSFile) -> _SnapshotSignature:
    raw = _load_raw_slope_stats(stats_file)
    return _build_signature(stats_file=stats_file, raw=raw)


def _load_slope_stats_snapshot(stats_file: BIDSFile) -> _RegressionStatsSnapshot:
    raw = _load_raw_slope_stats(stats_file)
    raw_subject = str(stats_file.get("subject") or stats_file.get("sub") or "").strip()
    subject = normalize_subject_value(raw_subject)

    channel_index_by_norm: dict[str, int] = {}
    for idx, name in enumerate(raw.channels):
        key = normalize_channel_name(name)
        channel_index_by_norm.setdefault(key, idx)

    signature = _build_signature(stats_file=stats_file, raw=raw)
    return _RegressionStatsSnapshot(
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


def _build_signature(
    *,
    stats_file: BIDSFile,
    raw: _RawRegressionStatsData,
) -> _SnapshotSignature:
    return _SnapshotSignature(
        task=str(stats_file.get("task") or ""),
        source_desc=str(stats_file.get("desc") or ""),
        condition_labels=raw.condition_labels,
        time_axis_hash=hash_time_axis(raw.time_axis_s),
        time_axis_len=int(len(raw.time_axis_s)),
        binning_mode=str(raw.binning_mode or "none"),
        window_ms=float(raw.window_ms),
        n_bins=int(raw.n_bins),
        effective_n_bins=int(raw.effective_n_bins),
        predictor=str(raw.predictor or ""),
        predictor_zscore=str(raw.predictor_zscore or "none"),
        predictor_transform_by_condition_json=json.dumps(
            raw.predictor_transform_by_condition,
            sort_keys=True,
            separators=(",", ":"),
        ),
        activity_zscore=str(raw.activity_zscore or "none"),
        activity_baseline_tmin_s=float(raw.activity_baseline_tmin_s),
        activity_baseline_tmax_s=float(raw.activity_baseline_tmax_s),
        trial_activity_summary_kind=str(raw.trial_activity_summary_kind or "epoch_mean"),
        trial_activity_summary_missing_response_policy=(
            normalize_trial_activity_summary_missing_response_policy(
                raw.trial_activity_summary_missing_response_policy or "nan_if_missing"
            )
        ),
        trial_activity_summary_source_json=json.dumps(
            raw.trial_activity_summary_source,
            sort_keys=True,
            separators=(",", ":"),
        ),
        trial_activity_summary_label=str(
            raw.trial_activity_summary_label or "Epoch mean activity"
        ),
        analysis_level=str(raw.analysis_level or "channel"),
    )


def _load_raw_slope_stats(stats_file: BIDSFile) -> _RawRegressionStatsData:
    extension = (stats_file.extension or "").lower()
    if extension == ".mat":
        return _load_raw_from_matlab(stats_file)
    return _load_raw_from_hdf5(stats_file)


def _load_raw_from_hdf5(stats_file: BIDSFile) -> _RawRegressionStatsData:
    with stats_file.ensure_loaded() as fh:
        analysis_type = str_scalar(dataset_or_none(fh, "meta/analysis_type"), default="")
        if analysis_type and analysis_type != "slope_regression":
            raise ValueError(
                f"{stats_file.path.name}: not a regression file "
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

        ds_cond_a_slope = dataset_or_none(fh, "stats/regression/condition_a/slope")
        ds_cond_b_slope = dataset_or_none(fh, "stats/regression/condition_b/slope")
        ds_cond_a_r = dataset_or_none(fh, "stats/regression/condition_a/r_value")
        ds_cond_b_r = dataset_or_none(fh, "stats/regression/condition_b/r_value")

        condition_a_slope = _read_2d("stats/regression/condition_a/slope")
        condition_b_slope = _read_2d("stats/regression/condition_b/slope")
        condition_a_r_value = _read_2d("stats/regression/condition_a/r_value")
        condition_b_r_value = _read_2d("stats/regression/condition_b/r_value")

        condition_a_mean = _read_2d("data/activity/condition_a/mean")
        condition_b_mean = _read_2d("data/activity/condition_b/mean")

        binning_mode = str_scalar(dataset_or_none(fh, "meta/binning_mode"), default="none")
        window_ms = float_scalar(dataset_or_none(fh, "meta/window_ms"), default=0.0)
        n_bins = int_scalar(dataset_or_none(fh, "meta/n_bins"), default=0)
        effective_n_bins = int_scalar(
            dataset_or_none(fh, "meta/effective_n_bins"), default=n_t
        )
        predictor_ds = dataset_or_none(fh, "meta/predictor")
        if predictor_ds is None:
            raise ValueError(
                f"{stats_file.path.name}: unsupported legacy regression_group input; "
                "meta/predictor is required."
            )
        predictor = str_scalar(predictor_ds, default="")
        predictor_zscore_ds = dataset_or_none(fh, "meta/predictor_zscore")
        if predictor_zscore_ds is None:
            raise ValueError(
                f"{stats_file.path.name}: unsupported legacy regression_group input; "
                "meta/predictor_zscore is required."
            )
        predictor_zscore = str_scalar(predictor_zscore_ds, default="none")
        predictor_transform_ds = dataset_or_none(
            fh,
            "meta/predictor_transform_by_condition_json",
        )
        if predictor_transform_ds is None:
            raise ValueError(
                f"{stats_file.path.name}: unsupported legacy regression_group input; "
                "meta/predictor_transform_by_condition_json is required."
            )
        predictor_transform_raw = str_scalar(predictor_transform_ds, default="{}")
        try:
            predictor_transform_by_condition = (
                json.loads(predictor_transform_raw) if predictor_transform_raw else {}
            )
        except json.JSONDecodeError:
            predictor_transform_by_condition = {}
        activity_zscore_ds = dataset_or_none(fh, "meta/activity_zscore")
        if activity_zscore_ds is None:
            raise ValueError(
                f"{stats_file.path.name}: unsupported legacy regression_group input; "
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
        trial_activity_summary_kind = "epoch_mean"
        trial_activity_summary_missing_response_policy = "nan_if_missing"
        trial_activity_summary_source: dict[str, str] = {}
        trial_activity_summary_label = "Epoch mean activity"
        if "trial_activity_summary" in fh:
            tg = fh["trial_activity_summary"]
            condition_a_trial_activity_summary_values = (
                np.asarray(tg["condition_a_values"][:], dtype=np.float64)
                if "condition_a_values" in tg
                else np.empty((n_ch, 0), dtype=np.float64)
            )
            condition_b_trial_activity_summary_values = (
                np.asarray(tg["condition_b_values"][:], dtype=np.float64)
                if "condition_b_values" in tg
                else np.empty((n_ch, 0), dtype=np.float64)
            )
            trial_activity_summary_kind = str_scalar(
                dataset_or_none(tg, "kind"),
                default=trial_activity_summary_kind,
            )
            trial_activity_summary_missing_response_policy = (
                normalize_trial_activity_summary_missing_response_policy(
                    str_scalar(
                        dataset_or_none(tg, "missing_response_policy"),
                        default=trial_activity_summary_missing_response_policy,
                    )
                )
            )
            trial_activity_summary_source_raw = str_scalar(
                dataset_or_none(tg, "source_json"),
                default="{}",
            )
            try:
                loaded_summary_source = (
                    json.loads(trial_activity_summary_source_raw)
                    if trial_activity_summary_source_raw
                    else {}
                )
                if isinstance(loaded_summary_source, dict):
                    trial_activity_summary_source = {
                        str(key): str(value)
                        for key, value in loaded_summary_source.items()
                    }
            except json.JSONDecodeError:
                trial_activity_summary_source = {}
            trial_activity_summary_label = (
                str_scalar(
                    dataset_or_none(tg, "label"),
                    default=trial_activity_summary_label,
                )
                or trial_activity_summary_label
            )
        else:
            condition_a_trial_activity_summary_values = np.empty((n_ch, 0), dtype=np.float64)
            condition_b_trial_activity_summary_values = np.empty((n_ch, 0), dtype=np.float64)

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

        perm_a_ds = dataset_or_none(fh, "stats/regression/condition_a/permuted_slopes")
        raw_condition_a_permuted_slopes: np.ndarray | None = (
            np.asarray(perm_a_ds[:], dtype=np.float32) if perm_a_ds is not None else None
        )
        perm_b_ds = dataset_or_none(fh, "stats/regression/condition_b/permuted_slopes")
        raw_condition_b_permuted_slopes: np.ndarray | None = (
            np.asarray(perm_b_ds[:], dtype=np.float32) if perm_b_ds is not None else None
        )

    return _RawRegressionStatsData(
        analysis_level=analysis_level,
        available_metrics=frozenset(
            metric
            for metric, present in {
                "slope": ds_cond_a_slope is not None and ds_cond_b_slope is not None,
                "r_value": ds_cond_a_r is not None and ds_cond_b_r is not None,
            }.items()
            if present
        ),
        channels=channels,
        time_axis_s=time_axis_s,
        condition_labels=condition_labels,
        condition_a_slope=condition_a_slope,
        condition_b_slope=condition_b_slope,
        condition_a_mean=condition_a_mean,
        condition_b_mean=condition_b_mean,
        condition_a_r_value=condition_a_r_value,
        condition_b_r_value=condition_b_r_value,
        condition_a_predictor_raw_values=_read_predictor_values_hdf5(
            stats_file,
            "predictor/condition_a_raw_values",
        ),
        condition_b_predictor_raw_values=_read_predictor_values_hdf5(
            stats_file,
            "predictor/condition_b_raw_values",
        ),
        condition_a_predictor_transformed_values=_read_predictor_values_hdf5(
            stats_file,
            "predictor/condition_a_transformed_values",
        ),
        condition_b_predictor_transformed_values=_read_predictor_values_hdf5(
            stats_file,
            "predictor/condition_b_transformed_values",
        ),
        condition_a_predictor_values=_read_predictor_values_hdf5(stats_file, "predictor/condition_a_values"),
        condition_b_predictor_values=_read_predictor_values_hdf5(stats_file, "predictor/condition_b_values"),
        condition_a_trial_activity_summary_values=condition_a_trial_activity_summary_values,
        condition_b_trial_activity_summary_values=condition_b_trial_activity_summary_values,
        condition_a_permuted_slopes=raw_condition_a_permuted_slopes,
        condition_b_permuted_slopes=raw_condition_b_permuted_slopes,
        binning_mode=binning_mode,
        window_ms=window_ms,
        n_bins=n_bins,
        effective_n_bins=effective_n_bins,
        predictor=predictor,
        predictor_zscore=predictor_zscore,
        predictor_transform_by_condition=predictor_transform_by_condition,
        activity_zscore=activity_zscore,
        activity_baseline_tmin_s=activity_baseline_tmin_s,
        activity_baseline_tmax_s=activity_baseline_tmax_s,
        trial_activity_summary_kind=trial_activity_summary_kind,
        trial_activity_summary_missing_response_policy=(
            trial_activity_summary_missing_response_policy
        ),
        trial_activity_summary_source=trial_activity_summary_source,
        trial_activity_summary_label=trial_activity_summary_label,
        source_ieeg_files=source_ieeg_files,
        source_electrodes_files=source_electrodes_files,
    )


def _load_raw_from_matlab(stats_file: BIDSFile) -> _RawRegressionStatsData:
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

        def _read_feature_trial_2d(obj: Any, attr: str) -> np.ndarray:
            arr = getattr(obj, attr, None) if obj is not None else None
            if arr is None:
                return np.empty((n_ch, 0), dtype=np.float64)
            out = np.asarray(arr, dtype=np.float64)
            if out.size == 0:
                return np.empty((n_ch, 0), dtype=np.float64)
            if out.ndim != 2:
                if out.size % max(n_ch, 1) != 0:
                    return np.empty((n_ch, 0), dtype=np.float64)
                return out.reshape(n_ch, -1)
            if out.shape[0] == n_ch:
                return out
            if out.shape[1] == n_ch:
                return out.T
            if out.size % max(n_ch, 1) != 0:
                return np.empty((n_ch, 0), dtype=np.float64)
            return out.reshape(n_ch, -1)

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
        predictor_raw_meta = getattr(meta, "predictor", None)
        if predictor_raw_meta is None:
            raise ValueError(
                f"{stats_file.path.name}: unsupported legacy regression_group input; "
                "meta.predictor is required."
            )
        predictor = mat_str(predictor_raw_meta, default="")
        predictor_zscore_raw = getattr(meta, "predictor_zscore", None)
        if predictor_zscore_raw is None:
            raise ValueError(
                f"{stats_file.path.name}: unsupported legacy regression_group input; "
                "meta.predictor_zscore is required."
            )
        predictor_zscore = mat_str(predictor_zscore_raw, default="none")
        predictor_transform_raw = getattr(meta, "predictor_transform_by_condition_json", None)
        if predictor_transform_raw is None:
            raise ValueError(
                f"{stats_file.path.name}: unsupported legacy regression_group input; "
                "meta.predictor_transform_by_condition_json is required."
            )
        predictor_transform_json = mat_str(predictor_transform_raw, default="{}")
        try:
            predictor_transform_by_condition = (
                json.loads(predictor_transform_json) if predictor_transform_json else {}
            )
        except json.JSONDecodeError:
            predictor_transform_by_condition = {}
        activity_zscore_raw = getattr(meta, "activity_zscore", None)
        if activity_zscore_raw is None:
            raise ValueError(
                f"{stats_file.path.name}: unsupported legacy regression_group input; "
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

        predictor_raw = getattr(data, "predictor", None)
        condition_a_predictor_raw_values: np.ndarray
        condition_b_predictor_raw_values: np.ndarray
        condition_a_predictor_transformed_values: np.ndarray
        condition_b_predictor_transformed_values: np.ndarray
        condition_a_predictor_values: np.ndarray
        condition_b_predictor_values: np.ndarray
        if predictor_raw is not None:
            raw_raw_a = getattr(predictor_raw, "condition_a_raw_values", None)
            raw_raw_b = getattr(predictor_raw, "condition_b_raw_values", None)
            raw_trans_a = getattr(predictor_raw, "condition_a_transformed_values", None)
            raw_trans_b = getattr(predictor_raw, "condition_b_transformed_values", None)
            raw_a = getattr(predictor_raw, "condition_a_values", None)
            raw_b = getattr(predictor_raw, "condition_b_values", None)
            condition_a_predictor_raw_values = np.asarray(raw_raw_a, dtype=np.float64).ravel() if raw_raw_a is not None else np.empty(0, dtype=np.float64)
            condition_b_predictor_raw_values = np.asarray(raw_raw_b, dtype=np.float64).ravel() if raw_raw_b is not None else np.empty(0, dtype=np.float64)
            condition_a_predictor_transformed_values = np.asarray(raw_trans_a, dtype=np.float64).ravel() if raw_trans_a is not None else np.empty(0, dtype=np.float64)
            condition_b_predictor_transformed_values = np.asarray(raw_trans_b, dtype=np.float64).ravel() if raw_trans_b is not None else np.empty(0, dtype=np.float64)
            condition_a_predictor_values = np.asarray(raw_a, dtype=np.float64).ravel() if raw_a is not None else np.empty(0, dtype=np.float64)
            condition_b_predictor_values = np.asarray(raw_b, dtype=np.float64).ravel() if raw_b is not None else np.empty(0, dtype=np.float64)
        else:
            condition_a_predictor_raw_values = np.empty(0, dtype=np.float64)
            condition_b_predictor_raw_values = np.empty(0, dtype=np.float64)
            condition_a_predictor_transformed_values = np.empty(0, dtype=np.float64)
            condition_b_predictor_transformed_values = np.empty(0, dtype=np.float64)
            condition_a_predictor_values = np.empty(0, dtype=np.float64)
            condition_b_predictor_values = np.empty(0, dtype=np.float64)

        trial_activity_summary_kind = "epoch_mean"
        trial_activity_summary_missing_response_policy = "nan_if_missing"
        trial_activity_summary_source: dict[str, str] = {}
        trial_activity_summary_label = "Epoch mean activity"
        trial_activity_summary = getattr(data, "trial_activity_summary", None)
        if trial_activity_summary is not None:
            condition_a_trial_activity_summary_values = _read_feature_trial_2d(
                trial_activity_summary,
                "condition_a_values",
            )
            condition_b_trial_activity_summary_values = _read_feature_trial_2d(
                trial_activity_summary,
                "condition_b_values",
            )
            trial_activity_summary_kind = mat_str(
                getattr(trial_activity_summary, "kind", None),
                default=trial_activity_summary_kind,
            )
            trial_activity_summary_missing_response_policy = (
                normalize_trial_activity_summary_missing_response_policy(
                    mat_str(
                        getattr(trial_activity_summary, "missing_response_policy", None),
                        default=trial_activity_summary_missing_response_policy,
                    )
                )
            )
            trial_activity_summary_source_raw = mat_str(
                getattr(trial_activity_summary, "source_json", None),
                default="{}",
            )
            try:
                loaded_summary_source = (
                    json.loads(trial_activity_summary_source_raw)
                    if trial_activity_summary_source_raw
                    else {}
                )
                if isinstance(loaded_summary_source, dict):
                    trial_activity_summary_source = {
                        str(key): str(value)
                        for key, value in loaded_summary_source.items()
                    }
            except json.JSONDecodeError:
                trial_activity_summary_source = {}
            trial_activity_summary_label = (
                mat_str(
                    getattr(trial_activity_summary, "label", None),
                    default=trial_activity_summary_label,
                )
                or trial_activity_summary_label
            )
        else:
            condition_a_trial_activity_summary_values = np.empty((n_ch, 0), dtype=np.float64)
            condition_b_trial_activity_summary_values = np.empty((n_ch, 0), dtype=np.float64)

        _mat_perm_a = getattr(cond_a_reg, "permuted_slopes", None) if cond_a_reg is not None else None
        mat_condition_a_permuted_slopes: np.ndarray | None = (
            np.asarray(_mat_perm_a, dtype=np.float32)
            if _mat_perm_a is not None and np.asarray(_mat_perm_a).size > 0
            else None
        )
        _mat_perm_b = getattr(cond_b_reg, "permuted_slopes", None) if cond_b_reg is not None else None
        mat_condition_b_permuted_slopes: np.ndarray | None = (
            np.asarray(_mat_perm_b, dtype=np.float32)
            if _mat_perm_b is not None and np.asarray(_mat_perm_b).size > 0
            else None
        )

    return _RawRegressionStatsData(
        analysis_level=analysis_level,
        available_metrics=frozenset(
            metric
            for metric, present in {
                "slope": cond_a_reg is not None and cond_b_reg is not None and getattr(cond_a_reg, "slope", None) is not None and getattr(cond_b_reg, "slope", None) is not None,
                "r_value": cond_a_reg is not None and cond_b_reg is not None and getattr(cond_a_reg, "r_value", None) is not None and getattr(cond_b_reg, "r_value", None) is not None,
            }.items()
            if present
        ),
        channels=channels,
        time_axis_s=time_axis_s,
        condition_labels=condition_labels,
        condition_a_slope=condition_a_slope,
        condition_b_slope=condition_b_slope,
        condition_a_mean=condition_a_mean,
        condition_b_mean=condition_b_mean,
        condition_a_r_value=condition_a_r_value,
        condition_b_r_value=condition_b_r_value,
        condition_a_predictor_raw_values=condition_a_predictor_raw_values,
        condition_b_predictor_raw_values=condition_b_predictor_raw_values,
        condition_a_predictor_transformed_values=condition_a_predictor_transformed_values,
        condition_b_predictor_transformed_values=condition_b_predictor_transformed_values,
        condition_a_predictor_values=condition_a_predictor_values,
        condition_b_predictor_values=condition_b_predictor_values,
        condition_a_trial_activity_summary_values=condition_a_trial_activity_summary_values,
        condition_b_trial_activity_summary_values=condition_b_trial_activity_summary_values,
        condition_a_permuted_slopes=mat_condition_a_permuted_slopes,
        condition_b_permuted_slopes=mat_condition_b_permuted_slopes,
        binning_mode=binning_mode,
        window_ms=window_ms,
        n_bins=n_bins,
        effective_n_bins=effective_n_bins,
        predictor=predictor,
        predictor_zscore=predictor_zscore,
        predictor_transform_by_condition=predictor_transform_by_condition,
        activity_zscore=activity_zscore,
        activity_baseline_tmin_s=activity_baseline_tmin_s,
        activity_baseline_tmax_s=activity_baseline_tmax_s,
        trial_activity_summary_kind=trial_activity_summary_kind,
        trial_activity_summary_missing_response_policy=(
            trial_activity_summary_missing_response_policy
        ),
        trial_activity_summary_source=trial_activity_summary_source,
        trial_activity_summary_label=trial_activity_summary_label,
        source_ieeg_files=source_ieeg_files,
        source_electrodes_files=source_electrodes_files,
    )


def _read_predictor_values_hdf5(stats_file: BIDSFile, path: str) -> np.ndarray:
    """Read a 1-D predictor values array from a subject slope-stats HDF5 file."""
    with stats_file.ensure_loaded() as fh:
        ds = dataset_or_none(fh, path)
        if ds is None:
            return np.empty(0, dtype=np.float64)
        return np.asarray(ds[:], dtype=np.float64).ravel()


def _read_condition_labels_hdf5(fh: h5py.File) -> tuple[str, str]:
    labels_ds = dataset_or_none(fh, "meta/condition_labels")
    if labels_ds is not None:
        labels = decode_str_array(np.asarray(labels_ds[:], dtype=object))
        if len(labels) >= 2:
            return labels[0], labels[1]
    return "condition_a", "condition_b"


def _mat_condition_labels(meta: Any) -> tuple[str, str]:
    labels = mat_str_list(getattr(meta, "trial_count_labels", None))
    if len(labels) >= 2:
        return labels[0], labels[1]
    return "condition_a", "condition_b"

