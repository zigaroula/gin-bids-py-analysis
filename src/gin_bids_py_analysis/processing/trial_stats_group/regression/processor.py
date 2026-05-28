from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from typing import Sequence

import numpy as np

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup

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
from gin_bids_py_analysis.processing.trial_stats.regression import (
    RegressionProcessingResult,
    load_regression_result,
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
    find_missing_manual_roi_channels,
    format_manual_roi_missing_channels_message,
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
    RegressionMetricStats,
    RegressionGroupProcessingResult,
    ScatterData,
    VsZeroStatsPair,
)


# ---------------------------------------------------------------------------
# Internal data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _RegressionStatsInput(SubjectStatsInput):
    result: RegressionProcessingResult


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


@dataclass
class _ROISamples:
    """Per-ROI stacked arrays — shape (n_channels, n_times) each."""
    metric_a: np.ndarray
    metric_b: np.ndarray
    slope_a: np.ndarray
    slope_b: np.ndarray
    mean_a: np.ndarray
    mean_b: np.ndarray
    r_a: np.ndarray
    r_b: np.ndarray


@dataclass
class _ClusterPermResult:
    best_p: float
    windows: list[tuple[float, float]]
    null: np.ndarray


@dataclass
class _ROIAccumulator:
    """Accumulates per-ROI results during the main processing loop."""
    region_names: list[str] = field(default_factory=list)
    # Timecourse two-sample stats
    slope_t: list[np.ndarray] = field(default_factory=list)
    slope_p: list[np.ndarray] = field(default_factory=list)
    activity_t: list[np.ndarray] = field(default_factory=list)
    activity_p: list[np.ndarray] = field(default_factory=list)
    # Group mean/sem estimates
    slope_mean_a: list[np.ndarray] = field(default_factory=list)
    slope_sem_a: list[np.ndarray] = field(default_factory=list)
    slope_mean_b: list[np.ndarray] = field(default_factory=list)
    slope_sem_b: list[np.ndarray] = field(default_factory=list)
    activity_mean_a: list[np.ndarray] = field(default_factory=list)
    activity_sem_a: list[np.ndarray] = field(default_factory=list)
    activity_mean_b: list[np.ndarray] = field(default_factory=list)
    activity_sem_b: list[np.ndarray] = field(default_factory=list)
    r_mean_a: list[np.ndarray] = field(default_factory=list)
    r_sem_a: list[np.ndarray] = field(default_factory=list)
    r_mean_b: list[np.ndarray] = field(default_factory=list)
    r_sem_b: list[np.ndarray] = field(default_factory=list)
    # Epoch summaries
    epoch_slope_t: list[float] = field(default_factory=list)
    epoch_slope_p: list[float] = field(default_factory=list)
    epoch_slope_df: list[float] = field(default_factory=list)
    epoch_activity_t: list[float] = field(default_factory=list)
    epoch_activity_p: list[float] = field(default_factory=list)
    epoch_activity_df: list[float] = field(default_factory=list)
    # Per-condition vs-zero stats
    vs_zero_t_a: list[np.ndarray] = field(default_factory=list)
    vs_zero_p_a: list[np.ndarray] = field(default_factory=list)
    vs_zero_t_b: list[np.ndarray] = field(default_factory=list)
    vs_zero_p_b: list[np.ndarray] = field(default_factory=list)
    # Cluster permutation inputs
    perm_slope_a: list[list[np.ndarray] | None] = field(default_factory=list)
    perm_slope_b: list[list[np.ndarray] | None] = field(default_factory=list)
    cluster_observed: list[np.ndarray | None] = field(default_factory=list)
    # Contribution metadata
    channel_counts: list[int] = field(default_factory=list)
    subject_counts: list[int] = field(default_factory=list)
    contributions: list[ROIChannelContribution] = field(default_factory=list)
    contribution_labels: list[list[str]] = field(default_factory=list)
    # Per-ROI sample matrices (used for contributions and vs-zero cluster perm)
    slope_a_samples: list[np.ndarray] = field(default_factory=list)
    slope_b_samples: list[np.ndarray] = field(default_factory=list)
    r_a_samples: list[np.ndarray] = field(default_factory=list)
    r_b_samples: list[np.ndarray] = field(default_factory=list)
    activity_a_samples: list[np.ndarray] = field(default_factory=list)
    activity_b_samples: list[np.ndarray] = field(default_factory=list)
    # Scatter data
    scatter_pred_a: list[np.ndarray] = field(default_factory=list)
    scatter_act_a: list[np.ndarray] = field(default_factory=list)
    scatter_pred_b: list[np.ndarray] = field(default_factory=list)
    scatter_act_b: list[np.ndarray] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public grouping helper
# ---------------------------------------------------------------------------

def build_regression_compatible_groups(
    stats_files: Sequence[BIDSFile],
) -> list[BIDSFileGroup]:
    """Group subject-level regression files by compatibility."""
    return build_compatible_groups(stats_files, read_signature=_read_input_signature)


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

        inputs = [_load_slope_stats_input(file) for file in files]
        _validate_group_compatibility(inputs)
        _validate_source_metric_availability(
            inputs,
            primary_regression_metric=self.params.primary_regression_metric,
        )

        first = inputs[0]
        excluded_rois: dict[str, str] = {}
        missing_manual_channels: dict[str, dict[str, list[str]]] = {}

        if self.params.roi_mode == "manual":
            missing_manual_channels = find_missing_manual_roi_channels(
                inputs=inputs,
                manual_region_channels=self.params.manual_region_channels,
            )
            missing_message = format_manual_roi_missing_channels_message(
                missing_manual_channels
            )
            if missing_message:
                print(missing_message)
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

        method = self.params.p_value_correction_method
        rng = (
            np.random.default_rng(self.params.permutation_seed)
            if method == "cluster_permutation"
            else None
        )

        acc = _ROIAccumulator()
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
            _accumulate_roi(acc, roi, records, self.params, method)

        n_times = int(len(first.time_axis_s))

        source_metric_t_values = self.stack_rows(acc.slope_t, n_times)
        source_metric_p_values_uncorr = self.stack_rows(acc.slope_p, n_times)
        t_values_activity = self.stack_rows(acc.activity_t, n_times)
        p_values_activity_uncorr = self.stack_rows(acc.activity_p, n_times)

        vs_zero_t_a = self.stack_rows(acc.vs_zero_t_a, n_times)
        vs_zero_p_raw_a = self.stack_rows(acc.vs_zero_p_a, n_times)
        vs_zero_t_b = self.stack_rows(acc.vs_zero_t_b, n_times)
        vs_zero_p_raw_b = self.stack_rows(acc.vs_zero_p_b, n_times)

        # For cluster permutation, skip multi-comparison correction here;
        # the cluster pass below produces the final significant mask.
        vz_correction = method if method != "cluster_permutation" else "none"
        vs_zero_p_a = _apply_correction_2d(vs_zero_p_raw_a, method=vz_correction)
        vs_zero_p_b = _apply_correction_2d(vs_zero_p_raw_b, method=vz_correction)

        source_metric_p_values = _apply_correction_2d(source_metric_p_values_uncorr, method=method)
        p_values_activity = _apply_correction_2d(p_values_activity_uncorr, method=method)

        alpha = self.params.significance_alpha
        n_rois = len(acc.region_names)

        vs_zero_sig_a = np.isfinite(vs_zero_p_a) & (vs_zero_p_a < alpha)
        vs_zero_sig_b = np.isfinite(vs_zero_p_b) & (vs_zero_p_b < alpha)

        # --- Cluster permutation pass (main contrast) ---
        cluster_p_values_out: np.ndarray | None = None
        cluster_windows_out: list[list[tuple[float, float]]] | None = None
        cluster_null_dists_out: list[np.ndarray] | None = None
        if method == "cluster_permutation" and acc.slope_t and rng is not None:
            cluster_p_values_out, cluster_windows_out, cluster_null_dists_out = (
                _run_main_cluster_perm_pass(acc, first.time_axis_s, self.params, alpha, rng)
            )

        if method == "cluster_permutation" and cluster_windows_out is not None:
            source_metric_significant_mask = _masks_from_windows(
                n_rois, n_times, first.time_axis_s, cluster_windows_out
            )
        else:
            source_metric_significant_mask = (
                np.isfinite(source_metric_p_values) & (source_metric_p_values < alpha)
            )
        sig_mask_activity = np.isfinite(p_values_activity) & (p_values_activity < alpha)

        # --- Per-condition vs-zero cluster permutation pass ---
        vs_zero_cluster_p_a_out: np.ndarray | None = None
        vs_zero_cluster_windows_a_out: list[list[tuple[float, float]]] | None = None
        vs_zero_cluster_null_dists_a_out: list[np.ndarray] | None = None
        vs_zero_cluster_p_b_out: np.ndarray | None = None
        vs_zero_cluster_windows_b_out: list[list[tuple[float, float]]] | None = None
        vs_zero_cluster_null_dists_b_out: list[np.ndarray] | None = None
        if method == "cluster_permutation" and acc.vs_zero_t_a and rng is not None:
            (
                vs_zero_cluster_p_a_out, vs_zero_cluster_windows_a_out, vs_zero_cluster_null_dists_a_out,
                vs_zero_cluster_p_b_out, vs_zero_cluster_windows_b_out, vs_zero_cluster_null_dists_b_out,
            ) = _run_vs_zero_cluster_perm_pass(acc, first.time_axis_s, self.params, alpha, rng)

        if method == "cluster_permutation" and vs_zero_cluster_windows_a_out is not None:
            vs_zero_sig_a = _masks_from_windows(
                n_rois, n_times, first.time_axis_s, vs_zero_cluster_windows_a_out
            )
        if method == "cluster_permutation" and vs_zero_cluster_windows_b_out is not None:
            vs_zero_sig_b = _masks_from_windows(
                n_rois, n_times, first.time_axis_s, vs_zero_cluster_windows_b_out
            )

        return RegressionGroupProcessingResult(
            source_group=BIDSFileGroup(primary=files[0], secondaries=files[1:]),
            metadata=_build_result_metadata(self.params, first, method, alpha),
            regression_stats=RegressionMetricStats(
                contrast=GroupTimecourseStats(
                    t_values=source_metric_t_values,
                    p_values=source_metric_p_values,
                    p_values_uncorrected=source_metric_p_values_uncorr,
                    significant_mask=source_metric_significant_mask,
                ),
                epoch_summary=GroupEpochStats(
                    t=self.array_1d(acc.epoch_slope_t),
                    p=self.array_1d(acc.epoch_slope_p),
                    df=self.array_1d(acc.epoch_slope_df),
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
            signal_activity_stats=GroupTimecourseStats(
                t_values=t_values_activity,
                p_values=p_values_activity,
                p_values_uncorrected=p_values_activity_uncorr,
                significant_mask=sig_mask_activity,
            ),
            slope=GroupEstimatePair(
                condition_a=GroupEstimate(
                    mean=self.stack_rows(acc.slope_mean_a, n_times),
                    sem=self.stack_rows(acc.slope_sem_a, n_times),
                ),
                condition_b=GroupEstimate(
                    mean=self.stack_rows(acc.slope_mean_b, n_times),
                    sem=self.stack_rows(acc.slope_sem_b, n_times),
                ),
            ),
            signal_activity=GroupEstimatePair(
                condition_a=GroupEstimate(
                    mean=self.stack_rows(acc.activity_mean_a, n_times),
                    sem=self.stack_rows(acc.activity_sem_a, n_times),
                ),
                condition_b=GroupEstimate(
                    mean=self.stack_rows(acc.activity_mean_b, n_times),
                    sem=self.stack_rows(acc.activity_sem_b, n_times),
                ),
            ),
            r_value=GroupEstimatePair(
                condition_a=GroupEstimate(
                    mean=self.stack_rows(acc.r_mean_a, n_times),
                    sem=self.stack_rows(acc.r_sem_a, n_times),
                ),
                condition_b=GroupEstimate(
                    mean=self.stack_rows(acc.r_mean_b, n_times),
                    sem=self.stack_rows(acc.r_sem_b, n_times),
                ),
            ),
            signal_activity_epoch=GroupEpochStats(
                t=self.array_1d(acc.epoch_activity_t),
                p=self.array_1d(acc.epoch_activity_p),
                df=self.array_1d(acc.epoch_activity_df),
            ),
            time_axis_s=first.time_axis_s.copy(),
            region_names=acc.region_names,
            condition_labels=first.condition_labels,
            roi_channel_counts=np.array(acc.channel_counts, dtype=np.int64),
            roi_subject_counts=np.array(acc.subject_counts, dtype=np.int64),
            contributions=acc.contributions,
            slope_contributions=IndexedConditionContributions(
                condition_a=acc.slope_a_samples,
                condition_b=acc.slope_b_samples,
                labels=acc.contribution_labels,
            ),
            r_value_contributions=IndexedConditionContributions(
                condition_a=acc.r_a_samples,
                condition_b=acc.r_b_samples,
                labels=acc.contribution_labels,
            ),
            signal_activity_contributions=IndexedConditionContributions(
                condition_a=acc.activity_a_samples,
                condition_b=acc.activity_b_samples,
                labels=acc.contribution_labels,
            ),
            scatter=ScatterData(
                condition_a_predictor=acc.scatter_pred_a,
                condition_a_signal_activity_summary=acc.scatter_act_a,
                condition_b_predictor=acc.scatter_pred_b,
                condition_b_signal_activity_summary=acc.scatter_act_b,
            ),
            primary_regression_metric=self.params.primary_regression_metric,
            contrast_mode=self.params.contrast_mode,
            p_value_correction_method=method,
            significance_alpha=alpha,
            roi_mode=self.params.roi_mode,
            atlas_name=self.params.atlas_name,
            source_subject_stats_files=[str(item.stats_file.path) for item in inputs],
            source_electrodes_files=sorted(used_electrode_paths),
            excluded_rois=excluded_rois,
            cluster_p_values=cluster_p_values_out,
            cluster_windows_s=cluster_windows_out,
            cluster_null_distributions=cluster_null_dists_out,
            manual_roi_missing_channels=missing_manual_channels,
        )


# ---------------------------------------------------------------------------
# Per-ROI processing
# ---------------------------------------------------------------------------

def _stack_roi_samples(
    records: list[_ContributionRecord],
    primary_regression_metric: str,
) -> _ROISamples:
    def stack(arrays: list) -> np.ndarray:
        return np.stack(arrays, axis=0).astype(np.float64)

    return _ROISamples(
        metric_a=stack([_metric_values_for_record(r, primary_regression_metric=primary_regression_metric, condition="a") for r in records]),
        metric_b=stack([_metric_values_for_record(r, primary_regression_metric=primary_regression_metric, condition="b") for r in records]),
        slope_a=stack([r.slope_a_values for r in records]),
        slope_b=stack([r.slope_b_values for r in records]),
        mean_a=stack([r.mean_a_values for r in records]),
        mean_b=stack([r.mean_b_values for r in records]),
        r_a=stack([r.r_value_a_values for r in records]),
        r_b=stack([r.r_value_b_values for r in records]),
    )


def _compute_contrast_stats(
    contrast_mode: str,
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    if contrast_mode == "paired":
        t, p = compute_paired_timecourse(samples_a, samples_b)
        ep_t, ep_p, ep_df = compute_paired_epoch_summary(samples_a, samples_b)
    else:
        t, p = compute_two_sample_timecourse(samples_a, samples_b)
        ep_t, ep_p, ep_df = compute_two_sample_epoch_summary(samples_a, samples_b)
    return t, p, ep_t, ep_p, ep_df


def _collect_scatter(
    records: list[_ContributionRecord],
    pred_attr: str,
    act_attr: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate per-trial (predictor, activity_summary) pairs across records."""
    valid_pred: list[np.ndarray] = []
    valid_act: list[np.ndarray] = []
    for r in records:
        pred = np.asarray(getattr(r, pred_attr), dtype=np.float64).ravel()
        act = np.asarray(getattr(r, act_attr), dtype=np.float64).ravel()
        if pred.size > 0 and act.size > 0 and pred.size == act.size:
            valid_pred.append(pred)
            valid_act.append(act)
    empty = np.empty(0, dtype=np.float64)
    return (
        np.concatenate(valid_pred) if valid_pred else empty,
        np.concatenate(valid_act) if valid_act else empty,
    )


def _accumulate_roi(
    acc: _ROIAccumulator,
    roi: str,
    records: list[_ContributionRecord],
    params: RegressionGroupParams,
    method: str,
) -> None:
    samples = _stack_roi_samples(records, params.primary_regression_metric)

    t_slope, p_slope, ep_slope_t, ep_slope_p, ep_slope_df = _compute_contrast_stats(
        params.contrast_mode, samples.metric_a, samples.metric_b
    )
    t_act, p_act, ep_act_t, ep_act_p, ep_act_df = _compute_contrast_stats(
        params.contrast_mode, samples.mean_a, samples.mean_b
    )

    slope_mean_a, slope_sem_a = compute_condition_group_stats(samples.slope_a)
    slope_mean_b, slope_sem_b = compute_condition_group_stats(samples.slope_b)
    act_mean_a, act_sem_a = compute_condition_group_stats(samples.mean_a)
    act_mean_b, act_sem_b = compute_condition_group_stats(samples.mean_b)
    r_mean_a, r_sem_a = compute_condition_group_stats(samples.r_a)
    r_mean_b, r_sem_b = compute_condition_group_stats(samples.r_b)

    vz_t_a, vz_p_a, _, _ = compute_one_sample_timecourse(samples.metric_a)
    vz_t_b, vz_p_b, _, _ = compute_one_sample_timecourse(samples.metric_b)

    if method == "cluster_permutation":
        if params.cluster_permutation_method == "custom":
            perm_a: list[np.ndarray] | None = [
                np.asarray(r.perm_slope_a_values, dtype=np.float64)
                for r in records
                if r.perm_slope_a_values is not None
            ] or None
            perm_b: list[np.ndarray] | None = [
                np.asarray(r.perm_slope_b_values, dtype=np.float64)
                for r in records
                if r.perm_slope_b_values is not None
            ] or None
            observed_s: np.ndarray | None = None
        else:  # sign_flip
            perm_a = None
            perm_b = None
            observed_s = samples.metric_a - samples.metric_b
    else:
        perm_a = None
        perm_b = None
        observed_s = None

    scatter_pred_a, scatter_act_a = _collect_scatter(records, "predictor_a_values", "scatter_activity_a")
    scatter_pred_b, scatter_act_b = _collect_scatter(records, "predictor_b_values", "scatter_activity_b")

    acc.region_names.append(roi)
    acc.slope_t.append(t_slope)
    acc.slope_p.append(p_slope)
    acc.slope_mean_a.append(slope_mean_a)
    acc.slope_sem_a.append(slope_sem_a)
    acc.slope_mean_b.append(slope_mean_b)
    acc.slope_sem_b.append(slope_sem_b)
    acc.activity_t.append(t_act)
    acc.activity_p.append(p_act)
    acc.activity_mean_a.append(act_mean_a)
    acc.activity_sem_a.append(act_sem_a)
    acc.activity_mean_b.append(act_mean_b)
    acc.activity_sem_b.append(act_sem_b)
    acc.r_mean_a.append(r_mean_a)
    acc.r_sem_a.append(r_sem_a)
    acc.r_mean_b.append(r_mean_b)
    acc.r_sem_b.append(r_sem_b)
    acc.epoch_slope_t.append(ep_slope_t)
    acc.epoch_slope_p.append(ep_slope_p)
    acc.epoch_slope_df.append(ep_slope_df)
    acc.epoch_activity_t.append(ep_act_t)
    acc.epoch_activity_p.append(ep_act_p)
    acc.epoch_activity_df.append(ep_act_df)
    acc.vs_zero_t_a.append(vz_t_a)
    acc.vs_zero_p_a.append(vz_p_a)
    acc.vs_zero_t_b.append(vz_t_b)
    acc.vs_zero_p_b.append(vz_p_b)
    acc.perm_slope_a.append(perm_a)
    acc.perm_slope_b.append(perm_b)
    acc.cluster_observed.append(observed_s)
    acc.channel_counts.append(len(records))
    acc.subject_counts.append(len({r.subject for r in records}))
    acc.slope_a_samples.append(samples.slope_a)
    acc.slope_b_samples.append(samples.slope_b)
    acc.r_a_samples.append(samples.r_a)
    acc.r_b_samples.append(samples.r_b)
    acc.activity_a_samples.append(samples.mean_a)
    acc.activity_b_samples.append(samples.mean_b)
    acc.contribution_labels.append([f"{r.subject}/{r.channel}" for r in records])
    acc.scatter_pred_a.append(scatter_pred_a)
    acc.scatter_act_a.append(scatter_act_a)
    acc.scatter_pred_b.append(scatter_pred_b)
    acc.scatter_act_b.append(scatter_act_b)
    acc.contributions.extend(
        ROIChannelContribution(
            roi=record.roi,
            subject=record.subject,
            channel=record.channel,
            source_stats_file=record.source_stats_file,
        )
        for record in records
    )


# ---------------------------------------------------------------------------
# Cluster permutation helpers
# ---------------------------------------------------------------------------

def _run_sign_flip_perm_roi(
    obs_samples: np.ndarray,
    time_axis_s: np.ndarray,
    cluster_threshold_alpha: float,
    n_group_permutations: int,
    n_clusters_to_keep: int,
    alpha: float,
    rng: np.random.Generator,
) -> _ClusterPermResult:
    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    best_p, top_windows_idx, top_p_vals, null = compute_mne_cluster_permutation(
        obs_samples,
        cluster_threshold_alpha=cluster_threshold_alpha,
        n_group_perm=n_group_permutations,
        seed=seed,
        n_clusters_to_keep=n_clusters_to_keep,
    )
    windows = [
        (float(time_axis_s[win_idx[0]]), float(time_axis_s[win_idx[1]]))
        for win_idx, p in zip(top_windows_idx, top_p_vals)
        if p < alpha
    ]
    return _ClusterPermResult(best_p=best_p, windows=windows, null=null)


def _cluster_result_from_null(
    null: np.ndarray,
    top_candidates: list,
    time_axis_s: np.ndarray,
    alpha: float,
) -> _ClusterPermResult:
    windows: list[tuple[float, float]] = []
    best_p = 1.0
    for i, (start, end, tsum) in enumerate(top_candidates):
        p = compute_cluster_permutation_pvalue(tsum, null)
        if i == 0:
            best_p = p
        if p < alpha:
            windows.append((float(time_axis_s[start]), float(time_axis_s[end])))
    return _ClusterPermResult(best_p=best_p, windows=windows, null=null)


def _run_main_cluster_perm_pass(
    acc: _ROIAccumulator,
    time_axis_s: np.ndarray,
    params: RegressionGroupParams,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[list[tuple[float, float]]], list[np.ndarray]]:
    n_keep = params.n_clusters_to_keep
    cp_list: list[float] = []
    cw_list: list[list[tuple[float, float]]] = []
    cn_list: list[np.ndarray] = []

    for roi_idx in range(len(acc.region_names)):
        roi_t = acc.slope_t[roi_idx]
        roi_p = acc.slope_p[roi_idx]
        h_mask = roi_p < params.cluster_threshold_alpha
        obs_clusters = find_temporal_clusters(h_mask, roi_t)

        if params.cluster_permutation_method == "sign_flip":
            obs_samples = acc.cluster_observed[roi_idx]
            if obs_samples is None:
                cp_list.append(1.0)
                cw_list.append([])
                cn_list.append(np.zeros(0, dtype=np.float64))
                continue
            result = _run_sign_flip_perm_roi(
                obs_samples, time_axis_s,
                params.cluster_threshold_alpha, params.n_group_permutations, n_keep, alpha, rng,
            )
        else:  # custom
            perm_a_roi = acc.perm_slope_a[roi_idx]
            perm_b_roi = acc.perm_slope_b[roi_idx]
            if not perm_a_roi or not perm_b_roi:
                cp_list.append(1.0)
                cw_list.append([])
                cn_list.append(np.zeros(0, dtype=np.float64))
                continue
            null = compute_cluster_null_distribution_paired(
                perm_a_roi,
                perm_b_roi,
                cluster_threshold_alpha=params.cluster_threshold_alpha,
                n_group_perm=params.n_group_permutations,
                rng=rng,
            )
            result = _cluster_result_from_null(null, obs_clusters[:n_keep], time_axis_s, alpha)

        cp_list.append(result.best_p)
        cw_list.append(result.windows)
        cn_list.append(result.null)

    return np.asarray(cp_list, dtype=np.float64), cw_list, cn_list


def _run_vs_zero_perm_for_condition(
    roi_t_list: list[np.ndarray],
    roi_p_list: list[np.ndarray],
    obs_samples_list: list[np.ndarray],
    perm_list_per_roi: list[list[np.ndarray] | None],
    time_axis_s: np.ndarray,
    params: RegressionGroupParams,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[list[tuple[float, float]]], list[np.ndarray]]:
    n_keep = params.n_clusters_to_keep
    cp_list: list[float] = []
    cw_list: list[list[tuple[float, float]]] = []
    cn_list: list[np.ndarray] = []

    for roi_idx in range(len(roi_t_list)):
        h_mask = roi_p_list[roi_idx] < params.cluster_threshold_alpha
        obs_clusters = find_temporal_clusters(h_mask, roi_t_list[roi_idx])

        if params.cluster_permutation_method == "sign_flip":
            obs = obs_samples_list[roi_idx]
            if obs is None or len(obs) < 2:
                cp_list.append(1.0)
                cw_list.append([])
                cn_list.append(np.zeros(0, dtype=np.float64))
                continue
            result = _run_sign_flip_perm_roi(
                obs, time_axis_s,
                params.cluster_threshold_alpha, params.n_group_permutations, n_keep, alpha, rng,
            )
        else:  # custom
            perm_list = perm_list_per_roi[roi_idx]
            if not perm_list:
                cp_list.append(1.0)
                cw_list.append([])
                cn_list.append(np.zeros(0, dtype=np.float64))
                continue
            null = compute_cluster_null_distribution(
                perm_list,
                cluster_threshold_alpha=params.cluster_threshold_alpha,
                n_group_perm=params.n_group_permutations,
                rng=rng,
            )
            result = _cluster_result_from_null(null, obs_clusters[:n_keep], time_axis_s, alpha)

        cp_list.append(result.best_p)
        cw_list.append(result.windows)
        cn_list.append(result.null)

    return np.asarray(cp_list, dtype=np.float64), cw_list, cn_list


def _run_vs_zero_cluster_perm_pass(
    acc: _ROIAccumulator,
    time_axis_s: np.ndarray,
    params: RegressionGroupParams,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list, list, np.ndarray, list, list]:
    # vs-zero always uses slope samples regardless of primary_regression_metric
    p_a, w_a, n_a = _run_vs_zero_perm_for_condition(
        acc.vs_zero_t_a, acc.vs_zero_p_a,
        obs_samples_list=acc.slope_a_samples,
        perm_list_per_roi=acc.perm_slope_a,
        time_axis_s=time_axis_s, params=params, alpha=alpha, rng=rng,
    )
    p_b, w_b, n_b = _run_vs_zero_perm_for_condition(
        acc.vs_zero_t_b, acc.vs_zero_p_b,
        obs_samples_list=acc.slope_b_samples,
        perm_list_per_roi=acc.perm_slope_b,
        time_axis_s=time_axis_s, params=params, alpha=alpha, rng=rng,
    )
    return p_a, w_a, n_a, p_b, w_b, n_b


def _masks_from_windows(
    n_rois: int,
    n_times: int,
    time_axis_s: np.ndarray,
    windows_list: list[list[tuple[float, float]]],
) -> np.ndarray:
    mask = np.zeros((n_rois, n_times), dtype=bool)
    for roi_idx, roi_windows in enumerate(windows_list):
        for t_start_s, t_end_s in roi_windows:
            mask[roi_idx, (time_axis_s >= t_start_s) & (time_axis_s <= t_end_s)] = True
    return mask


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
    primary_regression_metric: str,
    condition: str,
) -> np.ndarray:
    if condition not in {"a", "b"}:
        raise ValueError(f"Unsupported condition key {condition!r}.")
    if primary_regression_metric == "slope":
        return np.asarray(getattr(record, f"slope_{condition}_values"), dtype=np.float64)
    if primary_regression_metric == "r_value":
        return np.asarray(getattr(record, f"r_value_{condition}_values"), dtype=np.float64)
    raise ValueError(f"Unsupported primary_regression_metric={primary_regression_metric!r}.")


# ---------------------------------------------------------------------------
# Result metadata
# ---------------------------------------------------------------------------

def _build_result_metadata(
    params: RegressionGroupParams,
    first: _RegressionStatsInput,
    method: str,
    alpha: float,
) -> dict:
    return {
        "p_value_correction_method": method,
        "significance_alpha": alpha,
        "primary_regression_metric": params.primary_regression_metric,
        "contrast_mode": params.contrast_mode,
        "roi_mode": params.roi_mode,
        "atlas_name": params.atlas_name,
        "binning_mode": first.binning_mode,
        "window_ms": first.window_ms,
        "n_bins": first.n_bins,
        "effective_n_bins": first.effective_n_bins,
        "predictor": first.result.predictor,
        "predictor_zscore": first.result.predictor_zscore,
        "predictor_transform_by_condition_json": json.dumps(
            first.result.predictor_transform_by_condition,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "activity_zscore": first.result.activity_zscore,
        "activity_baseline_tmin_s": first.result.activity_baseline_tmin_s,
        "activity_baseline_tmax_s": first.result.activity_baseline_tmax_s,
        "trial_activity_summary_kind": first.result.trial_activity_summary_kind,
        "trial_activity_summary_missing_response_policy": (
            first.result.trial_activity_summary_missing_response_policy
        ),
        "trial_activity_summary_source_json": json.dumps(
            first.result.trial_activity_summary_source,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "trial_activity_summary_label": first.result.trial_activity_summary_label,
        "scatter_aggregation": "trial_pool",
    }


# ---------------------------------------------------------------------------
# ROI record collection
# ---------------------------------------------------------------------------

def _collect_manual_roi_records(
    *,
    inputs: Sequence[_RegressionStatsInput],
    manual_region_channels: dict[str, dict[str, list[str]]],
) -> dict[str, list[_ContributionRecord]]:
    return collect_manual_roi_records(
        inputs=inputs,
        manual_region_channels=manual_region_channels,
        create_record=_create_contribution_record,
    )


def _collect_atlas_roi_records(
    *,
    inputs: Sequence[_RegressionStatsInput],
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
    item: _RegressionStatsInput,
    idx: int,
) -> _ContributionRecord:
    result = item.result
    cond_a = result.regression.condition_a
    cond_b = result.regression.condition_b
    tas = result.trial_activity_summary_values

    def _safe_scatter(arr: np.ndarray, i: int) -> np.ndarray:
        if arr.ndim == 2 and arr.shape[0] > i:
            return np.asarray(arr[i, :], dtype=np.float64)
        return np.empty(0, dtype=np.float64)

    return _ContributionRecord(
        roi=roi,
        subject=subject,
        channel=item.channel_names[idx],
        source_stats_file=str(item.stats_file.path),
        slope_a_values=np.asarray(cond_a.slope[idx, :], dtype=np.float64),
        slope_b_values=np.asarray(cond_b.slope[idx, :], dtype=np.float64),
        mean_a_values=np.asarray(result.signal_activity.condition_a.mean[idx, :], dtype=np.float64),
        mean_b_values=np.asarray(result.signal_activity.condition_b.mean[idx, :], dtype=np.float64),
        r_value_a_values=np.asarray(cond_a.r_value[idx, :], dtype=np.float64),
        r_value_b_values=np.asarray(cond_b.r_value[idx, :], dtype=np.float64),
        predictor_a_raw_values=np.asarray(result.predictor_values.condition_a.raw_values, dtype=np.float64),
        predictor_b_raw_values=np.asarray(result.predictor_values.condition_b.raw_values, dtype=np.float64),
        predictor_a_transformed_values=np.asarray(
            result.predictor_values.condition_a.transformed_values, dtype=np.float64,
        ),
        predictor_b_transformed_values=np.asarray(
            result.predictor_values.condition_b.transformed_values, dtype=np.float64,
        ),
        predictor_a_values=np.asarray(result.predictor_values.condition_a.values, dtype=np.float64),
        predictor_b_values=np.asarray(result.predictor_values.condition_b.values, dtype=np.float64),
        scatter_activity_a=_safe_scatter(tas.condition_a, idx),
        scatter_activity_b=_safe_scatter(tas.condition_b, idx),
        perm_slope_a_values=(
            np.asarray(cond_a.permuted_slopes[:, idx, :], dtype=np.float32)
            if cond_a.permuted_slopes is not None
            else None
        ),
        perm_slope_b_values=(
            np.asarray(cond_b.permuted_slopes[:, idx, :], dtype=np.float32)
            if cond_b.permuted_slopes is not None
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Compatibility validation
# ---------------------------------------------------------------------------

def _validate_group_compatibility(inputs: Sequence[_RegressionStatsInput]) -> None:
    validate_group_compatibility(
        inputs,
        empty_message="At least one regression input is required.",
        non_channel_message="regression_group requires channel-level regression inputs.",
        incompatible_message=(
            "Incompatible regression inputs in one processing group. "
            "Use build_regression_compatible_groups() to split heterogeneous files."
        ),
    )


def _validate_source_metric_availability(
    inputs: Sequence[_RegressionStatsInput],
    *,
    primary_regression_metric: str,
) -> None:
    missing = [
        item.stats_file.path.name
        for item in inputs
        if primary_regression_metric not in _available_regression_metrics(item.result)
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"primary_regression_metric={primary_regression_metric!r} is not available in these regression "
            f"files: {joined}."
        )


def _read_input_signature(stats_file: BIDSFile) -> SubjectStatsSignature:
    item = _load_slope_stats_input(stats_file)
    return item.signature


def _load_slope_stats_input(stats_file: BIDSFile) -> _RegressionStatsInput:
    result = load_result_via_public_loader(stats_file, load_regression_result)
    base = build_subject_stats_input(
        stats_file,
        result,
        extra_key_parts={
            "predictor": result.predictor,
            "predictor_zscore": result.predictor_zscore,
            "predictor_transform_by_condition": result.predictor_transform_by_condition,
        },
    )
    return _RegressionStatsInput(**{f.name: getattr(base, f.name) for f in fields(base)})


def _available_regression_metrics(result: RegressionProcessingResult) -> set[str]:
    available = result.metadata.get("available_regression_metrics")
    if not available:
        available = result.available_regression_metrics()
    return {str(item) for item in available}


# ---------------------------------------------------------------------------
