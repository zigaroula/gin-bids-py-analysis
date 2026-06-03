from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup

from bidsforge.processing.utils.group_stats import (
    compute_condition_group_stats,
    compute_one_sample_epoch_summary,
    compute_one_sample_timecourse,
)
from bidsforge.processing.trial_stats.condition_test import (
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
from bidsforge.processing.utils.cluster_permutation import (
    compute_cluster_null_distribution,
    compute_mne_cluster_permutation,
    compute_cluster_permutation_pvalue,
    find_temporal_clusters,
)
from bidsforge.processing.utils.statistics import correct_p_values


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


@dataclass
class _ROIStats:
    """All collected statistics for one valid ROI, built during the main processing loop."""

    name: str
    t: np.ndarray
    p_uncorrected: np.ndarray
    mean: np.ndarray
    sem: np.ndarray
    cond_a_mean: np.ndarray
    cond_a_sem: np.ndarray
    cond_b_mean: np.ndarray
    cond_b_sem: np.ndarray
    cond_a_samples: np.ndarray
    cond_b_samples: np.ndarray
    channel_count: int
    subject_count: int
    t_summary: float
    p_summary: float
    df_summary: float
    mean_summary: float
    sem_summary: float
    contributions: list[ROIChannelContribution]
    labels: list[str]
    perm_t_list: list[np.ndarray] | None
    observed_samples: np.ndarray | None


_METRIC_EXTRACTORS = {
    "mean_difference": lambda r: np.asarray(r.difference.mean, dtype=np.float64),
    "t_values": lambda r: np.asarray(r.contrast.t_values, dtype=np.float64),
    "condition_a_mean": lambda r: np.asarray(r.signal_activity.condition_a.mean, dtype=np.float64),
    "condition_b_mean": lambda r: np.asarray(r.signal_activity.condition_b.mean, dtype=np.float64),
}


def build_condition_test_compatible_groups(
    stats_files: Sequence[BIDSFile],
    *,
    primary_condition_metric: str,
) -> list[BIDSFileGroup]:
    """Group subject-level condition_test files by compatibility."""
    return build_compatible_groups(
        stats_files,
        read_signature=lambda file: _read_input_signature(file, source_metric=primary_condition_metric),
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
            _load_trial_stats_input(f, source_metric=self.params.primary_condition_metric)
            for f in files
        ]
        _validate_group_compatibility(inputs)

        method = self.params.p_value_correction_method
        rng = self._init_cluster_rng(method, inputs)
        first = inputs[0]
        n_times = int(len(first.time_axis_s))

        roi_records, used_electrode_paths = self._collect_roi_records(inputs)
        excluded_rois: dict[str, str] = {}
        roi_stats_list: list[_ROIStats] = []
        for roi, records in roi_records.items():
            exclusion = self._check_roi_exclusion(records)
            if exclusion:
                excluded_rois[roi] = exclusion
                continue
            roi_stats_list.append(
                _process_roi_records(roi, records, method, self.params.cluster_permutation_method)
            )

        t_values = self.stack_rows([s.t for s in roi_stats_list], n_times)
        p_values_uncorrected = self.stack_rows([s.p_uncorrected for s in roi_stats_list], n_times)
        metric_mean = self.stack_rows([s.mean for s in roi_stats_list], n_times)
        metric_sem = self.stack_rows([s.sem for s in roi_stats_list], n_times)
        condition_a_activity_mean = self.stack_rows([s.cond_a_mean for s in roi_stats_list], n_times)
        condition_a_activity_sem = self.stack_rows([s.cond_a_sem for s in roi_stats_list], n_times)
        condition_b_activity_mean = self.stack_rows([s.cond_b_mean for s in roi_stats_list], n_times)
        condition_b_activity_sem = self.stack_rows([s.cond_b_sem for s in roi_stats_list], n_times)

        p_values = correct_p_values(p_values_uncorrected, method=method)

        cluster_p_values_out: np.ndarray | None = None
        cluster_windows_out: list[list[tuple[float, float]]] | None = None
        cluster_null_dists_out: list[np.ndarray] | None = None
        if method == "cluster_permutation" and roi_stats_list and rng is not None:
            perm_results = [
                _compute_roi_cluster_perm(s, first.time_axis_s, self.params, rng)
                for s in roi_stats_list
            ]
            best_ps, windows_list, null_dists = zip(*perm_results)
            cluster_p_values_out = np.asarray(best_ps, dtype=np.float64)
            cluster_windows_out = list(windows_list)
            cluster_null_dists_out = list(null_dists)

        significant_mask = _build_significant_mask(
            p_values=p_values,
            method=method,
            cluster_windows_out=cluster_windows_out,
            time_axis_s=first.time_axis_s,
            significance_alpha=self.params.significance_alpha,
        )

        return ConditionTestGroupProcessingResult(
            source_group=group,
            metadata=_build_metadata(self.params, first, excluded_rois),
            signal_activity_stats=GroupTimecourseStats(
                t_values=t_values,
                p_values=p_values,
                p_values_uncorrected=p_values_uncorrected,
                significant_mask=significant_mask,
            ),
            condition_difference=GroupEstimate(mean=metric_mean, sem=metric_sem),
            time_axis_s=first.time_axis_s.astype(np.float64),
            region_names=[s.name for s in roi_stats_list],
            primary_condition_metric=self.params.primary_condition_metric,
            condition_labels=first.condition_labels,
            p_value_correction_method=self.params.p_value_correction_method,
            significance_alpha=self.params.significance_alpha,
            roi_mode=self.params.roi_mode,
            atlas_name=self.params.atlas_name,
            signal_activity_epoch=GroupEpochStats(
                t=self.array_1d([s.t_summary for s in roi_stats_list]),
                p=self.array_1d([s.p_summary for s in roi_stats_list]),
                df=self.array_1d([s.df_summary for s in roi_stats_list]),
            ),
            summary_epoch=ConditionTestEpochSummary(
                t_values=self.array_1d([s.t_summary for s in roi_stats_list]),
                p_values=self.array_1d([s.p_summary for s in roi_stats_list]),
                df=self.array_1d([s.df_summary for s in roi_stats_list]),
                condition_difference=GroupEstimate(
                    mean=self.array_1d([s.mean_summary for s in roi_stats_list]),
                    sem=self.array_1d([s.sem_summary for s in roi_stats_list]),
                ),
            ),
            roi_channel_counts=np.asarray([s.channel_count for s in roi_stats_list], dtype=np.int64),
            roi_subject_counts=np.asarray([s.subject_count for s in roi_stats_list], dtype=np.int64),
            contributions=[c for s in roi_stats_list for c in s.contributions],
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
                condition_a=[s.cond_a_samples for s in roi_stats_list],
                condition_b=[s.cond_b_samples for s in roi_stats_list],
                labels=[s.labels for s in roi_stats_list],
            ),
            source_subject_stats_files=[str(item.stats_file.path) for item in inputs],
            source_electrodes_files=sorted(used_electrode_paths),
            excluded_rois=excluded_rois,
            cluster_p_values=cluster_p_values_out,
            cluster_windows_s=cluster_windows_out,
            cluster_null_distributions=cluster_null_dists_out,
        )

    def _init_cluster_rng(
        self,
        method: str,
        inputs: Sequence[_ConditionTestStatsInput],
    ) -> np.random.Generator | None:
        if method != "cluster_permutation":
            return None
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
        return np.random.default_rng(self.params.permutation_seed)

    def _collect_roi_records(
        self,
        inputs: Sequence[_ConditionTestStatsInput],
    ) -> tuple[dict[str, list[_ContributionRecord]], set[str]]:
        if self.params.roi_mode == "manual":
            return (
                collect_manual_roi_records(
                    inputs=inputs,
                    manual_region_channels=self.params.manual_region_channels,
                    create_record=_create_contribution_record,
                ),
                set(),
            )
        assert self.params.atlas_name is not None
        return collect_atlas_roi_records(
            inputs=inputs,
            atlas_name=self.params.atlas_name,
            create_record=_create_contribution_record,
        )

    def _check_roi_exclusion(self, records: list[_ContributionRecord]) -> str | None:
        if not records:
            return "no_channels"
        n_channels = len(records)
        n_subjects = len({r.subject for r in records})
        if n_channels < self.params.min_channels_per_roi:
            return f"insufficient_channels:{n_channels}<{self.params.min_channels_per_roi}"
        if n_subjects < self.params.min_subjects_per_roi:
            return f"insufficient_subjects:{n_subjects}<{self.params.min_subjects_per_roi}"
        return None


def _process_roi_records(
    roi: str,
    records: list[_ContributionRecord],
    method: str,
    cluster_permutation_method: str,
) -> _ROIStats:
    samples = np.stack([r.values for r in records], axis=0).astype(np.float64)
    t_values, p_values_raw, mean_values, sem_values = compute_one_sample_timecourse(samples)
    t_summary, p_summary, df_summary, mean_summary, sem_summary = compute_one_sample_epoch_summary(
        samples
    )

    samples_a = np.stack([r.condition_a_values for r in records], axis=0).astype(np.float64)
    samples_b = np.stack([r.condition_b_values for r in records], axis=0).astype(np.float64)
    cond_a_mean, cond_a_sem = compute_condition_group_stats(samples_a)
    cond_b_mean, cond_b_sem = compute_condition_group_stats(samples_b)

    if method == "cluster_permutation":
        if cluster_permutation_method == "custom":
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

    return _ROIStats(
        name=roi,
        t=t_values,
        p_uncorrected=p_values_raw,
        mean=mean_values,
        sem=sem_values,
        cond_a_mean=cond_a_mean,
        cond_a_sem=cond_a_sem,
        cond_b_mean=cond_b_mean,
        cond_b_sem=cond_b_sem,
        cond_a_samples=samples_a,
        cond_b_samples=samples_b,
        channel_count=len(records),
        subject_count=len({r.subject for r in records}),
        t_summary=t_summary,
        p_summary=p_summary,
        df_summary=df_summary,
        mean_summary=mean_summary,
        sem_summary=sem_summary,
        contributions=[
            ROIChannelContribution(
                roi=r.roi,
                subject=r.subject,
                channel=r.channel,
                source_stats_file=r.source_stats_file,
            )
            for r in records
        ],
        labels=[f"{r.subject}/{r.channel}" for r in records],
        perm_t_list=perm_t_list,
        observed_samples=observed_samples,
    )


def _compute_roi_cluster_perm(
    roi_stats: _ROIStats,
    time_axis_s: np.ndarray,
    params: ConditionTestGroupParams,
    rng: np.random.Generator,
) -> tuple[float, list[tuple[float, float]], np.ndarray]:
    """Returns (best_p, significant_windows, null_distribution) for one ROI."""
    h_mask = roi_stats.p_uncorrected < params.cluster_threshold_alpha
    observed_clusters = find_temporal_clusters(h_mask, roi_stats.t)
    n_keep = params.n_clusters_to_keep

    if not roi_stats.perm_t_list:
        if params.cluster_permutation_method == "sign_flip" and roi_stats.observed_samples is not None:
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            best_p, top_windows_idx, top_p_vals, null = compute_mne_cluster_permutation(
                roi_stats.observed_samples,
                cluster_threshold_alpha=params.cluster_threshold_alpha,
                n_group_perm=params.n_group_permutations,
                seed=seed,
                n_clusters_to_keep=n_keep,
            )
            windows = [
                (float(time_axis_s[win_idx[0]]), float(time_axis_s[win_idx[1]]))
                for win_idx, p in zip(top_windows_idx, top_p_vals)
                if p < params.significance_alpha
            ]
            return best_p, windows, null
        return 1.0, [], np.zeros(0, dtype=np.float64)

    null = compute_cluster_null_distribution(
        roi_stats.perm_t_list,
        cluster_threshold_alpha=params.cluster_threshold_alpha,
        n_group_perm=params.n_group_permutations,
        rng=rng,
    )
    windows = []
    best_p = 1.0
    for i, (start, end, tsum) in enumerate(observed_clusters[:n_keep]):
        p = compute_cluster_permutation_pvalue(tsum, null)
        if i == 0:
            best_p = p
        if p < params.significance_alpha:
            windows.append((float(time_axis_s[start]), float(time_axis_s[end])))
    return best_p, windows, null


def _build_significant_mask(
    p_values: np.ndarray,
    method: str,
    cluster_windows_out: list[list[tuple[float, float]]] | None,
    time_axis_s: np.ndarray,
    significance_alpha: float,
) -> np.ndarray:
    if method == "cluster_permutation" and cluster_windows_out is not None:
        mask = np.zeros_like(p_values, dtype=bool)
        for roi_idx, roi_windows in enumerate(cluster_windows_out):
            for t_start_s, t_end_s in roi_windows:
                in_window = (time_axis_s >= t_start_s) & (time_axis_s <= t_end_s)
                mask[roi_idx, in_window] = True
        return mask
    return np.isfinite(p_values) & (p_values < significance_alpha)


def _build_metadata(
    params: ConditionTestGroupParams,
    first: _ConditionTestStatsInput,
    excluded_rois: dict[str, str],
) -> dict[str, object]:
    return {
        "primary_condition_metric": params.primary_condition_metric,
        "roi_mode": params.roi_mode,
        "atlas_name": params.atlas_name,
        "min_channels_per_roi": params.min_channels_per_roi,
        "min_subjects_per_roi": params.min_subjects_per_roi,
        "p_value_correction_method": params.p_value_correction_method,
        "significance_alpha": params.significance_alpha,
        "n_group_permutations": params.n_group_permutations,
        "cluster_threshold_alpha": params.cluster_threshold_alpha,
        "cluster_permutation_method": params.cluster_permutation_method,
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
    }


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
    return _load_trial_stats_input(stats_file, source_metric=source_metric).signature


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
    values = {f.name: getattr(base_input, f.name) for f in dataclasses.fields(base_input)}
    values["result"] = result
    values["source_metric"] = source_metric
    return _ConditionTestStatsInput(**values)


def _condition_metric_values(
    result: ConditionTestProcessingResult,
    source_metric: str,
) -> np.ndarray:
    extractor = _METRIC_EXTRACTORS.get(source_metric)
    if extractor is None:
        raise ValueError(f"Unsupported primary_condition_metric={source_metric!r}.")
    return extractor(result)


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
