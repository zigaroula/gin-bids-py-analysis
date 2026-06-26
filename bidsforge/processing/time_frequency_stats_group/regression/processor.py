from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.processing.time_frequency_stats.regression import (
    TimeFrequencyRegressionResult,
    load_time_frequency_regression_result,
)
from bidsforge.processing.utils.group_stats import ROIChannelContribution
from bidsforge.processing.utils.time_frequency_group_stats import (
    build_tf_cluster_null_from_permutations,
    compute_condition_tf_stats,
    compute_one_sample_tf_maps,
    compute_paired_tf_maps,
    compute_tf_epoch_summary,
    correct_tf_p_values,
    signed_percentile_cluster_mask,
)

from ..compatibility import (
    TFSubjectStatsInput,
    TFSubjectStatsSignature,
    build_compatible_groups,
    build_tf_subject_input,
    load_result_via_public_loader,
    validate_group_compatibility,
)
from ..processor import (
    BaseTFGroupContributionRecord,
    BaseTimeFrequencyStatsGroupProcessing,
    check_roi_exclusion,
    collect_atlas_roi_records,
    collect_manual_roi_records,
)
from ..result import (
    TFGroupEpochStats,
    TFGroupEstimate,
    TFGroupEstimatePair,
    TFGroupStats,
    TFIndexedConditionContributions,
)
from .params import TimeFrequencyRegressionGroupParams
from .result import TFRegressionGroupStats, TimeFrequencyRegressionGroupResult


@dataclass(frozen=True)
class _RegressionInput(TFSubjectStatsInput):
    result: TimeFrequencyRegressionResult
    source_metric: str


@dataclass(frozen=True)
class _ContributionRecord(BaseTFGroupContributionRecord):
    metric_a: np.ndarray
    metric_b: np.ndarray
    mean_a: np.ndarray
    mean_b: np.ndarray
    r_a: np.ndarray
    r_b: np.ndarray
    perm_a: np.ndarray | None
    perm_b: np.ndarray | None


@dataclass
class _ROIStats:
    name: str
    metric_a_mean: np.ndarray
    metric_a_sem: np.ndarray
    metric_b_mean: np.ndarray
    metric_b_sem: np.ndarray
    r_a_mean: np.ndarray
    r_a_sem: np.ndarray
    r_b_mean: np.ndarray
    r_b_sem: np.ndarray
    activity_a_mean: np.ndarray
    activity_a_sem: np.ndarray
    activity_b_mean: np.ndarray
    activity_b_sem: np.ndarray
    contrast_t: np.ndarray
    contrast_p_raw: np.ndarray
    contrast_p: np.ndarray
    contrast_sig: np.ndarray
    vz_a_t: np.ndarray
    vz_a_p_raw: np.ndarray
    vz_a_p: np.ndarray
    vz_a_sig: np.ndarray
    vz_b_t: np.ndarray
    vz_b_p_raw: np.ndarray
    vz_b_p: np.ndarray
    vz_b_sig: np.ndarray
    epoch: tuple[float, float, float, float, float]
    metric_a_samples: np.ndarray
    metric_b_samples: np.ndarray
    activity_a_samples: np.ndarray
    activity_b_samples: np.ndarray
    labels: list[str]
    contributions: list[ROIChannelContribution]
    cluster_labels: np.ndarray | None
    cluster_sums: np.ndarray | None
    cluster_null: np.ndarray | None


def build_time_frequency_regression_compatible_groups(
    stats_files: Sequence[BIDSFile],
    *,
    primary_regression_metric: str = "t_values",
) -> list[BIDSFileGroup]:
    return build_compatible_groups(
        stats_files,
        read_signature=lambda file: _read_input_signature(
            file,
            source_metric=primary_regression_metric,
        ),
    )


class TimeFrequencyRegressionGroupProcessing(BaseTimeFrequencyStatsGroupProcessing):
    def __init__(self, params: TimeFrequencyRegressionGroupParams) -> None:
        self.params = params

    def process_group(
        self,
        group: BIDSFileGroup,
        progress_tracking_position: int = 0,
    ) -> TimeFrequencyRegressionGroupResult:
        del progress_tracking_position
        files = self.sorted_group_files(group)
        inputs = [
            _load_input(file, source_metric=self.params.primary_regression_metric)
            for file in files
        ]
        validate_group_compatibility(inputs)
        first = inputs[0]
        roi_records, used_electrodes = self._collect_records(inputs)

        excluded: dict[str, str] = {}
        roi_stats: list[_ROIStats] = []
        for roi, records in roi_records.items():
            exclusion = check_roi_exclusion(
                records,
                min_channels_per_roi=self.params.min_channels_per_roi,
                min_subjects_per_roi=self.params.min_subjects_per_roi,
            )
            if exclusion:
                excluded[roi] = exclusion
                continue
            roi_stats.append(_process_roi(roi, records, self.params))

        n_freqs = int(first.frequency_hz.size)
        n_times = int(first.time_axis_s.size)
        return TimeFrequencyRegressionGroupResult(
            source_group=group,
            metadata=_build_metadata(self.params, first),
            frequency_hz=first.frequency_hz.copy(),
            time_axis_s=first.time_axis_s.copy(),
            region_names=[item.name for item in roi_stats],
            condition_labels=first.condition_labels,
            roi_channel_counts=np.asarray([len(item.contributions) for item in roi_stats], dtype=np.int64),
            roi_subject_counts=np.asarray(
                [len({contrib.subject for contrib in item.contributions}) for item in roi_stats],
                dtype=np.int64,
            ),
            contributions=[contrib for item in roi_stats for contrib in item.contributions],
            source_subject_stats_files=[str(item.stats_file.path) for item in inputs],
            source_electrodes_files=sorted(used_electrodes),
            excluded_rois=excluded,
            p_value_correction_method=self.params.p_value_correction_method,
            significance_alpha=self.params.significance_alpha,
            roi_mode=self.params.roi_mode,
            atlas_name=self.params.atlas_name,
            source_metric=TFGroupEstimatePair(
                condition_a=TFGroupEstimate(
                    mean=self.stack_maps([item.metric_a_mean for item in roi_stats], n_freqs, n_times),
                    sem=self.stack_maps([item.metric_a_sem for item in roi_stats], n_freqs, n_times),
                ),
                condition_b=TFGroupEstimate(
                    mean=self.stack_maps([item.metric_b_mean for item in roi_stats], n_freqs, n_times),
                    sem=self.stack_maps([item.metric_b_sem for item in roi_stats], n_freqs, n_times),
                ),
            ),
            signal_activity=TFGroupEstimatePair(
                condition_a=TFGroupEstimate(
                    mean=self.stack_maps([item.activity_a_mean for item in roi_stats], n_freqs, n_times),
                    sem=self.stack_maps([item.activity_a_sem for item in roi_stats], n_freqs, n_times),
                ),
                condition_b=TFGroupEstimate(
                    mean=self.stack_maps([item.activity_b_mean for item in roi_stats], n_freqs, n_times),
                    sem=self.stack_maps([item.activity_b_sem for item in roi_stats], n_freqs, n_times),
                ),
            ),
            r_value=TFGroupEstimatePair(
                condition_a=TFGroupEstimate(
                    mean=self.stack_maps([item.r_a_mean for item in roi_stats], n_freqs, n_times),
                    sem=self.stack_maps([item.r_a_sem for item in roi_stats], n_freqs, n_times),
                ),
                condition_b=TFGroupEstimate(
                    mean=self.stack_maps([item.r_b_mean for item in roi_stats], n_freqs, n_times),
                    sem=self.stack_maps([item.r_b_sem for item in roi_stats], n_freqs, n_times),
                ),
            ),
            stats=TFRegressionGroupStats(
                condition_contrast=TFGroupStats(
                    t_values=self.stack_maps([item.contrast_t for item in roi_stats], n_freqs, n_times),
                    p_values=self.stack_maps([item.contrast_p for item in roi_stats], n_freqs, n_times),
                    p_values_uncorrected=self.stack_maps([item.contrast_p_raw for item in roi_stats], n_freqs, n_times),
                    significant_mask=self.stack_maps(
                        [item.contrast_sig.astype(np.float64) for item in roi_stats],
                        n_freqs,
                        n_times,
                    ).astype(bool),
                    cluster_labels=[item.cluster_labels for item in roi_stats if item.cluster_labels is not None] or None,
                    cluster_sums=[item.cluster_sums for item in roi_stats if item.cluster_sums is not None] or None,
                    cluster_null_distributions=[item.cluster_null for item in roi_stats if item.cluster_null is not None] or None,
                ),
                condition_a_vs_zero=TFGroupStats(
                    t_values=self.stack_maps([item.vz_a_t for item in roi_stats], n_freqs, n_times),
                    p_values=self.stack_maps([item.vz_a_p for item in roi_stats], n_freqs, n_times),
                    p_values_uncorrected=self.stack_maps([item.vz_a_p_raw for item in roi_stats], n_freqs, n_times),
                    significant_mask=self.stack_maps(
                        [item.vz_a_sig.astype(np.float64) for item in roi_stats],
                        n_freqs,
                        n_times,
                    ).astype(bool),
                ),
                condition_b_vs_zero=TFGroupStats(
                    t_values=self.stack_maps([item.vz_b_t for item in roi_stats], n_freqs, n_times),
                    p_values=self.stack_maps([item.vz_b_p for item in roi_stats], n_freqs, n_times),
                    p_values_uncorrected=self.stack_maps([item.vz_b_p_raw for item in roi_stats], n_freqs, n_times),
                    significant_mask=self.stack_maps(
                        [item.vz_b_sig.astype(np.float64) for item in roi_stats],
                        n_freqs,
                        n_times,
                    ).astype(bool),
                ),
                epoch_summary=TFGroupEpochStats(
                    t=self.array_1d([item.epoch[0] for item in roi_stats]),
                    p=self.array_1d([item.epoch[1] for item in roi_stats]),
                    df=self.array_1d([item.epoch[2] for item in roi_stats]),
                    mean=self.array_1d([item.epoch[3] for item in roi_stats]),
                    sem=self.array_1d([item.epoch[4] for item in roi_stats]),
                ),
            ),
            source_metric_contributions=TFIndexedConditionContributions(
                condition_a=[item.metric_a_samples for item in roi_stats],
                condition_b=[item.metric_b_samples for item in roi_stats],
                labels=[item.labels for item in roi_stats],
            ),
            signal_activity_contributions=TFIndexedConditionContributions(
                condition_a=[item.activity_a_samples for item in roi_stats],
                condition_b=[item.activity_b_samples for item in roi_stats],
                labels=[item.labels for item in roi_stats],
            ),
            primary_regression_metric=self.params.primary_regression_metric,
            contrast_mode=self.params.contrast_mode,
        )

    def _collect_records(
        self,
        inputs: Sequence[_RegressionInput],
    ) -> tuple[dict[str, list[_ContributionRecord]], set[str]]:
        if self.params.roi_mode == "manual":
            return (
                collect_manual_roi_records(
                    inputs=inputs,
                    manual_region_channels=self.params.manual_region_channels,
                    create_record=_create_record,
                ),
                set(),
            )
        assert self.params.atlas_name is not None
        return collect_atlas_roi_records(
            inputs=inputs,
            atlas_name=self.params.atlas_name,
            create_record=_create_record,
        )


def _process_roi(
    roi: str,
    records: list[_ContributionRecord],
    params: TimeFrequencyRegressionGroupParams,
) -> _ROIStats:
    a = np.stack([record.metric_a for record in records], axis=0)
    b = np.stack([record.metric_b for record in records], axis=0)
    mean_a, sem_a = compute_condition_tf_stats(a)
    mean_b, sem_b = compute_condition_tf_stats(b)
    vz_a_t, vz_a_p_raw, _mean, _sem = compute_one_sample_tf_maps(a)
    vz_b_t, vz_b_p_raw, _mean, _sem = compute_one_sample_tf_maps(b)
    contrast_t, contrast_p_raw = compute_paired_tf_maps(a, b)

    vz_a_p = correct_tf_p_values(vz_a_p_raw, method=params.p_value_correction_method)
    vz_b_p = correct_tf_p_values(vz_b_p_raw, method=params.p_value_correction_method)
    contrast_p = correct_tf_p_values(contrast_p_raw, method=params.p_value_correction_method)
    contrast_sig = np.isfinite(contrast_p) & (contrast_p < params.significance_alpha)
    cluster_labels = cluster_sums = cluster_null = None
    if params.p_value_correction_method == "cluster_permutation":
        perm_a, perm_b = _permutation_lists(records, params)
        perm_diff = [pa - pb for pa, pb in zip(perm_a, perm_b)]
        cluster_null = build_tf_cluster_null_from_permutations(
            perm_diff,
            cluster_threshold_alpha=params.cluster_threshold_alpha,
        )
        source_map = mean_a - mean_b
        contrast_sig, cluster_labels, cluster_sums = signed_percentile_cluster_mask(
            source_map=source_map,
            p_values=contrast_p_raw,
            null_distribution=cluster_null,
            cluster_threshold_alpha=params.cluster_threshold_alpha,
            cluster_percentile_alpha=params.cluster_percentile_alpha,
        )
        contrast_p = contrast_p_raw.copy()

    activity_a = np.stack([record.mean_a for record in records], axis=0)
    activity_b = np.stack([record.mean_b for record in records], axis=0)
    act_a_mean, act_a_sem = compute_condition_tf_stats(activity_a)
    act_b_mean, act_b_sem = compute_condition_tf_stats(activity_b)
    r_a = np.stack([record.r_a for record in records], axis=0)
    r_b = np.stack([record.r_b for record in records], axis=0)
    r_a_mean, r_a_sem = compute_condition_tf_stats(r_a)
    r_b_mean, r_b_sem = compute_condition_tf_stats(r_b)
    return _ROIStats(
        name=roi,
        metric_a_mean=mean_a,
        metric_a_sem=sem_a,
        metric_b_mean=mean_b,
        metric_b_sem=sem_b,
        r_a_mean=r_a_mean,
        r_a_sem=r_a_sem,
        r_b_mean=r_b_mean,
        r_b_sem=r_b_sem,
        activity_a_mean=act_a_mean,
        activity_a_sem=act_a_sem,
        activity_b_mean=act_b_mean,
        activity_b_sem=act_b_sem,
        contrast_t=contrast_t,
        contrast_p_raw=contrast_p_raw,
        contrast_p=contrast_p,
        contrast_sig=contrast_sig,
        vz_a_t=vz_a_t,
        vz_a_p_raw=vz_a_p_raw,
        vz_a_p=vz_a_p,
        vz_a_sig=np.isfinite(vz_a_p) & (vz_a_p < params.significance_alpha),
        vz_b_t=vz_b_t,
        vz_b_p_raw=vz_b_p_raw,
        vz_b_p=vz_b_p,
        vz_b_sig=np.isfinite(vz_b_p) & (vz_b_p < params.significance_alpha),
        epoch=compute_tf_epoch_summary(a - b),
        metric_a_samples=a,
        metric_b_samples=b,
        activity_a_samples=activity_a,
        activity_b_samples=activity_b,
        labels=[f"{record.subject}/{record.channel}" for record in records],
        contributions=[record.as_summary() for record in records],
        cluster_labels=cluster_labels,
        cluster_sums=cluster_sums,
        cluster_null=cluster_null,
    )


def _permutation_lists(
    records: list[_ContributionRecord],
    params: TimeFrequencyRegressionGroupParams,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if params.cluster_permutation_method == "custom":
        if any(record.perm_a is None or record.perm_b is None for record in records):
            raise ValueError(
                "cluster_permutation/custom requires permuted_slopes in every source regression file."
            )
        return (
            [np.asarray(record.perm_a, dtype=np.float64) for record in records],
            [np.asarray(record.perm_b, dtype=np.float64) for record in records],
        )
    rng = np.random.default_rng(params.permutation_seed)
    out_a = []
    out_b = []
    for record in records:
        signs = rng.choice([-1.0, 1.0], size=(params.n_group_permutations, 1, 1))
        out_a.append(signs * record.metric_a[np.newaxis, :, :])
        out_b.append(signs * record.metric_b[np.newaxis, :, :])
    return out_a, out_b


def _read_input_signature(stats_file: BIDSFile, *, source_metric: str) -> TFSubjectStatsSignature:
    return _load_input(stats_file, source_metric=source_metric).signature


def _load_input(stats_file: BIDSFile, *, source_metric: str) -> _RegressionInput:
    result = load_result_via_public_loader(stats_file, load_time_frequency_regression_result)
    _validate_metric(source_metric, filename=stats_file.path.name)
    base = build_tf_subject_input(
        stats_file,
        result,
        extra_key_parts={
            "primary_regression_metric": source_metric,
            "predictor": result.predictor,
            "predictor_zscore": result.predictor_zscore,
            "predictor_transform_by_condition": result.predictor_transform_by_condition,
        },
    )
    values = dict(base.__dict__)
    values["result"] = result
    values["source_metric"] = source_metric
    return _RegressionInput(**values)


def _create_record(
    roi: str,
    subject: str,
    item: _RegressionInput,
    idx: int,
) -> _ContributionRecord:
    cond_a = item.result.regression.condition_a
    cond_b = item.result.regression.condition_b
    metric_a = _condition_metric(cond_a, item.source_metric)
    metric_b = _condition_metric(cond_b, item.source_metric)
    return _ContributionRecord(
        roi=roi,
        subject=subject,
        channel=item.channel_names[idx],
        source_stats_file=str(item.stats_file.path),
        metric_a=np.asarray(metric_a[idx], dtype=np.float64),
        metric_b=np.asarray(metric_b[idx], dtype=np.float64),
        mean_a=np.asarray(item.result.signal_activity.condition_a.mean[idx], dtype=np.float64),
        mean_b=np.asarray(item.result.signal_activity.condition_b.mean[idx], dtype=np.float64),
        r_a=np.asarray(cond_a.r_value[idx], dtype=np.float64),
        r_b=np.asarray(cond_b.r_value[idx], dtype=np.float64),
        perm_a=(
            np.asarray(cond_a.permuted_slopes[:, idx], dtype=np.float32)
            if cond_a.permuted_slopes is not None
            else None
        ),
        perm_b=(
            np.asarray(cond_b.permuted_slopes[:, idx], dtype=np.float32)
            if cond_b.permuted_slopes is not None
            else None
        ),
    )


def _condition_metric(condition: object, metric: str) -> np.ndarray:
    if metric == "t_values":
        return np.asarray(condition.t_values, dtype=np.float64)
    if metric == "slope":
        return np.asarray(condition.slope, dtype=np.float64)
    if metric == "r_value":
        return np.asarray(condition.r_value, dtype=np.float64)
    raise ValueError(f"Unsupported primary_regression_metric={metric!r}.")


def _validate_metric(source_metric: str, *, filename: str) -> None:
    if source_metric not in {"t_values", "slope", "r_value"}:
        raise ValueError(
            f"primary_regression_metric={source_metric!r} is not available in {filename}."
        )


def _build_metadata(
    params: TimeFrequencyRegressionGroupParams,
    first: _RegressionInput,
) -> dict[str, object]:
    return {
        "primary_regression_metric": params.primary_regression_metric,
        "contrast_mode": params.contrast_mode,
        "cluster_permutation_method": params.cluster_permutation_method,
        "cluster_threshold_alpha": float(params.cluster_threshold_alpha),
        "cluster_percentile_alpha": float(params.cluster_percentile_alpha),
        "n_group_permutations": int(params.n_group_permutations),
        "power_mode": first.result.power_mode,
        "time_selection": str(first.result.metadata.get("time_selection", "")),
        "baseline_grand_average": bool(first.result.metadata.get("baseline_grand_average", False)),
        "predictor": first.result.predictor,
        "predictor_zscore": first.result.predictor_zscore,
    }
