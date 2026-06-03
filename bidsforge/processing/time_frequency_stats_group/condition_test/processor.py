from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.processing.time_frequency_stats.condition_test import (
    TimeFrequencyConditionTestResult,
    load_time_frequency_condition_test_result,
)
from bidsforge.processing.utils.group_stats import ROIChannelContribution
from bidsforge.processing.utils.time_frequency_group_stats import (
    build_tf_cluster_null_from_permutations,
    compute_condition_tf_stats,
    compute_one_sample_tf_maps,
    compute_tf_epoch_summary,
    correct_tf_p_values,
    matlab_style_cluster_mask,
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
from .params import TimeFrequencyConditionTestGroupParams
from .result import TimeFrequencyConditionTestGroupResult
from ..result import (
    TFGroupEpochStats,
    TFGroupEstimate,
    TFGroupEstimatePair,
    TFGroupStats,
    TFIndexedConditionContributions,
    TFIndexedContributions,
)


@dataclass(frozen=True)
class _ConditionInput(TFSubjectStatsInput):
    result: TimeFrequencyConditionTestResult
    source_metric: str


@dataclass(frozen=True)
class _ContributionRecord(BaseTFGroupContributionRecord):
    values: np.ndarray
    condition_a_values: np.ndarray
    condition_b_values: np.ndarray
    permuted_values: np.ndarray | None


@dataclass
class _ROIStats:
    name: str
    source_mean: np.ndarray
    source_sem: np.ndarray
    t_values: np.ndarray
    p_uncorrected: np.ndarray
    p_values: np.ndarray
    significant_mask: np.ndarray
    epoch: tuple[float, float, float, float, float]
    condition_a_mean: np.ndarray
    condition_a_sem: np.ndarray
    condition_b_mean: np.ndarray
    condition_b_sem: np.ndarray
    source_samples: np.ndarray
    condition_a_samples: np.ndarray
    condition_b_samples: np.ndarray
    labels: list[str]
    contributions: list[ROIChannelContribution]
    cluster_labels: np.ndarray | None
    cluster_sums: np.ndarray | None
    cluster_null: np.ndarray | None


_METRIC_EXTRACTORS = {
    "t_values": lambda r: np.asarray(r.contrast.t_values, dtype=np.float64),
    "mean_difference": lambda r: np.asarray(r.difference.mean, dtype=np.float64),
    "condition_a_mean": lambda r: np.asarray(r.signal_activity.condition_a.mean, dtype=np.float64),
    "condition_b_mean": lambda r: np.asarray(r.signal_activity.condition_b.mean, dtype=np.float64),
}


def build_time_frequency_condition_test_compatible_groups(
    stats_files: Sequence[BIDSFile],
    *,
    primary_condition_metric: str = "t_values",
) -> list[BIDSFileGroup]:
    return build_compatible_groups(
        stats_files,
        read_signature=lambda file: _read_input_signature(
            file,
            source_metric=primary_condition_metric,
        ),
    )


class TimeFrequencyConditionTestGroupProcessing(BaseTimeFrequencyStatsGroupProcessing):
    def __init__(self, params: TimeFrequencyConditionTestGroupParams) -> None:
        self.params = params

    def process_group(
        self,
        group: BIDSFileGroup,
        progress_tracking_position: int = 0,
    ) -> TimeFrequencyConditionTestGroupResult:
        del progress_tracking_position
        files = self.sorted_group_files(group)
        inputs = [
            _load_input(file, source_metric=self.params.primary_condition_metric)
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
        region_names = [item.name for item in roi_stats]
        cluster_labels = [item.cluster_labels for item in roi_stats if item.cluster_labels is not None]
        cluster_sums = [item.cluster_sums for item in roi_stats if item.cluster_sums is not None]
        cluster_null = [item.cluster_null for item in roi_stats if item.cluster_null is not None]

        return TimeFrequencyConditionTestGroupResult(
            source_group=group,
            metadata=_build_metadata(self.params, first),
            frequency_hz=first.frequency_hz.copy(),
            time_axis_s=first.time_axis_s.copy(),
            region_names=region_names,
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
            source_metric=TFGroupEstimate(
                mean=self.stack_maps([item.source_mean for item in roi_stats], n_freqs, n_times),
                sem=self.stack_maps([item.source_sem for item in roi_stats], n_freqs, n_times),
            ),
            source_metric_stats=TFGroupStats(
                t_values=self.stack_maps([item.t_values for item in roi_stats], n_freqs, n_times),
                p_values=self.stack_maps([item.p_values for item in roi_stats], n_freqs, n_times),
                p_values_uncorrected=self.stack_maps(
                    [item.p_uncorrected for item in roi_stats],
                    n_freqs,
                    n_times,
                ),
                significant_mask=self.stack_maps(
                    [item.significant_mask.astype(np.float64) for item in roi_stats],
                    n_freqs,
                    n_times,
                ).astype(bool),
                cluster_labels=cluster_labels or None,
                cluster_sums=cluster_sums or None,
                cluster_null_distributions=cluster_null or None,
            ),
            source_metric_epoch=TFGroupEpochStats(
                t=self.array_1d([item.epoch[0] for item in roi_stats]),
                p=self.array_1d([item.epoch[1] for item in roi_stats]),
                df=self.array_1d([item.epoch[2] for item in roi_stats]),
                mean=self.array_1d([item.epoch[3] for item in roi_stats]),
                sem=self.array_1d([item.epoch[4] for item in roi_stats]),
            ),
            signal_activity=TFGroupEstimatePair(
                condition_a=TFGroupEstimate(
                    mean=self.stack_maps([item.condition_a_mean for item in roi_stats], n_freqs, n_times),
                    sem=self.stack_maps([item.condition_a_sem for item in roi_stats], n_freqs, n_times),
                ),
                condition_b=TFGroupEstimate(
                    mean=self.stack_maps([item.condition_b_mean for item in roi_stats], n_freqs, n_times),
                    sem=self.stack_maps([item.condition_b_sem for item in roi_stats], n_freqs, n_times),
                ),
            ),
            source_metric_contributions=TFIndexedContributions(
                values=[item.source_samples for item in roi_stats],
                labels=[item.labels for item in roi_stats],
            ),
            signal_activity_contributions=TFIndexedConditionContributions(
                condition_a=[item.condition_a_samples for item in roi_stats],
                condition_b=[item.condition_b_samples for item in roi_stats],
                labels=[item.labels for item in roi_stats],
            ),
            primary_condition_metric=self.params.primary_condition_metric,
        )

    def _collect_records(
        self,
        inputs: Sequence[_ConditionInput],
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
    params: TimeFrequencyConditionTestGroupParams,
) -> _ROIStats:
    samples = np.stack([record.values for record in records], axis=0)
    t_values, p_raw, mean_values, sem_values = compute_one_sample_tf_maps(samples)
    p_values = correct_tf_p_values(p_raw, method=params.p_value_correction_method)
    sig = np.isfinite(p_values) & (p_values < params.significance_alpha)
    cluster_labels = cluster_sums = cluster_null = None
    if params.p_value_correction_method == "cluster_permutation":
        perm_list = _permutation_list(records, params)
        cluster_null = build_tf_cluster_null_from_permutations(
            perm_list,
            cluster_threshold_alpha=params.cluster_threshold_alpha,
        )
        sig, cluster_labels, cluster_sums = matlab_style_cluster_mask(
            source_map=mean_values,
            p_values=p_raw,
            null_distribution=cluster_null,
            cluster_threshold_alpha=params.cluster_threshold_alpha,
            cluster_percentile_alpha=params.cluster_percentile_alpha,
        )
        p_values = p_raw.copy()

    samples_a = np.stack([record.condition_a_values for record in records], axis=0)
    samples_b = np.stack([record.condition_b_values for record in records], axis=0)
    mean_a, sem_a = compute_condition_tf_stats(samples_a)
    mean_b, sem_b = compute_condition_tf_stats(samples_b)
    return _ROIStats(
        name=roi,
        source_mean=mean_values,
        source_sem=sem_values,
        t_values=t_values,
        p_uncorrected=p_raw,
        p_values=p_values,
        significant_mask=sig,
        epoch=compute_tf_epoch_summary(samples),
        condition_a_mean=mean_a,
        condition_a_sem=sem_a,
        condition_b_mean=mean_b,
        condition_b_sem=sem_b,
        source_samples=samples,
        condition_a_samples=samples_a,
        condition_b_samples=samples_b,
        labels=[f"{record.subject}/{record.channel}" for record in records],
        contributions=[record.as_summary() for record in records],
        cluster_labels=cluster_labels,
        cluster_sums=cluster_sums,
        cluster_null=cluster_null,
    )


def _permutation_list(
    records: list[_ContributionRecord],
    params: TimeFrequencyConditionTestGroupParams,
) -> list[np.ndarray]:
    if params.cluster_permutation_method == "custom":
        if any(record.permuted_values is None for record in records):
            raise ValueError(
                "cluster_permutation/custom requires permuted_t_values in every source condition_test file."
            )
        return [np.asarray(record.permuted_values, dtype=np.float64) for record in records]
    rng = np.random.default_rng(params.permutation_seed)
    samples = np.stack([record.values for record in records], axis=0)
    out = []
    for sample in samples:
        signs = rng.choice([-1.0, 1.0], size=(params.n_group_permutations, 1, 1))
        out.append(signs * sample[np.newaxis, :, :])
    return out


def _read_input_signature(stats_file: BIDSFile, *, source_metric: str) -> TFSubjectStatsSignature:
    return _load_input(stats_file, source_metric=source_metric).signature


def _load_input(stats_file: BIDSFile, *, source_metric: str) -> _ConditionInput:
    result = load_result_via_public_loader(stats_file, load_time_frequency_condition_test_result)
    _validate_metric(result, source_metric, filename=stats_file.path.name)
    base = build_tf_subject_input(
        stats_file,
        result,
        extra_key_parts={"primary_condition_metric": source_metric},
    )
    values = dict(base.__dict__)
    values["result"] = result
    values["source_metric"] = source_metric
    return _ConditionInput(**values)


def _create_record(
    roi: str,
    subject: str,
    item: _ConditionInput,
    idx: int,
) -> _ContributionRecord:
    values = _metric_values(item.result, item.source_metric)
    return _ContributionRecord(
        roi=roi,
        subject=subject,
        channel=item.channel_names[idx],
        source_stats_file=str(item.stats_file.path),
        values=np.asarray(values[idx], dtype=np.float64),
        condition_a_values=np.asarray(item.result.signal_activity.condition_a.mean[idx], dtype=np.float64),
        condition_b_values=np.asarray(item.result.signal_activity.condition_b.mean[idx], dtype=np.float64),
        permuted_values=(
            np.asarray(item.result.contrast.permuted_t_values[:, idx], dtype=np.float32)
            if item.result.contrast.permuted_t_values is not None
            else None
        ),
    )


def _metric_values(result: TimeFrequencyConditionTestResult, source_metric: str) -> np.ndarray:
    extractor = _METRIC_EXTRACTORS.get(source_metric)
    if extractor is None:
        raise ValueError(f"Unsupported primary_condition_metric={source_metric!r}.")
    return extractor(result)


def _validate_metric(result: TimeFrequencyConditionTestResult, source_metric: str, *, filename: str) -> None:
    available = {"t_values", "mean_difference", "condition_a_mean", "condition_b_mean"}
    if source_metric not in available:
        raise ValueError(
            f"primary_condition_metric={source_metric!r} is not available in {filename}."
        )


def _build_metadata(
    params: TimeFrequencyConditionTestGroupParams,
    first: _ConditionInput,
) -> dict[str, object]:
    return {
        "primary_condition_metric": params.primary_condition_metric,
        "cluster_permutation_method": params.cluster_permutation_method,
        "cluster_threshold_alpha": float(params.cluster_threshold_alpha),
        "cluster_percentile_alpha": float(params.cluster_percentile_alpha),
        "n_group_permutations": int(params.n_group_permutations),
        "power_mode": first.result.power_mode,
        "time_selection": str(first.result.metadata.get("time_selection", "")),
        "baseline_grand_average": bool(first.result.metadata.get("baseline_grand_average", False)),
    }
