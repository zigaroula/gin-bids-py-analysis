from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bidsforge.processing.base import BaseProcessingResult
from bidsforge.processing.utils.group_stats import ROIChannelContribution
from bidsforge.processing.utils.serialization import OutputTree, compressed


@dataclass
class TFGroupEstimate:
    mean: np.ndarray = field(default_factory=lambda: np.array([]))
    sem: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_output_dict(self) -> dict[str, object]:
        return {
            "mean": np.asarray(self.mean, dtype=np.float64),
            "sem": np.asarray(self.sem, dtype=np.float64),
        }


@dataclass
class TFGroupEstimatePair:
    condition_a: TFGroupEstimate = field(default_factory=TFGroupEstimate)
    condition_b: TFGroupEstimate = field(default_factory=TFGroupEstimate)

    def to_output_dict(self, *, label_a: str, label_b: str) -> dict[str, object]:
        return {
            label_a: self.condition_a.to_output_dict(),
            label_b: self.condition_b.to_output_dict(),
        }


@dataclass
class TFGroupStats:
    t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values_uncorrected: np.ndarray = field(default_factory=lambda: np.array([]))
    significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    cluster_labels: list[np.ndarray] | None = None
    cluster_sums: list[np.ndarray] | None = None
    cluster_null_distributions: list[np.ndarray] | None = None

    def to_output_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "t_values": np.asarray(self.t_values, dtype=np.float64),
            "p_values": np.asarray(self.p_values, dtype=np.float64),
            "p_values_uncorrected": np.asarray(self.p_values_uncorrected, dtype=np.float64),
            "significant_mask": np.asarray(self.significant_mask, dtype=bool),
        }
        if self.cluster_labels is not None:
            out["cluster"] = {
                "labels": _stack_ragged_maps(self.cluster_labels, dtype=np.int64),
                "sums": _stack_ragged_vectors(self.cluster_sums or []),
                "null_distributions": _stack_ragged_vectors(
                    self.cluster_null_distributions or []
                ),
            }
        return out


@dataclass
class TFGroupEpochStats:
    t: np.ndarray = field(default_factory=lambda: np.array([]))
    p: np.ndarray = field(default_factory=lambda: np.array([]))
    df: np.ndarray = field(default_factory=lambda: np.array([]))
    mean: np.ndarray = field(default_factory=lambda: np.array([]))
    sem: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_output_dict(self) -> dict[str, object]:
        return {
            "t": np.asarray(self.t, dtype=np.float64),
            "p": np.asarray(self.p, dtype=np.float64),
            "df": np.asarray(self.df, dtype=np.float64),
            "mean": np.asarray(self.mean, dtype=np.float64),
            "sem": np.asarray(self.sem, dtype=np.float64),
        }


@dataclass
class TFIndexedContributions:
    values: list[np.ndarray] = field(default_factory=list)
    labels: list[list[str]] = field(default_factory=list)

    def to_output_dict(self, *, region_names: list[str], key: str = "values") -> dict[str, object]:
        out: dict[str, object] = {}
        for idx, roi in enumerate(region_names):
            if idx >= len(self.values):
                continue
            out[roi] = {
                key: compressed(np.asarray(self.values[idx], dtype=np.float32)),
                "labels": np.asarray(
                    self.labels[idx] if idx < len(self.labels) else [],
                    dtype=object,
                ),
            }
        return out


@dataclass
class TFIndexedConditionContributions:
    condition_a: list[np.ndarray] = field(default_factory=list)
    condition_b: list[np.ndarray] = field(default_factory=list)
    labels: list[list[str]] = field(default_factory=list)

    def to_output_dict(
        self,
        *,
        region_names: list[str],
        label_a: str,
        label_b: str,
    ) -> dict[str, object]:
        out: dict[str, object] = {}
        for idx, roi in enumerate(region_names):
            out[roi] = {
                label_a: compressed(np.asarray(self.condition_a[idx], dtype=np.float32)),
                label_b: compressed(np.asarray(self.condition_b[idx], dtype=np.float32)),
                "labels": np.asarray(
                    self.labels[idx] if idx < len(self.labels) else [],
                    dtype=object,
                ),
            }
        return out


@dataclass
class BaseTimeFrequencyStatsGroupResult(BaseProcessingResult):
    frequency_hz: np.ndarray = field(default_factory=lambda: np.array([]))
    time_axis_s: np.ndarray = field(default_factory=lambda: np.array([]))
    region_names: list[str] = field(default_factory=list)
    condition_labels: tuple[str, str] = ("condition_a", "condition_b")
    roi_channel_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    roi_subject_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    contributions: list[ROIChannelContribution] = field(default_factory=list)
    source_subject_stats_files: list[str] = field(default_factory=list)
    source_electrodes_files: list[str] = field(default_factory=list)
    excluded_rois: dict[str, str] = field(default_factory=dict)
    p_value_correction_method: str = "none"
    significance_alpha: float = 0.05
    roi_mode: str = "manual"
    atlas_name: str | None = None

    def _base_tree(
        self,
        *,
        analysis_type: str,
        pipeline_name: str,
        pipeline_version: str,
    ) -> dict[str, object]:
        return {
            "axes": {
                "region": np.asarray(self.region_names, dtype=object),
                "frequency_hz": np.asarray(self.frequency_hz, dtype=np.float64),
                "time_s": np.asarray(self.time_axis_s, dtype=np.float64),
            },
            "meta": {
                "schema_name": "time_frequency_stats_group",
                "schema_version": "1.0",
                "analysis_type": analysis_type,
                "condition_labels": np.asarray(list(self.condition_labels), dtype=object),
                "p_value_correction_method": str(self.p_value_correction_method),
                "significance_alpha": float(self.significance_alpha),
                "roi_mode": str(self.roi_mode),
                "atlas_name": str(self.atlas_name or ""),
                "roi_channel_counts": np.asarray(self.roi_channel_counts, dtype=np.int64),
                "roi_subject_counts": np.asarray(self.roi_subject_counts, dtype=np.int64),
                **self.metadata,
            },
            "contributions": {
                "summary": {
                    "region": np.asarray([item.roi for item in self.contributions], dtype=object),
                    "subject": np.asarray([item.subject for item in self.contributions], dtype=object),
                    "channel": np.asarray([item.channel for item in self.contributions], dtype=object),
                    "source_stats_file": np.asarray(
                        [item.source_stats_file for item in self.contributions],
                        dtype=object,
                    ),
                },
            },
            "excluded_rois": {
                "region": np.asarray(list(self.excluded_rois.keys()), dtype=object),
                "reason": np.asarray(list(self.excluded_rois.values()), dtype=object),
            },
            "provenance": {
                "source_subject_stats_files": np.asarray(
                    self.source_subject_stats_files,
                    dtype=object,
                ),
                "source_electrodes_files": np.asarray(
                    self.source_electrodes_files,
                    dtype=object,
                ),
                "pipeline_name": pipeline_name,
                "pipeline_version": pipeline_version,
            },
        }

    def to_output_tree(
        self,
        *,
        pipeline_name: str = "time_frequency_stats_group",
        pipeline_version: str = "unknown",
    ) -> OutputTree:
        return self._base_tree(
            analysis_type="time_frequency_stats_group",
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
        )


def _stack_ragged_maps(values: list[np.ndarray], *, dtype: object) -> np.ndarray:
    if not values:
        return np.empty((0, 0, 0), dtype=dtype)
    n_freqs, n_times = values[0].shape
    out = np.zeros((len(values), n_freqs, n_times), dtype=dtype)
    for idx, value in enumerate(values):
        out[idx] = np.asarray(value, dtype=dtype)
    return out


def _stack_ragged_vectors(values: list[np.ndarray]) -> np.ndarray:
    max_len = max((np.asarray(item).size for item in values), default=0)
    out = np.full((len(values), max_len), np.nan, dtype=np.float64)
    for idx, value in enumerate(values):
        arr = np.asarray(value, dtype=np.float64).ravel()
        out[idx, : arr.size] = arr
    return out
