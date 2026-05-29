from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bidsforge.processing.base import BaseProcessingResult
from bidsforge.processing.utils.group_stats import ROIChannelContribution
from bidsforge.processing.utils.serialization import OutputTree

__all__ = [
    "BaseTrialStatsGroupProcessingResult",
    "GroupEstimate",
    "GroupEstimatePair",
    "GroupEpochStats",
    "GroupTimecourseStats",
    "IndexedConditionContributions",
    "ROIChannelContribution",
    "_cluster_stats_tree",
    "_deep_merge",
]


def _deep_merge(dest: dict[str, object], src: dict[str, object]) -> dict[str, object]:
    """Recursively merge *src* into *dest* in-place; *src* wins on conflicts."""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dest.get(key), dict):
            _deep_merge(dest[key], value)  # type: ignore[arg-type]
        else:
            dest[key] = value
    return dest


@dataclass
class GroupEstimate:
    """Mean and uncertainty for one group-level quantity."""

    mean: np.ndarray = field(default_factory=lambda: np.array([]))
    sem: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_output_dict(self) -> dict[str, object]:
        return {
            "mean": np.asarray(self.mean, dtype=np.float64),
            "sem": np.asarray(self.sem, dtype=np.float64),
        }


@dataclass
class GroupEstimatePair:
    """Per-condition group-level estimates."""

    condition_a: GroupEstimate = field(default_factory=GroupEstimate)
    condition_b: GroupEstimate = field(default_factory=GroupEstimate)

    def to_output_dict(
        self,
        *,
        label_a: str = "condition_a",
        label_b: str = "condition_b",
    ) -> dict[str, object]:
        return {
            label_a: self.condition_a.to_output_dict(),
            label_b: self.condition_b.to_output_dict(),
        }


@dataclass
class GroupEpochStats:
    """Scalar epoch-summary statistics per ROI."""

    t: np.ndarray = field(default_factory=lambda: np.array([]))
    p: np.ndarray = field(default_factory=lambda: np.array([]))
    df: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_output_dict(self) -> dict[str, object]:
        return {
            "t": np.asarray(self.t, dtype=np.float64),
            "p": np.asarray(self.p, dtype=np.float64),
            "df": np.asarray(self.df, dtype=np.float64),
        }


@dataclass
class GroupTimecourseStats:
    """Time-resolved t-test statistics per ROI."""

    t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values_uncorrected: np.ndarray = field(default_factory=lambda: np.array([]))
    significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_output_dict(
        self,
        *,
        epoch_summary: GroupEpochStats | None = None,
    ) -> dict[str, object]:
        out: dict[str, object] = {
            "t_values": np.asarray(self.t_values, dtype=np.float64),
            "p_values": np.asarray(self.p_values, dtype=np.float64),
            "p_values_uncorrected": np.asarray(
                self.p_values_uncorrected,
                dtype=np.float64,
            ),
            "significant_mask": np.asarray(self.significant_mask, dtype=bool),
        }
        if epoch_summary is not None:
            out["epoch_summary"] = epoch_summary.to_output_dict()
        return out


@dataclass
class IndexedConditionContributions:
    """Per-ROI ragged contribution matrices for both conditions."""

    condition_a: list[np.ndarray] = field(default_factory=list)
    condition_b: list[np.ndarray] = field(default_factory=list)
    labels: list[list[str]] = field(default_factory=list)

    def to_output_dict(
        self,
        *,
        region_names: list[str],
        label_a: str = "condition_a",
        label_b: str = "condition_b",
    ) -> dict[str, object]:
        entries: dict[str, object] = {}
        for idx, roi_name in enumerate(region_names):
            if idx >= len(self.condition_a) or idx >= len(self.condition_b):
                continue
            entries[roi_name] = {
                label_a: np.asarray(self.condition_a[idx], dtype=np.float64),
                label_b: np.asarray(self.condition_b[idx], dtype=np.float64),
                "labels": np.array(
                    self.labels[idx] if idx < len(self.labels) else [],
                    dtype=object,
                ),
            }
        return entries

    def condition_to_output_dict(
        self,
        condition: str,
        *,
        region_names: list[str],
    ) -> dict[str, object]:
        values = self.condition_a if condition == "condition_a" else self.condition_b
        entries: dict[str, object] = {}
        for idx, roi_name in enumerate(region_names):
            if idx >= len(values):
                continue
            entries[roi_name] = {
                condition: np.asarray(values[idx], dtype=np.float64),
                "labels": np.array(
                    self.labels[idx] if idx < len(self.labels) else [],
                    dtype=object,
                ),
            }
        return entries


def _cluster_stats_tree(
    *,
    p_values: np.ndarray,
    windows_s: list[list[tuple[float, float]]] | None,
    null_distributions: list[np.ndarray] | None,
) -> dict[str, object]:
    from bidsforge.processing.utils.serialization import compressed

    n_rois = len(p_values)
    windows_list = windows_s or []
    n_max_clusters = max((len(w) for w in windows_list), default=0)
    starts_2d = np.full((n_rois, n_max_clusters), np.nan, dtype=np.float64)
    ends_2d = np.full((n_rois, n_max_clusters), np.nan, dtype=np.float64)
    for i, roi_windows in enumerate(windows_list):
        for j, (t_start, t_end) in enumerate(roi_windows):
            starts_2d[i, j] = t_start
            ends_2d[i, j] = t_end

    null_dists = null_distributions or []
    max_len = max((len(nd) for nd in null_dists), default=0)
    null_matrix = np.full((len(null_dists), max_len), np.nan, dtype=np.float64)
    for i, nd in enumerate(null_dists):
        if len(nd) > 0:
            null_matrix[i, : len(nd)] = np.asarray(nd, dtype=np.float64)
    return {
        "p_values": np.asarray(p_values, dtype=np.float64),
        "cluster_starts_s": starts_2d,
        "cluster_ends_s": ends_2d,
        "null_distributions": compressed(null_matrix),
    }


@dataclass
class BaseTrialStatsGroupProcessingResult(BaseProcessingResult):
    """Shared result surface for group-level trial-statistics pipelines."""

    # --- Axes ---
    time_axis_s: np.ndarray = field(default_factory=lambda: np.array([]))
    """Time axis for the epoch window, shape (n_times,), in seconds."""
    region_names: list[str] = field(default_factory=list)
    """Ordered list of ROI names corresponding to the first axis of all 2-D arrays."""
    condition_labels: tuple[str, str] = ("condition_a", "condition_b")
    """Human-readable labels for condition A and condition B."""

    # --- ROI composition ---
    roi_channel_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    """Number of contributing channels per ROI, shape (n_rois,), dtype int64."""
    roi_subject_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    """Number of contributing subjects per ROI, shape (n_rois,), dtype int64."""
    contributions: list[ROIChannelContribution] = field(default_factory=list)
    """Flat list of every (roi, subject, channel) triple that contributed to the group stats."""

    # --- Output-shaped signal-activity data and stats ---
    signal_activity: GroupEstimatePair = field(default_factory=GroupEstimatePair)
    """Per-condition signal-activity means and SEMs."""
    signal_activity_stats: GroupTimecourseStats = field(default_factory=GroupTimecourseStats)
    """Group-level signal-activity contrast statistics."""
    signal_activity_epoch: GroupEpochStats = field(default_factory=GroupEpochStats)
    """Epoch-level signal-activity contrast summary."""
    signal_activity_contributions: IndexedConditionContributions = field(
        default_factory=IndexedConditionContributions
    )
    """Per-ROI signal-activity contribution matrices."""

    # --- Configuration and provenance ---
    p_value_correction_method: str = "none"
    """Multiple-comparison correction applied to p-values ('none', 'fdr_bh', 'bonferroni', 'cluster_permutation')."""
    significance_alpha: float = 0.05
    """Significance threshold used to build the significant_mask fields."""
    roi_mode: str = "manual"
    """How ROIs were defined: 'manual' (explicit channel lists) or 'atlas' (electrode TSV column)."""
    atlas_name: str | None = None
    """Name of the atlas column in _electrodes.tsv used when roi_mode='atlas'; None otherwise."""
    source_subject_stats_files: list[str] = field(default_factory=list)
    """Absolute paths of the subject-level stats files that were aggregated."""
    source_electrodes_files: list[str] = field(default_factory=list)
    """Absolute paths of the _electrodes.tsv files consulted for atlas or channel metadata."""
    excluded_rois: dict[str, str] = field(default_factory=dict)
    """ROIs that were dropped before statistics, keyed by ROI name; value is the exclusion reason."""

    def to_output_tree(
        self,
        *,
        pipeline_name: str = "unknown",
        pipeline_version: str = "unknown",
    ) -> OutputTree:
        """Return the common output tree for group-level trial statistics."""
        label_a, label_b = self.condition_labels
        tree: dict[str, object] = {
            "axes": {
                "region": np.array(self.region_names, dtype=object),
                "time_s": np.asarray(self.time_axis_s, dtype=np.float64),
            },
            "data": {
                "signal_activity": self.signal_activity.to_output_dict(
                    label_a=label_a,
                    label_b=label_b,
                ),
            },
            "stats": {
                "signal_activity": self.signal_activity_stats.to_output_dict(
                    epoch_summary=self.signal_activity_epoch
                ),
            },
            "meta": self._build_meta_tree(),
            "excluded_rois": {
                "region": np.array(list(self.excluded_rois.keys()), dtype=object),
                "reason": np.array(list(self.excluded_rois.values()), dtype=object),
            },
            "contributions": {
                "summary": {
                    "region": np.array(
                        [item.roi for item in self.contributions],
                        dtype=object,
                    ),
                    "subject": np.array(
                        [item.subject for item in self.contributions],
                        dtype=object,
                    ),
                    "channel": np.array(
                        [item.channel for item in self.contributions],
                        dtype=object,
                    ),
                    "source_stats_file": np.array(
                        [item.source_stats_file for item in self.contributions],
                        dtype=object,
                    ),
                },
                "signal_activity": self.signal_activity_contributions.to_output_dict(
                    region_names=self.region_names,
                    label_a=label_a,
                    label_b=label_b,
                ),
            },
            "provenance": {
                "source_subject_stats_files": np.array(
                    self.source_subject_stats_files,
                    dtype=object,
                ),
                "source_electrodes_files": np.array(
                    self.source_electrodes_files,
                    dtype=object,
                ),
                "pipeline_name": pipeline_name,
                "pipeline_version": pipeline_version,
            },
        }
        return tree

    def _build_meta_tree(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_name": "trial_stats_group",
            "schema_version": "1.0",
            "analysis_level": "roi_group",
            "condition_labels": np.array(list(self.condition_labels), dtype=object),
            "p_value_correction_method": str(self.p_value_correction_method),
            "significance_alpha": float(self.significance_alpha),
            "roi_mode": str(self.roi_mode),
            "atlas_name": str(self.atlas_name or ""),
            "roi_channel_counts": np.asarray(self.roi_channel_counts, dtype=np.int64),
            "roi_subject_counts": np.asarray(self.roi_subject_counts, dtype=np.int64),
            "included_roi_count": int(len(self.region_names)),
            "excluded_roi_count": int(len(self.excluded_rois)),
        }
        for key in (
            "binning_mode",
            "window_ms",
            "n_bins",
            "effective_n_bins",
            "activity_zscore",
            "activity_baseline_tmin_s",
            "activity_baseline_tmax_s",
        ):
            value = self.metadata.get(key)
            if value is not None:
                out[key] = str(value) if isinstance(value, str) else value
        return out

