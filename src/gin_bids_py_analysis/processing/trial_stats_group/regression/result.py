from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gin_bids_py_analysis.processing.utils.serialization import OutputTree

from ..result import (
    BaseTrialStatsGroupProcessingResult,
    GroupEpochStats,
    GroupEstimatePair,
    GroupTimecourseStats,
    IndexedConditionContributions,
    _cluster_stats_tree,
    _deep_merge,
)

__all__ = [
    "RegressionGroupProcessingResult",
    "RegressionSourceMetricStats",
    "ScatterData",
    "VsZeroStatsPair",
]


@dataclass
class VsZeroStatsPair:
    """Per-condition one-sample tests against zero for the source metric."""

    condition_a: GroupTimecourseStats = field(default_factory=GroupTimecourseStats)
    condition_b: GroupTimecourseStats = field(default_factory=GroupTimecourseStats)
    condition_a_cluster_p_values: np.ndarray | None = None
    condition_a_cluster_windows_s: list[list[tuple[float, float]]] | None = None
    condition_a_cluster_null_distributions: list[np.ndarray] | None = None
    condition_b_cluster_p_values: np.ndarray | None = None
    condition_b_cluster_windows_s: list[list[tuple[float, float]]] | None = None
    condition_b_cluster_null_distributions: list[np.ndarray] | None = None

    def condition_to_output_dict(self, condition: str) -> dict[str, object]:
        stats = self.condition_a if condition == "condition_a" else self.condition_b
        out = stats.to_output_dict()
        cluster_p_values = getattr(self, f"{condition}_cluster_p_values")
        if cluster_p_values is not None:
            out["cluster"] = _cluster_stats_tree(
                p_values=cluster_p_values,
                windows_s=getattr(self, f"{condition}_cluster_windows_s"),
                null_distributions=getattr(
                    self,
                    f"{condition}_cluster_null_distributions",
                ),
            )
        return out


@dataclass
class RegressionSourceMetricStats:
    """Source-metric contrast plus epoch summary and vs-zero tests."""

    contrast: GroupTimecourseStats = field(default_factory=GroupTimecourseStats)
    epoch_summary: GroupEpochStats = field(default_factory=GroupEpochStats)
    vs_zero: VsZeroStatsPair = field(default_factory=VsZeroStatsPair)

    def to_output_dict(self) -> dict[str, object]:
        out = self.contrast.to_output_dict(epoch_summary=self.epoch_summary)
        out["condition_a_vs_zero"] = self.vs_zero.condition_to_output_dict(
            "condition_a"
        )
        out["condition_b_vs_zero"] = self.vs_zero.condition_to_output_dict(
            "condition_b"
        )
        return out


@dataclass
class ScatterData:
    """Per-ROI scatter arrays used by regression visualisation."""

    condition_a_predictor: list[np.ndarray] = field(default_factory=list)
    condition_a_activity: list[np.ndarray] = field(default_factory=list)
    condition_b_predictor: list[np.ndarray] = field(default_factory=list)
    condition_b_activity: list[np.ndarray] = field(default_factory=list)

    def to_output_dict(self, *, region_names: list[str]) -> dict[str, object]:
        out: dict[str, object] = {
            "region_names": np.array(region_names, dtype=object),
        }
        for idx, roi_name in enumerate(region_names):
            out[str(idx)] = {
                "region": str(roi_name),
                "condition_a_predictor": np.asarray(
                    self.condition_a_predictor[idx],
                    dtype=np.float64,
                ),
                "condition_a_activity": np.asarray(
                    self.condition_a_activity[idx],
                    dtype=np.float64,
                ),
                "condition_b_predictor": np.asarray(
                    self.condition_b_predictor[idx],
                    dtype=np.float64,
                ),
                "condition_b_activity": np.asarray(
                    self.condition_b_activity[idx],
                    dtype=np.float64,
                ),
            }
        return out


@dataclass
class RegressionGroupProcessingResult(BaseTrialStatsGroupProcessingResult):
    """Structured outputs for group-level ROI statistics on regression data."""

    source_metric_stats: RegressionSourceMetricStats = field(
        default_factory=RegressionSourceMetricStats
    )
    """Group-level statistics for the configured regression source metric."""
    source_metric_data: GroupEstimatePair = field(default_factory=GroupEstimatePair)
    """Per-condition source-metric estimates."""
    r_values: GroupEstimatePair = field(default_factory=GroupEstimatePair)
    """Per-condition Pearson-r estimates."""
    source_metric_contributions: IndexedConditionContributions = field(
        default_factory=IndexedConditionContributions
    )
    """Per-ROI source-metric contribution matrices."""
    scatter: ScatterData = field(default_factory=ScatterData)
    """Per-ROI scatter arrays."""

    # --- Regression-specific configuration ---
    contrast_mode: str = "paired"
    """How conditions are contrasted: 'paired' (within-subject) or 'independent' (between-subject)."""

    # --- Cluster permutation stats (only populated when p_value_correction_method='cluster_permutation') ---
    cluster_p_values: np.ndarray | None = None
    """Cluster-based permutation p-values (best cluster), one per ROI, shape (n_rois,). None when not computed."""
    cluster_windows_s: list[list[tuple[float, float]]] | None = None
    """Significant cluster windows per ROI for the slope contrast. Each inner list holds (start_s, end_s)
    tuples for up to n_clusters_to_keep clusters that passed p < significance_alpha."""
    cluster_null_distributions: list[np.ndarray] | None = None
    """Per-ROI null distributions of max-cluster t-sum statistics. Each array has shape (n_permutations,)."""
    manual_roi_missing_channels: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    """Manual ROI channel assignments absent from the available subject result files."""

    def to_output_tree(
        self,
        *,
        pipeline_name: str = "regression_group",
        pipeline_version: str = "unknown",
    ) -> OutputTree:
        """Return the output-shaped tree for group-level regression results."""
        tree = super().to_output_tree(
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
        )
        _deep_merge(
            tree,
            {
                "stats": {
                    "source_metric": self.source_metric_stats.to_output_dict(),
                },
                "data": {
                    "source_metric": self.source_metric_data.to_output_dict(),
                    "r_values": self.r_values.to_output_dict(),
                },
                "meta": {
                    "analysis_type": "regression_group",
                    "contrast_mode": str(self.contrast_mode),
                    **{
                        key: str(value)
                        for key in (
                            "predictor",
                            "predictor_zscore",
                            "predictor_transform_by_condition_json",
                            "trial_activity_summary_kind",
                            "trial_activity_summary_missing_response_policy",
                            "trial_activity_summary_source_json",
                            "trial_activity_summary_label",
                            "scatter_aggregation",
                        )
                        if (value := self.metadata.get(key)) is not None
                    },
                },
                "contributions": {
                    "source_metric": self.source_metric_contributions.to_output_dict(
                        region_names=self.region_names
                    ),
                },
            },
        )
        if self.scatter.condition_a_predictor:
            tree["scatter_data"] = self.scatter.to_output_dict(
                region_names=self.region_names,
            )
        if self.cluster_p_values is not None:
            tree["stats"]["cluster"] = _cluster_stats_tree(
                p_values=self.cluster_p_values,
                windows_s=self.cluster_windows_s,
                null_distributions=self.cluster_null_distributions,
            )
        return tree

    # Temporary read-only compatibility aliases for existing visualization/tests.
    # Writers, loaders, and processors use source_metric_stats/data objects.
    @property
    def source_metric_t_values(self) -> np.ndarray:
        return self.source_metric_stats.contrast.t_values

    @property
    def source_metric_p_values(self) -> np.ndarray:
        return self.source_metric_stats.contrast.p_values

    @property
    def source_metric_p_values_uncorrected(self) -> np.ndarray:
        return self.source_metric_stats.contrast.p_values_uncorrected

    @property
    def source_metric_significant_mask(self) -> np.ndarray:
        return self.source_metric_stats.contrast.significant_mask

    @property
    def condition_a_source_metric_mean(self) -> np.ndarray:
        return self.source_metric_data.condition_a.mean

    @property
    def condition_a_source_metric_sem(self) -> np.ndarray:
        return self.source_metric_data.condition_a.sem

    @property
    def condition_b_source_metric_mean(self) -> np.ndarray:
        return self.source_metric_data.condition_b.mean

    @property
    def condition_b_source_metric_sem(self) -> np.ndarray:
        return self.source_metric_data.condition_b.sem

    @property
    def condition_a_r_value_mean(self) -> np.ndarray:
        return self.r_values.condition_a.mean

    @property
    def condition_a_r_value_sem(self) -> np.ndarray:
        return self.r_values.condition_a.sem

    @property
    def condition_b_r_value_mean(self) -> np.ndarray:
        return self.r_values.condition_b.mean

    @property
    def condition_b_r_value_sem(self) -> np.ndarray:
        return self.r_values.condition_b.sem

    @property
    def epoch_source_metric_t(self) -> np.ndarray:
        return self.source_metric_stats.epoch_summary.t

    @property
    def epoch_source_metric_p(self) -> np.ndarray:
        return self.source_metric_stats.epoch_summary.p

    @property
    def epoch_source_metric_df(self) -> np.ndarray:
        return self.source_metric_stats.epoch_summary.df

    @property
    def condition_a_source_metric_contributions(self) -> list[np.ndarray]:
        return self.source_metric_contributions.condition_a

    @property
    def condition_b_source_metric_contributions(self) -> list[np.ndarray]:
        return self.source_metric_contributions.condition_b

    @property
    def condition_a_scatter_predictor(self) -> list[np.ndarray]:
        return self.scatter.condition_a_predictor

    @property
    def condition_a_scatter_activity(self) -> list[np.ndarray]:
        return self.scatter.condition_a_activity

    @property
    def condition_b_scatter_predictor(self) -> list[np.ndarray]:
        return self.scatter.condition_b_predictor

    @property
    def condition_b_scatter_activity(self) -> list[np.ndarray]:
        return self.scatter.condition_b_activity

    @property
    def condition_a_source_metric_vs_zero_t_values(self) -> np.ndarray:
        return self.source_metric_stats.vs_zero.condition_a.t_values

    @property
    def condition_a_source_metric_vs_zero_p_values_uncorrected(self) -> np.ndarray:
        return self.source_metric_stats.vs_zero.condition_a.p_values_uncorrected

    @property
    def condition_a_source_metric_vs_zero_p_values(self) -> np.ndarray:
        return self.source_metric_stats.vs_zero.condition_a.p_values

    @property
    def condition_a_source_metric_vs_zero_significant_mask(self) -> np.ndarray:
        return self.source_metric_stats.vs_zero.condition_a.significant_mask

    @property
    def condition_a_source_metric_vs_zero_cluster_p_values(self) -> np.ndarray | None:
        return self.source_metric_stats.vs_zero.condition_a_cluster_p_values

    @property
    def condition_a_source_metric_vs_zero_cluster_windows_s(
        self,
    ) -> list[list[tuple[float, float]]] | None:
        return self.source_metric_stats.vs_zero.condition_a_cluster_windows_s

    @property
    def condition_a_source_metric_vs_zero_cluster_null_distributions(
        self,
    ) -> list[np.ndarray] | None:
        return self.source_metric_stats.vs_zero.condition_a_cluster_null_distributions

    @property
    def condition_b_source_metric_vs_zero_t_values(self) -> np.ndarray:
        return self.source_metric_stats.vs_zero.condition_b.t_values

    @property
    def condition_b_source_metric_vs_zero_p_values_uncorrected(self) -> np.ndarray:
        return self.source_metric_stats.vs_zero.condition_b.p_values_uncorrected

    @property
    def condition_b_source_metric_vs_zero_p_values(self) -> np.ndarray:
        return self.source_metric_stats.vs_zero.condition_b.p_values

    @property
    def condition_b_source_metric_vs_zero_significant_mask(self) -> np.ndarray:
        return self.source_metric_stats.vs_zero.condition_b.significant_mask

    @property
    def condition_b_source_metric_vs_zero_cluster_p_values(self) -> np.ndarray | None:
        return self.source_metric_stats.vs_zero.condition_b_cluster_p_values

    @property
    def condition_b_source_metric_vs_zero_cluster_windows_s(
        self,
    ) -> list[list[tuple[float, float]]] | None:
        return self.source_metric_stats.vs_zero.condition_b_cluster_windows_s

    @property
    def condition_b_source_metric_vs_zero_cluster_null_distributions(
        self,
    ) -> list[np.ndarray] | None:
        return self.source_metric_stats.vs_zero.condition_b_cluster_null_distributions
