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
    "RegressionMetricStats",
    "ScatterData",
    "VsZeroStatsPair",
]


@dataclass
class VsZeroStatsPair:
    """Per-condition one-sample tests against zero for the primary regression metric."""

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
class RegressionMetricStats:
    """Primary regression metric contrast plus epoch summary and vs-zero tests."""

    contrast: GroupTimecourseStats = field(default_factory=GroupTimecourseStats)
    epoch_summary: GroupEpochStats = field(default_factory=GroupEpochStats)
    vs_zero: VsZeroStatsPair = field(default_factory=VsZeroStatsPair)

    def to_output_dict(self) -> dict[str, object]:
        return {
            "condition_contrast": self.contrast.to_output_dict(
                epoch_summary=self.epoch_summary
            ),
            "condition_a_vs_zero": self.vs_zero.condition_to_output_dict(
                "condition_a"
            ),
            "condition_b_vs_zero": self.vs_zero.condition_to_output_dict(
                "condition_b"
            ),
        }


@dataclass
class ScatterData:
    """Per-ROI scatter arrays used by regression visualisation."""

    condition_a_predictor: list[np.ndarray] = field(default_factory=list)
    condition_a_signal_activity_summary: list[np.ndarray] = field(default_factory=list)
    condition_b_predictor: list[np.ndarray] = field(default_factory=list)
    condition_b_signal_activity_summary: list[np.ndarray] = field(default_factory=list)

    def to_output_dict(self, *, region_names: list[str]) -> dict[str, object]:
        out: dict[str, object] = {}
        for idx, roi_name in enumerate(region_names):
            out[roi_name] = {
                "condition_a": {
                    "predictor": np.asarray(self.condition_a_predictor[idx], dtype=np.float64),
                    "signal_activity_summary": np.asarray(
                        self.condition_a_signal_activity_summary[idx],
                        dtype=np.float64,
                    ),
                },
                "condition_b": {
                    "predictor": np.asarray(self.condition_b_predictor[idx], dtype=np.float64),
                    "signal_activity_summary": np.asarray(
                        self.condition_b_signal_activity_summary[idx],
                        dtype=np.float64,
                    ),
                },
            }
        return out


@dataclass
class RegressionGroupProcessingResult(BaseTrialStatsGroupProcessingResult):
    """Structured outputs for group-level ROI statistics on regression data."""

    regression_stats: RegressionMetricStats = field(default_factory=RegressionMetricStats)
    """Group-level statistics for the configured primary regression metric."""
    slope: GroupEstimatePair = field(default_factory=GroupEstimatePair)
    """Per-condition slope estimates."""
    r_value: GroupEstimatePair = field(default_factory=GroupEstimatePair)
    """Per-condition Pearson-r estimates."""
    slope_contributions: IndexedConditionContributions = field(
        default_factory=IndexedConditionContributions
    )
    """Per-ROI slope contribution matrices."""
    r_value_contributions: IndexedConditionContributions = field(
        default_factory=IndexedConditionContributions
    )
    """Per-ROI Pearson-r contribution matrices."""
    scatter: ScatterData = field(default_factory=ScatterData)
    """Per-ROI scatter arrays."""

    # --- Regression-specific configuration ---
    contrast_mode: str = "paired"
    """How conditions are contrasted: 'paired' (within-subject) or 'independent' (between-subject)."""
    primary_regression_metric: str = "slope"
    """Primary regression metric used by stats/regression."""

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
                    "regression": self.regression_stats.to_output_dict(),
                },
                "data": {
                    "regression": {
                        "condition_a": {
                            "slope": self.slope.condition_a.to_output_dict(),
                            "r_value": self.r_value.condition_a.to_output_dict(),
                        },
                        "condition_b": {
                            "slope": self.slope.condition_b.to_output_dict(),
                            "r_value": self.r_value.condition_b.to_output_dict(),
                        },
                    },
                },
                "meta": {
                    "analysis_type": "regression_group",
                    "contrast_mode": str(self.contrast_mode),
                    "primary_regression_metric": str(self.primary_regression_metric),
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
                    "regression": {
                        "slope": self.slope_contributions.to_output_dict(
                            region_names=self.region_names,
                        ),
                        "r_value": self.r_value_contributions.to_output_dict(
                            region_names=self.region_names,
                        ),
                    },
                },
            },
        )
        if self.scatter.condition_a_predictor:
            tree["scatter"] = self.scatter.to_output_dict(
                region_names=self.region_names,
            )
        if self.cluster_p_values is not None:
            tree["stats"]["regression"]["condition_contrast"]["cluster"] = _cluster_stats_tree(
                p_values=self.cluster_p_values,
                windows_s=self.cluster_windows_s,
                null_distributions=self.cluster_null_distributions,
            )
        return tree
