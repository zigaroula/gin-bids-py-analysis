from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gin_bids_py_analysis.processing.utils.serialization import OutputTree

from ..result import (
    BaseTrialStatsGroupProcessingResult,
    GroupEstimate,
    _cluster_stats_tree,
    _deep_merge,
)

__all__ = ["ConditionTestEpochSummary", "ConditionTestGroupProcessingResult"]


@dataclass
class ConditionTestEpochSummary:
    """Epoch-level condition-difference summary plus its one-sample test."""

    t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    df: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_difference: GroupEstimate = field(default_factory=GroupEstimate)

    def to_output_dict(self) -> dict[str, object]:
        return {
            "t_values": np.asarray(self.t_values, dtype=np.float64),
            "p_values": np.asarray(self.p_values, dtype=np.float64),
            "df": np.asarray(self.df, dtype=np.float64),
            "condition_difference_mean": np.asarray(
                self.condition_difference.mean,
                dtype=np.float64,
            ),
            "condition_difference_sem": np.asarray(self.condition_difference.sem, dtype=np.float64),
        }


@dataclass
class ConditionTestGroupProcessingResult(BaseTrialStatsGroupProcessingResult):
    """Structured outputs for group-level condition-test ROI statistics."""

    condition_difference: GroupEstimate = field(default_factory=GroupEstimate)
    """Mean/SEM of the configured condition metric across channels/subjects."""
    summary_epoch: ConditionTestEpochSummary = field(
        default_factory=ConditionTestEpochSummary
    )
    """Epoch-level summary for the configured source metric."""
    cluster_p_values: np.ndarray | None = None
    cluster_windows_s: list[list[tuple[float, float]]] | None = None
    cluster_null_distributions: list[np.ndarray] | None = None
    primary_condition_metric: str = "mean_difference"

    def to_output_tree(
        self,
        *,
        pipeline_name: str = "condition_test_group",
        pipeline_version: str = "unknown",
    ) -> OutputTree:
        """Return the output-shaped tree for group-level condition-test results."""
        tree = super().to_output_tree(
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
        )
        _deep_merge(
            tree,
            {
                "data": {
                    "condition_difference": self.condition_difference.to_output_dict(),
                    "summary_epoch": self.summary_epoch.to_output_dict(),
                },
                "meta": {
                    "analysis_type": "condition_test_group",
                    "primary_condition_metric": str(self.primary_condition_metric),
                    "n_group_permutations": int(
                        self.metadata.get("n_group_permutations", 0)
                    ),
                    "cluster_threshold_alpha": float(
                        self.metadata.get("cluster_threshold_alpha", 0.05)
                    ),
                },
            },
        )
        if self.cluster_p_values is not None:
            tree["stats"]["signal_activity"]["cluster"] = _cluster_stats_tree(
                p_values=self.cluster_p_values,
                windows_s=self.cluster_windows_s,
                null_distributions=self.cluster_null_distributions,
            )
        return tree
