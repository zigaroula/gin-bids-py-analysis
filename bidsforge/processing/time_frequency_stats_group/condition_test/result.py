from __future__ import annotations

from dataclasses import dataclass, field

from bidsforge.processing.utils.serialization import OutputTree

from ..result import (
    BaseTimeFrequencyStatsGroupResult,
    TFGroupEpochStats,
    TFGroupEstimate,
    TFGroupEstimatePair,
    TFGroupStats,
    TFIndexedConditionContributions,
    TFIndexedContributions,
)


@dataclass
class TimeFrequencyConditionTestGroupResult(BaseTimeFrequencyStatsGroupResult):
    source_metric: TFGroupEstimate = field(default_factory=TFGroupEstimate)
    source_metric_stats: TFGroupStats = field(default_factory=TFGroupStats)
    source_metric_epoch: TFGroupEpochStats = field(default_factory=TFGroupEpochStats)
    signal_activity: TFGroupEstimatePair = field(default_factory=TFGroupEstimatePair)
    source_metric_contributions: TFIndexedContributions = field(default_factory=TFIndexedContributions)
    signal_activity_contributions: TFIndexedConditionContributions = field(
        default_factory=TFIndexedConditionContributions
    )
    primary_condition_metric: str = "t_values"

    def to_output_tree(
        self,
        *,
        pipeline_name: str = "time_frequency_condition_test_group",
        pipeline_version: str = "unknown",
    ) -> OutputTree:
        label_a, label_b = self.condition_labels
        tree = self._base_tree(
            analysis_type="time_frequency_condition_test_group",
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
        )
        tree["data"] = {
            "source_metric": self.source_metric.to_output_dict(),
            "signal_activity": self.signal_activity.to_output_dict(
                label_a=label_a,
                label_b=label_b,
            ),
        }
        tree["stats"] = {
            "source_metric": self.source_metric_stats.to_output_dict(),
            "source_metric_epoch": self.source_metric_epoch.to_output_dict(),
        }
        tree["contributions"]["source_metric"] = self.source_metric_contributions.to_output_dict(  # type: ignore[index]
            region_names=self.region_names,
            key=self.primary_condition_metric,
        )
        tree["contributions"]["signal_activity"] = (  # type: ignore[index]
            self.signal_activity_contributions.to_output_dict(
                region_names=self.region_names,
                label_a=label_a,
                label_b=label_b,
            )
        )
        tree["meta"]["primary_condition_metric"] = str(self.primary_condition_metric)  # type: ignore[index]
        return tree
