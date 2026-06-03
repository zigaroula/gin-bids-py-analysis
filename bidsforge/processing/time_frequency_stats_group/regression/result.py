from __future__ import annotations

from dataclasses import dataclass, field

from bidsforge.processing.utils.serialization import OutputTree

from ..result import (
    BaseTimeFrequencyStatsGroupResult,
    TFGroupEpochStats,
    TFGroupEstimatePair,
    TFGroupStats,
    TFIndexedConditionContributions,
)


@dataclass
class TFRegressionGroupStats:
    condition_contrast: TFGroupStats = field(default_factory=TFGroupStats)
    condition_a_vs_zero: TFGroupStats = field(default_factory=TFGroupStats)
    condition_b_vs_zero: TFGroupStats = field(default_factory=TFGroupStats)
    epoch_summary: TFGroupEpochStats = field(default_factory=TFGroupEpochStats)


@dataclass
class TimeFrequencyRegressionGroupResult(BaseTimeFrequencyStatsGroupResult):
    source_metric: TFGroupEstimatePair = field(default_factory=TFGroupEstimatePair)
    signal_activity: TFGroupEstimatePair = field(default_factory=TFGroupEstimatePair)
    r_value: TFGroupEstimatePair = field(default_factory=TFGroupEstimatePair)
    stats: TFRegressionGroupStats = field(default_factory=TFRegressionGroupStats)
    source_metric_contributions: TFIndexedConditionContributions = field(
        default_factory=TFIndexedConditionContributions
    )
    signal_activity_contributions: TFIndexedConditionContributions = field(
        default_factory=TFIndexedConditionContributions
    )
    primary_regression_metric: str = "t_values"
    contrast_mode: str = "paired"

    def to_output_tree(
        self,
        *,
        pipeline_name: str = "time_frequency_regression_group",
        pipeline_version: str = "unknown",
    ) -> OutputTree:
        label_a, label_b = self.condition_labels
        tree = self._base_tree(
            analysis_type="time_frequency_regression_group",
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
        )
        tree["data"] = {
            "regression": {
                "source_metric": self.source_metric.to_output_dict(
                    label_a=label_a,
                    label_b=label_b,
                ),
                "r_value": self.r_value.to_output_dict(label_a=label_a, label_b=label_b),
            },
            "signal_activity": self.signal_activity.to_output_dict(
                label_a=label_a,
                label_b=label_b,
            ),
        }
        tree["stats"] = {
            "regression": {
                "condition_contrast": self.stats.condition_contrast.to_output_dict(),
                f"{label_a}_vs_zero": self.stats.condition_a_vs_zero.to_output_dict(),
                f"{label_b}_vs_zero": self.stats.condition_b_vs_zero.to_output_dict(),
                "epoch_summary": self.stats.epoch_summary.to_output_dict(),
            }
        }
        tree["contributions"]["regression"] = {  # type: ignore[index]
            "source_metric": self.source_metric_contributions.to_output_dict(
                region_names=self.region_names,
                label_a=label_a,
                label_b=label_b,
            ),
        }
        tree["contributions"]["signal_activity"] = (  # type: ignore[index]
            self.signal_activity_contributions.to_output_dict(
                region_names=self.region_names,
                label_a=label_a,
                label_b=label_b,
            )
        )
        tree["meta"]["primary_regression_metric"] = str(self.primary_regression_metric)  # type: ignore[index]
        tree["meta"]["contrast_mode"] = str(self.contrast_mode)  # type: ignore[index]
        return tree
