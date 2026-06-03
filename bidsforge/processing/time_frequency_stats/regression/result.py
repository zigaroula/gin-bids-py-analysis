from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bidsforge.processing.utils.serialization import OutputTree, compressed

from ..result import BaseTimeFrequencyStatsResult


@dataclass
class TFConditionPredictorValues:
    raw_values: np.ndarray = field(default_factory=lambda: np.array([]))
    transformed_values: np.ndarray = field(default_factory=lambda: np.array([]))
    values: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class TFPredictorValues:
    condition_a: TFConditionPredictorValues = field(default_factory=TFConditionPredictorValues)
    condition_b: TFConditionPredictorValues = field(default_factory=TFConditionPredictorValues)


@dataclass
class TFConditionRegressionStats:
    slope: np.ndarray = field(default_factory=lambda: np.array([]))
    intercept: np.ndarray = field(default_factory=lambda: np.array([]))
    r_value: np.ndarray = field(default_factory=lambda: np.array([]))
    t_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_value: np.ndarray = field(default_factory=lambda: np.array([]))
    p_value_corrected: np.ndarray = field(default_factory=lambda: np.array([]))
    significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    n_trials_used: int = 0
    stats_valid: bool = False
    permuted_slopes: np.ndarray | None = None

    def to_output_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "slope": np.asarray(self.slope, dtype=np.float64),
            "intercept": np.asarray(self.intercept, dtype=np.float64),
            "r_value": np.asarray(self.r_value, dtype=np.float64),
            "t_values": np.asarray(self.t_values, dtype=np.float64),
            "p_value": np.asarray(self.p_value, dtype=np.float64),
            "p_value_corrected": np.asarray(self.p_value_corrected, dtype=np.float64),
            "significant_mask": np.asarray(self.significant_mask, dtype=bool),
            "n_trials_used": int(self.n_trials_used),
            "stats_valid": bool(self.stats_valid),
        }
        if self.permuted_slopes is not None:
            out["permuted_slopes"] = compressed(
                np.asarray(self.permuted_slopes, dtype=np.float32)
            )
        return out


@dataclass
class TFRegressionStats:
    condition_a: TFConditionRegressionStats = field(default_factory=TFConditionRegressionStats)
    condition_b: TFConditionRegressionStats = field(default_factory=TFConditionRegressionStats)


@dataclass
class TimeFrequencyRegressionResult(BaseTimeFrequencyStatsResult):
    """Structured outputs for TF regression subject statistics."""

    regression: TFRegressionStats = field(default_factory=TFRegressionStats)
    predictor_values: TFPredictorValues = field(default_factory=TFPredictorValues)
    predictor: str = "predictor_value"
    predictor_zscore: str = "none"
    predictor_transform_by_condition: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_output_tree(
        self,
        *,
        include_epochs: bool = False,
        pipeline_name: str = "time_frequency_regression",
        pipeline_version: str = "unknown",
    ) -> OutputTree:
        tree = self._base_output_tree(
            include_epochs=include_epochs,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
        )
        tree["stats"] = {
            "regression": {
                self.condition_a: self.regression.condition_a.to_output_dict(),
                self.condition_b: self.regression.condition_b.to_output_dict(),
            }
        }
        tree["predictor"] = {
            self.condition_a: {
                "raw_values": np.asarray(
                    self.predictor_values.condition_a.raw_values,
                    dtype=np.float64,
                ),
                "transformed_values": np.asarray(
                    self.predictor_values.condition_a.transformed_values,
                    dtype=np.float64,
                ),
                "values": np.asarray(
                    self.predictor_values.condition_a.values,
                    dtype=np.float64,
                ),
            },
            self.condition_b: {
                "raw_values": np.asarray(
                    self.predictor_values.condition_b.raw_values,
                    dtype=np.float64,
                ),
                "transformed_values": np.asarray(
                    self.predictor_values.condition_b.transformed_values,
                    dtype=np.float64,
                ),
                "values": np.asarray(
                    self.predictor_values.condition_b.values,
                    dtype=np.float64,
                ),
            },
        }
        tree["meta"]["analysis_type"] = "time_frequency_regression"  # type: ignore[index]
        tree["meta"]["predictor"] = str(self.predictor)  # type: ignore[index]
        tree["meta"]["predictor_zscore"] = str(self.predictor_zscore)  # type: ignore[index]
        return tree

    def _trial_table_extra_tree(self) -> dict[str, object]:
        return {
            "predictor_raw_value": np.asarray(
                [
                    _to_float_or_nan(trial.metadata.get("predictor_raw_value"))
                    for trial in self.resolved_trials
                ],
                dtype=np.float64,
            ),
            "predictor_transformed_value": np.asarray(
                [
                    _to_float_or_nan(trial.metadata.get("predictor_transformed_value"))
                    for trial in self.resolved_trials
                ],
                dtype=np.float64,
            ),
            "predictor_value": np.asarray(
                [
                    _to_float_or_nan(trial.metadata.get("predictor_value"))
                    for trial in self.resolved_trials
                ],
                dtype=np.float64,
            ),
        }


def _to_float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")

