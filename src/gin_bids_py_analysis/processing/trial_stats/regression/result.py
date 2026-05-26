from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from gin_bids_py_analysis.processing.utils.serialization import OutputTree, compressed

from ..result import BaseTrialStatsProcessingResult, _deep_merge, _to_float_or_nan


@dataclass
class ConditionRegressionStats:
    """Slope regression statistics for one experimental condition."""

    slope: np.ndarray = field(default_factory=lambda: np.array([]))
    intercept: np.ndarray = field(default_factory=lambda: np.array([]))
    r_value: np.ndarray = field(default_factory=lambda: np.array([]))
    p_value: np.ndarray = field(default_factory=lambda: np.array([]))
    p_value_corrected: np.ndarray = field(default_factory=lambda: np.array([]))
    significant_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    n_trials_used: int = 0
    stats_valid: bool = False
    permuted_slopes: np.ndarray | None = None
    """Permuted slope maps, shape ``(n_perm, n_channels, n_times)``, float32.
    None when ``n_permutations == 0`` or ``stats_valid`` is False."""

    def to_output_dict(self) -> dict[str, object]:
        """Return a serializable dict for this condition's regression stats."""
        out: dict[str, object] = {
            "slope": self.slope.astype(np.float64),
            "intercept": self.intercept.astype(np.float64),
            "r_value": self.r_value.astype(np.float64),
            "p_value": self.p_value.astype(np.float64),
            "p_value_corrected": self.p_value_corrected.astype(np.float64),
            "significant_mask": self.significant_mask.astype(bool),
            "n_trials_used": int(self.n_trials_used),
            "stats_valid": bool(self.stats_valid),
        }
        if self.permuted_slopes is not None:
            out["permuted_slopes"] = compressed(self.permuted_slopes.astype(np.float32))
        return out


@dataclass
class RegressionStats:
    """Regression statistics for both experimental conditions."""

    condition_a: ConditionRegressionStats = field(default_factory=ConditionRegressionStats)
    condition_b: ConditionRegressionStats = field(default_factory=ConditionRegressionStats)


@dataclass
class ConditionPredictorValues:
    """Per-trial predictor values for one experimental condition."""

    raw_values: np.ndarray = field(default_factory=lambda: np.array([]))
    transformed_values: np.ndarray = field(default_factory=lambda: np.array([]))
    values: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_output_dict(self) -> dict[str, object]:
        """Return a serializable dict for this condition's predictor values."""
        return {
            "raw_values": np.asarray(self.raw_values, dtype=np.float64),
            "transformed_values": np.asarray(self.transformed_values, dtype=np.float64),
            "values": np.asarray(self.values, dtype=np.float64),
        }


@dataclass
class RegressionPredictor:
    """Predictor values for both experimental conditions."""

    condition_a: ConditionPredictorValues = field(default_factory=ConditionPredictorValues)
    condition_b: ConditionPredictorValues = field(default_factory=ConditionPredictorValues)


@dataclass
class RegressionProcessingResult(BaseTrialStatsProcessingResult):
    """Structured outputs for subject-level trial slope regression statistics."""

    regression: RegressionStats = field(default_factory=RegressionStats)
    predictor_values: RegressionPredictor = field(default_factory=RegressionPredictor)
    analysis_type: str = "slope_regression"
    excluded_channels: dict[str, str] = field(default_factory=dict)
    excluded_trial_channel_pairs: dict[str, list[int]] = field(default_factory=dict)
    predictor: str = "predictor_value"
    predictor_zscore: str = "none"
    predictor_transform_by_condition: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_output_tree(
        self,
        *,
        include_epochs: bool = False,
        pipeline_name: str = "unknown",
        pipeline_version: str = "unknown",
    ) -> OutputTree:
        """Return an output-shaped tree for this result."""
        tree = super().to_output_tree(
            include_epochs=include_epochs,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
        )
        _deep_merge(
            tree,
            {
                "stats": {
                    "regression": {
                        self.condition_a: self.regression.condition_a.to_output_dict(),
                        self.condition_b: self.regression.condition_b.to_output_dict(),
                    }
                },
                "predictor": {
                    self.condition_a: self.predictor_values.condition_a.to_output_dict(),
                    self.condition_b: self.predictor_values.condition_b.to_output_dict(),
                },
                "meta": {
                    "analysis_type": "slope_regression",
                    "predictor": str(self.predictor),
                    "predictor_zscore": str(self.predictor_zscore),
                    "predictor_transform_by_condition_json": json.dumps(
                        self.predictor_transform_by_condition,
                        sort_keys=True,
                        ensure_ascii=True,
                    ),
                    f"{self.condition_a}_stats_valid": bool(self.regression.condition_a.stats_valid),
                    f"{self.condition_b}_stats_valid": bool(self.regression.condition_b.stats_valid),
                },
            },
        )
        return tree

    def _build_trial_table_extra_tree(self) -> dict[str, object]:
        """Add predictor trial columns to the trial table."""
        return {
            "predictor_raw": np.array(
                [
                    str(trial.metadata.get("predictor_raw", ""))
                    for trial in self.resolved_trials
                ],
                dtype=object,
            ),
            "predictor_value": np.array(
                [
                    _to_float_or_nan(trial.metadata.get("predictor_value"))
                    for trial in self.resolved_trials
                ],
                dtype=np.float64,
            ),
            "predictor_raw_value": np.array(
                [
                    _to_float_or_nan(trial.metadata.get("predictor_raw_value"))
                    for trial in self.resolved_trials
                ],
                dtype=np.float64,
            ),
            "predictor_transformed_value": np.array(
                [
                    _to_float_or_nan(trial.metadata.get("predictor_transformed_value"))
                    for trial in self.resolved_trials
                ],
                dtype=np.float64,
            ),
            "predictor_transform_scale": np.array(
                [
                    _to_float_or_nan(trial.metadata.get("predictor_transform_scale"))
                    for trial in self.resolved_trials
                ],
                dtype=np.float64,
            ),
            "predictor_transform_offset": np.array(
                [
                    _to_float_or_nan(trial.metadata.get("predictor_transform_offset"))
                    for trial in self.resolved_trials
                ],
                dtype=np.float64,
            ),
        }
