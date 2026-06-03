from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from bidsforge.processing.utils.regression_params import PredictorAffineTransform

from ..params import BaseTimeFrequencyStatsParams, BaseTimeFrequencyStatsWriterParams


class TimeFrequencyRegressionParams(BaseTimeFrequencyStatsParams):
    """Parameters for trial-wise regression on stored TFR derivatives."""

    min_trials_per_condition: int = Field(default=3, ge=3)
    predictor: str = Field(default="predictor_value")
    predictor_transform_by_condition: dict[str, PredictorAffineTransform] = Field(
        default_factory=dict
    )
    predictor_zscore: Literal["none", "condition", "global"] = "none"

    @field_validator("predictor_transform_by_condition", mode="before")
    @classmethod
    def _coerce_predictor_transform_by_condition(
        cls,
        value: object,
    ) -> dict[str, dict[str, float]]:
        if value in (None, ""):
            return {}
        if not isinstance(value, dict):
            raise ValueError("predictor_transform_by_condition must be a mapping.")
        cleaned: dict[str, dict[str, float]] = {}
        for raw_condition, raw_transform in value.items():
            condition = str(raw_condition).strip()
            if not condition:
                continue
            if raw_transform is None:
                cleaned[condition] = {}
            elif isinstance(raw_transform, dict):
                item: dict[str, float] = {}
                if "scale" in raw_transform:
                    item["scale"] = float(raw_transform["scale"])
                if "offset" in raw_transform:
                    item["offset"] = float(raw_transform["offset"])
                cleaned[condition] = item
            else:
                raise ValueError("Each predictor transform must be a mapping.")
        return cleaned

    @model_validator(mode="after")
    def _validate_regression(self) -> "TimeFrequencyRegressionParams":
        self.predictor = str(self.predictor).strip()
        if not self.predictor:
            raise ValueError("predictor must be a non-empty string.")
        return self


class TimeFrequencyRegressionWriterParams(BaseTimeFrequencyStatsWriterParams):
    """Writer configuration for TF regression outputs."""

    pipeline_label: str = "time_frequency_regression"
    output_description: str = "tfregression"
    output_format: Literal["hdf5", "matlab"] = "hdf5"
