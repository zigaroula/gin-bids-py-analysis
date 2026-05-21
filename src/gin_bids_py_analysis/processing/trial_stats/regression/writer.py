from __future__ import annotations

from ..result import _to_float_or_nan
from ..writer import BaseTrialStatsProcessingWriter
from .result import RegressionProcessingResult


class RegressionProcessingWriter(BaseTrialStatsProcessingWriter):
    """Write regression statistics plus a companion TSV trial audit table."""

    def _pipeline_name(self) -> str:
        return "regression"

    def _validate_result(self, result: RegressionProcessingResult) -> None:
        if not isinstance(result, RegressionProcessingResult):
            raise TypeError(
                f"Expected RegressionProcessingResult, got {type(result).__name__!r}"
            )
        _validate_shape_consistency(result)

    def _trial_table_extra_header(self) -> list[str]:
        return [
            "predictor_raw",
            "predictor_raw_value",
            "predictor_transformed_value",
            "predictor_value",
            "predictor_transform_scale",
            "predictor_transform_offset",
        ]

    def _trial_table_extra_row(self, trial: object) -> list[object]:
        metadata = getattr(trial, "metadata", {})
        return [
            str(metadata.get("predictor_raw", "")),
            _to_float_or_nan(metadata.get("predictor_raw_value")),
            _to_float_or_nan(metadata.get("predictor_transformed_value")),
            _to_float_or_nan(metadata.get("predictor_value")),
            _to_float_or_nan(metadata.get("predictor_transform_scale")),
            _to_float_or_nan(metadata.get("predictor_transform_offset")),
        ]


def _validate_shape_consistency(result: RegressionProcessingResult) -> None:
    expected = result.regression.condition_a.slope.shape
    shapes = {
        "regression.condition_a.intercept": result.regression.condition_a.intercept.shape,
        "regression.condition_a.r_value": result.regression.condition_a.r_value.shape,
        "regression.condition_a.p_value": result.regression.condition_a.p_value.shape,
        "regression.condition_a.p_value_corrected": result.regression.condition_a.p_value_corrected.shape,
        "regression.condition_a.significant_mask": result.regression.condition_a.significant_mask.shape,
        "regression.condition_b.slope": result.regression.condition_b.slope.shape,
        "regression.condition_b.intercept": result.regression.condition_b.intercept.shape,
        "regression.condition_b.r_value": result.regression.condition_b.r_value.shape,
        "regression.condition_b.p_value": result.regression.condition_b.p_value.shape,
        "regression.condition_b.p_value_corrected": result.regression.condition_b.p_value_corrected.shape,
        "regression.condition_b.significant_mask": result.regression.condition_b.significant_mask.shape,
        "activity.condition_a.mean": result.activity.condition_a.mean.shape,
        "activity.condition_b.mean": result.activity.condition_b.mean.shape,
        "activity.condition_a.sem": result.activity.condition_a.sem.shape,
        "activity.condition_b.sem": result.activity.condition_b.sem.shape,
    }
    mismatched = [name for name, shape in shapes.items() if shape != expected]
    if mismatched:
        details = ", ".join(f"{name}={shapes[name]!r}" for name in mismatched)
        raise ValueError(
            "All regression and summary arrays must share shape "
            f"{expected!r}; got {details}."
        )
