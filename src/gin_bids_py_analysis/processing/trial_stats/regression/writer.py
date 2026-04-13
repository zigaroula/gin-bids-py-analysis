from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np
from scipy.io import savemat

from gin_bids_py_analysis.processing.utils.matlab import make_struct, matlab_safe_name

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
            "trial_activity_summary_response_time_s",
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
            _to_float_or_nan(metadata.get("trial_activity_summary_response_time_s")),
        ]

    def _write_hdf5_meta_extra(
        self,
        meta_grp: h5py.Group,
        result: RegressionProcessingResult,
        str_dtype: h5py.DatatypeLike,
    ) -> None:
        meta_grp.create_dataset("analysis_type", data="slope_regression", dtype=str_dtype)
        meta_grp.create_dataset("predictor", data=str(result.predictor), dtype=str_dtype)
        meta_grp.create_dataset(
            "predictor_zscore",
            data=str(result.predictor_zscore),
            dtype=str_dtype,
        )
        meta_grp.create_dataset(
            "predictor_transform_by_condition_json",
            data=json.dumps(
                result.predictor_transform_by_condition,
                sort_keys=True,
                ensure_ascii=True,
            ),
            dtype=str_dtype,
        )
        meta_grp.create_dataset(
            "trial_activity_summary_kind",
            data=str(result.trial_activity_summary_kind),
            dtype=str_dtype,
        )
        meta_grp.create_dataset(
            "trial_activity_summary_missing_response_policy",
            data=str(result.trial_activity_summary_missing_response_policy),
            dtype=str_dtype,
        )
        meta_grp.create_dataset(
            "trial_activity_summary_source_json",
            data=json.dumps(
                result.trial_activity_summary_source,
                sort_keys=True,
                ensure_ascii=True,
            ),
            dtype=str_dtype,
        )
        meta_grp.create_dataset(
            "trial_activity_summary_label",
            data=str(result.trial_activity_summary_label),
            dtype=str_dtype,
        )
        meta_grp.create_dataset("condition_a_stats_valid", data=bool(result.condition_a_stats_valid))
        meta_grp.create_dataset("condition_b_stats_valid", data=bool(result.condition_b_stats_valid))

    def _write_hdf5_trial_extra(
        self,
        trial_grp: h5py.Group,
        result: RegressionProcessingResult,
        str_dtype: h5py.DatatypeLike,
    ) -> None:
        trial_grp.create_dataset(
            "predictor_raw",
            data=np.array(
                [str(trial.metadata.get("predictor_raw", "")) for trial in result.resolved_trials],
                dtype=object,
            ),
            dtype=str_dtype,
        )
        trial_grp.create_dataset(
            "predictor_value",
            data=np.array(
                [_to_float_or_nan(trial.metadata.get("predictor_value")) for trial in result.resolved_trials],
                dtype=np.float64,
            ),
        )
        trial_grp.create_dataset(
            "predictor_raw_value",
            data=np.array(
                [_to_float_or_nan(trial.metadata.get("predictor_raw_value")) for trial in result.resolved_trials],
                dtype=np.float64,
            ),
        )
        trial_grp.create_dataset(
            "predictor_transformed_value",
            data=np.array(
                [
                    _to_float_or_nan(trial.metadata.get("predictor_transformed_value"))
                    for trial in result.resolved_trials
                ],
                dtype=np.float64,
            ),
        )
        trial_grp.create_dataset(
            "predictor_transform_scale",
            data=np.array(
                [
                    _to_float_or_nan(trial.metadata.get("predictor_transform_scale"))
                    for trial in result.resolved_trials
                ],
                dtype=np.float64,
            ),
        )
        trial_grp.create_dataset(
            "predictor_transform_offset",
            data=np.array(
                [
                    _to_float_or_nan(trial.metadata.get("predictor_transform_offset"))
                    for trial in result.resolved_trials
                ],
                dtype=np.float64,
            ),
        )
        trial_grp.create_dataset(
            "trial_activity_summary_response_time_s",
            data=np.array(
                [
                    _to_float_or_nan(trial.metadata.get("trial_activity_summary_response_time_s"))
                    for trial in result.resolved_trials
                ],
                dtype=np.float64,
            ),
        )

    def _write_hdf5_specific(
        self,
        fh: h5py.File,
        result: RegressionProcessingResult,
        str_dtype: h5py.DatatypeLike,
    ) -> None:
        regression_grp = fh.create_group("regression")
        _write_condition_regression_hdf5(
            regression_grp.create_group("condition_a"),
            slope=result.condition_a_slope,
            intercept=result.condition_a_intercept,
            r_value=result.condition_a_r_value,
            p_value=result.condition_a_p_value,
            p_value_corrected=result.condition_a_p_value_corrected,
            significant_mask=result.condition_a_significant_mask,
            n_trials_used=result.condition_a_trials_used,
            stats_valid=result.condition_a_stats_valid,
        )
        _write_condition_regression_hdf5(
            regression_grp.create_group("condition_b"),
            slope=result.condition_b_slope,
            intercept=result.condition_b_intercept,
            r_value=result.condition_b_r_value,
            p_value=result.condition_b_p_value,
            p_value_corrected=result.condition_b_p_value_corrected,
            significant_mask=result.condition_b_significant_mask,
            n_trials_used=result.condition_b_trials_used,
            stats_valid=result.condition_b_stats_valid,
        )

        predictor_grp = fh.create_group("predictor")
        predictor_grp.create_dataset(
            "condition_a_raw_values",
            data=np.asarray(result.condition_a_predictor_raw_values, dtype=np.float64),
        )
        predictor_grp.create_dataset(
            "condition_b_raw_values",
            data=np.asarray(result.condition_b_predictor_raw_values, dtype=np.float64),
        )
        predictor_grp.create_dataset(
            "condition_a_transformed_values",
            data=np.asarray(result.condition_a_predictor_transformed_values, dtype=np.float64),
        )
        predictor_grp.create_dataset(
            "condition_b_transformed_values",
            data=np.asarray(result.condition_b_predictor_transformed_values, dtype=np.float64),
        )
        predictor_grp.create_dataset(
            "condition_a_values",
            data=np.asarray(result.condition_a_predictor_values, dtype=np.float64),
        )
        predictor_grp.create_dataset(
            "condition_b_values",
            data=np.asarray(result.condition_b_predictor_values, dtype=np.float64),
        )

        if result.condition_a_epoch_means.ndim == 2 and result.condition_a_epoch_means.size > 0:
            scatter_grp = fh.create_group("scatter")
            scatter_grp.create_dataset(
                "condition_a_epoch_means",
                data=result.condition_a_epoch_means.astype(np.float64),
            )
            scatter_grp.create_dataset(
                "condition_b_epoch_means",
                data=result.condition_b_epoch_means.astype(np.float64),
            )

        if (
            result.condition_a_trial_activity_summary_values.ndim == 2
            and result.condition_b_trial_activity_summary_values.ndim == 2
        ):
            summary_grp = fh.create_group("trial_activity_summary")
            summary_grp.create_dataset(
                "condition_a_values",
                data=result.condition_a_trial_activity_summary_values.astype(np.float64),
            )
            summary_grp.create_dataset(
                "condition_b_values",
                data=result.condition_b_trial_activity_summary_values.astype(np.float64),
            )
            summary_grp.create_dataset(
                "kind",
                data=str(result.trial_activity_summary_kind),
                dtype=str_dtype,
            )
            summary_grp.create_dataset(
                "missing_response_policy",
                data=str(result.trial_activity_summary_missing_response_policy),
                dtype=str_dtype,
            )
            summary_grp.create_dataset(
                "source_json",
                data=json.dumps(
                    result.trial_activity_summary_source,
                    sort_keys=True,
                    ensure_ascii=True,
                ),
                dtype=str_dtype,
            )
            summary_grp.create_dataset(
                "label",
                data=str(result.trial_activity_summary_label),
                dtype=str_dtype,
            )

    def _build_matlab_meta_extra(
        self,
        result: RegressionProcessingResult,
    ) -> dict[str, Any]:
        return {
            "analysis_type": np.str_("slope_regression"),
            "predictor": np.str_(result.predictor),
            "predictor_zscore": np.str_(result.predictor_zscore),
            "predictor_transform_by_condition_json": np.str_(
                json.dumps(
                    result.predictor_transform_by_condition,
                    sort_keys=True,
                    ensure_ascii=True,
                )
            ),
            "trial_activity_summary_kind": np.str_(result.trial_activity_summary_kind),
            "trial_activity_summary_missing_response_policy": np.str_(
                result.trial_activity_summary_missing_response_policy
            ),
            "trial_activity_summary_source_json": np.str_(
                json.dumps(
                    result.trial_activity_summary_source,
                    sort_keys=True,
                    ensure_ascii=True,
                )
            ),
            "trial_activity_summary_label": np.str_(result.trial_activity_summary_label),
            "condition_a_stats_valid": bool(result.condition_a_stats_valid),
            "condition_b_stats_valid": bool(result.condition_b_stats_valid),
        }

    def _build_matlab_trial_extra(
        self,
        result: RegressionProcessingResult,
    ) -> dict[str, Any]:
        return {
            "predictor_raw": np.array(
                [str(trial.metadata.get("predictor_raw", "")) for trial in result.resolved_trials],
                dtype=object,
            ),
            "predictor_raw_value": np.array(
                [_to_float_or_nan(trial.metadata.get("predictor_raw_value")) for trial in result.resolved_trials],
                dtype=np.float64,
            ),
            "predictor_transformed_value": np.array(
                [
                    _to_float_or_nan(trial.metadata.get("predictor_transformed_value"))
                    for trial in result.resolved_trials
                ],
                dtype=np.float64,
            ),
            "predictor_value": np.array(
                [_to_float_or_nan(trial.metadata.get("predictor_value")) for trial in result.resolved_trials],
                dtype=np.float64,
            ),
            "predictor_transform_scale": np.array(
                [
                    _to_float_or_nan(trial.metadata.get("predictor_transform_scale"))
                    for trial in result.resolved_trials
                ],
                dtype=np.float64,
            ),
            "predictor_transform_offset": np.array(
                [
                    _to_float_or_nan(trial.metadata.get("predictor_transform_offset"))
                    for trial in result.resolved_trials
                ],
                dtype=np.float64,
            ),
            "trial_activity_summary_response_time_s": np.array(
                [
                    _to_float_or_nan(trial.metadata.get("trial_activity_summary_response_time_s"))
                    for trial in result.resolved_trials
                ],
                dtype=np.float64,
            ),
        }

    def _build_matlab_specific(
        self,
        result: RegressionProcessingResult,
    ) -> dict[str, Any]:
        cond_a = matlab_safe_name(result.condition_a)
        cond_b = matlab_safe_name(result.condition_b)
        if result.condition_a_epoch_means.ndim == 2 and result.condition_a_epoch_means.size > 0:
            scatter_struct: object = make_struct(
                condition_a_epoch_means=result.condition_a_epoch_means.astype(np.float64),
                condition_b_epoch_means=result.condition_b_epoch_means.astype(np.float64),
            )
        else:
            scatter_struct = make_struct(
                condition_a_epoch_means=np.empty((0, 0), dtype=np.float64),
                condition_b_epoch_means=np.empty((0, 0), dtype=np.float64),
            )

        return {
            "regression": make_struct(
                condition_a=make_struct(
                    slope=result.condition_a_slope.astype(np.float64),
                    intercept=result.condition_a_intercept.astype(np.float64),
                    r_value=result.condition_a_r_value.astype(np.float64),
                    p_value=result.condition_a_p_value.astype(np.float64),
                    p_value_corrected=result.condition_a_p_value_corrected.astype(np.float64),
                    significant_mask=result.condition_a_significant_mask.astype(np.uint8),
                    n_trials_used=int(result.condition_a_trials_used),
                    stats_valid=bool(result.condition_a_stats_valid),
                ),
                condition_b=make_struct(
                    slope=result.condition_b_slope.astype(np.float64),
                    intercept=result.condition_b_intercept.astype(np.float64),
                    r_value=result.condition_b_r_value.astype(np.float64),
                    p_value=result.condition_b_p_value.astype(np.float64),
                    p_value_corrected=result.condition_b_p_value_corrected.astype(np.float64),
                    significant_mask=result.condition_b_significant_mask.astype(np.uint8),
                    n_trials_used=int(result.condition_b_trials_used),
                    stats_valid=bool(result.condition_b_stats_valid),
                ),
            ),
            "predictor": make_struct(
                condition_a_raw_values=np.asarray(result.condition_a_predictor_raw_values, dtype=np.float64),
                condition_b_raw_values=np.asarray(result.condition_b_predictor_raw_values, dtype=np.float64),
                condition_a_transformed_values=np.asarray(result.condition_a_predictor_transformed_values, dtype=np.float64),
                condition_b_transformed_values=np.asarray(result.condition_b_predictor_transformed_values, dtype=np.float64),
                condition_a_values=np.asarray(result.condition_a_predictor_values, dtype=np.float64),
                condition_b_values=np.asarray(result.condition_b_predictor_values, dtype=np.float64),
            ),
            "scatter": scatter_struct,
            "trial_activity_summary": make_struct(
                condition_a_values=result.condition_a_trial_activity_summary_values.astype(np.float64),
                condition_b_values=result.condition_b_trial_activity_summary_values.astype(np.float64),
                kind=np.str_(result.trial_activity_summary_kind),
                missing_response_policy=np.str_(result.trial_activity_summary_missing_response_policy),
                source_json=np.str_(
                    json.dumps(
                        result.trial_activity_summary_source,
                        sort_keys=True,
                        ensure_ascii=True,
                    )
                ),
                label=np.str_(result.trial_activity_summary_label),
            ),
            "means": make_struct(
                **{
                    cond_a: result.condition_a_mean.astype(np.float64),
                    cond_b: result.condition_b_mean.astype(np.float64),
                }
            ),
        }


def _write_condition_regression_hdf5(
    group: h5py.Group,
    *,
    slope: np.ndarray,
    intercept: np.ndarray,
    r_value: np.ndarray,
    p_value: np.ndarray,
    p_value_corrected: np.ndarray,
    significant_mask: np.ndarray,
    n_trials_used: int,
    stats_valid: bool,
) -> None:
    group.create_dataset("slope", data=np.asarray(slope, dtype=np.float64))
    group.create_dataset("intercept", data=np.asarray(intercept, dtype=np.float64))
    group.create_dataset("r_value", data=np.asarray(r_value, dtype=np.float64))
    group.create_dataset("p_value", data=np.asarray(p_value, dtype=np.float64))
    group.create_dataset("p_value_corrected", data=np.asarray(p_value_corrected, dtype=np.float64))
    group.create_dataset("significant_mask", data=np.asarray(significant_mask, dtype=bool))
    group.create_dataset("n_trials_used", data=int(n_trials_used))
    group.create_dataset("stats_valid", data=bool(stats_valid))


def _validate_shape_consistency(result: RegressionProcessingResult) -> None:
    expected = result.condition_a_slope.shape
    shapes = {
        "condition_a_intercept": result.condition_a_intercept.shape,
        "condition_a_r_value": result.condition_a_r_value.shape,
        "condition_a_p_value": result.condition_a_p_value.shape,
        "condition_a_p_value_corrected": result.condition_a_p_value_corrected.shape,
        "condition_a_significant_mask": result.condition_a_significant_mask.shape,
        "condition_b_slope": result.condition_b_slope.shape,
        "condition_b_intercept": result.condition_b_intercept.shape,
        "condition_b_r_value": result.condition_b_r_value.shape,
        "condition_b_p_value": result.condition_b_p_value.shape,
        "condition_b_p_value_corrected": result.condition_b_p_value_corrected.shape,
        "condition_b_significant_mask": result.condition_b_significant_mask.shape,
        "condition_a_mean": result.condition_a_mean.shape,
        "condition_b_mean": result.condition_b_mean.shape,
        "condition_a_sem": result.condition_a_sem.shape,
        "condition_b_sem": result.condition_b_sem.shape,
    }
    mismatched = [name for name, shape in shapes.items() if shape != expected]
    if mismatched:
        details = ", ".join(f"{name}={shapes[name]!r}" for name in mismatched)
        raise ValueError(
            "All regression and summary arrays must share shape "
            f"{expected!r}; got {details}."
        )


def _to_float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")
