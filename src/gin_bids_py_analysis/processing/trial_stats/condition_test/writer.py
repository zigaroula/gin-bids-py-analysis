from __future__ import annotations

from typing import Any

import h5py
import numpy as np

from gin_bids_py_analysis.processing.utils.matlab import make_struct, matlab_safe_name

from ..writer import BaseTrialStatsProcessingWriter
from .result import ConditionTestProcessingResult


class ConditionTestProcessingWriter(BaseTrialStatsProcessingWriter):
    """Write condition-test statistics plus a companion TSV trial audit table."""

    def _pipeline_name(self) -> str:
        return "conditiontest"

    def _validate_result(self, result: ConditionTestProcessingResult) -> None:
        if not isinstance(result, ConditionTestProcessingResult):
            raise TypeError(
                f"Expected ConditionTestProcessingResult, got {type(result).__name__!r}"
            )
        _validate_uncertainty_shapes(result)

    def _write_hdf5_meta_extra(
        self,
        meta_grp: h5py.Group,
        result: ConditionTestProcessingResult,
        str_dtype: h5py.DatatypeLike,
    ) -> None:
        meta_grp.create_dataset(
            "n_permutations",
            data=int(result.metadata.get("n_permutations", 0)),
        )
        meta_grp.create_dataset(
            "channel_significance_mode",
            data=str(result.metadata.get("channel_significance_mode", "none")),
            dtype=str_dtype,
        )
        meta_grp.create_dataset(
            "channel_significance_duration_threshold_ms",
            data=float(
                result.metadata.get("channel_significance_duration_threshold_ms", 100.0)
            ),
        )

    def _write_hdf5_specific(
        self,
        fh: h5py.File,
        result: ConditionTestProcessingResult,
        str_dtype: h5py.DatatypeLike,
    ) -> None:
        del str_dtype
        p_values_uncorrected = (
            result.p_values_uncorrected if result.p_values_uncorrected.size else result.p_values
        )
        significant_mask = (
            result.significant_mask
            if result.significant_mask.size
            else (np.isfinite(result.p_values) & (result.p_values < result.significance_alpha))
        )

        stats_grp = fh.create_group("stats")
        stats_grp.create_dataset("t_values", data=result.t_values.astype(np.float64))
        stats_grp.create_dataset("p_values", data=result.p_values.astype(np.float64))
        stats_grp.create_dataset(
            "p_values_uncorrected",
            data=p_values_uncorrected.astype(np.float64),
        )
        stats_grp.create_dataset(
            "significant_mask",
            data=significant_mask.astype(bool),
        )
        if result.channel_significant_mask is not None:
            stats_grp.create_dataset(
                "channel_significant_mask",
                data=result.channel_significant_mask.astype(bool),
            )
        if result.permuted_t_values is not None:
            stats_grp.create_dataset(
                "permuted_t_values",
                data=result.permuted_t_values.astype(np.float32),
                compression="gzip",
                compression_opts=4,
            )

        fh["means"].create_dataset(
            "difference",
            data=result.mean_difference.astype(np.float64),
        )
        fh["uncertainty"].create_dataset(
            "difference_sem",
            data=result.difference_sem.astype(np.float64),
        )
        fh["uncertainty"].create_dataset(
            "difference_ci95_low",
            data=result.difference_ci95_low.astype(np.float64),
        )
        fh["uncertainty"].create_dataset(
            "difference_ci95_high",
            data=result.difference_ci95_high.astype(np.float64),
        )

    def _build_matlab_meta_extra(
        self,
        result: ConditionTestProcessingResult,
    ) -> dict[str, Any]:
        return {
            "n_permutations": int(result.metadata.get("n_permutations", 0)),
            "channel_significance_mode": str(
                result.metadata.get("channel_significance_mode", "none")
            ),
            "ch_sig_duration_threshold_ms": float(
                result.metadata.get("channel_significance_duration_threshold_ms", 100.0)
            ),
        }

    def _build_matlab_specific(
        self,
        result: ConditionTestProcessingResult,
    ) -> dict[str, Any]:
        p_values_uncorrected = (
            result.p_values_uncorrected if result.p_values_uncorrected.size else result.p_values
        )
        significant_mask = (
            result.significant_mask
            if result.significant_mask.size
            else (np.isfinite(result.p_values) & (result.p_values < result.significance_alpha))
        )
        cond_a = matlab_safe_name(result.condition_a)
        cond_b = matlab_safe_name(result.condition_b)

        stats_fields: dict[str, Any] = {
            "t_values": result.t_values.astype(np.float64),
            "p_values": result.p_values.astype(np.float64),
            "p_values_uncorrected": p_values_uncorrected.astype(np.float64),
            "significant_mask": significant_mask.astype(np.uint8),
        }
        if result.channel_significant_mask is not None:
            stats_fields["channel_significant_mask"] = result.channel_significant_mask.astype(
                np.uint8
            )

        return {
            "stats": make_struct(**stats_fields),
            "means": make_struct(
                **{
                    cond_a: result.condition_a_mean.astype(np.float64),
                    cond_b: result.condition_b_mean.astype(np.float64),
                    "difference": result.mean_difference.astype(np.float64),
                }
            ),
            "uncertainty": make_struct(
                **{
                    cond_a + "_sem": result.condition_a_sem.astype(np.float64),
                    cond_b + "_sem": result.condition_b_sem.astype(np.float64),
                    "difference_sem": result.difference_sem.astype(np.float64),
                    "difference_ci95_low": result.difference_ci95_low.astype(np.float64),
                    "difference_ci95_high": result.difference_ci95_high.astype(np.float64),
                }
            ),
        }


def _validate_uncertainty_shapes(result: ConditionTestProcessingResult) -> None:
    expected = result.mean_difference.shape
    shapes = {
        "condition_a_sem": result.condition_a_sem.shape,
        "condition_b_sem": result.condition_b_sem.shape,
        "difference_sem": result.difference_sem.shape,
        "difference_ci95_low": result.difference_ci95_low.shape,
        "difference_ci95_high": result.difference_ci95_high.shape,
    }
    mismatched = [name for name, shape in shapes.items() if shape != expected]
    if mismatched:
        details = ", ".join(f"{name}={shapes[name]!r}" for name in mismatched)
        raise ValueError(
            "Uncertainty arrays must match mean_difference shape "
            f"{expected!r}; got {details}."
        )
