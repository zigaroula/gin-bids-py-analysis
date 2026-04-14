from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat

from gin_bids_py_analysis.processing.base import BaseProcessingResult
from gin_bids_py_analysis.processing.utils.matlab import make_struct

from ..writer import BaseTrialStatsGroupProcessingWriter
from .result import RegressionGroupProcessingResult


class RegressionGroupProcessingWriter(BaseTrialStatsGroupProcessingWriter):
    """Write group-level regression ROI outputs to HDF5 or MATLAB."""

    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        if not isinstance(result, RegressionGroupProcessingResult):
            raise TypeError(
                f"Expected RegressionGroupProcessingResult, got {type(result).__name__!r}"
            )
        if self.params.output_format == "matlab":
            self._write_matlab(result, output_path)
        else:
            self._write_hdf5(result, output_path)

    def _write_hdf5(
        self,
        result: RegressionGroupProcessingResult,
        output_path: Path,
    ) -> None:
        str_dtype = self.string_dtype()

        with h5py.File(output_path, "w") as fh:
            source_metric = fh.create_group("source_metric")
            source_metric.create_dataset(
                "t_values",
                data=result.source_metric_t_values.astype(np.float64),
            )
            source_metric.create_dataset(
                "p_values",
                data=result.source_metric_p_values.astype(np.float64),
            )
            source_metric.create_dataset(
                "p_values_uncorrected",
                data=result.source_metric_p_values_uncorrected.astype(np.float64),
            )
            source_metric.create_dataset(
                "significant_mask",
                data=result.source_metric_significant_mask.astype(bool),
            )
            source_metric.create_dataset(
                "condition_a_mean",
                data=result.condition_a_source_metric_mean.astype(np.float64),
            )
            source_metric.create_dataset(
                "condition_a_sem",
                data=result.condition_a_source_metric_sem.astype(np.float64),
            )
            source_metric.create_dataset(
                "condition_b_mean",
                data=result.condition_b_source_metric_mean.astype(np.float64),
            )
            source_metric.create_dataset(
                "condition_b_sem",
                data=result.condition_b_source_metric_sem.astype(np.float64),
            )
            ep_reg = source_metric.create_group("epoch_summary")
            ep_reg.create_dataset(
                "t",
                data=result.epoch_source_metric_t.astype(np.float64),
            )
            ep_reg.create_dataset(
                "p",
                data=result.epoch_source_metric_p.astype(np.float64),
            )
            ep_reg.create_dataset(
                "df",
                data=result.epoch_source_metric_df.astype(np.float64),
            )

            self.write_activity_stats_hdf5(fh, result=result, group_name="activity")
            ep_act = fh["activity"].create_group("epoch_summary")
            ep_act.create_dataset("t", data=result.epoch_activity_t.astype(np.float64))
            ep_act.create_dataset("p", data=result.epoch_activity_p.astype(np.float64))
            ep_act.create_dataset("df", data=result.epoch_activity_df.astype(np.float64))

            self.write_activity_means_hdf5(fh, result=result)

            r_vals = fh.create_group("r_values")
            r_vals.create_dataset(
                "condition_a_mean",
                data=result.condition_a_r_value_mean.astype(np.float64),
            )
            r_vals.create_dataset(
                "condition_a_sem",
                data=result.condition_a_r_value_sem.astype(np.float64),
            )
            r_vals.create_dataset(
                "condition_b_mean",
                data=result.condition_b_r_value_mean.astype(np.float64),
            )
            r_vals.create_dataset(
                "condition_b_sem",
                data=result.condition_b_r_value_sem.astype(np.float64),
            )

            self.write_axes_hdf5(fh, result=result, str_dtype=str_dtype)

            meta = fh.create_group("meta")
            self.write_common_meta_hdf5(meta, result=result, str_dtype=str_dtype)
            meta.create_dataset(
                "contrast_mode",
                data=str(result.contrast_mode),
                dtype=str_dtype,
            )
            for key in (
                "predictor",
                "predictor_zscore",
                "predictor_transform_by_condition_json",
                "trial_activity_summary_kind",
                "trial_activity_summary_missing_response_policy",
                "trial_activity_summary_source_json",
                "trial_activity_summary_label",
                "scatter_aggregation",
            ):
                value = result.metadata.get(key)
                if value is not None:
                    meta.create_dataset(key, data=str(value), dtype=str_dtype)

            self.write_excluded_rois_hdf5(fh, result=result, str_dtype=str_dtype)
            self.write_contributions_hdf5(fh, result=result, str_dtype=str_dtype)
            self.write_activity_contributions_hdf5(
                fh,
                result=result,
                str_dtype=str_dtype,
            )
            _write_metric_contributions_hdf5(
                fh.create_group("source_metric_contributions"),
                condition_a_values=result.condition_a_source_metric_contributions,
                condition_b_values=result.condition_b_source_metric_contributions,
                labels=result.contribution_labels,
                region_names=result.region_names,
                str_dtype=str_dtype,
            )

            if result.condition_a_scatter_predictor:
                _write_scatter_hdf5(
                    fh.create_group("scatter_data"),
                    condition_a_predictor=result.condition_a_scatter_predictor,
                    condition_a_activity=result.condition_a_scatter_activity,
                    condition_b_predictor=result.condition_b_scatter_predictor,
                    condition_b_activity=result.condition_b_scatter_activity,
                    region_names=result.region_names,
                    str_dtype=str_dtype,
                )

            if result.cluster_p_values is not None:
                cs = fh.create_group("cluster_stats")
                cs.create_dataset(
                    "p_values",
                    data=result.cluster_p_values.astype(np.float64),
                )
                windows = result.cluster_best_cluster_windows_s or []
                cs.create_dataset(
                    "best_cluster_start_s",
                    data=np.array(
                        [w[0] if w is not None else np.nan for w in windows],
                        dtype=np.float64,
                    ),
                )
                cs.create_dataset(
                    "best_cluster_end_s",
                    data=np.array(
                        [w[1] if w is not None else np.nan for w in windows],
                        dtype=np.float64,
                    ),
                )
                null_dists = result.cluster_null_distributions or []
                max_len = max((len(nd) for nd in null_dists), default=0)
                null_matrix = np.full(
                    (len(null_dists), max_len), np.nan, dtype=np.float64
                )
                for i, nd in enumerate(null_dists):
                    null_matrix[i, : len(nd)] = nd
                cs.create_dataset(
                    "null_distributions",
                    data=null_matrix,
                    compression="gzip",
                    compression_opts=4,
                )

            self.write_provenance_hdf5(
                fh,
                result=result,
                pipeline_name="regression_group",
                str_dtype=str_dtype,
            )

    def _write_matlab(
        self,
        result: RegressionGroupProcessingResult,
        output_path: Path,
    ) -> None:
        source_metric_struct = make_struct(
            t_values=result.source_metric_t_values.astype(np.float64),
            p_values=result.source_metric_p_values.astype(np.float64),
            p_values_uncorrected=result.source_metric_p_values_uncorrected.astype(
                np.float64
            ),
            significant_mask=result.source_metric_significant_mask.astype(np.uint8),
            condition_a_mean=result.condition_a_source_metric_mean.astype(np.float64),
            condition_a_sem=result.condition_a_source_metric_sem.astype(np.float64),
            condition_b_mean=result.condition_b_source_metric_mean.astype(np.float64),
            condition_b_sem=result.condition_b_source_metric_sem.astype(np.float64),
            epoch_summary=make_struct(
                t=result.epoch_source_metric_t.astype(np.float64),
                p=result.epoch_source_metric_p.astype(np.float64),
                df=result.epoch_source_metric_df.astype(np.float64),
            ),
        )

        activity_struct = self.make_activity_stats_struct(
            result=result,
            epoch_summary=make_struct(
                t=result.epoch_activity_t.astype(np.float64),
                p=result.epoch_activity_p.astype(np.float64),
                df=result.epoch_activity_df.astype(np.float64),
            ),
        )

        r_values_struct = make_struct(
            condition_a_mean=result.condition_a_r_value_mean.astype(np.float64),
            condition_a_sem=result.condition_a_r_value_sem.astype(np.float64),
            condition_b_mean=result.condition_b_r_value_mean.astype(np.float64),
            condition_b_sem=result.condition_b_r_value_sem.astype(np.float64),
        )

        meta_kwargs = self.common_meta_kwargs(result=result)
        meta_kwargs["contrast_mode"] = np.str_(result.contrast_mode)
        for key in (
            "predictor",
            "predictor_zscore",
            "predictor_transform_by_condition_json",
            "trial_activity_summary_kind",
            "trial_activity_summary_missing_response_policy",
            "trial_activity_summary_source_json",
            "trial_activity_summary_label",
            "scatter_aggregation",
        ):
            value = result.metadata.get(key)
            if value is not None:
                meta_kwargs[key] = np.str_(str(value))
        data = make_struct(
            source_metric=source_metric_struct,
            activity=activity_struct,
            means=self.make_activity_means_struct(result=result),
            r_values=r_values_struct,
            axes=self.make_axes_struct(result=result),
            meta=make_struct(**meta_kwargs),
            contributions=self.make_contributions_struct(result=result),
            activity_contributions=self.make_activity_contributions_struct(result=result),
            source_metric_contributions=_make_metric_contributions_struct(result=result),
            scatter_data=_make_scatter_struct(result=result),
            excluded_rois=self.make_excluded_rois_struct(result=result),
            provenance=self.make_provenance_struct(
                result=result,
                pipeline_name="regression_group",
            ),
            **(
                {
                    "cluster_stats": _make_cluster_stats_struct(result),
                }
                if result.cluster_p_values is not None
                else {}
            ),
        )
        savemat(str(output_path), {"data": data}, do_compression=True, long_field_names=True)


def _write_metric_contributions_hdf5(
    grp: h5py.Group,
    *,
    condition_a_values: list,
    condition_b_values: list,
    labels: list,
    region_names: list[str],
    str_dtype: object,
) -> None:
    if not condition_a_values:
        return
    grp.create_dataset(
        "region_names",
        data=np.array(region_names, dtype=object),
        dtype=str_dtype,
    )
    for i, roi_name in enumerate(region_names):
        roi_grp = grp.create_group(str(i))
        roi_grp.attrs["roi"] = roi_name
        roi_grp.create_dataset(
            "condition_a",
            data=np.asarray(condition_a_values[i], dtype=np.float64),
        )
        roi_grp.create_dataset(
            "condition_b",
            data=np.asarray(condition_b_values[i], dtype=np.float64),
        )
        roi_grp.create_dataset(
            "labels",
            data=np.array(labels[i], dtype=object),
            dtype=str_dtype,
        )


def _make_metric_contributions_struct(
    result: RegressionGroupProcessingResult,
) -> np.ndarray:
    n_rois = len(result.region_names)
    if not result.condition_a_source_metric_contributions:
        return make_struct(
            condition_a=np.array([], dtype=object),
            condition_b=np.array([], dtype=object),
            labels=np.array([], dtype=object),
            region_names=np.array([], dtype=object),
        )
    cond_a_cell: np.ndarray = np.empty(n_rois, dtype=object)
    cond_b_cell: np.ndarray = np.empty(n_rois, dtype=object)
    labels_cell: np.ndarray = np.empty(n_rois, dtype=object)
    for i in range(n_rois):
        cond_a_cell[i] = np.asarray(
            result.condition_a_source_metric_contributions[i],
            dtype=np.float64,
        )
        cond_b_cell[i] = np.asarray(
            result.condition_b_source_metric_contributions[i],
            dtype=np.float64,
        )
        labels_cell[i] = np.array(result.contribution_labels[i], dtype=object)
    return make_struct(
        condition_a=cond_a_cell,
        condition_b=cond_b_cell,
        labels=labels_cell,
        region_names=np.array(result.region_names, dtype=object),
    )


def _make_scatter_struct(result: RegressionGroupProcessingResult) -> np.ndarray:
    n_rois = len(result.region_names)
    if not result.condition_a_scatter_predictor:
        return make_struct(
            condition_a_predictor=np.array([], dtype=object),
            condition_a_activity=np.array([], dtype=object),
            condition_b_predictor=np.array([], dtype=object),
            condition_b_activity=np.array([], dtype=object),
            region_names=np.array([], dtype=object),
        )
    scatter_pred_a_cell: np.ndarray = np.empty(n_rois, dtype=object)
    scatter_act_a_cell: np.ndarray = np.empty(n_rois, dtype=object)
    scatter_pred_b_cell: np.ndarray = np.empty(n_rois, dtype=object)
    scatter_act_b_cell: np.ndarray = np.empty(n_rois, dtype=object)
    for i in range(n_rois):
        scatter_pred_a_cell[i] = np.asarray(
            result.condition_a_scatter_predictor[i],
            dtype=np.float64,
        )
        scatter_act_a_cell[i] = np.asarray(
            result.condition_a_scatter_activity[i],
            dtype=np.float64,
        )
        scatter_pred_b_cell[i] = np.asarray(
            result.condition_b_scatter_predictor[i],
            dtype=np.float64,
        )
        scatter_act_b_cell[i] = np.asarray(
            result.condition_b_scatter_activity[i],
            dtype=np.float64,
        )
    return make_struct(
        condition_a_predictor=scatter_pred_a_cell,
        condition_a_activity=scatter_act_a_cell,
        condition_b_predictor=scatter_pred_b_cell,
        condition_b_activity=scatter_act_b_cell,
        region_names=np.array(result.region_names, dtype=object),
    )


def _write_scatter_hdf5(
    grp: h5py.Group,
    *,
    condition_a_predictor: list,
    condition_a_activity: list,
    condition_b_predictor: list,
    condition_b_activity: list,
    region_names: list[str],
    str_dtype: object,
) -> None:
    grp.create_dataset(
        "region_names",
        data=np.array(region_names, dtype=object),
        dtype=str_dtype,
    )
    for i, roi_name in enumerate(region_names):
        roi_grp = grp.create_group(str(i))
        roi_grp.attrs["roi"] = roi_name
        roi_grp.create_dataset(
            "condition_a_predictor",
            data=np.asarray(condition_a_predictor[i], dtype=np.float64),
        )
        roi_grp.create_dataset(
            "condition_a_activity",
            data=np.asarray(condition_a_activity[i], dtype=np.float64),
        )
        roi_grp.create_dataset(
            "condition_b_predictor",
            data=np.asarray(condition_b_predictor[i], dtype=np.float64),
        )
        roi_grp.create_dataset(
            "condition_b_activity",
            data=np.asarray(condition_b_activity[i], dtype=np.float64),
        )


def _make_cluster_stats_struct(result: RegressionGroupProcessingResult) -> object:
    """Build a MATLAB struct for cluster-permutation outputs."""
    from gin_bids_py_analysis.processing.utils.matlab import make_struct

    p_values = result.cluster_p_values
    windows = result.cluster_best_cluster_windows_s or []
    null_dists = result.cluster_null_distributions or []

    max_len = max((len(nd) for nd in null_dists), default=0)
    null_matrix = np.full((len(null_dists), max_len), np.nan, dtype=np.float64)
    for i, nd in enumerate(null_dists):
        null_matrix[i, : len(nd)] = nd

    return make_struct(
        p_values=np.asarray(p_values, dtype=np.float64),
        best_cluster_start_s=np.array(
            [w[0] if w is not None else np.nan for w in windows], dtype=np.float64
        ),
        best_cluster_end_s=np.array(
            [w[1] if w is not None else np.nan for w in windows], dtype=np.float64
        ),
        null_distributions=null_matrix,
    )
