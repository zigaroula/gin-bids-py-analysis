from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat

from gin_bids_py_analysis.processing.base import BaseProcessingResult
from gin_bids_py_analysis.processing.utils.matlab import make_struct

from ..writer import BaseTrialStatsGroupProcessingWriter
from .result import ConditionTestGroupProcessingResult


class ConditionTestGroupProcessingWriter(BaseTrialStatsGroupProcessingWriter):
    """Write group-level condition_test ROI outputs to HDF5 or MATLAB."""

    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        if not isinstance(result, ConditionTestGroupProcessingResult):
            raise TypeError(
                f"Expected ConditionTestGroupProcessingResult, got {type(result).__name__!r}"
            )

        if self.params.output_format == "matlab":
            self._write_matlab(result, output_path)
        else:
            self._write_hdf5(result, output_path)

    def _write_matlab(
        self,
        result: ConditionTestGroupProcessingResult,
        output_path: Path,
    ) -> None:
        stats_struct = self.make_activity_stats_struct(result=result)

        means_struct = make_struct(
            metric_mean=result.metric_mean.astype(np.float64),
            condition_a_mean=result.condition_a_activity_mean.astype(np.float64),
            condition_a_sem=result.condition_a_activity_sem.astype(np.float64),
            condition_b_mean=result.condition_b_activity_mean.astype(np.float64),
            condition_b_sem=result.condition_b_activity_sem.astype(np.float64),
        )

        uncertainty_struct = make_struct(
            metric_sem=result.metric_sem.astype(np.float64),
        )

        summary_struct = make_struct(
            t_values=result.epoch_activity_t.astype(np.float64),
            p_values=result.epoch_activity_p.astype(np.float64),
            df=result.epoch_activity_df.astype(np.float64),
            metric_mean=result.epoch_mean_metric_mean.astype(np.float64),
            metric_sem=result.epoch_mean_metric_sem.astype(np.float64),
        )

        meta_kwargs = self.common_meta_kwargs(result=result)
        meta_kwargs.update(
            n_group_permutations=int(result.metadata.get("n_group_permutations", 0)),
            cluster_threshold_alpha=float(
                result.metadata.get("cluster_threshold_alpha", 0.05)
            ),
        )
        meta_struct = make_struct(**meta_kwargs)

        data = make_struct(
            stats=stats_struct,
            means=means_struct,
            uncertainty=uncertainty_struct,
            summary_epoch=summary_struct,
            axes=self.make_axes_struct(result=result),
            meta=meta_struct,
            contributions=self.make_contributions_struct(result=result),
            activity_contributions=self.make_activity_contributions_struct(result=result),
            excluded_rois=self.make_excluded_rois_struct(result=result),
            provenance=self.make_provenance_struct(
                result=result,
                pipeline_name="condition_test_group",
            ),
        )
        if result.cluster_p_values is not None:
            n_rois_cs = len(result.cluster_p_values)
            windows_list = result.cluster_windows_s or []
            n_max_clusters = max((len(w) for w in windows_list), default=0)
            starts_2d = np.full((n_rois_cs, n_max_clusters), np.nan, dtype=np.float64)
            ends_2d = np.full((n_rois_cs, n_max_clusters), np.nan, dtype=np.float64)
            for i, roi_windows in enumerate(windows_list):
                for j, (t_start, t_end) in enumerate(roi_windows):
                    starts_2d[i, j] = t_start
                    ends_2d[i, j] = t_end
            null_cell: np.ndarray = np.empty(n_rois_cs, dtype=object)
            for i, nd in enumerate(result.cluster_null_distributions or []):
                null_cell[i] = nd.astype(np.float64)
            data.cluster_stats = make_struct(
                p_values=result.cluster_p_values.astype(np.float64),
                cluster_starts_s=starts_2d,
                cluster_ends_s=ends_2d,
                null_distributions=null_cell,
            )
        savemat(str(output_path), {"data": data}, do_compression=True, long_field_names=True)

    def _write_hdf5(self, result: ConditionTestGroupProcessingResult, output_path: Path) -> None:
        str_dtype = self.string_dtype()
        with h5py.File(output_path, "w") as fh:
            self.write_activity_stats_hdf5(fh, result=result, group_name="stats")

            self.write_activity_means_hdf5(fh, result=result)
            fh["means"].create_dataset(
                "metric_mean",
                data=result.metric_mean.astype(np.float64),
            )

            uncertainty_grp = fh.create_group("uncertainty")
            uncertainty_grp.create_dataset(
                "metric_sem",
                data=result.metric_sem.astype(np.float64),
            )

            summary_grp = fh.create_group("summary_epoch")
            summary_grp.create_dataset(
                "t_values",
                data=result.epoch_activity_t.astype(np.float64),
            )
            summary_grp.create_dataset(
                "p_values",
                data=result.epoch_activity_p.astype(np.float64),
            )
            summary_grp.create_dataset(
                "df",
                data=result.epoch_activity_df.astype(np.float64),
            )
            summary_grp.create_dataset(
                "metric_mean",
                data=result.epoch_mean_metric_mean.astype(np.float64),
            )
            summary_grp.create_dataset(
                "metric_sem",
                data=result.epoch_mean_metric_sem.astype(np.float64),
            )

            self.write_axes_hdf5(fh, result=result, str_dtype=str_dtype)

            meta_grp = fh.create_group("meta")
            self.write_common_meta_hdf5(meta_grp, result=result, str_dtype=str_dtype)
            meta_grp.create_dataset(
                "n_group_permutations",
                data=int(result.metadata.get("n_group_permutations", 0)),
            )
            meta_grp.create_dataset(
                "cluster_threshold_alpha",
                data=float(result.metadata.get("cluster_threshold_alpha", 0.05)),
            )

            self.write_excluded_rois_hdf5(
                fh,
                result=result,
                str_dtype=str_dtype,
            )
            self.write_contributions_hdf5(fh, result=result, str_dtype=str_dtype)
            self.write_activity_contributions_hdf5(
                fh,
                result=result,
                str_dtype=str_dtype,
            )
            self.write_provenance_hdf5(
                fh,
                result=result,
                pipeline_name="condition_test_group",
                str_dtype=str_dtype,
            )

            if result.cluster_p_values is not None:
                cs_grp = fh.create_group("cluster_stats")
                cs_grp.create_dataset(
                    "p_values",
                    data=result.cluster_p_values.astype(np.float64),
                )
                n_rois_cs = len(result.cluster_p_values)
                windows_list = result.cluster_windows_s or []
                n_max_clusters = max((len(w) for w in windows_list), default=0)
                starts_2d = np.full((n_rois_cs, n_max_clusters), np.nan, dtype=np.float64)
                ends_2d = np.full((n_rois_cs, n_max_clusters), np.nan, dtype=np.float64)
                for i, roi_windows in enumerate(windows_list):
                    for j, (t_start, t_end) in enumerate(roi_windows):
                        starts_2d[i, j] = t_start
                        ends_2d[i, j] = t_end
                cs_grp.create_dataset("cluster_starts_s", data=starts_2d)
                cs_grp.create_dataset("cluster_ends_s", data=ends_2d)
                if result.cluster_null_distributions:
                    all_sizes = [nd.size for nd in result.cluster_null_distributions]
                    max_len = max(all_sizes, default=0)
                    if max_len > 0:
                        null_matrix = np.full(
                            (len(result.cluster_null_distributions), max_len),
                            np.nan,
                            dtype=np.float32,
                        )
                        for i, nd in enumerate(result.cluster_null_distributions):
                            if nd.size > 0:
                                null_matrix[i, : nd.size] = nd.astype(np.float32)
                        cs_grp.create_dataset(
                            "null_distributions",
                            data=null_matrix,
                            compression="gzip",
                            compression_opts=4,
                        )
