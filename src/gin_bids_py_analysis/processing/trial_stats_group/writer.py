"""Shared writer helpers for group-level trial statistics."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import h5py
import numpy as np

from gin_bids_py_analysis.processing.base import BaseProcessingWriter
from gin_bids_py_analysis.processing.utils.matlab import make_struct

from .result import BaseTrialStatsGroupProcessingResult


def package_version() -> str:
    """Return the installed package version when available."""

    try:
        return version("gin-bids-py-analysis")
    except PackageNotFoundError:
        return "unknown"


class BaseTrialStatsGroupProcessingWriter(BaseProcessingWriter):
    """Shared helper methods for group-level result writers."""

    @staticmethod
    def string_dtype() -> h5py.Datatype:
        return h5py.string_dtype(encoding="utf-8")

    @staticmethod
    def write_axes_hdf5(
        fh: h5py.File,
        *,
        result: BaseTrialStatsGroupProcessingResult,
        str_dtype: h5py.Datatype,
    ) -> None:
        axes = fh.create_group("axes")
        axes.create_dataset(
            "region",
            data=np.array(result.region_names, dtype=object),
            dtype=str_dtype,
        )
        axes.create_dataset("time_s", data=result.time_axis_s.astype(np.float64))

    @staticmethod
    def make_axes_struct(*, result: BaseTrialStatsGroupProcessingResult) -> np.ndarray:
        return make_struct(
            region=np.array(result.region_names, dtype=object),
            time_s=result.time_axis_s.astype(np.float64),
        )

    @staticmethod
    def write_excluded_rois_hdf5(
        fh: h5py.File,
        *,
        result: BaseTrialStatsGroupProcessingResult,
        str_dtype: h5py.Datatype,
        group_name: str = "excluded_rois",
        region_key: str = "region",
    ) -> None:
        excl = fh.create_group(group_name)
        excl.create_dataset(
            region_key,
            data=np.array(list(result.excluded_rois.keys()), dtype=object),
            dtype=str_dtype,
        )
        excl.create_dataset(
            "reason",
            data=np.array(list(result.excluded_rois.values()), dtype=object),
            dtype=str_dtype,
        )

    @staticmethod
    def make_excluded_rois_struct(*, result: BaseTrialStatsGroupProcessingResult) -> np.ndarray:
        return make_struct(
            region=np.array(list(result.excluded_rois.keys()), dtype=object),
            reason=np.array(list(result.excluded_rois.values()), dtype=object),
        )

    @staticmethod
    def write_contributions_hdf5(
        fh: h5py.File,
        *,
        result: BaseTrialStatsGroupProcessingResult,
        str_dtype: h5py.Datatype,
        roi_key: str = "region",
    ) -> None:
        contribs = fh.create_group("contributions")
        contribs.create_dataset(
            roi_key,
            data=np.array([c.roi for c in result.contributions], dtype=object),
            dtype=str_dtype,
        )
        contribs.create_dataset(
            "subject",
            data=np.array([c.subject for c in result.contributions], dtype=object),
            dtype=str_dtype,
        )
        contribs.create_dataset(
            "channel",
            data=np.array([c.channel for c in result.contributions], dtype=object),
            dtype=str_dtype,
        )
        contribs.create_dataset(
            "source_stats_file",
            data=np.array([c.source_stats_file for c in result.contributions], dtype=object),
            dtype=str_dtype,
        )

    @staticmethod
    def make_contributions_struct(*, result: BaseTrialStatsGroupProcessingResult) -> np.ndarray:
        return make_struct(
            region=np.array([item.roi for item in result.contributions], dtype=object),
            subject=np.array([item.subject for item in result.contributions], dtype=object),
            channel=np.array([item.channel for item in result.contributions], dtype=object),
            source_stats_file=np.array(
                [item.source_stats_file for item in result.contributions], dtype=object
            ),
        )

    @staticmethod
    def write_common_meta_hdf5(
        meta: h5py.Group,
        *,
        result: BaseTrialStatsGroupProcessingResult,
        str_dtype: h5py.Datatype,
    ) -> None:
        meta.create_dataset("analysis_level", data="roi_group", dtype=str_dtype)
        meta.create_dataset(
            "condition_labels",
            data=np.array(list(result.condition_labels), dtype=object),
            dtype=str_dtype,
        )
        meta.create_dataset("source_metric", data=str(result.source_metric), dtype=str_dtype)
        meta.create_dataset(
            "p_value_correction_method",
            data=str(result.p_value_correction_method),
            dtype=str_dtype,
        )
        meta.create_dataset("significance_alpha", data=float(result.significance_alpha))
        meta.create_dataset("roi_mode", data=str(result.roi_mode), dtype=str_dtype)
        meta.create_dataset("atlas_name", data=str(result.atlas_name or ""), dtype=str_dtype)
        meta.create_dataset(
            "roi_channel_counts",
            data=result.roi_channel_counts.astype(np.int64),
        )
        meta.create_dataset(
            "roi_subject_counts",
            data=result.roi_subject_counts.astype(np.int64),
        )
        meta.create_dataset("included_roi_count", data=int(len(result.region_names)))
        meta.create_dataset("excluded_roi_count", data=int(len(result.excluded_rois)))
        for key in (
            "binning_mode",
            "window_ms",
            "n_bins",
            "effective_n_bins",
            "activity_zscore",
            "activity_baseline_tmin_s",
            "activity_baseline_tmax_s",
        ):
            value = result.metadata.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                meta.create_dataset(key, data=value, dtype=str_dtype)
            elif isinstance(value, float):
                meta.create_dataset(key, data=float(value))
            else:
                meta.create_dataset(key, data=int(value))

    @staticmethod
    def common_meta_kwargs(*, result: BaseTrialStatsGroupProcessingResult) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "analysis_level": np.str_("roi_group"),
            "source_metric": np.str_(result.source_metric),
            "condition_labels": np.array(list(result.condition_labels), dtype=object),
            "p_value_correction_method": np.str_(result.p_value_correction_method),
            "significance_alpha": float(result.significance_alpha),
            "roi_mode": np.str_(result.roi_mode),
            "atlas_name": np.str_(result.atlas_name or ""),
            "roi_channel_counts": result.roi_channel_counts.astype(np.int64),
            "roi_subject_counts": result.roi_subject_counts.astype(np.int64),
            "included_roi_count": int(len(result.region_names)),
            "excluded_roi_count": int(len(result.excluded_rois)),
        }
        for key in (
            "binning_mode",
            "window_ms",
            "n_bins",
            "effective_n_bins",
            "activity_zscore",
            "activity_baseline_tmin_s",
            "activity_baseline_tmax_s",
        ):
            value = result.metadata.get(key)
            if value is None:
                continue
            kwargs[key] = np.str_(value) if isinstance(value, str) else value
        return kwargs

    @staticmethod
    def write_activity_stats_hdf5(
        fh: h5py.File,
        *,
        result: BaseTrialStatsGroupProcessingResult,
        group_name: str,
    ) -> None:
        grp = fh.create_group(group_name)
        grp.create_dataset(
            "t_values",
            data=result.activity_t_values.astype(np.float64),
        )
        grp.create_dataset(
            "p_values",
            data=result.activity_p_values.astype(np.float64),
        )
        grp.create_dataset(
            "p_values_uncorrected",
            data=result.activity_p_values_uncorrected.astype(np.float64),
        )
        grp.create_dataset(
            "significant_mask",
            data=result.activity_significant_mask.astype(bool),
        )

    @staticmethod
    def make_activity_stats_struct(
        *,
        result: BaseTrialStatsGroupProcessingResult,
        epoch_summary: "np.ndarray | None" = None,
    ) -> "np.ndarray":
        kwargs: dict[str, object] = dict(
            t_values=result.activity_t_values.astype(np.float64),
            p_values=result.activity_p_values.astype(np.float64),
            p_values_uncorrected=result.activity_p_values_uncorrected.astype(np.float64),
            significant_mask=result.activity_significant_mask.astype(np.uint8),
        )
        if epoch_summary is not None:
            kwargs["epoch_summary"] = epoch_summary
        return make_struct(**kwargs)

    @staticmethod
    def write_activity_means_hdf5(
        fh: h5py.File,
        *,
        result: BaseTrialStatsGroupProcessingResult,
    ) -> None:
        means = fh.create_group("means")
        means.create_dataset(
            "condition_a_mean",
            data=result.condition_a_activity_mean.astype(np.float64),
        )
        means.create_dataset(
            "condition_a_sem",
            data=result.condition_a_activity_sem.astype(np.float64),
        )
        means.create_dataset(
            "condition_b_mean",
            data=result.condition_b_activity_mean.astype(np.float64),
        )
        means.create_dataset(
            "condition_b_sem",
            data=result.condition_b_activity_sem.astype(np.float64),
        )

    @staticmethod
    def make_activity_means_struct(
        *,
        result: BaseTrialStatsGroupProcessingResult,
    ) -> np.ndarray:
        return make_struct(
            condition_a_mean=result.condition_a_activity_mean.astype(np.float64),
            condition_a_sem=result.condition_a_activity_sem.astype(np.float64),
            condition_b_mean=result.condition_b_activity_mean.astype(np.float64),
            condition_b_sem=result.condition_b_activity_sem.astype(np.float64),
        )

    @staticmethod
    def write_activity_contributions_hdf5(
        fh: h5py.File,
        *,
        result: BaseTrialStatsGroupProcessingResult,
        str_dtype: h5py.Datatype,
        group_name: str = "activity_contributions",
    ) -> None:
        if not result.condition_a_activity_contributions:
            return
        contribs = fh.create_group(group_name)
        contribs.create_dataset(
            "region_names",
            data=np.array(result.region_names, dtype=object),
            dtype=str_dtype,
        )
        for roi_idx, roi_name in enumerate(result.region_names):
            roi_grp = contribs.create_group(str(roi_idx))
            roi_grp.attrs["roi"] = roi_name
            roi_grp.create_dataset(
                "condition_a",
                data=np.asarray(
                    result.condition_a_activity_contributions[roi_idx],
                    dtype=np.float64,
                ),
            )
            roi_grp.create_dataset(
                "condition_b",
                data=np.asarray(
                    result.condition_b_activity_contributions[roi_idx],
                    dtype=np.float64,
                ),
            )
            roi_grp.create_dataset(
                "labels",
                data=np.array(result.contribution_labels[roi_idx], dtype=object),
                dtype=str_dtype,
            )

    @staticmethod
    def make_activity_contributions_struct(
        *,
        result: BaseTrialStatsGroupProcessingResult,
    ) -> np.ndarray:
        n_rois = len(result.region_names)
        if not result.condition_a_activity_contributions:
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
                result.condition_a_activity_contributions[i],
                dtype=np.float64,
            )
            cond_b_cell[i] = np.asarray(
                result.condition_b_activity_contributions[i],
                dtype=np.float64,
            )
            labels_cell[i] = np.array(result.contribution_labels[i], dtype=object)
        return make_struct(
            condition_a=cond_a_cell,
            condition_b=cond_b_cell,
            labels=labels_cell,
            region_names=np.array(result.region_names, dtype=object),
        )

    @staticmethod
    def write_provenance_hdf5(
        fh: h5py.File,
        *,
        result: BaseTrialStatsGroupProcessingResult,
        pipeline_name: str,
        str_dtype: h5py.Datatype,
    ) -> None:
        prov = fh.create_group("provenance")
        prov.create_dataset(
            "source_subject_stats_files",
            data=np.array(result.source_subject_stats_files, dtype=object),
            dtype=str_dtype,
        )
        prov.create_dataset(
            "source_electrodes_files",
            data=np.array(result.source_electrodes_files, dtype=object),
            dtype=str_dtype,
        )
        prov.create_dataset("pipeline_name", data=pipeline_name, dtype=str_dtype)
        prov.create_dataset("pipeline_version", data=package_version(), dtype=str_dtype)

    @staticmethod
    def make_provenance_struct(
        *,
        result: BaseTrialStatsGroupProcessingResult,
        pipeline_name: str,
    ) -> np.ndarray:
        return make_struct(
            **{
                "source_subject_stats_files": np.array(
                    result.source_subject_stats_files,
                    dtype=object,
                ),
                "source_electrodes_files": np.array(
                    result.source_electrodes_files,
                    dtype=object,
                ),
                "pipeline_name": np.str_(pipeline_name),
                "pipeline_version": np.str_(package_version()),
            }
        )
