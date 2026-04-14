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
        region_key: str = "name",
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
        roi_key: str,
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
    def write_provenance_hdf5(
        fh: h5py.File,
        *,
        source_stats_key: str,
        source_stats_files: list[str],
        source_electrodes_files: list[str],
        pipeline_name: str,
        str_dtype: h5py.Datatype,
    ) -> None:
        prov = fh.create_group("provenance")
        prov.create_dataset(
            source_stats_key,
            data=np.array(source_stats_files, dtype=object),
            dtype=str_dtype,
        )
        prov.create_dataset(
            "source_electrodes_files",
            data=np.array(source_electrodes_files, dtype=object),
            dtype=str_dtype,
        )
        prov.create_dataset("pipeline_name", data=pipeline_name, dtype=str_dtype)
        prov.create_dataset("pipeline_version", data=package_version(), dtype=str_dtype)

    @staticmethod
    def make_provenance_struct(
        *,
        source_stats_key: str,
        source_stats_files: list[str],
        source_electrodes_files: list[str],
        pipeline_name: str,
    ) -> np.ndarray:
        return make_struct(
            **{
                source_stats_key: np.array(source_stats_files, dtype=object),
                "source_electrodes_files": np.array(source_electrodes_files, dtype=object),
                "pipeline_name": np.str_(pipeline_name),
                "pipeline_version": np.str_(package_version()),
            }
        )
