from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py
import numpy as np

from gin_bids_py_analysis.processing.base import BaseProcessingResult, BaseProcessingWriter

from .result import TrialStatsGroupProcessingResult


def _package_version() -> str:
    try:
        return version("gin-bids-py-analysis")
    except PackageNotFoundError:
        return "unknown"


class TrialStatsGroupProcessingWriter(BaseProcessingWriter):
    """Write group-level trial-stats ROI outputs to HDF5."""

    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        if not isinstance(result, TrialStatsGroupProcessingResult):
            raise TypeError(
                f"Expected TrialStatsGroupProcessingResult, got {type(result).__name__!r}"
            )

        str_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(output_path, "w") as fh:
            stats_grp = fh.create_group("stats")
            stats_grp.create_dataset("t_values", data=result.t_values.astype(np.float64))
            stats_grp.create_dataset("p_values", data=result.p_values.astype(np.float64))
            stats_grp.create_dataset(
                "p_values_uncorrected",
                data=result.p_values_uncorrected.astype(np.float64),
            )
            stats_grp.create_dataset(
                "significant_mask",
                data=result.significant_mask.astype(bool),
            )

            means_grp = fh.create_group("means")
            means_grp.create_dataset(
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
                data=result.epoch_mean_t_values.astype(np.float64),
            )
            summary_grp.create_dataset(
                "p_values",
                data=result.epoch_mean_p_values.astype(np.float64),
            )
            summary_grp.create_dataset(
                "df",
                data=result.epoch_mean_df.astype(np.float64),
            )
            summary_grp.create_dataset(
                "metric_mean",
                data=result.epoch_mean_metric_mean.astype(np.float64),
            )
            summary_grp.create_dataset(
                "metric_sem",
                data=result.epoch_mean_metric_sem.astype(np.float64),
            )
            summary_grp.create_dataset(
                "roi_channel_counts",
                data=result.roi_channel_counts.astype(np.int64),
            )
            summary_grp.create_dataset(
                "roi_subject_counts",
                data=result.roi_subject_counts.astype(np.int64),
            )

            axes_grp = fh.create_group("axes")
            axes_grp.create_dataset(
                "region",
                data=np.array(result.region_names, dtype=object),
                dtype=str_dtype,
            )
            axes_grp.create_dataset(
                "time_s",
                data=result.time_axis_s.astype(np.float64),
            )

            meta_grp = fh.create_group("meta")
            meta_grp.create_dataset(
                "analysis_level",
                data="roi_group",
                dtype=str_dtype,
            )
            meta_grp.create_dataset(
                "source_metric",
                data=result.source_metric,
                dtype=str_dtype,
            )
            meta_grp.create_dataset(
                "condition_labels",
                data=np.array(list(result.condition_labels), dtype=object),
                dtype=str_dtype,
            )
            meta_grp.create_dataset(
                "p_value_correction_method",
                data=result.p_value_correction_method,
                dtype=str_dtype,
            )
            meta_grp.create_dataset(
                "significance_alpha",
                data=float(result.significance_alpha),
            )
            meta_grp.create_dataset(
                "roi_mode",
                data=result.roi_mode,
                dtype=str_dtype,
            )
            meta_grp.create_dataset(
                "atlas_name",
                data=str(result.atlas_name or ""),
                dtype=str_dtype,
            )
            meta_grp.create_dataset(
                "included_roi_count",
                data=int(len(result.region_names)),
            )
            meta_grp.create_dataset(
                "excluded_roi_count",
                data=int(len(result.excluded_rois)),
            )

            excluded_grp = meta_grp.create_group("excluded_rois")
            excluded_grp.create_dataset(
                "region",
                data=np.array(list(result.excluded_rois.keys()), dtype=object),
                dtype=str_dtype,
            )
            excluded_grp.create_dataset(
                "reason",
                data=np.array(list(result.excluded_rois.values()), dtype=object),
                dtype=str_dtype,
            )

            if "binning_mode" in result.metadata:
                meta_grp.create_dataset(
                    "binning_mode",
                    data=str(result.metadata["binning_mode"]),
                    dtype=str_dtype,
                )
            if "window_ms" in result.metadata:
                meta_grp.create_dataset(
                    "window_ms",
                    data=float(result.metadata["window_ms"]),
                )
            if "n_bins" in result.metadata:
                meta_grp.create_dataset(
                    "n_bins",
                    data=int(result.metadata["n_bins"]),
                )
            if "effective_n_bins" in result.metadata:
                meta_grp.create_dataset(
                    "effective_n_bins",
                    data=int(result.metadata["effective_n_bins"]),
                )

            contrib_grp = fh.create_group("contributions")
            contrib_grp.create_dataset(
                "region",
                data=np.array([item.roi for item in result.contributions], dtype=object),
                dtype=str_dtype,
            )
            contrib_grp.create_dataset(
                "subject",
                data=np.array([item.subject for item in result.contributions], dtype=object),
                dtype=str_dtype,
            )
            contrib_grp.create_dataset(
                "channel",
                data=np.array([item.channel for item in result.contributions], dtype=object),
                dtype=str_dtype,
            )
            contrib_grp.create_dataset(
                "source_stats_file",
                data=np.array(
                    [item.source_stats_file for item in result.contributions],
                    dtype=object,
                ),
                dtype=str_dtype,
            )

            prov_grp = fh.create_group("provenance")
            prov_grp.create_dataset(
                "source_trial_stats_files",
                data=np.array(result.source_trial_stats_files, dtype=object),
                dtype=str_dtype,
            )
            prov_grp.create_dataset(
                "source_electrodes_files",
                data=np.array(result.source_electrodes_files, dtype=object),
                dtype=str_dtype,
            )
            prov_grp.create_dataset(
                "pipeline_name",
                data="trial_stats_group",
                dtype=str_dtype,
            )
            prov_grp.create_dataset(
                "pipeline_version",
                data=_package_version(),
                dtype=str_dtype,
            )
