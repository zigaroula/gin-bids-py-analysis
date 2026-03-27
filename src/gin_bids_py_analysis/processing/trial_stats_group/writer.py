from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat

from gin_bids_py_analysis.processing.base import BaseProcessingResult, BaseProcessingWriter
from gin_bids_py_analysis.processing.utils.matlab import make_struct

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

        if self.params.output_format == "matlab":
            self._write_matlab(result, output_path)
        else:
            self._write_hdf5(result, output_path)

    def _write_matlab(
        self,
        result: TrialStatsGroupProcessingResult,
        output_path: Path,
    ) -> None:
        stats_struct = make_struct(
            t_values=result.t_values.astype(np.float64),
            p_values=result.p_values.astype(np.float64),
            p_values_uncorrected=result.p_values_uncorrected.astype(np.float64),
            significant_mask=result.significant_mask.astype(np.uint8),
        )

        means_struct = make_struct(
            metric_mean=result.metric_mean.astype(np.float64),
            condition_a_mean=result.condition_a_group_mean.astype(np.float64),
            condition_a_sem=result.condition_a_group_sem.astype(np.float64),
            condition_b_mean=result.condition_b_group_mean.astype(np.float64),
            condition_b_sem=result.condition_b_group_sem.astype(np.float64),
        )

        uncertainty_struct = make_struct(
            metric_sem=result.metric_sem.astype(np.float64),
        )

        summary_struct = make_struct(
            t_values=result.epoch_mean_t_values.astype(np.float64),
            p_values=result.epoch_mean_p_values.astype(np.float64),
            df=result.epoch_mean_df.astype(np.float64),
            metric_mean=result.epoch_mean_metric_mean.astype(np.float64),
            metric_sem=result.epoch_mean_metric_sem.astype(np.float64),
            roi_channel_counts=result.roi_channel_counts.astype(np.int64),
            roi_subject_counts=result.roi_subject_counts.astype(np.int64),
        )

        axes_struct = make_struct(
            region=np.array(result.region_names, dtype=object),
            time_s=result.time_axis_s.astype(np.float64),
        )

        excluded_rois_struct = make_struct(
            region=np.array(list(result.excluded_rois.keys()), dtype=object),
            reason=np.array(list(result.excluded_rois.values()), dtype=object),
        )

        meta_kwargs: dict[str, object] = dict(
            analysis_level=np.str_("roi_group"),
            source_metric=np.str_(result.source_metric),
            condition_labels=np.array(list(result.condition_labels), dtype=object),
            p_value_correction_method=np.str_(result.p_value_correction_method),
            significance_alpha=float(result.significance_alpha),
            roi_mode=np.str_(result.roi_mode),
            atlas_name=np.str_(result.atlas_name or ""),
            included_roi_count=int(len(result.region_names)),
            excluded_roi_count=int(len(result.excluded_rois)),
            excluded_rois=excluded_rois_struct,
        )
        if "binning_mode" in result.metadata:
            meta_kwargs["binning_mode"] = np.str_(str(result.metadata["binning_mode"]))
        if "window_ms" in result.metadata:
            meta_kwargs["window_ms"] = float(result.metadata["window_ms"])
        if "n_bins" in result.metadata:
            meta_kwargs["n_bins"] = int(result.metadata["n_bins"])
        if "effective_n_bins" in result.metadata:
            meta_kwargs["effective_n_bins"] = int(result.metadata["effective_n_bins"])
        meta_struct = make_struct(**meta_kwargs)

        contributions_struct = make_struct(
            region=np.array([item.roi for item in result.contributions], dtype=object),
            subject=np.array([item.subject for item in result.contributions], dtype=object),
            channel=np.array([item.channel for item in result.contributions], dtype=object),
            source_stats_file=np.array(
                [item.source_stats_file for item in result.contributions], dtype=object
            ),
        )

        prov_struct = make_struct(
            source_trial_stats_files=np.array(result.source_trial_stats_files, dtype=object),
            source_electrodes_files=np.array(result.source_electrodes_files, dtype=object),
            pipeline_name=np.str_("trial_stats_group"),
            pipeline_version=np.str_(_package_version()),
        )

        if result.condition_a_contributions:
            n_rois = len(result.condition_a_contributions)
            cond_a_cell: np.ndarray = np.empty(n_rois, dtype=object)
            cond_b_cell: np.ndarray = np.empty(n_rois, dtype=object)
            labels_cell: np.ndarray = np.empty(n_rois, dtype=object)
            for i, (a, b, lbl) in enumerate(
                zip(
                    result.condition_a_contributions,
                    result.condition_b_contributions,
                    result.contribution_labels,
                )
            ):
                cond_a_cell[i] = a.astype(np.float64)
                cond_b_cell[i] = b.astype(np.float64)
                labels_cell[i] = np.array(lbl, dtype=object)
            contrib_epochs_struct = make_struct(
                condition_a=cond_a_cell,
                condition_b=cond_b_cell,
                labels=labels_cell,
                region=np.array(result.region_names, dtype=object),
            )
        else:
            contrib_epochs_struct = make_struct(
                condition_a=np.array([], dtype=object),
                condition_b=np.array([], dtype=object),
                labels=np.array([], dtype=object),
                region=np.array([], dtype=object),
            )

        data = make_struct(
            stats=stats_struct,
            means=means_struct,
            uncertainty=uncertainty_struct,
            summary_epoch=summary_struct,
            axes=axes_struct,
            meta=meta_struct,
            contributions=contributions_struct,
            contribution_epochs=contrib_epochs_struct,
            provenance=prov_struct,
        )
        savemat(str(output_path), {"data": data}, do_compression=True)

    def _write_hdf5(self, result: TrialStatsGroupProcessingResult, output_path: Path) -> None:
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
            means_grp.create_dataset(
                "condition_a_mean",
                data=result.condition_a_group_mean.astype(np.float64),
            )
            means_grp.create_dataset(
                "condition_a_sem",
                data=result.condition_a_group_sem.astype(np.float64),
            )
            means_grp.create_dataset(
                "condition_b_mean",
                data=result.condition_b_group_mean.astype(np.float64),
            )
            means_grp.create_dataset(
                "condition_b_sem",
                data=result.condition_b_group_sem.astype(np.float64),
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

            if result.condition_a_contributions:
                contrib_epochs_grp = fh.create_group("contribution_epochs")
                for roi_idx, roi_name in enumerate(result.region_names):
                    roi_grp = contrib_epochs_grp.create_group(roi_name)
                    roi_grp.create_dataset(
                        "condition_a",
                        data=result.condition_a_contributions[roi_idx].astype(np.float64),
                    )
                    roi_grp.create_dataset(
                        "condition_b",
                        data=result.condition_b_contributions[roi_idx].astype(np.float64),
                    )
                    roi_grp.create_dataset(
                        "labels",
                        data=np.array(result.contribution_labels[roi_idx], dtype=object),
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
