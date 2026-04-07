from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat

from gin_bids_py_analysis.processing.base import BaseProcessingResult, BaseProcessingWriter
from gin_bids_py_analysis.processing.utils.matlab import make_struct

from .result import TrialSlopeStatsGroupProcessingResult


def _package_version() -> str:
    try:
        return version("gin-bids-py-analysis")
    except PackageNotFoundError:
        return "unknown"


class TrialSlopeStatsGroupProcessingWriter(BaseProcessingWriter):
    """Write group-level trial-slope-stats ROI outputs to HDF5 or MATLAB."""

    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        if not isinstance(result, TrialSlopeStatsGroupProcessingResult):
            raise TypeError(
                f"Expected TrialSlopeStatsGroupProcessingResult, got {type(result).__name__!r}"
            )
        if self.params.output_format == "matlab":
            self._write_matlab(result, output_path)
        else:
            self._write_hdf5(result, output_path)

    # ------------------------------------------------------------------
    # HDF5
    # ------------------------------------------------------------------

    def _write_hdf5(
        self,
        result: TrialSlopeStatsGroupProcessingResult,
        output_path: Path,
    ) -> None:
        str_dtype = h5py.string_dtype(encoding="utf-8")

        with h5py.File(output_path, "w") as fh:
            # --- /regression (condition_a vs condition_b slope t-test) ---
            reg = fh.create_group("regression")
            reg.create_dataset("t_values", data=result.slope_t_values.astype(np.float64))
            reg.create_dataset("p_values", data=result.slope_p_values.astype(np.float64))
            reg.create_dataset("p_values_uncorrected", data=result.slope_p_values_uncorrected.astype(np.float64))
            reg.create_dataset("significant_mask", data=result.slope_significant_mask.astype(bool))
            reg.create_dataset("slope_mean_a", data=result.condition_a_slope_mean.astype(np.float64))
            reg.create_dataset("slope_sem_a", data=result.condition_a_slope_sem.astype(np.float64))
            reg.create_dataset("slope_mean_b", data=result.condition_b_slope_mean.astype(np.float64))
            reg.create_dataset("slope_sem_b", data=result.condition_b_slope_sem.astype(np.float64))
            ep_reg = reg.create_group("epoch_summary")
            ep_reg.create_dataset("t", data=result.epoch_slope_t.astype(np.float64))
            ep_reg.create_dataset("p", data=result.epoch_slope_p.astype(np.float64))
            ep_reg.create_dataset("df", data=result.epoch_slope_df.astype(np.float64))

            # --- /activity (condition_a vs condition_b activity t-test) ---
            act = fh.create_group("activity")
            act.create_dataset("t_values", data=result.activity_t_values.astype(np.float64))
            act.create_dataset("p_values", data=result.activity_p_values.astype(np.float64))
            act.create_dataset("p_values_uncorrected", data=result.activity_p_values_uncorrected.astype(np.float64))
            act.create_dataset("significant_mask", data=result.activity_significant_mask.astype(bool))
            ep_act = act.create_group("epoch_summary")
            ep_act.create_dataset("t", data=result.epoch_activity_t.astype(np.float64))
            ep_act.create_dataset("p", data=result.epoch_activity_p.astype(np.float64))
            ep_act.create_dataset("df", data=result.epoch_activity_df.astype(np.float64))

            # --- /means ---
            means = fh.create_group("means")
            means.create_dataset("condition_a_mean", data=result.condition_a_activity_mean.astype(np.float64))
            means.create_dataset("condition_a_sem", data=result.condition_a_activity_sem.astype(np.float64))
            means.create_dataset("condition_b_mean", data=result.condition_b_activity_mean.astype(np.float64))
            means.create_dataset("condition_b_sem", data=result.condition_b_activity_sem.astype(np.float64))

            # --- /r_values ---
            r_vals = fh.create_group("r_values")
            r_vals.create_dataset("condition_a_mean", data=result.condition_a_r_value_mean.astype(np.float64))
            r_vals.create_dataset("condition_a_sem", data=result.condition_a_r_value_sem.astype(np.float64))
            r_vals.create_dataset("condition_b_mean", data=result.condition_b_r_value_mean.astype(np.float64))
            r_vals.create_dataset("condition_b_sem", data=result.condition_b_r_value_sem.astype(np.float64))

            # --- /axes ---
            axes = fh.create_group("axes")
            axes.create_dataset(
                "region",
                data=np.array(result.region_names, dtype=object),
                dtype=str_dtype,
            )
            axes.create_dataset("time_s", data=result.time_axis_s.astype(np.float64))

            # --- /meta ---
            meta = fh.create_group("meta")
            meta.create_dataset("analysis_level", data="roi_group", dtype=str_dtype)
            meta.create_dataset(
                "condition_labels",
                data=np.array(list(result.condition_labels), dtype=object),
                dtype=str_dtype,
            )
            meta.create_dataset(
                "p_value_correction_method",
                data=str(result.p_value_correction_method),
                dtype=str_dtype,
            )
            meta.create_dataset("significance_alpha", data=float(result.significance_alpha))
            meta.create_dataset("roi_mode", data=str(result.roi_mode), dtype=str_dtype)
            meta.create_dataset(
                "atlas_name", data=str(result.atlas_name or ""), dtype=str_dtype
            )
            meta.create_dataset(
                "roi_channel_counts", data=result.roi_channel_counts.astype(np.int64)
            )
            meta.create_dataset(
                "roi_subject_counts", data=result.roi_subject_counts.astype(np.int64)
            )
            meta.create_dataset("included_roi_count", data=int(len(result.region_names)))
            meta.create_dataset("excluded_roi_count", data=int(len(result.excluded_rois)))
            for key in ("binning_mode", "window_ms", "n_bins", "effective_n_bins"):
                val = result.metadata.get(key)
                if val is not None:
                    if isinstance(val, str):
                        meta.create_dataset(key, data=val, dtype=str_dtype)
                    elif isinstance(val, float):
                        meta.create_dataset(key, data=float(val))
                    else:
                        meta.create_dataset(key, data=int(val))

            # --- /excluded_rois ---
            excl = fh.create_group("excluded_rois")
            excl.create_dataset(
                "name",
                data=np.array(list(result.excluded_rois.keys()), dtype=object),
                dtype=str_dtype,
            )
            excl.create_dataset(
                "reason",
                data=np.array(list(result.excluded_rois.values()), dtype=object),
                dtype=str_dtype,
            )

            # --- /contributions ---
            contribs = fh.create_group("contributions")
            contribs.create_dataset(
                "roi",
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
                data=np.array(
                    [c.source_stats_file for c in result.contributions], dtype=object
                ),
                dtype=str_dtype,
            )

            # --- /contribution_samples (ragged arrays as variable-length HDF5) ---
            if result.condition_a_slope_contributions:
                _write_ragged_hdf5(
                    fh.create_group("contribution_samples"),
                    condition_a_slope=result.condition_a_slope_contributions,
                    condition_b_slope=result.condition_b_slope_contributions,
                    condition_a_activity=result.condition_a_activity_contributions,
                    condition_b_activity=result.condition_b_activity_contributions,
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

            # --- /provenance ---
            prov = fh.create_group("provenance")
            prov.create_dataset(
                "source_trial_slope_stats_files",
                data=np.array(result.source_trial_slope_stats_files, dtype=object),
                dtype=str_dtype,
            )
            prov.create_dataset(
                "source_electrodes_files",
                data=np.array(result.source_electrodes_files, dtype=object),
                dtype=str_dtype,
            )
            prov.create_dataset("pipeline_name", data="trial_slope_stats_group", dtype=str_dtype)
            prov.create_dataset(
                "pipeline_version", data=_package_version(), dtype=str_dtype
            )

    # ------------------------------------------------------------------
    # MATLAB
    # ------------------------------------------------------------------

    def _write_matlab(
        self,
        result: TrialSlopeStatsGroupProcessingResult,
        output_path: Path,
    ) -> None:
        epoch_slope_struct = make_struct(
            t=result.epoch_slope_t.astype(np.float64),
            p=result.epoch_slope_p.astype(np.float64),
            df=result.epoch_slope_df.astype(np.float64),
        )
        regression_struct = make_struct(
            t_values=result.slope_t_values.astype(np.float64),
            p_values=result.slope_p_values.astype(np.float64),
            p_values_uncorrected=result.slope_p_values_uncorrected.astype(np.float64),
            significant_mask=result.slope_significant_mask.astype(np.uint8),
            slope_mean_a=result.condition_a_slope_mean.astype(np.float64),
            slope_sem_a=result.condition_a_slope_sem.astype(np.float64),
            slope_mean_b=result.condition_b_slope_mean.astype(np.float64),
            slope_sem_b=result.condition_b_slope_sem.astype(np.float64),
            epoch_summary=epoch_slope_struct,
        )

        epoch_activity_struct = make_struct(
            t=result.epoch_activity_t.astype(np.float64),
            p=result.epoch_activity_p.astype(np.float64),
            df=result.epoch_activity_df.astype(np.float64),
        )
        activity_struct = make_struct(
            t_values=result.activity_t_values.astype(np.float64),
            p_values=result.activity_p_values.astype(np.float64),
            p_values_uncorrected=result.activity_p_values_uncorrected.astype(np.float64),
            significant_mask=result.activity_significant_mask.astype(np.uint8),
            epoch_summary=epoch_activity_struct,
        )

        means_struct = make_struct(
            condition_a_mean=result.condition_a_activity_mean.astype(np.float64),
            condition_a_sem=result.condition_a_activity_sem.astype(np.float64),
            condition_b_mean=result.condition_b_activity_mean.astype(np.float64),
            condition_b_sem=result.condition_b_activity_sem.astype(np.float64),
        )

        r_values_struct = make_struct(
            condition_a_mean=result.condition_a_r_value_mean.astype(np.float64),
            condition_a_sem=result.condition_a_r_value_sem.astype(np.float64),
            condition_b_mean=result.condition_b_r_value_mean.astype(np.float64),
            condition_b_sem=result.condition_b_r_value_sem.astype(np.float64),
        )

        axes_struct = make_struct(
            region=np.array(result.region_names, dtype=object),
            time_s=result.time_axis_s.astype(np.float64),
        )

        excluded_rois_struct = make_struct(
            name=np.array(list(result.excluded_rois.keys()), dtype=object),
            reason=np.array(list(result.excluded_rois.values()), dtype=object),
        )

        meta_kwargs: dict[str, object] = dict(
            analysis_level=np.str_("roi_group"),
            condition_labels=np.array(list(result.condition_labels), dtype=object),
            p_value_correction_method=np.str_(result.p_value_correction_method),
            significance_alpha=float(result.significance_alpha),
            roi_mode=np.str_(result.roi_mode),
            atlas_name=np.str_(result.atlas_name or ""),
            roi_channel_counts=result.roi_channel_counts.astype(np.int64),
            roi_subject_counts=result.roi_subject_counts.astype(np.int64),
            included_roi_count=int(len(result.region_names)),
            excluded_roi_count=int(len(result.excluded_rois)),
            excluded_rois=excluded_rois_struct,
        )
        for key in ("binning_mode", "window_ms", "n_bins", "effective_n_bins"):
            val = result.metadata.get(key)
            if val is not None:
                meta_kwargs[key] = val
        meta_struct = make_struct(**meta_kwargs)

        contributions_struct = make_struct(
            roi=np.array([c.roi for c in result.contributions], dtype=object),
            subject=np.array([c.subject for c in result.contributions], dtype=object),
            channel=np.array([c.channel for c in result.contributions], dtype=object),
            source_stats_file=np.array(
                [c.source_stats_file for c in result.contributions], dtype=object
            ),
        )

        prov_struct = make_struct(
            source_trial_slope_stats_files=np.array(
                result.source_trial_slope_stats_files, dtype=object
            ),
            source_electrodes_files=np.array(result.source_electrodes_files, dtype=object),
            pipeline_name=np.str_("trial_slope_stats_group"),
            pipeline_version=np.str_(_package_version()),
        )

        n_rois = len(result.region_names)
        if result.condition_a_slope_contributions:
            slope_a_cell: np.ndarray = np.empty(n_rois, dtype=object)
            slope_b_cell: np.ndarray = np.empty(n_rois, dtype=object)
            activity_a_cell: np.ndarray = np.empty(n_rois, dtype=object)
            activity_b_cell: np.ndarray = np.empty(n_rois, dtype=object)
            labels_cell: np.ndarray = np.empty(n_rois, dtype=object)
            for i in range(n_rois):
                slope_a_cell[i] = result.condition_a_slope_contributions[i].astype(np.float64)
                slope_b_cell[i] = result.condition_b_slope_contributions[i].astype(np.float64)
                activity_a_cell[i] = result.condition_a_activity_contributions[i].astype(np.float64)
                activity_b_cell[i] = result.condition_b_activity_contributions[i].astype(np.float64)
                labels_cell[i] = np.array(result.contribution_labels[i], dtype=object)
            contrib_samples_struct = make_struct(
                condition_a_slope=slope_a_cell,
                condition_b_slope=slope_b_cell,
                condition_a_activity=activity_a_cell,
                condition_b_activity=activity_b_cell,
                labels=labels_cell,
                region=np.array(result.region_names, dtype=object),
            )
        else:
            contrib_samples_struct = make_struct(
                condition_a_slope=np.array([], dtype=object),
                condition_b_slope=np.array([], dtype=object),
                condition_a_activity=np.array([], dtype=object),
                condition_b_activity=np.array([], dtype=object),
                labels=np.array([], dtype=object),
                region=np.array([], dtype=object),
            )

        if result.condition_a_scatter_predictor:
            scatter_pred_a_cell: np.ndarray = np.empty(n_rois, dtype=object)
            scatter_act_a_cell: np.ndarray = np.empty(n_rois, dtype=object)
            scatter_pred_b_cell: np.ndarray = np.empty(n_rois, dtype=object)
            scatter_act_b_cell: np.ndarray = np.empty(n_rois, dtype=object)
            for i in range(n_rois):
                scatter_pred_a_cell[i] = np.asarray(result.condition_a_scatter_predictor[i], dtype=np.float64)
                scatter_act_a_cell[i] = np.asarray(result.condition_a_scatter_activity[i], dtype=np.float64)
                scatter_pred_b_cell[i] = np.asarray(result.condition_b_scatter_predictor[i], dtype=np.float64)
                scatter_act_b_cell[i] = np.asarray(result.condition_b_scatter_activity[i], dtype=np.float64)
            scatter_data_struct = make_struct(
                condition_a_predictor=scatter_pred_a_cell,
                condition_a_activity=scatter_act_a_cell,
                condition_b_predictor=scatter_pred_b_cell,
                condition_b_activity=scatter_act_b_cell,
                region=np.array(result.region_names, dtype=object),
            )
        else:
            scatter_data_struct = make_struct(
                condition_a_predictor=np.array([], dtype=object),
                condition_a_activity=np.array([], dtype=object),
                condition_b_predictor=np.array([], dtype=object),
                condition_b_activity=np.array([], dtype=object),
                region=np.array([], dtype=object),
            )

        data = make_struct(
            regression=regression_struct,
            activity=activity_struct,
            means=means_struct,
            r_values=r_values_struct,
            axes=axes_struct,
            meta=meta_struct,
            contributions=contributions_struct,
            contribution_samples=contrib_samples_struct,
            scatter_data=scatter_data_struct,
            provenance=prov_struct,
        )
        savemat(str(output_path), {"data": data}, do_compression=True)


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _write_ragged_hdf5(
    grp: h5py.Group,
    *,
    condition_a_slope: list,
    condition_b_slope: list,
    condition_a_activity: list,
    condition_b_activity: list,
    labels: list,
    region_names: list[str],
    str_dtype: object,
) -> None:
    """Write per-ROI contribution sample arrays as indexed HDF5 datasets."""
    n_rois = len(region_names)
    grp.create_dataset(
        "region_names",
        data=np.array(region_names, dtype=object),
        dtype=str_dtype,
    )
    for i in range(n_rois):
        roi_name = region_names[i]
        roi_grp = grp.create_group(str(i))
        roi_grp.attrs["roi"] = roi_name
        roi_grp.create_dataset(
            "condition_a_slope",
            data=np.asarray(condition_a_slope[i], dtype=np.float64),
        )
        roi_grp.create_dataset(
            "condition_b_slope",
            data=np.asarray(condition_b_slope[i], dtype=np.float64),
        )
        roi_grp.create_dataset(
            "condition_a_activity",
            data=np.asarray(condition_a_activity[i], dtype=np.float64),
        )
        roi_grp.create_dataset(
            "condition_b_activity",
            data=np.asarray(condition_b_activity[i], dtype=np.float64),
        )
        roi_grp.create_dataset(
            "labels",
            data=np.array(labels[i], dtype=object),
            dtype=str_dtype,
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
    """Write per-ROI scatter arrays (predictor × epoch-mean-activity) as indexed HDF5 datasets."""
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
