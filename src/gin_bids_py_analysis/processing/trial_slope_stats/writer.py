from __future__ import annotations

import csv
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat

from gin_bids_py_analysis.processing.base import BaseProcessingResult, BaseProcessingWriter
from gin_bids_py_analysis.processing.utils.matlab import make_struct, matlab_safe_name

from .result import TrialSlopeStatsProcessingResult


def _package_version() -> str:
    try:
        return version("gin-bids-py-analysis")
    except PackageNotFoundError:
        return "unknown"


class TrialSlopeStatsProcessingWriter(BaseProcessingWriter):
    """Write trial slope statistics plus a companion TSV trial audit table."""

    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        if not isinstance(result, TrialSlopeStatsProcessingResult):
            raise TypeError(
                f"Expected TrialSlopeStatsProcessingResult, got {type(result).__name__!r}"
            )

        if self.params.output_format == "matlab":
            self._write_matlab(result, output_path)
        else:
            self._write_hdf5(result, output_path)
        self._write_trial_table_tsv(result, _trial_table_path(output_path))

    def _write_hdf5(
        self,
        result: TrialSlopeStatsProcessingResult,
        output_path: Path,
    ) -> None:
        str_dtype = h5py.string_dtype(encoding="utf-8")
        _validate_shape_consistency(result)

        with h5py.File(output_path, "w") as fh:
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
                "condition_a_values",
                data=np.asarray(result.condition_a_predictor_values, dtype=np.float64),
            )
            predictor_grp.create_dataset(
                "condition_b_values",
                data=np.asarray(result.condition_b_predictor_values, dtype=np.float64),
            )

            means_grp = fh.create_group("means")
            means_grp.create_dataset(
                result.condition_a,
                data=result.condition_a_mean.astype(np.float64),
            )
            means_grp.create_dataset(
                result.condition_b,
                data=result.condition_b_mean.astype(np.float64),
            )

            uncertainty_grp = fh.create_group("uncertainty")
            uncertainty_grp.create_dataset(
                result.condition_a + "_sem",
                data=result.condition_a_sem.astype(np.float64),
            )
            uncertainty_grp.create_dataset(
                result.condition_b + "_sem",
                data=result.condition_b_sem.astype(np.float64),
            )

            axes_grp = fh.create_group("axes")
            primary_axis_name = "region" if result.analysis_level == "roi" else "channel"
            axes_grp.create_dataset(
                primary_axis_name,
                data=np.array(result.channel_names, dtype=object),
                dtype=str_dtype,
            )
            axes_grp.create_dataset(
                "time_s",
                data=result.time_axis_s.astype(np.float64),
            )

            meta_grp = fh.create_group("meta")
            meta_grp.create_dataset("analysis_type", data="slope_regression", dtype=str_dtype)
            meta_grp.create_dataset(
                "predictor",
                data=str(result.predictor),
                dtype=str_dtype,
            )
            meta_grp.create_dataset(
                "predictor_scaling",
                data=str(result.predictor_scaling),
                dtype=str_dtype,
            )
            meta_grp.create_dataset(
                "trial_counts",
                data=np.array(
                    [result.condition_a_trial_count, result.condition_b_trial_count],
                    dtype=np.int64,
                ),
            )
            meta_grp.create_dataset(
                "trial_count_labels",
                data=np.array([result.condition_a, result.condition_b], dtype=object),
                dtype=str_dtype,
            )
            meta_grp.create_dataset("sampling_frequency_hz", data=float(result.sfreq))
            meta_grp.create_dataset(
                "p_value_correction_method",
                data=str(result.p_value_correction_method),
                dtype=str_dtype,
            )
            meta_grp.create_dataset("significance_alpha", data=float(result.significance_alpha))
            meta_grp.create_dataset("analysis_level", data=str(result.analysis_level), dtype=str_dtype)
            meta_grp.create_dataset("atlas_name", data=str(result.atlas_name or ""), dtype=str_dtype)
            meta_grp.create_dataset(
                "atlas_regions",
                data=np.array(result.atlas_regions, dtype=object),
                dtype=str_dtype,
            )
            atlas_map_grp = meta_grp.create_group("atlas_region_channel_map")
            region_order = result.channel_names if result.analysis_level == "roi" else []
            atlas_map_grp.create_dataset(
                "region_order",
                data=np.array(region_order, dtype=object),
                dtype=str_dtype,
            )
            region_pairs: list[str] = []
            channel_pairs: list[str] = []
            ordered_regions = list(region_order)
            for region in result.region_channels:
                if region not in ordered_regions:
                    ordered_regions.append(region)
            for region in ordered_regions:
                for channel in result.region_channels.get(region, []):
                    region_pairs.append(str(region))
                    channel_pairs.append(str(channel))
            atlas_map_grp.create_dataset("region", data=np.array(region_pairs, dtype=object), dtype=str_dtype)
            atlas_map_grp.create_dataset("channel", data=np.array(channel_pairs, dtype=object), dtype=str_dtype)

            meta_grp.create_dataset("window_ms", data=float(result.window_ms))
            meta_grp.create_dataset("n_bins", data=int(result.n_bins))
            meta_grp.create_dataset(
                "activity_scaling",
                data=str(result.activity_scaling),
                dtype=str_dtype,
            )
            meta_grp.create_dataset(
                "activity_baseline_tmin_s",
                data=float(result.activity_baseline_tmin_s),
            )
            meta_grp.create_dataset(
                "activity_baseline_tmax_s",
                data=float(result.activity_baseline_tmax_s),
            )
            meta_grp.create_dataset("window_samples", data=int(result.metadata.get("window_samples", 0)))
            meta_grp.create_dataset(
                "effective_n_bins",
                data=int(result.metadata.get("effective_n_bins", len(result.time_axis_s))),
            )
            meta_grp.create_dataset(
                "binning_mode",
                data=str(
                    result.metadata.get(
                        "binning_mode",
                        "window_ms"
                        if result.window_ms > 0
                        else ("n_bins" if result.n_bins > 0 else "none"),
                    )
                ),
                dtype=str_dtype,
            )
            meta_grp.create_dataset("condition_a_stats_valid", data=bool(result.condition_a_stats_valid))
            meta_grp.create_dataset("condition_b_stats_valid", data=bool(result.condition_b_stats_valid))
            meta_grp.create_dataset("stats_valid", data=bool(result.stats_valid))

            trial_grp = fh.create_group("trials")
            trial_grp.create_dataset(
                "source_file",
                data=np.array([str(trial.source_file.path) for trial in result.resolved_trials], dtype=object),
                dtype=str_dtype,
            )
            trial_grp.create_dataset(
                "anchor_event_code",
                data=np.array([trial.anchor_event_code or "" for trial in result.resolved_trials], dtype=object),
                dtype=str_dtype,
            )
            trial_grp.create_dataset(
                "anchor_onset_s",
                data=np.array([trial.anchor_onset_s for trial in result.resolved_trials], dtype=np.float64),
            )
            trial_grp.create_dataset(
                "resolved_label",
                data=np.array([trial.label or "" for trial in result.resolved_trials], dtype=object),
                dtype=str_dtype,
            )
            trial_grp.create_dataset(
                "keep",
                data=np.array([trial.keep for trial in result.resolved_trials], dtype=bool),
            )
            trial_grp.create_dataset(
                "exclusion_reason",
                data=np.array([trial.exclusion_reason or "" for trial in result.resolved_trials], dtype=object),
                dtype=str_dtype,
            )
            trial_grp.create_dataset(
                "trial_id",
                data=np.array([trial.trial_id or "" for trial in result.resolved_trials], dtype=object),
                dtype=str_dtype,
            )
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
                    [
                        _to_float_or_nan(trial.metadata.get("predictor_value"))
                        for trial in result.resolved_trials
                    ],
                    dtype=np.float64,
                ),
            )

            if result.condition_a_epochs.ndim == 3 and self.params.include_epochs:
                epochs_grp = fh.create_group("epochs")
                epochs_grp.create_dataset("condition_a", data=result.condition_a_epochs.astype(np.float64))
                epochs_grp.create_dataset("condition_b", data=result.condition_b_epochs.astype(np.float64))
                epochs_grp.create_dataset(
                    "channel",
                    data=np.array(result.channel_names, dtype=object),
                    dtype=str_dtype,
                )
                epochs_grp.create_dataset("time_s", data=result.time_axis_s.astype(np.float64))

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

            prov_grp = fh.create_group("provenance")
            prov_grp.create_dataset(
                "source_ieeg_files",
                data=np.array(result.source_ieeg_files, dtype=object),
                dtype=str_dtype,
            )
            prov_grp.create_dataset(
                "source_table_files",
                data=np.array(result.source_table_files, dtype=object),
                dtype=str_dtype,
            )
            prov_grp.create_dataset(
                "source_electrodes_files",
                data=np.array(result.source_electrodes_files, dtype=object),
                dtype=str_dtype,
            )
            prov_grp.create_dataset("pipeline_name", data="trialslopestats", dtype=str_dtype)
            prov_grp.create_dataset("pipeline_version", data=_package_version(), dtype=str_dtype)

    def _write_matlab(
        self,
        result: TrialSlopeStatsProcessingResult,
        output_path: Path,
    ) -> None:
        _validate_shape_consistency(result)

        cond_a = matlab_safe_name(result.condition_a)
        cond_b = matlab_safe_name(result.condition_b)

        regression_struct = make_struct(
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
        )

        predictor_struct = make_struct(
            condition_a_values=np.asarray(result.condition_a_predictor_values, dtype=np.float64),
            condition_b_values=np.asarray(result.condition_b_predictor_values, dtype=np.float64),
        )

        means_struct = make_struct(
            **{
                cond_a: result.condition_a_mean.astype(np.float64),
                cond_b: result.condition_b_mean.astype(np.float64),
            }
        )

        uncertainty_struct = make_struct(
            **{
                cond_a + "_sem": result.condition_a_sem.astype(np.float64),
                cond_b + "_sem": result.condition_b_sem.astype(np.float64),
            }
        )

        primary_axis_name = "region" if result.analysis_level == "roi" else "channel"
        axes_struct = make_struct(
            **{
                primary_axis_name: np.array(result.channel_names, dtype=object),
                "time_s": result.time_axis_s.astype(np.float64),
            }
        )

        region_order = result.channel_names if result.analysis_level == "roi" else []
        ordered_regions = list(region_order)
        region_pairs: list[str] = []
        channel_pairs: list[str] = []
        for region in result.region_channels:
            if region not in ordered_regions:
                ordered_regions.append(region)
        for region in ordered_regions:
            for channel in result.region_channels.get(region, []):
                region_pairs.append(str(region))
                channel_pairs.append(str(channel))
        atlas_map_struct = make_struct(
            region_order=np.array(region_order, dtype=object),
            region=np.array(region_pairs, dtype=object),
            channel=np.array(channel_pairs, dtype=object),
        )

        meta_struct = make_struct(
            analysis_type=np.str_("slope_regression"),
            predictor=np.str_(result.predictor),
            predictor_scaling=np.str_(result.predictor_scaling),
            trial_counts=np.array(
                [result.condition_a_trial_count, result.condition_b_trial_count],
                dtype=np.int64,
            ),
            trial_count_labels=np.array([result.condition_a, result.condition_b], dtype=object),
            sampling_frequency_hz=float(result.sfreq),
            p_value_correction_method=np.str_(result.p_value_correction_method),
            significance_alpha=float(result.significance_alpha),
            analysis_level=np.str_(result.analysis_level),
            atlas_name=np.str_(result.atlas_name or ""),
            atlas_regions=np.array(result.atlas_regions, dtype=object),
            atlas_region_channel_map=atlas_map_struct,
            window_ms=float(result.window_ms),
            n_bins=int(result.n_bins),
            activity_scaling=np.str_(result.activity_scaling),
            activity_baseline_tmin_s=float(result.activity_baseline_tmin_s),
            activity_baseline_tmax_s=float(result.activity_baseline_tmax_s),
            window_samples=int(result.metadata.get("window_samples", 0)),
            effective_n_bins=int(result.metadata.get("effective_n_bins", len(result.time_axis_s))),
            binning_mode=np.str_(
                result.metadata.get(
                    "binning_mode",
                    "window_ms" if result.window_ms > 0 else ("n_bins" if result.n_bins > 0 else "none"),
                )
            ),
            condition_a_stats_valid=bool(result.condition_a_stats_valid),
            condition_b_stats_valid=bool(result.condition_b_stats_valid),
            stats_valid=bool(result.stats_valid),
        )

        trials_struct = make_struct(
            source_file=np.array([str(trial.source_file.path) for trial in result.resolved_trials], dtype=object),
            anchor_event_code=np.array([trial.anchor_event_code or "" for trial in result.resolved_trials], dtype=object),
            anchor_onset_s=np.array([trial.anchor_onset_s for trial in result.resolved_trials], dtype=np.float64),
            resolved_label=np.array([trial.label or "" for trial in result.resolved_trials], dtype=object),
            keep=np.array([trial.keep for trial in result.resolved_trials], dtype=np.uint8),
            exclusion_reason=np.array([trial.exclusion_reason or "" for trial in result.resolved_trials], dtype=object),
            trial_id=np.array([trial.trial_id or "" for trial in result.resolved_trials], dtype=object),
            predictor_raw=np.array([str(trial.metadata.get("predictor_raw", "")) for trial in result.resolved_trials], dtype=object),
            predictor_value=np.array(
                [_to_float_or_nan(trial.metadata.get("predictor_value")) for trial in result.resolved_trials],
                dtype=np.float64,
            ),
        )

        prov_struct = make_struct(
            source_ieeg_files=np.array(result.source_ieeg_files, dtype=object),
            source_table_files=np.array(result.source_table_files, dtype=object),
            source_electrodes_files=np.array(result.source_electrodes_files, dtype=object),
            pipeline_name=np.str_("trialslopestats"),
            pipeline_version=np.str_(_package_version()),
        )

        epochs_struct: object
        if result.condition_a_epochs.ndim == 3 and self.params.include_epochs:
            epochs_struct = make_struct(
                condition_a=result.condition_a_epochs.astype(np.float64),
                condition_b=result.condition_b_epochs.astype(np.float64),
                channel=np.array(result.channel_names, dtype=object),
                time_s=result.time_axis_s.astype(np.float64),
            )
        else:
            epochs_struct = make_struct(
                condition_a=np.empty((0, 0, 0), dtype=np.float64),
                condition_b=np.empty((0, 0, 0), dtype=np.float64),
                channel=np.array([], dtype=object),
                time_s=np.array([], dtype=np.float64),
            )

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

        data = make_struct(
            regression=regression_struct,
            predictor=predictor_struct,
            means=means_struct,
            uncertainty=uncertainty_struct,
            axes=axes_struct,
            meta=meta_struct,
            trials=trials_struct,
            epochs=epochs_struct,
            scatter=scatter_struct,
            provenance=prov_struct,
        )
        savemat(str(output_path), {"data": data}, do_compression=True)

    def _write_trial_table_tsv(
        self,
        result: TrialSlopeStatsProcessingResult,
        output_path: Path,
    ) -> None:
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(
                [
                    "source_file",
                    "anchor_event_index",
                    "anchor_event_code",
                    "anchor_onset_s",
                    "trial_id",
                    "resolved_label",
                    "predictor_raw",
                    "predictor_value",
                    "keep",
                    "exclusion_reason",
                ]
            )
            for trial in result.resolved_trials:
                writer.writerow(
                    [
                        str(trial.source_file.path),
                        trial.anchor_event_index,
                        trial.anchor_event_code or "",
                        trial.anchor_onset_s,
                        trial.trial_id or "",
                        trial.label or "",
                        str(trial.metadata.get("predictor_raw", "")),
                        _to_float_or_nan(trial.metadata.get("predictor_value")),
                        str(trial.keep).lower(),
                        trial.exclusion_reason or "",
                    ]
                )


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


def _trial_table_path(output_path: Path) -> Path:
    tsv_path = output_path.with_suffix(".tsv")
    return tsv_path.with_name(tsv_path.name.replace("_stats.tsv", "_trials.tsv"))


def _validate_shape_consistency(result: TrialSlopeStatsProcessingResult) -> None:
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


