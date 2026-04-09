from __future__ import annotations

import csv
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat

from gin_bids_py_analysis.processing.base import BaseProcessingResult, BaseProcessingWriter
from gin_bids_py_analysis.processing.utils.matlab import make_struct, matlab_safe_name

from .result import TrialStatsProcessingResult


def _package_version() -> str:
    try:
        return version("gin-bids-py-analysis")
    except PackageNotFoundError:
        return "unknown"


class TrialStatsProcessingWriter(BaseProcessingWriter):
    """Write HDF5 trial statistics plus a companion TSV trial audit table."""

    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        if not isinstance(result, TrialStatsProcessingResult):
            raise TypeError(
                f"Expected TrialStatsProcessingResult, got {type(result).__name__!r}"
            )

        if self.params.output_format == "matlab":
            self._write_matlab(result, output_path)
        else:
            self._write_hdf5(result, output_path)
        self._write_trial_table_tsv(result, _trial_table_path(output_path))

    def _write_matlab(
        self,
        result: TrialStatsProcessingResult,
        output_path: Path,
    ) -> None:
        p_values_uncorrected = (
            result.p_values_uncorrected
            if result.p_values_uncorrected.size
            else result.p_values
        )
        significant_mask = (
            result.significant_mask
            if result.significant_mask.size
            else (np.isfinite(result.p_values) & (result.p_values < result.significance_alpha))
        )
        _validate_uncertainty_shapes(result)

        binning_mode = str(
            result.metadata.get(
                "binning_mode",
                "window_ms"
                if result.window_ms > 0
                else ("n_bins" if result.n_bins > 0 else "none"),
            )
        )
        effective_n_bins = int(
            result.metadata.get("effective_n_bins", len(result.time_axis_s))
        )

        cond_a = matlab_safe_name(result.condition_a)
        cond_b = matlab_safe_name(result.condition_b)

        stats_struct_fields: dict = dict(
            t_values=result.t_values.astype(np.float64),
            p_values=result.p_values.astype(np.float64),
            p_values_uncorrected=p_values_uncorrected.astype(np.float64),
            significant_mask=significant_mask.astype(np.uint8),
        )
        if result.channel_significant_mask is not None:
            stats_struct_fields["channel_significant_mask"] = result.channel_significant_mask.astype(np.uint8)
        stats_struct = make_struct(**stats_struct_fields)

        means_struct = make_struct(
            **{
                cond_a: result.condition_a_mean.astype(np.float64),
                cond_b: result.condition_b_mean.astype(np.float64),
                "difference": result.mean_difference.astype(np.float64),
            }
        )

        uncertainty_struct = make_struct(
            **{
                cond_a + "_sem": result.condition_a_sem.astype(np.float64),
                cond_b + "_sem": result.condition_b_sem.astype(np.float64),
                "difference_sem": result.difference_sem.astype(np.float64),
                "difference_ci95_low": result.difference_ci95_low.astype(np.float64),
                "difference_ci95_high": result.difference_ci95_high.astype(np.float64),
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
            trial_counts=np.array(
                [result.condition_a_trial_count, result.condition_b_trial_count],
                dtype=np.int64,
            ),
            trial_count_labels=np.array([result.condition_a, result.condition_b], dtype=object),
            sampling_frequency_hz=float(result.sfreq),
            p_value_correction_method=str(result.p_value_correction_method),
            significance_alpha=float(result.significance_alpha),
            analysis_level=str(result.analysis_level),
            atlas_name=str(result.atlas_name or ""),
            atlas_regions=np.array(result.atlas_regions, dtype=object),
            atlas_region_channel_map=atlas_map_struct,
            window_ms=float(result.window_ms),
            n_bins=int(result.n_bins),
            activity_zscore=str(result.activity_zscore),
            activity_baseline_tmin_s=float(result.activity_baseline_tmin_s),
            activity_baseline_tmax_s=float(result.activity_baseline_tmax_s),
            activity_baseline_scope=str(result.activity_baseline_scope),
            activity_baseline_remove_outlier_trial_means=bool(
                result.activity_baseline_remove_outlier_trial_means
            ),
            window_samples=int(result.metadata.get("window_samples", 0)),
            effective_n_bins=effective_n_bins,
            binning_mode=binning_mode,
            stats_valid=bool(result.stats_valid),
            n_permutations=int(result.metadata.get("n_permutations", 0)),
            channel_significance_mode=str(result.metadata.get("channel_significance_mode", "none")),
            ch_sig_duration_threshold_ms=float(
                result.metadata.get("channel_significance_duration_threshold_ms", 100.0)
            ),
        )

        trials_struct = make_struct(
            source_file=np.array(
                [str(trial.source_file.path) for trial in result.resolved_trials],
                dtype=object,
            ),
            anchor_event_code=np.array(
                [trial.anchor_event_code or "" for trial in result.resolved_trials],
                dtype=object,
            ),
            anchor_onset_s=np.array(
                [trial.anchor_onset_s for trial in result.resolved_trials],
                dtype=np.float64,
            ),
            resolved_label=np.array(
                [trial.label or "" for trial in result.resolved_trials],
                dtype=object,
            ),
            keep=np.array(
                [trial.keep for trial in result.resolved_trials],
                dtype=np.uint8,
            ),
            exclusion_reason=np.array(
                [trial.exclusion_reason or "" for trial in result.resolved_trials],
                dtype=object,
            ),
            trial_id=np.array(
                [trial.trial_id or "" for trial in result.resolved_trials],
                dtype=object,
            ),
            condition_inputs=np.array(
                [_condition_inputs_json(trial) for trial in result.resolved_trials],
                dtype=object,
            ),
            condition_resolution_reason=np.array(
                [
                    str(trial.metadata.get("condition_resolution_reason", ""))
                    for trial in result.resolved_trials
                ],
                dtype=object,
            ),
        )

        prov_struct = make_struct(
            source_ieeg_files=np.array(result.source_ieeg_files, dtype=object),
            source_table_files=np.array(result.source_table_files, dtype=object),
            source_electrodes_files=np.array(result.source_electrodes_files, dtype=object),
            pipeline_name=np.str_("trialstats"),
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

        data = make_struct(
            stats=stats_struct,
            means=means_struct,
            uncertainty=uncertainty_struct,
            axes=axes_struct,
            meta=meta_struct,
            trials=trials_struct,
            epochs=epochs_struct,
            provenance=prov_struct,
        )
        savemat(str(output_path), {"data": data}, do_compression=True, long_field_names=True)

    def _write_hdf5(
        self,
        result: TrialStatsProcessingResult,
        output_path: Path,
    ) -> None:
        str_dtype = h5py.string_dtype(encoding="utf-8")
        p_values_uncorrected = (
            result.p_values_uncorrected
            if result.p_values_uncorrected.size
            else result.p_values
        )
        significant_mask = (
            result.significant_mask
            if result.significant_mask.size
            else (np.isfinite(result.p_values) & (result.p_values < result.significance_alpha))
        )
        _validate_uncertainty_shapes(result)

        with h5py.File(output_path, "w") as fh:
            binning_mode = str(
                result.metadata.get(
                    "binning_mode",
                    "window_ms"
                    if result.window_ms > 0
                    else ("n_bins" if result.n_bins > 0 else "none"),
                )
            )
            effective_n_bins = int(
                result.metadata.get(
                    "effective_n_bins",
                    len(result.time_axis_s),
                )
            )
            window_samples = int(result.metadata.get("window_samples", 0))

            stats_grp = fh.create_group("stats")
            stats_grp.create_dataset(
                "t_values",
                data=result.t_values.astype(np.float64),
            )
            stats_grp.create_dataset(
                "p_values",
                data=result.p_values.astype(np.float64),
            )
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

            means_grp = fh.create_group("means")
            means_grp.create_dataset(
                result.condition_a,
                data=result.condition_a_mean.astype(np.float64),
            )
            means_grp.create_dataset(
                result.condition_b,
                data=result.condition_b_mean.astype(np.float64),
            )
            means_grp.create_dataset(
                "difference",
                data=result.mean_difference.astype(np.float64),
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
            uncertainty_grp.create_dataset(
                "difference_sem",
                data=result.difference_sem.astype(np.float64),
            )
            uncertainty_grp.create_dataset(
                "difference_ci95_low",
                data=result.difference_ci95_low.astype(np.float64),
            )
            uncertainty_grp.create_dataset(
                "difference_ci95_high",
                data=result.difference_ci95_high.astype(np.float64),
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
            meta_grp.create_dataset(
                "sampling_frequency_hz",
                data=float(result.sfreq),
            )
            meta_grp.create_dataset(
                "p_value_correction_method",
                data=str(result.p_value_correction_method),
                dtype=str_dtype,
            )
            meta_grp.create_dataset(
                "significance_alpha",
                data=float(result.significance_alpha),
            )
            meta_grp.create_dataset(
                "analysis_level",
                data=str(result.analysis_level),
                dtype=str_dtype,
            )
            meta_grp.create_dataset(
                "atlas_name",
                data=str(result.atlas_name or ""),
                dtype=str_dtype,
            )
            meta_grp.create_dataset(
                "atlas_regions",
                data=np.array(result.atlas_regions, dtype=object),
                dtype=str_dtype,
            )
            atlas_map_grp = meta_grp.create_group("atlas_region_channel_map")
            region_order = (
                result.channel_names
                if result.analysis_level == "roi"
                else []
            )
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
            atlas_map_grp.create_dataset(
                "region",
                data=np.array(region_pairs, dtype=object),
                dtype=str_dtype,
            )
            atlas_map_grp.create_dataset(
                "channel",
                data=np.array(channel_pairs, dtype=object),
                dtype=str_dtype,
            )
            meta_grp.create_dataset(
                "window_ms",
                data=float(result.window_ms),
            )
            meta_grp.create_dataset(
                "n_bins",
                data=int(result.n_bins),
            )
            meta_grp.create_dataset(
                "activity_zscore",
                data=str(result.activity_zscore),
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
            meta_grp.create_dataset(
                "activity_baseline_scope",
                data=str(result.activity_baseline_scope),
                dtype=str_dtype,
            )
            meta_grp.create_dataset(
                "activity_baseline_remove_outlier_trial_means",
                data=bool(result.activity_baseline_remove_outlier_trial_means),
            )
            meta_grp.create_dataset(
                "window_samples",
                data=window_samples,
            )
            meta_grp.create_dataset(
                "effective_n_bins",
                data=effective_n_bins,
            )
            meta_grp.create_dataset(
                "binning_mode",
                data=binning_mode,
                dtype=str_dtype,
            )
            meta_grp.create_dataset("stats_valid", data=bool(result.stats_valid))
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
                data=float(result.metadata.get("channel_significance_duration_threshold_ms", 100.0)),
            )

            trial_grp = fh.create_group("trials")
            trial_grp.create_dataset(
                "source_file",
                data=np.array(
                    [str(trial.source_file.path) for trial in result.resolved_trials],
                    dtype=object,
                ),
                dtype=str_dtype,
            )
            trial_grp.create_dataset(
                "anchor_event_code",
                data=np.array(
                    [trial.anchor_event_code or "" for trial in result.resolved_trials],
                    dtype=object,
                ),
                dtype=str_dtype,
            )
            trial_grp.create_dataset(
                "anchor_onset_s",
                data=np.array(
                    [trial.anchor_onset_s for trial in result.resolved_trials],
                    dtype=np.float64,
                ),
            )
            trial_grp.create_dataset(
                "resolved_label",
                data=np.array(
                    [trial.label or "" for trial in result.resolved_trials],
                    dtype=object,
                ),
                dtype=str_dtype,
            )
            trial_grp.create_dataset(
                "keep",
                data=np.array(
                    [trial.keep for trial in result.resolved_trials],
                    dtype=bool,
                ),
            )
            trial_grp.create_dataset(
                "exclusion_reason",
                data=np.array(
                    [trial.exclusion_reason or "" for trial in result.resolved_trials],
                    dtype=object,
                ),
                dtype=str_dtype,
            )
            trial_grp.create_dataset(
                "trial_id",
                data=np.array(
                    [trial.trial_id or "" for trial in result.resolved_trials],
                    dtype=object,
                ),
                dtype=str_dtype,
            )
            trial_grp.create_dataset(
                "condition_inputs",
                data=np.array(
                    [_condition_inputs_json(trial) for trial in result.resolved_trials],
                    dtype=object,
                ),
                dtype=str_dtype,
            )
            trial_grp.create_dataset(
                "condition_resolution_reason",
                data=np.array(
                    [
                        str(trial.metadata.get("condition_resolution_reason", ""))
                        for trial in result.resolved_trials
                    ],
                    dtype=object,
                ),
                dtype=str_dtype,
            )

            if result.condition_a_epochs.ndim == 3 and self.params.include_epochs:
                epochs_grp = fh.create_group("epochs")
                epochs_grp.create_dataset(
                    "condition_a",
                    data=result.condition_a_epochs.astype(np.float64),
                )
                epochs_grp.create_dataset(
                    "condition_b",
                    data=result.condition_b_epochs.astype(np.float64),
                )
                epochs_grp.create_dataset(
                    "channel",
                    data=np.array(result.channel_names, dtype=object),
                    dtype=str_dtype,
                )
                epochs_grp.create_dataset(
                    "time_s",
                    data=result.time_axis_s.astype(np.float64),
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
            prov_grp.create_dataset(
                "pipeline_name",
                data="trialstats",
                dtype=str_dtype,
            )
            prov_grp.create_dataset(
                "pipeline_version",
                data=_package_version(),
                dtype=str_dtype,
            )

    def _write_trial_table_tsv(
        self,
        result: TrialStatsProcessingResult,
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
                    "condition_inputs",
                    "condition_resolution_reason",
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
                        _condition_inputs_json(trial),
                        str(trial.metadata.get("condition_resolution_reason", "")),
                        str(trial.keep).lower(),
                        trial.exclusion_reason or "",
                    ]
                )


def _trial_table_path(output_path: Path) -> Path:
    tsv_path = output_path.with_suffix(".tsv")
    return tsv_path.with_name(tsv_path.name.replace("_stats.tsv", "_trials.tsv"))


def _condition_inputs_json(trial: object) -> str:
    metadata = getattr(trial, "metadata", {})
    value = metadata.get("condition_inputs", {}) if isinstance(metadata, dict) else {}
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def _validate_uncertainty_shapes(result: TrialStatsProcessingResult) -> None:
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
