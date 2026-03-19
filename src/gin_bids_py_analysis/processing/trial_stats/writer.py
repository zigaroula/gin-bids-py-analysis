from __future__ import annotations

import csv
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py
import numpy as np

from gin_bids_py_analysis.processing.base import BaseProcessingResult, BaseProcessingWriter

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

        self._write_hdf5(result, output_path)
        self._write_trial_table_tsv(result, _trial_table_path(output_path))

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
            meta_grp.create_dataset(
                "window_ms",
                data=float(result.window_ms),
            )
            meta_grp.create_dataset(
                "n_bins",
                data=int(result.n_bins),
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
                        str(trial.keep).lower(),
                        trial.exclusion_reason or "",
                    ]
                )


def _trial_table_path(output_path: Path) -> Path:
    tsv_path = output_path.with_suffix(".tsv")
    return tsv_path.with_name(tsv_path.name.replace("_stats.tsv", "_trials.tsv"))
