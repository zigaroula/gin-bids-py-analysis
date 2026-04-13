from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import savemat

from gin_bids_py_analysis.processing.base import BaseProcessingResult, BaseProcessingWriter
from gin_bids_py_analysis.processing.utils.matlab import make_struct, matlab_safe_name

from .result import BaseTrialStatsProcessingResult


def _package_version() -> str:
    try:
        return version("gin-bids-py-analysis")
    except PackageNotFoundError:
        return "unknown"


class BaseTrialStatsProcessingWriter(BaseProcessingWriter, ABC):
    """Shared writer helpers for subject-level trial statistics outputs."""

    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        if not isinstance(result, BaseTrialStatsProcessingResult):
            raise TypeError(
                f"Expected BaseTrialStatsProcessingResult, got {type(result).__name__!r}"
            )

        self._validate_result(result)
        if self.params.output_format == "matlab":
            self._write_matlab(result, output_path)
        else:
            self._write_hdf5(result, output_path)
        self._write_trial_table_tsv(result, _trial_table_path(output_path))

    def _write_hdf5(
        self,
        result: BaseTrialStatsProcessingResult,
        output_path: Path,
    ) -> None:
        str_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(output_path, "w") as fh:
            self._write_shared_hdf5_groups(fh, result, str_dtype)
            self._write_hdf5_specific(fh, result, str_dtype)

    def _write_matlab(
        self,
        result: BaseTrialStatsProcessingResult,
        output_path: Path,
    ) -> None:
        data = self._build_shared_matlab_groups(result)
        data.update(self._build_matlab_specific(result))
        savemat(
            str(output_path),
            {"data": make_struct(**data)},
            do_compression=True,
            long_field_names=True,
        )

    def _write_trial_table_tsv(
        self,
        result: BaseTrialStatsProcessingResult,
        output_path: Path,
    ) -> None:
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(
                self._trial_table_prefix_header()
                + self._trial_table_extra_header()
                + self._trial_table_suffix_header()
            )
            for trial in result.resolved_trials:
                writer.writerow(
                    self._trial_table_prefix_row(trial)
                    + self._trial_table_extra_row(trial)
                    + self._trial_table_suffix_row(trial)
                )

    def _trial_table_prefix_header(self) -> list[str]:
        return [
            "source_file",
            "anchor_event_index",
            "anchor_event_code",
            "anchor_onset_s",
            "trial_id",
            "resolved_label",
        ]

    def _trial_table_prefix_row(self, trial: object) -> list[object]:
        return [
            str(getattr(getattr(trial, "source_file", None), "path", "")),
            getattr(trial, "anchor_event_index", ""),
            getattr(trial, "anchor_event_code", "") or "",
            getattr(trial, "anchor_onset_s", ""),
            getattr(trial, "trial_id", "") or "",
            getattr(trial, "label", "") or "",
        ]

    def _trial_table_extra_header(self) -> list[str]:
        return []

    def _trial_table_extra_row(self, trial: object) -> list[object]:
        del trial
        return []

    def _trial_table_suffix_header(self) -> list[str]:
        return [
            "condition_inputs",
            "condition_resolution_reason",
            "keep",
            "exclusion_reason",
        ]

    def _trial_table_suffix_row(self, trial: object) -> list[object]:
        metadata = getattr(trial, "metadata", {})
        return [
            _condition_inputs_json(trial),
            str(metadata.get("condition_resolution_reason", "")) if isinstance(metadata, dict) else "",
            str(getattr(trial, "keep", False)).lower(),
            getattr(trial, "exclusion_reason", "") or "",
        ]

    def _write_shared_hdf5_groups(
        self,
        fh: h5py.File,
        result: BaseTrialStatsProcessingResult,
        str_dtype: h5py.DatatypeLike,
    ) -> None:
        means_grp = fh.create_group("means")
        means_grp.create_dataset(
            result.condition_a,
            data=np.asarray(result.condition_a_mean, dtype=np.float64),
        )
        means_grp.create_dataset(
            result.condition_b,
            data=np.asarray(result.condition_b_mean, dtype=np.float64),
        )

        uncertainty_grp = fh.create_group("uncertainty")
        uncertainty_grp.create_dataset(
            result.condition_a + "_sem",
            data=np.asarray(result.condition_a_sem, dtype=np.float64),
        )
        uncertainty_grp.create_dataset(
            result.condition_b + "_sem",
            data=np.asarray(result.condition_b_sem, dtype=np.float64),
        )

        axes_grp = fh.create_group("axes")
        primary_axis_name = "region" if result.analysis_level == "roi" else "channel"
        axes_grp.create_dataset(
            primary_axis_name,
            data=np.array(result.channel_names, dtype=object),
            dtype=str_dtype,
        )
        axes_grp.create_dataset("time_s", data=np.asarray(result.time_axis_s, dtype=np.float64))

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
        meta_grp.create_dataset("activity_zscore", data=str(result.activity_zscore), dtype=str_dtype)
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
            data=int(result.metadata.get("window_samples", 0)),
        )
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
        meta_grp.create_dataset("stats_valid", data=bool(result.stats_valid))
        self._write_hdf5_meta_extra(meta_grp, result, str_dtype)

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
            data=np.array([trial.keep for trial in result.resolved_trials], dtype=bool),
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
        self._write_hdf5_trial_extra(trial_grp, result, str_dtype)

        if result.condition_a_epochs.ndim == 3 and self.params.include_epochs:
            epochs_grp = fh.create_group("epochs")
            epochs_grp.create_dataset(
                "condition_a",
                data=np.asarray(result.condition_a_epochs, dtype=np.float64),
            )
            epochs_grp.create_dataset(
                "condition_b",
                data=np.asarray(result.condition_b_epochs, dtype=np.float64),
            )
            epochs_grp.create_dataset(
                "channel",
                data=np.array(result.channel_names, dtype=object),
                dtype=str_dtype,
            )
            epochs_grp.create_dataset(
                "time_s",
                data=np.asarray(result.time_axis_s, dtype=np.float64),
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
        prov_grp.create_dataset("pipeline_name", data=self._pipeline_name(), dtype=str_dtype)
        prov_grp.create_dataset("pipeline_version", data=_package_version(), dtype=str_dtype)

    def _build_shared_matlab_groups(
        self,
        result: BaseTrialStatsProcessingResult,
    ) -> dict[str, Any]:
        cond_a = matlab_safe_name(result.condition_a)
        cond_b = matlab_safe_name(result.condition_b)
        primary_axis_name = "region" if result.analysis_level == "roi" else "channel"
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

        meta_fields: dict[str, Any] = {
            "trial_counts": np.array(
                [result.condition_a_trial_count, result.condition_b_trial_count],
                dtype=np.int64,
            ),
            "trial_count_labels": np.array([result.condition_a, result.condition_b], dtype=object),
            "sampling_frequency_hz": float(result.sfreq),
            "p_value_correction_method": str(result.p_value_correction_method),
            "significance_alpha": float(result.significance_alpha),
            "analysis_level": str(result.analysis_level),
            "atlas_name": str(result.atlas_name or ""),
            "atlas_regions": np.array(result.atlas_regions, dtype=object),
            "atlas_region_channel_map": atlas_map_struct,
            "window_ms": float(result.window_ms),
            "n_bins": int(result.n_bins),
            "activity_zscore": str(result.activity_zscore),
            "activity_baseline_tmin_s": float(result.activity_baseline_tmin_s),
            "activity_baseline_tmax_s": float(result.activity_baseline_tmax_s),
            "activity_baseline_scope": str(result.activity_baseline_scope),
            "activity_baseline_remove_outlier_trial_means": bool(
                result.activity_baseline_remove_outlier_trial_means
            ),
            "window_samples": int(result.metadata.get("window_samples", 0)),
            "effective_n_bins": int(result.metadata.get("effective_n_bins", len(result.time_axis_s))),
            "binning_mode": str(
                result.metadata.get(
                    "binning_mode",
                    "window_ms" if result.window_ms > 0 else ("n_bins" if result.n_bins > 0 else "none"),
                )
            ),
            "stats_valid": bool(result.stats_valid),
        }
        meta_fields.update(self._build_matlab_meta_extra(result))

        trials_fields: dict[str, Any] = {
            "source_file": np.array(
                [str(trial.source_file.path) for trial in result.resolved_trials],
                dtype=object,
            ),
            "anchor_event_code": np.array(
                [trial.anchor_event_code or "" for trial in result.resolved_trials],
                dtype=object,
            ),
            "anchor_onset_s": np.array(
                [trial.anchor_onset_s for trial in result.resolved_trials],
                dtype=np.float64,
            ),
            "resolved_label": np.array(
                [trial.label or "" for trial in result.resolved_trials],
                dtype=object,
            ),
            "keep": np.array([trial.keep for trial in result.resolved_trials], dtype=np.uint8),
            "exclusion_reason": np.array(
                [trial.exclusion_reason or "" for trial in result.resolved_trials],
                dtype=object,
            ),
            "trial_id": np.array(
                [trial.trial_id or "" for trial in result.resolved_trials],
                dtype=object,
            ),
            "condition_inputs": np.array(
                [_condition_inputs_json(trial) for trial in result.resolved_trials],
                dtype=object,
            ),
            "condition_resolution_reason": np.array(
                [
                    str(trial.metadata.get("condition_resolution_reason", ""))
                    for trial in result.resolved_trials
                ],
                dtype=object,
            ),
        }
        trials_fields.update(self._build_matlab_trial_extra(result))

        if result.condition_a_epochs.ndim == 3 and self.params.include_epochs:
            epochs_struct: object = make_struct(
                condition_a=np.asarray(result.condition_a_epochs, dtype=np.float64),
                condition_b=np.asarray(result.condition_b_epochs, dtype=np.float64),
                channel=np.array(result.channel_names, dtype=object),
                time_s=np.asarray(result.time_axis_s, dtype=np.float64),
            )
        else:
            epochs_struct = make_struct(
                condition_a=np.empty((0, 0, 0), dtype=np.float64),
                condition_b=np.empty((0, 0, 0), dtype=np.float64),
                channel=np.array([], dtype=object),
                time_s=np.array([], dtype=np.float64),
            )

        return {
            "means": make_struct(
                **{
                    cond_a: np.asarray(result.condition_a_mean, dtype=np.float64),
                    cond_b: np.asarray(result.condition_b_mean, dtype=np.float64),
                }
            ),
            "uncertainty": make_struct(
                **{
                    cond_a + "_sem": np.asarray(result.condition_a_sem, dtype=np.float64),
                    cond_b + "_sem": np.asarray(result.condition_b_sem, dtype=np.float64),
                }
            ),
            "axes": make_struct(
                **{
                    primary_axis_name: np.array(result.channel_names, dtype=object),
                    "time_s": np.asarray(result.time_axis_s, dtype=np.float64),
                }
            ),
            "meta": make_struct(**meta_fields),
            "trials": make_struct(**trials_fields),
            "epochs": epochs_struct,
            "provenance": make_struct(
                source_ieeg_files=np.array(result.source_ieeg_files, dtype=object),
                source_table_files=np.array(result.source_table_files, dtype=object),
                source_electrodes_files=np.array(result.source_electrodes_files, dtype=object),
                pipeline_name=np.str_(self._pipeline_name()),
                pipeline_version=np.str_(_package_version()),
            ),
        }

    def _write_hdf5_meta_extra(
        self,
        meta_grp: h5py.Group,
        result: BaseTrialStatsProcessingResult,
        str_dtype: h5py.DatatypeLike,
    ) -> None:
        del meta_grp, result, str_dtype

    def _write_hdf5_trial_extra(
        self,
        trial_grp: h5py.Group,
        result: BaseTrialStatsProcessingResult,
        str_dtype: h5py.DatatypeLike,
    ) -> None:
        del trial_grp, result, str_dtype

    def _build_matlab_meta_extra(
        self,
        result: BaseTrialStatsProcessingResult,
    ) -> dict[str, Any]:
        del result
        return {}

    def _build_matlab_trial_extra(
        self,
        result: BaseTrialStatsProcessingResult,
    ) -> dict[str, Any]:
        del result
        return {}

    @abstractmethod
    def _pipeline_name(self) -> str:
        """Return the subject-level provenance pipeline name."""

    @abstractmethod
    def _write_hdf5_specific(
        self,
        fh: h5py.File,
        result: BaseTrialStatsProcessingResult,
        str_dtype: h5py.DatatypeLike,
    ) -> None:
        """Write pipeline-specific HDF5 groups."""

    @abstractmethod
    def _build_matlab_specific(
        self,
        result: BaseTrialStatsProcessingResult,
    ) -> dict[str, Any]:
        """Return pipeline-specific MATLAB root groups."""

    def _validate_result(self, result: BaseTrialStatsProcessingResult) -> None:
        del result


def _trial_table_path(output_path: Path) -> Path:
    tsv_path = output_path.with_suffix(".tsv")
    return tsv_path.with_name(tsv_path.name.replace("_stats.tsv", "_trials.tsv"))


def _condition_inputs_json(trial: object) -> str:
    metadata = getattr(trial, "metadata", {})
    value = metadata.get("condition_inputs", {}) if isinstance(metadata, dict) else {}
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
