"""Processor for per-subject slope regression (gamma ~ predictor) on iEEG derivatives."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.bids.matching import files_matching_entities, shared_entities
from gin_bids_py_analysis.processing.base import BaseProcessing
from gin_bids_py_analysis.processing.utils.atlas import (
    aggregate_epochs_with_mne,
    resolve_atlas_grouping,
)
from gin_bids_py_analysis.processing.utils.epoching import (
    extract_anchor_events_with_mne,
    extract_epochs_with_mne,
    stack_epochs,
    temporal_bin_epochs,
    temporal_bin_epochs_by_n_bins,
    window_samples as compute_window_samples,
)
from gin_bids_py_analysis.processing.utils.statistics import (
    compute_condition_mean,
    compute_condition_sem,
    correct_p_values,
    zscore_activity_by_baseline,
)
from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial, TrialResolver

from .params import TrialSlopeStatsParams
from .result import TrialSlopeStatsProcessingResult
from .stats import compute_linear_regression_maps


class TrialSlopeStatsProcessing(BaseProcessing):
    """Compute per-subject per-condition slope regression on epoched iEEG data."""

    def __init__(
        self,
        params: TrialSlopeStatsParams,
        resolver: TrialResolver,
    ) -> None:
        self.params = params
        self.resolver = resolver
        self._validate_resolver_labels()

    def process_group(
        self,
        group: BIDSFileGroup,
        progress_tracking_position: int = 0,
    ) -> TrialSlopeStatsProcessingResult:
        del progress_tracking_position

        ieeg_files = files_matching_entities(
            group.all_files,
            extension=".vhdr",
            suffix="ieeg",
        )
        if not ieeg_files:
            raise ValueError(
                "TrialSlopeStatsProcessing requires at least one ieeg BrainVision file."
            )

        table_files = files_matching_entities(
            group.all_files,
            extension={".tsv", ".csv"},
        )
        source_table_files = sorted(
            {
                str(file.path)
                for file in table_files
                if file.suffix != "electrodes"
            }
        )
        electrodes_files = files_matching_entities(
            group.all_files,
            extension={".tsv", ".csv"},
            suffix="electrodes",
        )

        atlas_mode = bool(self.params.atlas_name)
        used_electrodes_paths: set[str] = set()
        missing_atlas_regions: set[str] = set()

        anchor_codes = set(self.params.anchor_event_codes)
        all_resolved_trials: list[ResolvedTrial] = []
        epochs_a: list[np.ndarray] = []
        epochs_b: list[np.ndarray] = []
        predictor_a: list[float] = []
        predictor_b: list[float] = []

        sfreq_ref: float | None = None
        channel_names_ref: list[str] | None = None
        feature_names_ref: list[str] | None = None
        feature_indices_ref: list[np.ndarray] | None = None
        time_axis_ref: np.ndarray | None = None

        for ieeg_file in ieeg_files:
            with ieeg_file.ensure_loaded() as raw:
                sfreq = float(raw.info["sfreq"])
                channel_names = list(raw.ch_names)
                if sfreq_ref is None:
                    sfreq_ref = sfreq
                    channel_names_ref = channel_names
                else:
                    if sfreq != sfreq_ref:
                        raise ValueError(
                            "All ieeg files in a subject group must share the same sampling frequency."
                        )
                    if channel_names != channel_names_ref:
                        raise ValueError(
                            "All ieeg files in a subject group must share the same channel ordering."
                        )

                if atlas_mode:
                    assert self.params.atlas_name is not None
                    grouping = resolve_atlas_grouping(
                        ieeg_file=ieeg_file,
                        electrodes_files=electrodes_files,
                        channel_names=channel_names,
                        atlas_name=self.params.atlas_name,
                        atlas_regions=self.params.atlas_regions,
                    )
                    used_electrodes_paths.add(str(grouping.source_file.path))
                    missing_atlas_regions.update(grouping.missing_regions)

                    if feature_names_ref is None:
                        feature_names_ref = grouping.feature_names
                        feature_indices_ref = grouping.feature_channel_indices
                    elif feature_names_ref != grouping.feature_names:
                        raise ValueError(
                            "Atlas region grouping must be consistent across all ieeg files in a subject group. "
                            f"Got {grouping.feature_names} for {ieeg_file.path.name} but expected {feature_names_ref}."
                        )

                anchor_events, anchor_samples = extract_anchor_events_with_mne(
                    raw,
                    anchor_codes=anchor_codes,
                    experiment_start_event_code=self.params.experiment_start_event_code,
                    experiment_end_event_code=self.params.experiment_end_event_code,
                )
                resolved_trials = self.resolver.resolve_trials(group, ieeg_file, anchor_events)
                if len(resolved_trials) != len(anchor_events):
                    raise ValueError(
                        f"Resolver returned {len(resolved_trials)} trial rows for "
                        f"{len(anchor_events)} anchor events in {ieeg_file.path.name}."
                    )

                normalized_trials = self._normalize_trials_for_slope(resolved_trials)
                extraction = extract_epochs_with_mne(
                    raw,
                    anchor_samples=anchor_samples,
                    trials=normalized_trials,
                    tmin_s=self.params.tmin_s,
                    tmax_s=self.params.tmax_s,
                    drop_partial_epochs=self.params.drop_partial_epochs,
                )
                if time_axis_ref is None:
                    time_axis_ref = extraction.time_axis_s
                elif not np.allclose(
                    np.asarray(extraction.time_axis_s, dtype=np.float64),
                    np.asarray(time_axis_ref, dtype=np.float64),
                    atol=1e-12,
                    rtol=0.0,
                ):
                    raise ValueError(
                        "All ieeg files in a subject group must yield the same epoch time axis."
                    )
                all_resolved_trials.extend(extraction.updated_trials)

                epochs_for_stats = np.asarray(extraction.epochs, dtype=np.float64)
                if atlas_mode and extraction.epochs.shape[0] > 0:
                    assert feature_names_ref is not None
                    assert feature_indices_ref is not None
                    epochs_for_stats = aggregate_epochs_with_mne(
                        epochs_for_stats,
                        channel_names=channel_names,
                        feature_names=feature_names_ref,
                        feature_channel_indices=feature_indices_ref,
                        sfreq=sfreq,
                        tmin_s=float(extraction.time_axis_s[0]),
                    )

                for epoch_for_stats, trial in zip(epochs_for_stats, extraction.kept_trials):
                    predictor_value = _to_float_or_nan(trial.metadata.get("predictor_value"))
                    if not np.isfinite(predictor_value):
                        continue
                    if trial.label == self.params.condition_a:
                        epochs_a.append(epoch_for_stats)
                        predictor_a.append(predictor_value)
                    elif trial.label == self.params.condition_b:
                        epochs_b.append(epoch_for_stats)
                        predictor_b.append(predictor_value)

        assert sfreq_ref is not None
        assert channel_names_ref is not None
        assert time_axis_ref is not None

        feature_names = feature_names_ref if atlas_mode else channel_names_ref
        assert feature_names is not None

        epochs_a_array = stack_epochs(epochs_a, len(feature_names), len(time_axis_ref))
        epochs_b_array = stack_epochs(epochs_b, len(feature_names), len(time_axis_ref))

        predictor_a_array = np.asarray(predictor_a, dtype=np.float64)
        predictor_b_array = np.asarray(predictor_b, dtype=np.float64)

        if self.params.activity_scaling == "zscore_by_baseline":
            epochs_a_array, epochs_b_array = zscore_activity_by_baseline(
                epochs_a_array,
                epochs_b_array,
                time_axis_ref,
                baseline_tmin_s=self.params.activity_baseline_tmin_s,
                baseline_tmax_s=self.params.activity_baseline_tmax_s,
            )

        time_axis_eval = time_axis_ref
        binning_mode = "none"
        window_sample_count = 0
        effective_n_bins = int(len(time_axis_ref))
        if self.params.window_ms > 0:
            binning_mode = "window_ms"
            window_sample_count = compute_window_samples(sfreq_ref, self.params.window_ms)
            epochs_a_array, time_axis_eval = temporal_bin_epochs(
                epochs_a_array,
                time_axis_ref,
                window_sample_count,
            )
            epochs_b_array, _ = temporal_bin_epochs(
                epochs_b_array,
                time_axis_ref,
                window_sample_count,
            )
            effective_n_bins = int(len(time_axis_eval))
        elif self.params.n_bins > 0:
            if self.params.n_bins > len(time_axis_ref):
                raise ValueError(
                    "n_bins cannot be greater than the number of epoch samples "
                    f"({len(time_axis_ref)})."
                )
            binning_mode = "n_bins"
            epochs_a_array, time_axis_eval = temporal_bin_epochs_by_n_bins(
                epochs_a_array,
                time_axis_ref,
                self.params.n_bins,
            )
            epochs_b_array, _ = temporal_bin_epochs_by_n_bins(
                epochs_b_array,
                time_axis_ref,
                self.params.n_bins,
            )
            effective_n_bins = int(len(time_axis_eval))

        condition_a_mean = compute_condition_mean(
            epochs_a_array,
            n_features=len(feature_names),
            n_times=len(time_axis_eval),
        )
        condition_b_mean = compute_condition_mean(
            epochs_b_array,
            n_features=len(feature_names),
            n_times=len(time_axis_eval),
        )
        condition_a_sem = compute_condition_sem(
            epochs_a_array,
            n_features=len(feature_names),
            n_times=len(time_axis_eval),
        )
        condition_b_sem = compute_condition_sem(
            epochs_b_array,
            n_features=len(feature_names),
            n_times=len(time_axis_eval),
        )

        cond_a_ready = epochs_a_array.shape[0] >= self.params.min_trials_per_condition
        cond_b_ready = epochs_b_array.shape[0] >= self.params.min_trials_per_condition

        condition_a_slope, condition_a_intercept, condition_a_r_value, condition_a_p_value, condition_a_stats_valid = (
            compute_linear_regression_maps(
                predictor_a_array if cond_a_ready else np.array([], dtype=np.float64),
                epochs_a_array if cond_a_ready else np.empty_like(epochs_a_array[:0]),
                n_features=len(feature_names),
                n_times=len(time_axis_eval),
            )
        )
        condition_b_slope, condition_b_intercept, condition_b_r_value, condition_b_p_value, condition_b_stats_valid = (
            compute_linear_regression_maps(
                predictor_b_array if cond_b_ready else np.array([], dtype=np.float64),
                epochs_b_array if cond_b_ready else np.empty_like(epochs_b_array[:0]),
                n_features=len(feature_names),
                n_times=len(time_axis_eval),
            )
        )

        condition_a_p_value_corrected = correct_p_values(
            condition_a_p_value,
            method=self.params.p_value_correction_method,
        )
        condition_b_p_value_corrected = correct_p_values(
            condition_b_p_value,
            method=self.params.p_value_correction_method,
        )
        condition_a_significant_mask = np.isfinite(condition_a_p_value_corrected) & (
            condition_a_p_value_corrected < self.params.significance_alpha
        )
        condition_b_significant_mask = np.isfinite(condition_b_p_value_corrected) & (
            condition_b_p_value_corrected < self.params.significance_alpha
        )

        source_electrodes_files = sorted(used_electrodes_paths)
        if not source_electrodes_files and electrodes_files:
            source_electrodes_files = sorted({str(file.path) for file in electrodes_files})

        region_channels: dict[str, list[str]] = {}
        if atlas_mode:
            assert feature_indices_ref is not None
            region_channels = {
                region: [channel_names_ref[int(index)] for index in indices]
                for region, indices in zip(feature_names, feature_indices_ref)
            }

        return TrialSlopeStatsProcessingResult(
            source_group=group,
            output_entities=shared_entities(
                ieeg_files,
                excluded_entities=frozenset(
                    {"suffix", "extension", "datatype", "desc", "description", "run"}
                ),
            ),
            metadata={
                "analysis_type": "slope_regression",
                "anchor_event_codes": list(self.params.anchor_event_codes),
                "experiment_start_event_code": self.params.experiment_start_event_code,
                "experiment_end_event_code": self.params.experiment_end_event_code,
                "tmin_s": self.params.tmin_s,
                "tmax_s": self.params.tmax_s,
                "min_trials_per_condition": self.params.min_trials_per_condition,
                "drop_partial_epochs": self.params.drop_partial_epochs,
                "predictor": self.params.predictor,
                "predictor_scaling": self.params.predictor_scaling,
                "p_value_correction_method": self.params.p_value_correction_method,
                "significance_alpha": self.params.significance_alpha,
                "analysis_level": "roi" if atlas_mode else "channel",
                "atlas_name": self.params.atlas_name,
                "atlas_regions": feature_names if atlas_mode else [],
                "atlas_regions_requested": list(self.params.atlas_regions),
                "atlas_regions_missing": sorted(missing_atlas_regions),
                "window_ms": self.params.window_ms,
                "n_bins": self.params.n_bins,
                "window_samples": window_sample_count,
                "effective_n_bins": effective_n_bins,
                "binning_mode": binning_mode,
                "activity_scaling": self.params.activity_scaling,
                "activity_baseline_tmin_s": self.params.activity_baseline_tmin_s,
                "activity_baseline_tmax_s": self.params.activity_baseline_tmax_s,
            },
            condition_a_slope=condition_a_slope,
            condition_a_intercept=condition_a_intercept,
            condition_a_r_value=condition_a_r_value,
            condition_a_p_value=condition_a_p_value,
            condition_a_p_value_corrected=condition_a_p_value_corrected,
            condition_a_significant_mask=condition_a_significant_mask,
            condition_b_slope=condition_b_slope,
            condition_b_intercept=condition_b_intercept,
            condition_b_r_value=condition_b_r_value,
            condition_b_p_value=condition_b_p_value,
            condition_b_p_value_corrected=condition_b_p_value_corrected,
            condition_b_significant_mask=condition_b_significant_mask,
            condition_a_mean=condition_a_mean,
            condition_b_mean=condition_b_mean,
            condition_a_sem=condition_a_sem,
            condition_b_sem=condition_b_sem,
            time_axis_s=time_axis_eval,
            channel_names=feature_names,
            condition_a=self.params.condition_a,
            condition_b=self.params.condition_b,
            condition_a_trial_count=int(epochs_a_array.shape[0]),
            condition_b_trial_count=int(epochs_b_array.shape[0]),
            condition_a_trials_used=int(epochs_a_array.shape[0]),
            condition_b_trials_used=int(epochs_b_array.shape[0]),
            sfreq=sfreq_ref,
            condition_a_predictor_values=predictor_a_array,
            condition_b_predictor_values=predictor_b_array,
            resolved_trials=all_resolved_trials,
            source_ieeg_files=[str(file.path) for file in ieeg_files],
            source_table_files=source_table_files,
            source_electrodes_files=source_electrodes_files,
            analysis_level="roi" if atlas_mode else "channel",
            analysis_type="slope_regression",
            atlas_name=self.params.atlas_name,
            atlas_regions=feature_names if atlas_mode else [],
            region_channels=region_channels,
            window_ms=self.params.window_ms,
            n_bins=self.params.n_bins,
            activity_scaling=self.params.activity_scaling,
            activity_baseline_tmin_s=self.params.activity_baseline_tmin_s,
            activity_baseline_tmax_s=self.params.activity_baseline_tmax_s,
            predictor=self.params.predictor,
            predictor_scaling=self.params.predictor_scaling,
            p_value_correction_method=self.params.p_value_correction_method,
            significance_alpha=self.params.significance_alpha,
            condition_a_stats_valid=condition_a_stats_valid,
            condition_b_stats_valid=condition_b_stats_valid,
            stats_valid=bool(condition_a_stats_valid or condition_b_stats_valid),
            condition_a_epochs=epochs_a_array,
            condition_b_epochs=epochs_b_array,
            condition_a_epoch_means=(
                epochs_a_array.mean(axis=2).T.astype(np.float64)
                if epochs_a_array.ndim == 3 and epochs_a_array.size > 0
                else np.empty((len(feature_names), 0), dtype=np.float64)
            ),
            condition_b_epoch_means=(
                epochs_b_array.mean(axis=2).T.astype(np.float64)
                if epochs_b_array.ndim == 3 and epochs_b_array.size > 0
                else np.empty((len(feature_names), 0), dtype=np.float64)
            ),
        )

    def _normalize_trials_for_slope(
        self,
        trials: list[ResolvedTrial],
    ) -> list[ResolvedTrial]:
        normalized: list[ResolvedTrial] = []
        supported_labels = {self.params.condition_a, self.params.condition_b}
        predictor_key = self.params.predictor
        for trial in trials:
            metadata = dict(trial.metadata)
            raw_predictor = metadata.get(predictor_key)
            metadata["predictor_raw"] = "" if raw_predictor is None else str(raw_predictor)
            predictor_value = _to_float_or_nan(raw_predictor)
            if np.isfinite(predictor_value):
                metadata["predictor_value"] = float(predictor_value)
            else:
                metadata["predictor_value"] = np.nan

            if not trial.keep:
                normalized.append(replace(trial, metadata=metadata))
                continue
            if trial.label not in supported_labels:
                normalized.append(
                    ResolvedTrial(
                        source_file=trial.source_file,
                        anchor_event_index=trial.anchor_event_index,
                        anchor_event_code=trial.anchor_event_code,
                        anchor_onset_s=trial.anchor_onset_s,
                        anchor_duration_s=trial.anchor_duration_s,
                        label=trial.label,
                        trial_id=trial.trial_id,
                        keep=False,
                        exclusion_reason=trial.exclusion_reason or "unsupported_label",
                        metadata=metadata,
                    )
                )
                continue
            if not np.isfinite(predictor_value):
                normalized.append(
                    ResolvedTrial(
                        source_file=trial.source_file,
                        anchor_event_index=trial.anchor_event_index,
                        anchor_event_code=trial.anchor_event_code,
                        anchor_onset_s=trial.anchor_onset_s,
                        anchor_duration_s=trial.anchor_duration_s,
                        label=trial.label,
                        trial_id=trial.trial_id,
                        keep=False,
                        exclusion_reason=trial.exclusion_reason or "invalid_predictor_value",
                        metadata=metadata,
                    )
                )
                continue
            normalized.append(replace(trial, metadata=metadata))
        return normalized

    def _validate_resolver_labels(self) -> None:
        resolver_labels = getattr(self.resolver, "condition_labels", None)
        if resolver_labels is None:
            return

        labels = tuple(str(label) for label in resolver_labels)
        expected = (self.params.condition_a, self.params.condition_b)
        if len(labels) != 2 or set(labels) != set(expected):
            raise ValueError(
                "Resolver condition labels must match TrialSlopeStatsParams.condition_a/"
                f"condition_b exactly. Expected {expected!r}, got {labels!r}."
            )


def _to_float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
        return out if np.isfinite(out) else float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        out = float(text)
    except ValueError:
        return float("nan")
    return out if np.isfinite(out) else float("nan")



