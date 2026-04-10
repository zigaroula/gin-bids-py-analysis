"""Processor for per-subject slope regression (gamma ~ predictor) on iEEG derivatives."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
from mne import Annotations

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
from gin_bids_py_analysis.processing.utils.events import (
    AnnotationEvent,
    parse_annotation_description,
)
from gin_bids_py_analysis.processing.utils.epoch_quality import (
    apply_channel_exclusions,
    apply_trial_nan_mask,
    detect_outlier_trial_channel_pairs_by_max,
    detect_outlier_trial_channel_pairs_by_mean,
    reject_channels_by_nan_trial_ratio,
    reject_channels_by_trial_max_spread,
    reject_channels_by_trial_mean_spread,
)
from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial, TrialResolver

from .params import TrialSlopeStatsParams
from .result import TrialSlopeStatsProcessingResult
from .stats import (
    compute_linear_regression_maps,
    zscore_epochs_across_trials,
    zscore_predictor_values_by_scope,
)


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
        predictor_a_raw: list[float] = []
        predictor_b_raw: list[float] = []
        predictor_a_transformed: list[float] = []
        predictor_b_transformed: list[float] = []
        kept_trials_a: list[ResolvedTrial] = []
        kept_trials_b: list[ResolvedTrial] = []

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
                normalized_trials = self._attach_trial_activity_summary_metadata(
                    raw_annotations=raw.annotations,
                    anchor_events=anchor_events,
                    trials=normalized_trials,
                )
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
                    raw_predictor_value = _to_float_or_nan(
                        trial.metadata.get("predictor_raw_value")
                    )
                    transformed_predictor_value = _to_float_or_nan(
                        trial.metadata.get("predictor_transformed_value")
                    )
                    if not np.isfinite(transformed_predictor_value):
                        continue
                    if trial.label == self.params.condition_a:
                        epochs_a.append(epoch_for_stats)
                        predictor_a_raw.append(raw_predictor_value)
                        predictor_a_transformed.append(transformed_predictor_value)
                        kept_trials_a.append(trial)
                    elif trial.label == self.params.condition_b:
                        epochs_b.append(epoch_for_stats)
                        predictor_b_raw.append(raw_predictor_value)
                        predictor_b_transformed.append(transformed_predictor_value)
                        kept_trials_b.append(trial)

        assert sfreq_ref is not None
        assert channel_names_ref is not None
        assert time_axis_ref is not None

        feature_names = feature_names_ref if atlas_mode else channel_names_ref
        assert feature_names is not None

        epochs_a_array = stack_epochs(epochs_a, len(feature_names), len(time_axis_ref))
        epochs_b_array = stack_epochs(epochs_b, len(feature_names), len(time_axis_ref))

        predictor_a_raw_array = np.asarray(predictor_a_raw, dtype=np.float64)
        predictor_b_raw_array = np.asarray(predictor_b_raw, dtype=np.float64)
        predictor_a_transformed_array = np.asarray(predictor_a_transformed, dtype=np.float64)
        predictor_b_transformed_array = np.asarray(predictor_b_transformed, dtype=np.float64)

        # -------------------------------------------------------------------
        # Epoch cleaning (Level A: trial-channel NaN masking;
        #                 Level B: channel exclusion)
        # Applied on the pooled conditions before activity z-scoring so that
        # channel-level decisions are consistent across conditions.
        # -------------------------------------------------------------------
        excluded_channels: dict[str, str] = {}
        excluded_trial_channel_pairs: dict[str, list[int]] = {}

        cfg = self.params.epoch_cleaning
        any_level_a = cfg.reject_trials_by_epoch_mean or cfg.reject_trials_by_epoch_max
        any_level_b = (
            cfg.reject_by_trial_mean_spread
            or cfg.reject_by_trial_max_spread
            or cfg.max_nan_trial_ratio is not None
        )

        if (any_level_a or any_level_b) and (
            epochs_a_array.shape[0] + epochs_b_array.shape[0]
        ) > 0:
            n_a = epochs_a_array.shape[0]
            epochs_pooled = np.concatenate([epochs_a_array, epochs_b_array], axis=0)

            # --- Level A ---
            if any_level_a:
                nan_tc_mask = np.zeros(
                    (epochs_pooled.shape[0], len(feature_names)), dtype=bool
                )
                if cfg.reject_trials_by_epoch_mean:
                    nan_tc_mask |= detect_outlier_trial_channel_pairs_by_mean(
                        epochs_pooled,
                        threshold_factor=cfg.epoch_mean_threshold_factor,
                    )
                if cfg.reject_trials_by_epoch_max:
                    nan_tc_mask |= detect_outlier_trial_channel_pairs_by_max(
                        epochs_pooled,
                        threshold_factor=cfg.epoch_max_threshold_factor,
                    )
                if np.any(nan_tc_mask):
                    epochs_pooled = apply_trial_nan_mask(epochs_pooled, nan_tc_mask)
                    for ch_idx, ch_name in enumerate(feature_names):
                        trial_indices = [
                            int(t) for t in np.where(nan_tc_mask[:, ch_idx])[0]
                        ]
                        if trial_indices:
                            excluded_trial_channel_pairs[ch_name] = trial_indices

            # --- Level B ---
            if any_level_b:
                reason_masks: dict[str, np.ndarray] = {}
                ch_excl_mask = np.zeros(len(feature_names), dtype=bool)
                if cfg.reject_by_trial_mean_spread:
                    m = reject_channels_by_trial_mean_spread(
                        epochs_pooled,
                        threshold_factor=cfg.trial_mean_spread_threshold,
                    )
                    ch_excl_mask |= m
                    reason_masks["trial_mean_spread"] = m
                if cfg.reject_by_trial_max_spread:
                    m = reject_channels_by_trial_max_spread(
                        epochs_pooled,
                        threshold_factor=cfg.trial_max_spread_threshold,
                    )
                    ch_excl_mask |= m
                    reason_masks["trial_max_spread"] = m
                if cfg.max_nan_trial_ratio is not None:
                    m = reject_channels_by_nan_trial_ratio(
                        epochs_pooled,
                        max_ratio=cfg.max_nan_trial_ratio,
                    )
                    ch_excl_mask |= m
                    reason_masks["nan_trial_ratio"] = m
                if np.any(ch_excl_mask):
                    for ch_idx, ch_name in enumerate(feature_names):
                        if ch_excl_mask[ch_idx]:
                            reasons = [
                                r for r, mask in reason_masks.items() if mask[ch_idx]
                            ]
                            excluded_channels[ch_name] = ",".join(reasons)
                    # Excluded channels entirely subsume their Level A entries.
                    for ch_name in excluded_channels:
                        excluded_trial_channel_pairs.pop(ch_name, None)
                    epochs_pooled, feature_names, _ = apply_channel_exclusions(
                        epochs_pooled,
                        list(feature_names),
                        ch_excl_mask,
                    )
                    # Keep feature_indices_ref aligned with feature_names (atlas mode).
                    if atlas_mode and feature_indices_ref is not None:
                        feature_indices_ref = [
                            indices
                            for indices, excl in zip(feature_indices_ref, ch_excl_mask)
                            if not excl
                        ]

            epochs_a_array = epochs_pooled[:n_a]
            epochs_b_array = epochs_pooled[n_a:]

        if self.params.activity_zscore == "baseline":
            epochs_a_array, epochs_b_array = zscore_activity_by_baseline(
                epochs_a_array,
                epochs_b_array,
                time_axis_ref,
                baseline_tmin_s=self.params.activity_baseline_tmin_s,
                baseline_tmax_s=self.params.activity_baseline_tmax_s,
                baseline_scope=self.params.activity_baseline_scope,
                remove_outlier_trial_means=self.params.activity_baseline_remove_outlier_trial_means,
            )
            activity_a_zscore_valid = True
            activity_b_zscore_valid = True
        elif self.params.activity_zscore == "across_trials":
            epochs_a_array, activity_a_zscore_valid = zscore_epochs_across_trials(
                epochs_a_array
            )
            epochs_b_array, activity_b_zscore_valid = zscore_epochs_across_trials(
                epochs_b_array
            )
        else:
            activity_a_zscore_valid = True
            activity_b_zscore_valid = True

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

        cond_a_ready = (
            epochs_a_array.shape[0] >= self.params.min_trials_per_condition
            and activity_a_zscore_valid
        )
        cond_b_ready = (
            epochs_b_array.shape[0] >= self.params.min_trials_per_condition
            and activity_b_zscore_valid
        )

        (
            predictor_a_effective_array,
            predictor_b_effective_array,
            predictor_a_zscore_valid,
            predictor_b_zscore_valid,
        ) = self._scale_predictor_values_pair(
            predictor_a_transformed_array,
            predictor_b_transformed_array,
        )
        self._store_effective_predictor_values(
            kept_trials_a,
            predictor_a_effective_array,
        )
        self._store_effective_predictor_values(
            kept_trials_b,
            predictor_b_effective_array,
        )

        condition_a_slope, condition_a_intercept, condition_a_r_value, condition_a_p_value, condition_a_stats_valid = (
            compute_linear_regression_maps(
                predictor_a_effective_array
                if cond_a_ready and predictor_a_zscore_valid
                else np.array([], dtype=np.float64),
                epochs_a_array if cond_a_ready else np.empty_like(epochs_a_array[:0]),
                n_features=len(feature_names),
                n_times=len(time_axis_eval),
            )
        )
        condition_b_slope, condition_b_intercept, condition_b_r_value, condition_b_p_value, condition_b_stats_valid = (
            compute_linear_regression_maps(
                predictor_b_effective_array
                if cond_b_ready and predictor_b_zscore_valid
                else np.array([], dtype=np.float64),
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

        condition_a_epoch_means = self._compute_epoch_means(epochs_a_array, len(feature_names))
        condition_b_epoch_means = self._compute_epoch_means(epochs_b_array, len(feature_names))
        condition_a_trial_activity_summary_values = self._compute_trial_activity_summary_values(
            epochs=epochs_a_array,
            time_axis_s=time_axis_eval,
            trials=kept_trials_a,
            fallback_epoch_means=condition_a_epoch_means,
            n_features=len(feature_names),
        )
        condition_b_trial_activity_summary_values = self._compute_trial_activity_summary_values(
            epochs=epochs_b_array,
            time_axis_s=time_axis_eval,
            trials=kept_trials_b,
            fallback_epoch_means=condition_b_epoch_means,
            n_features=len(feature_names),
        )
        trial_activity_summary_source = self._serialize_trial_activity_summary_source()
        trial_activity_summary_label = self._trial_activity_summary_label()

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
                "predictor_zscore": self.params.predictor_zscore,
                "predictor_transform_by_condition": {
                    condition: {
                        "scale": float(transform.scale),
                        "offset": float(transform.offset),
                    }
                    for condition, transform in self.params.predictor_transform_by_condition.items()
                },
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
                "activity_zscore": self.params.activity_zscore,
                "activity_baseline_tmin_s": self.params.activity_baseline_tmin_s,
                "activity_baseline_tmax_s": self.params.activity_baseline_tmax_s,
                "activity_baseline_scope": self.params.activity_baseline_scope,
                "activity_baseline_remove_outlier_trial_means": self.params.activity_baseline_remove_outlier_trial_means,
                "epoch_cleaning": json.loads(
                    self.params.epoch_cleaning.model_dump_json()
                ),
                "trial_activity_summary": json.loads(
                    self.params.trial_activity_summary.model_dump_json()
                ),
                "trial_activity_summary_kind": self.params.trial_activity_summary.kind,
                "trial_activity_summary_missing_response_policy": self.params.trial_activity_summary.missing_response_policy,
                "trial_activity_summary_source": trial_activity_summary_source,
                "trial_activity_summary_label": trial_activity_summary_label,
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
            condition_a_predictor_raw_values=predictor_a_raw_array,
            condition_b_predictor_raw_values=predictor_b_raw_array,
            condition_a_predictor_transformed_values=predictor_a_transformed_array,
            condition_b_predictor_transformed_values=predictor_b_transformed_array,
            condition_a_predictor_values=predictor_a_effective_array,
            condition_b_predictor_values=predictor_b_effective_array,
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
            activity_zscore=self.params.activity_zscore,
            activity_baseline_tmin_s=self.params.activity_baseline_tmin_s,
            activity_baseline_tmax_s=self.params.activity_baseline_tmax_s,
            activity_baseline_scope=self.params.activity_baseline_scope,
            activity_baseline_remove_outlier_trial_means=self.params.activity_baseline_remove_outlier_trial_means,
            excluded_channels=excluded_channels,
            excluded_trial_channel_pairs=excluded_trial_channel_pairs,
            predictor=self.params.predictor,
            predictor_zscore=self.params.predictor_zscore,
            predictor_transform_by_condition={
                condition: {
                    "scale": float(transform.scale),
                    "offset": float(transform.offset),
                }
                for condition, transform in self.params.predictor_transform_by_condition.items()
            },
            trial_activity_summary_kind=self.params.trial_activity_summary.kind,
            trial_activity_summary_missing_response_policy=self.params.trial_activity_summary.missing_response_policy,
            trial_activity_summary_source=trial_activity_summary_source,
            trial_activity_summary_label=trial_activity_summary_label,
            p_value_correction_method=self.params.p_value_correction_method,
            significance_alpha=self.params.significance_alpha,
            condition_a_stats_valid=condition_a_stats_valid,
            condition_b_stats_valid=condition_b_stats_valid,
            stats_valid=bool(condition_a_stats_valid or condition_b_stats_valid),
            condition_a_epochs=epochs_a_array,
            condition_b_epochs=epochs_b_array,
            condition_a_trial_activity_summary_values=condition_a_trial_activity_summary_values,
            condition_b_trial_activity_summary_values=condition_b_trial_activity_summary_values,
            condition_a_epoch_means=condition_a_epoch_means,
            condition_b_epoch_means=condition_b_epoch_means,
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
            raw_predictor_value = _to_float_or_nan(raw_predictor)
            metadata["predictor_raw_value"] = float(raw_predictor_value) if np.isfinite(raw_predictor_value) else np.nan
            transform = self.params.predictor_transform_by_condition.get(
                trial.label,
                self.params.predictor_transform_by_condition.get(
                    str(trial.label).strip(),
                ),
            )
            scale = float(transform.scale) if transform is not None else 1.0
            offset = float(transform.offset) if transform is not None else 0.0
            metadata["predictor_transform_scale"] = scale
            metadata["predictor_transform_offset"] = offset
            predictor_value = _apply_affine_transform(
                raw_predictor_value,
                scale=scale,
                offset=offset,
            )
            metadata["predictor_transformed_value"] = (
                float(predictor_value) if np.isfinite(predictor_value) else np.nan
            )
            metadata["predictor_value"] = (
                float(predictor_value) if np.isfinite(predictor_value) else np.nan
            )

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

    def _scale_predictor_values_pair(
        self,
        predictor_values_a: np.ndarray,
        predictor_values_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        return zscore_predictor_values_by_scope(
            predictor_values_a,
            predictor_values_b,
            scope=self.params.predictor_zscore,
        )

    def _store_effective_predictor_values(
        self,
        trials: list[ResolvedTrial],
        predictor_values: np.ndarray,
    ) -> None:
        for trial, value in zip(trials, np.asarray(predictor_values, dtype=np.float64).ravel()):
            trial.metadata["predictor_value"] = (
                float(value) if np.isfinite(value) else np.nan
            )

    def _attach_trial_activity_summary_metadata(
        self,
        *,
        raw_annotations: Annotations,
        anchor_events: list[AnnotationEvent],
        trials: list[ResolvedTrial],
    ) -> list[ResolvedTrial]:
        if self.params.trial_activity_summary.kind != "anchor_to_response_mean":
            return trials

        response_source = self.params.trial_activity_summary.response
        if response_source is None:
            return trials

        if response_source.source == "table_column":
            response_times_s = self._response_times_from_table_column(
                trials,
                column=response_source.column,
                units=response_source.units,
            )
        else:
            response_times_s = self._response_times_from_annotations(
                raw_annotations,
                anchor_events,
                event_code=response_source.event_code,
            )

        updated_trials: list[ResolvedTrial] = []
        for trial, response_time_s in zip(trials, response_times_s):
            metadata = dict(trial.metadata)
            metadata["trial_activity_summary_response_time_s"] = (
                float(response_time_s) if np.isfinite(response_time_s) else np.nan
            )
            updated_trials.append(replace(trial, metadata=metadata))
        return updated_trials

    def _response_times_from_table_column(
        self,
        trials: list[ResolvedTrial],
        *,
        column: str,
        units: str,
    ) -> np.ndarray:
        scale = 1.0 if units == "s" else 0.001
        response_times = np.full((len(trials),), np.nan, dtype=np.float64)
        for idx, trial in enumerate(trials):
            value = _to_float_or_nan(trial.metadata.get(column))
            if np.isfinite(value):
                response_times[idx] = float(value) * scale
        return response_times

    def _response_times_from_annotations(
        self,
        raw_annotations: Annotations,
        anchor_events: list[AnnotationEvent],
        *,
        event_code: str,
    ) -> np.ndarray:
        response_times = np.full((len(anchor_events),), np.nan, dtype=np.float64)
        if not anchor_events:
            return response_times

        target = str(event_code).strip()
        if not target:
            return response_times

        response_onsets_s: list[float] = []
        for annotation in raw_annotations:
            onset_s = float(annotation["onset"])
            description = str(annotation["description"])
            if _annotation_matches_event_code(description, target):
                response_onsets_s.append(onset_s)
        if not response_onsets_s:
            return response_times

        response_onsets = np.asarray(sorted(response_onsets_s), dtype=np.float64)
        anchor_onsets = np.asarray([event.onset_s for event in anchor_events], dtype=np.float64)

        for idx, anchor_onset_s in enumerate(anchor_onsets):
            next_anchor_onset_s = (
                float(anchor_onsets[idx + 1]) if idx + 1 < anchor_onsets.size else float("inf")
            )
            insert_at = int(np.searchsorted(response_onsets, anchor_onset_s, side="right"))
            if insert_at >= response_onsets.size:
                continue
            response_onset_s = float(response_onsets[insert_at])
            if response_onset_s >= next_anchor_onset_s:
                continue
            response_times[idx] = response_onset_s - float(anchor_onset_s)
        return response_times

    def _compute_epoch_means(
        self,
        epochs: np.ndarray,
        n_features: int,
    ) -> np.ndarray:
        if epochs.ndim == 3 and epochs.size > 0:
            return epochs.mean(axis=2).T.astype(np.float64)
        return np.empty((n_features, 0), dtype=np.float64)

    def _compute_trial_activity_summary_values(
        self,
        *,
        epochs: np.ndarray,
        time_axis_s: np.ndarray,
        trials: list[ResolvedTrial],
        fallback_epoch_means: np.ndarray,
        n_features: int,
    ) -> np.ndarray:
        if self.params.trial_activity_summary.kind == "epoch_mean":
            return np.asarray(fallback_epoch_means, dtype=np.float64)

        if epochs.ndim != 3 or epochs.size == 0:
            return np.empty((n_features, 0), dtype=np.float64)

        summary = np.full((epochs.shape[0], n_features), np.nan, dtype=np.float64)
        time_axis = np.asarray(time_axis_s, dtype=np.float64).ravel()
        summary_window_start_s = 0.0
        summary_window_end_s = float(self.params.tmax_s)
        boundary_tol_s = 1e-12
        missing_response_policy = (
            str(self.params.trial_activity_summary.missing_response_policy)
            .strip()
            .lower()
        )

        for trial_idx, trial in enumerate(trials):
            response_time_s = _to_float_or_nan(
                trial.metadata.get("trial_activity_summary_response_time_s")
            )
            if not np.isfinite(response_time_s):
                continue
            if response_time_s <= summary_window_start_s:
                continue
            if response_time_s > summary_window_end_s:
                if missing_response_policy == "clamp_to_epoch":
                    response_time_s = summary_window_end_s
                else:
                    continue
            time_mask = (
                (time_axis >= summary_window_start_s - boundary_tol_s)
                & (time_axis <= response_time_s + boundary_tol_s)
            )
            if not np.any(time_mask):
                continue
            summary[trial_idx, :] = np.nanmean(
                epochs[trial_idx][:, time_mask],
                axis=1,
                dtype=np.float64,
            )

        return summary.T.astype(np.float64)

    def _serialize_trial_activity_summary_source(self) -> dict[str, str]:
        response_source = self.params.trial_activity_summary.response
        if response_source is None:
            return {}
        serialized = response_source.model_dump()
        return {str(key): str(value) for key, value in serialized.items() if value is not None}

    def _trial_activity_summary_label(self) -> str:
        if self.params.trial_activity_summary.kind == "anchor_to_response_mean":
            return "Mean activity (trigger to response)"
        return "Epoch mean activity"


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


def _apply_affine_transform(
    value: float,
    *,
    scale: float,
    offset: float,
) -> float:
    if not np.isfinite(value):
        return float("nan")
    transformed = (float(scale) * float(value)) + float(offset)
    return transformed if np.isfinite(transformed) else float("nan")


def _annotation_matches_event_code(description: str, event_code: str) -> bool:
    full_description = str(description).strip()
    target = str(event_code).strip()
    if not full_description or not target:
        return False
    event_type, parsed_description, parsed_code = parse_annotation_description(full_description)
    del event_type
    candidates = {
        full_description,
        parsed_description.strip(),
    }
    if parsed_code is not None:
        candidates.add(str(parsed_code).strip())
    return target in candidates



