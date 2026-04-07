"""Processor for per-subject condition-A vs condition-B statistics on iEEG derivatives.

Orchestrates file loading, trial resolution, epoch extraction, optional atlas-region
aggregation, temporal binning, and statistical testing across all iEEG files in one
``BIDSFileGroup``.  Heavy numerical work is delegated to ``stats.py``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.bids.matching import (
    files_matching_entities,
    shared_entities,
)
from gin_bids_py_analysis.processing.base import BaseProcessing
from gin_bids_py_analysis.processing.utils.atlas import (
    AtlasGrouping,
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
    compute_condition_sem,
    correct_p_values,
)
from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial, TrialResolver

from .params import TrialStatsParams
from .result import TrialStatsProcessingResult
from .stats import (
    compute_bootstrap_difference_ci95,
    compute_condition_statistics,
    compute_duration_channel_significance,
    compute_permutation_p_values,
    compute_permuted_statistics,
    compute_single_bin_channel_significance,
)


class TrialStatsProcessing(BaseProcessing):
    """Compute per-subject condition_a-vs-condition_b statistics on ieeg data."""

    def __init__(
        self,
        params: TrialStatsParams,
        resolver: TrialResolver,
    ) -> None:
        self.params = params
        self.resolver = resolver

    def process_group(
        self,
        group: BIDSFileGroup,
        progress_tracking_position: int = 0,
    ) -> TrialStatsProcessingResult:
        """Run the full pipeline for one subject group and return a result object.

        Steps:
        1. Partition the group's files into iEEG recordings, table files, and electrodes tables.
        2. For each iEEG file: load data, validate cross-file consistency, optionally resolve
           atlas grouping, extract anchor events, resolve trial labels, and extract epochs.
        3. Stack per-condition epoch lists, apply optional temporal binning.
        4. Compute t-tests, correct p-values, and build the result.
        """
        del progress_tracking_position

        # --- Step 1: partition files in the group ---
        ieeg_files = files_matching_entities(
            group.all_files,
            extension=".vhdr",
            suffix="ieeg",
        )
        if not ieeg_files:
            raise ValueError(
                "TrialStatsProcessing requires at least one ieeg BrainVision file."
            )

        table_files = files_matching_entities(
            group.all_files,
            extension={".tsv", ".csv"},
        )
        # Non-electrodes table files are recorded for provenance only.
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

        atlas_mode = bool(self.params.atlas_name)  # True when an atlas column name is provided.
        used_electrodes_paths: set[str] = set()
        missing_atlas_regions: set[str] = set()

        anchor_codes = set(self.params.anchor_event_codes)
        all_resolved_trials: list[ResolvedTrial] = []
        epochs_a: list[np.ndarray] = []
        epochs_b: list[np.ndarray] = []

        sfreq_ref: float | None = None
        channel_names_ref: list[str] | None = None
        feature_names_ref: list[str] | None = None
        feature_indices_ref: list[np.ndarray] | None = None
        time_axis_ref: np.ndarray | None = None

        # --- Step 2: iterate over iEEG files ---
        for ieeg_file in ieeg_files:
            with ieeg_file.ensure_loaded() as raw:
                sfreq = float(raw.info["sfreq"])
                channel_names = list(raw.ch_names)
                # Capture reference values from the first file; validate consistency for the rest.
                if sfreq_ref is None:
                    sfreq_ref = sfreq
                    channel_names_ref = channel_names
                else:
                    if sfreq != sfreq_ref:
                        raise ValueError(
                            "All ieeg files in a subject group must share the same "
                            "sampling frequency."
                        )
                    if channel_names != channel_names_ref:
                        raise ValueError(
                            "All ieeg files in a subject group must share the same "
                            "channel ordering."
                        )

                # Atlas mode: map channels to brain regions, replacing channel-level features.
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
                            "Atlas region grouping must be consistent across all ieeg files "
                            f"in a subject group. Got {grouping.feature_names} for "
                            f"{ieeg_file.path.name} but expected {feature_names_ref}."
                        )

                # Extract only the annotations whose codes mark trial onsets.
                anchor_events, anchor_samples = extract_anchor_events_with_mne(
                    raw,
                    anchor_codes=anchor_codes,
                )
                resolved_trials = self.resolver.resolve_trials(
                    group,
                    ieeg_file,
                    anchor_events,
                )
                if len(resolved_trials) != len(anchor_events):
                    raise ValueError(
                        f"Resolver returned {len(resolved_trials)} trial rows for "
                        f"{len(anchor_events)} anchor events in {ieeg_file.path.name}."
                    )

                # Cut the continuous recording into per-trial windows.
                extraction = extract_epochs_with_mne(
                    raw,
                    anchor_samples=anchor_samples,
                    trials=self._normalize_trial_labels(resolved_trials),
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

                # Route each kept epoch to the appropriate condition list.
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
                    if atlas_mode:
                        assert epoch_for_stats.shape[0] == len(feature_names_ref or [])

                    if trial.label == self.params.condition_a:
                        epochs_a.append(epoch_for_stats)
                    elif trial.label == self.params.condition_b:
                        epochs_b.append(epoch_for_stats)

        assert sfreq_ref is not None
        assert channel_names_ref is not None
        assert time_axis_ref is not None

        # --- Step 3: stack epochs and apply optional temporal binning ---
        if atlas_mode:
            assert feature_names_ref is not None
            feature_names = feature_names_ref
        else:
            feature_names = channel_names_ref

        epochs_a_array = stack_epochs(
            epochs_a,
            len(feature_names),
            len(time_axis_ref),
        )
        epochs_b_array = stack_epochs(
            epochs_b,
            len(feature_names),
            len(time_axis_ref),
        )

        time_axis_eval = time_axis_ref
        binning_mode = "none"
        window_sample_count = 0
        effective_n_bins = int(len(time_axis_ref))
        if self.params.window_ms > 0:
            # Reduce temporal resolution by averaging consecutive time windows.
            binning_mode = "window_ms"
            window_sample_count = compute_window_samples(
                sfreq_ref,
                self.params.window_ms,
            )
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

        # --- Step 4: compute statistics ---
        # Only run the t-test when both conditions have enough trials.
        stats_valid = (
            epochs_a_array.shape[0] >= self.params.min_trials_per_condition
            and epochs_b_array.shape[0] >= self.params.min_trials_per_condition
        )
        t_values, p_values_raw, mean_a, mean_b, mean_difference = (
            compute_condition_statistics(
                epochs_a_array if stats_valid else np.empty_like(epochs_a_array[:0]),
                epochs_b_array if stats_valid else np.empty_like(epochs_b_array[:0]),
                n_channels=len(feature_names),
                n_times=len(time_axis_eval),
                equal_var=self.params.equal_var,
            )
        )
        # Recompute per-condition means over all trials (stats may have used empty arrays).
        if epochs_a_array.size:
            mean_a = np.nanmean(epochs_a_array, axis=0, dtype=np.float64)
        if epochs_b_array.size:
            mean_b = np.nanmean(epochs_b_array, axis=0, dtype=np.float64)
        mean_difference = mean_a - mean_b
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
        difference_sem = np.sqrt(
            np.square(condition_a_sem, dtype=np.float64)
            + np.square(condition_b_sem, dtype=np.float64)
        )
        difference_ci95_low, difference_ci95_high = compute_bootstrap_difference_ci95(
            epochs_a_array,
            epochs_b_array,
            n_bootstraps=2000,
            random_state=self.params.permutation_seed,
        )

        p_values = correct_p_values(
            p_values_raw,
            method=self.params.p_value_correction_method,
        )

        # --- Build permuted null distribution (when n_permutations > 0) ---
        permuted_t_values: np.ndarray | None = None
        if self.params.n_permutations > 0 and stats_valid:
            rng = np.random.default_rng(self.params.permutation_seed)
            permuted_t_values = compute_permuted_statistics(
                epochs_a_array,
                epochs_b_array,
                self.params.n_permutations,
                rng,
                equal_var=self.params.equal_var,
            )

        # When method='permutation', overwrite p_values with pointwise permutation p-values.
        if self.params.p_value_correction_method == "permutation":
            if permuted_t_values is not None:
                p_values = compute_permutation_p_values(t_values, permuted_t_values)
            else:
                # Stats invalid or n_permutations==0 (the validator prevents n_perm==0 here,
                # so this only fires when stats_valid is False — leave p_values as NaN).
                p_values = p_values_raw.copy()

        significant_mask = np.isfinite(p_values) & (p_values < self.params.significance_alpha)

        # --- Compute optional per-channel significance flag ---
        channel_significant_mask: np.ndarray | None = None
        if self.params.channel_significance_mode == "single_bin" and stats_valid:
            rng_csm = np.random.default_rng(self.params.permutation_seed)
            n_perm_csm = (
                self.params.n_permutations
                if self.params.p_value_correction_method == "permutation"
                else 0
            )
            channel_significant_mask = compute_single_bin_channel_significance(
                epochs_a_array,
                epochs_b_array,
                equal_var=self.params.equal_var,
                p_value_correction_method=self.params.p_value_correction_method,
                significance_alpha=self.params.significance_alpha,
                n_permutations=n_perm_csm,
                rng=rng_csm,
            )
        elif self.params.channel_significance_mode == "duration" and stats_valid:
            channel_significant_mask = compute_duration_channel_significance(
                significant_mask,
                time_axis_eval,
                threshold_ms=self.params.channel_significance_duration_threshold_ms,
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

        return TrialStatsProcessingResult(
            source_group=group,
            output_entities=shared_entities(
                ieeg_files,
                excluded_entities=frozenset(
                    {"suffix", "extension", "datatype", "desc", "description", "run"}
                ),
            ),
            metadata={
                "anchor_event_codes": list(self.params.anchor_event_codes),
                "tmin_s": self.params.tmin_s,
                "tmax_s": self.params.tmax_s,
                "min_trials_per_condition": self.params.min_trials_per_condition,
                "drop_partial_epochs": self.params.drop_partial_epochs,
                "equal_var": self.params.equal_var,
                "p_value_correction_method": self.params.p_value_correction_method,
                "significance_alpha": self.params.significance_alpha,
                "n_permutations": self.params.n_permutations,
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
                "channel_significance_mode": self.params.channel_significance_mode,
                "channel_significance_duration_threshold_ms": self.params.channel_significance_duration_threshold_ms,
            },
            t_values=t_values,
            p_values=p_values,
            p_values_uncorrected=p_values_raw,
            condition_a_mean=mean_a,
            condition_b_mean=mean_b,
            mean_difference=mean_difference,
            condition_a_sem=condition_a_sem,
            condition_b_sem=condition_b_sem,
            difference_sem=difference_sem,
            difference_ci95_low=difference_ci95_low,
            difference_ci95_high=difference_ci95_high,
            significant_mask=significant_mask,
            time_axis_s=time_axis_eval,
            channel_names=feature_names,
            condition_a=self.params.condition_a,
            condition_b=self.params.condition_b,
            condition_a_trial_count=int(epochs_a_array.shape[0]),
            condition_b_trial_count=int(epochs_b_array.shape[0]),
            sfreq=sfreq_ref,
            resolved_trials=all_resolved_trials,
            source_ieeg_files=[str(file.path) for file in ieeg_files],
            source_table_files=source_table_files,
            source_electrodes_files=source_electrodes_files,
            analysis_level="roi" if atlas_mode else "channel",
            atlas_name=self.params.atlas_name,
            atlas_regions=feature_names if atlas_mode else [],
            region_channels=region_channels,
            window_ms=self.params.window_ms,
            n_bins=self.params.n_bins,
            p_value_correction_method=self.params.p_value_correction_method,
            significance_alpha=self.params.significance_alpha,
            stats_valid=stats_valid,
            condition_a_epochs=epochs_a_array,
            condition_b_epochs=epochs_b_array,
            permuted_t_values=permuted_t_values,
            channel_significant_mask=channel_significant_mask,
        )

    def _normalize_trial_labels(
        self,
        trials: Sequence[ResolvedTrial],
    ) -> list[ResolvedTrial]:
        """Mark trials whose label is not condition_a or condition_b as excluded."""
        normalized: list[ResolvedTrial] = []
        supported_labels = {self.params.condition_a, self.params.condition_b}
        for trial in trials:
            if not trial.keep:
                normalized.append(trial)
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
                        metadata=dict(trial.metadata),
                    )
                )
            else:
                normalized.append(trial)
        return normalized

