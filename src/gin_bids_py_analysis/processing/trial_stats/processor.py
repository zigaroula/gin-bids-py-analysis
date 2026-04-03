"""Processor for per-subject condition-A vs condition-B statistics on iEEG derivatives.

Orchestrates file loading, trial resolution, epoch extraction, optional atlas-region
aggregation, temporal binning, and statistical testing across all iEEG files in one
``BIDSFileGroup``.  Heavy numerical work is delegated to ``stats.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import mne
import numpy as np

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.bids.matching import (
    files_matching_entities,
    find_best_entity_match,
    shared_entities,
)
from gin_bids_py_analysis.processing.base import BaseProcessing
from gin_bids_py_analysis.processing.utils.channels import normalize_channel_name
from gin_bids_py_analysis.processing.utils.events import (
    AnnotationEvent,
    parse_annotation_description,
)
from gin_bids_py_analysis.processing.utils.tables import select_column

from .params import TrialStatsParams
from .resolver import ResolvedTrial, TrialLabelResolver
from .result import TrialStatsProcessingResult
from .stats import (
    EpochExtractionResult,
    build_time_axis_s,
    compute_bootstrap_difference_ci95,
    compute_condition_statistics,
    compute_duration_channel_significance,
    compute_permutation_p_values,
    compute_permuted_statistics,
    compute_single_bin_channel_significance,
    correct_p_values,
)


@dataclass(frozen=True)
class _AtlasGrouping:
    """Intermediate result of mapping iEEG channels to atlas regions for one file."""

    source_file: BIDSFile              # Electrodes TSV/CSV the mapping was built from.
    feature_names: list[str]           # Ordered list of atlas region names.
    feature_channel_indices: list[np.ndarray]  # Per-region channel row indices into the data matrix.
    missing_regions: list[str]         # Requested regions absent from the electrodes table.


class TrialStatsProcessing(BaseProcessing):
    """Compute per-subject condition_a-vs-condition_b statistics on ieeg data."""

    def __init__(
        self,
        params: TrialStatsParams,
        resolver: TrialLabelResolver,
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
                    grouping = _resolve_atlas_grouping(
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
                anchor_events, anchor_samples = _extract_anchor_events_with_mne(
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
                extraction = _extract_epochs_with_mne(
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
                    epochs_for_stats = _aggregate_epochs_with_mne(
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

        epochs_a_array = _stack_epochs(
            epochs_a,
            len(feature_names),
            len(time_axis_ref),
        )
        epochs_b_array = _stack_epochs(
            epochs_b,
            len(feature_names),
            len(time_axis_ref),
        )

        time_axis_eval = time_axis_ref
        binning_mode = "none"
        window_samples = 0
        effective_n_bins = int(len(time_axis_ref))
        if self.params.window_ms > 0:
            # Reduce temporal resolution by averaging consecutive time windows.
            binning_mode = "window_ms"
            window_samples = _window_samples(
                sfreq_ref,
                self.params.window_ms,
            )
            epochs_a_array, time_axis_eval = _temporal_bin_epochs(
                epochs_a_array,
                time_axis_ref,
                window_samples,
            )
            epochs_b_array, _ = _temporal_bin_epochs(
                epochs_b_array,
                time_axis_ref,
                window_samples,
            )
            effective_n_bins = int(len(time_axis_eval))
        elif self.params.n_bins > 0:
            if self.params.n_bins > len(time_axis_ref):
                raise ValueError(
                    "n_bins cannot be greater than the number of epoch samples "
                    f"({len(time_axis_ref)})."
                )
            binning_mode = "n_bins"
            epochs_a_array, time_axis_eval = _temporal_bin_epochs_by_n_bins(
                epochs_a_array,
                time_axis_ref,
                self.params.n_bins,
            )
            epochs_b_array, _ = _temporal_bin_epochs_by_n_bins(
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
        condition_a_sem = _compute_condition_sem(
            epochs_a_array,
            n_channels=len(feature_names),
            n_times=len(time_axis_eval),
        )
        condition_b_sem = _compute_condition_sem(
            epochs_b_array,
            n_channels=len(feature_names),
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
                "window_samples": window_samples,
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


def _extract_anchor_events_with_mne(
    raw: mne.io.BaseRaw,
    *,
    anchor_codes: set[str],
) -> tuple[list[AnnotationEvent], np.ndarray]:
    """Extract anchor events using MNE's sample-accurate annotation parser."""
    sfreq = float(raw.info["sfreq"])

    events, _ = mne.events_from_annotations(
        raw,
        event_id=_annotation_code_parser,
        use_rounding=True,
        verbose=False,
    )
    if events.size == 0:
        return [], np.empty((0,), dtype=np.int64)

    durations_by_key: dict[tuple[int, str], list[float]] = {}
    for annotation in raw.annotations:
        _, _, code = parse_annotation_description(str(annotation["description"]))
        if code is None or code not in anchor_codes:
            continue
        sample = int(round(float(annotation["onset"]) * sfreq))
        durations_by_key.setdefault((sample, code), []).append(float(annotation["duration"]))

    anchor_events: list[AnnotationEvent] = []
    anchor_samples: list[int] = []
    for sample, _, event_code in events:
        code = str(int(event_code))
        if code not in anchor_codes:
            continue
        key = (int(sample), code)
        durations = durations_by_key.get(key, [])
        duration_s = durations.pop(0) if durations else 0.0
        anchor_events.append(
            AnnotationEvent(
                onset_s=float(sample) / sfreq,
                duration_s=duration_s,
                event_type="Stimulus",
                description=f"S {code}",
                code=code,
            )
        )
        anchor_samples.append(int(sample))

    return anchor_events, np.asarray(anchor_samples, dtype=np.int64)


def _annotation_code_parser(description: str) -> int | None:
    """Return integer event code parsed from an annotation description."""
    _, _, code = parse_annotation_description(description)
    if code is None:
        return None
    try:
        return int(code)
    except ValueError:
        return None


def _extract_epochs_with_mne(
    raw: mne.io.BaseRaw,
    *,
    anchor_samples: np.ndarray,
    trials: list[ResolvedTrial],
    tmin_s: float,
    tmax_s: float,
    drop_partial_epochs: bool,
) -> EpochExtractionResult:
    """Extract per-trial epochs via ``mne.Epochs`` and map dropped trials to exclusions."""
    sfreq = float(raw.info["sfreq"])
    fallback_time_axis = build_time_axis_s(sfreq, tmin_s, tmax_s)
    if not trials:
        return EpochExtractionResult(
            epochs=np.empty((0, len(raw.ch_names), len(fallback_time_axis)), dtype=np.float64),
            kept_trials=[],
            updated_trials=[],
            time_axis_s=fallback_time_axis,
        )
    if len(trials) != int(anchor_samples.size):
        raise ValueError(
            f"Trial/event length mismatch: {len(trials)} trials for {anchor_samples.size} anchor samples."
        )

    kept_indices = [idx for idx, trial in enumerate(trials) if trial.keep]
    updated_trials = list(trials)
    if not kept_indices:
        return EpochExtractionResult(
            epochs=np.empty((0, len(raw.ch_names), len(fallback_time_axis)), dtype=np.float64),
            kept_trials=[],
            updated_trials=updated_trials,
            time_axis_s=fallback_time_axis,
        )

    events = np.column_stack(
        [
            anchor_samples[np.asarray(kept_indices, dtype=np.int64)],
            np.zeros(len(kept_indices), dtype=np.int64),
            np.ones(len(kept_indices), dtype=np.int64),
        ]
    )
    epochs_obj = mne.Epochs(
        raw,
        events=events,
        event_id={"anchor": 1},
        tmin=tmin_s,
        tmax=tmax_s,
        baseline=None,
        preload=True,
        reject_by_annotation=False,
        verbose=False,
    )
    time_axis_s = np.asarray(epochs_obj.times, dtype=np.float64)
    drop_log = list(epochs_obj.drop_log)
    kept_trials: list[ResolvedTrial] = []
    kept_epoch_rows: list[np.ndarray] = []
    kept_data = np.asarray(epochs_obj.get_data(copy=True), dtype=np.float64)
    kept_data_cursor = 0

    for local_idx, trial_idx in enumerate(kept_indices):
        reasons = tuple(drop_log[local_idx]) if local_idx < len(drop_log) else ()
        if reasons:
            is_partial = any(reason in {"TOO_SHORT", "NO_DATA"} for reason in reasons)
            if not drop_partial_epochs and is_partial:
                raise ValueError(
                    f"Trial at {trials[trial_idx].anchor_onset_s:.6f}s would create a partial epoch."
                )
            updated_trials[trial_idx] = replace(
                trials[trial_idx],
                keep=False,
                exclusion_reason=trials[trial_idx].exclusion_reason or "partial_epoch",
            )
            continue

        kept_trials.append(updated_trials[trial_idx])
        if kept_data_cursor >= kept_data.shape[0]:
            raise ValueError("Internal epoch extraction mismatch: missing kept epoch row.")
        kept_epoch_rows.append(kept_data[kept_data_cursor])
        kept_data_cursor += 1

    epochs_array = (
        np.stack(kept_epoch_rows, axis=0).astype(np.float64)
        if kept_epoch_rows
        else np.empty((0, len(raw.ch_names), len(time_axis_s)), dtype=np.float64)
    )
    return EpochExtractionResult(
        epochs=epochs_array,
        kept_trials=kept_trials,
        updated_trials=updated_trials,
        time_axis_s=time_axis_s,
    )


def _aggregate_epochs_with_mne(
    epochs: np.ndarray,
    *,
    channel_names: Sequence[str],
    feature_names: Sequence[str],
    feature_channel_indices: Sequence[np.ndarray],
    sfreq: float,
    tmin_s: float,
) -> np.ndarray:
    """Aggregate channels into ROI means via ``mne.channels.combine_channels``."""
    if epochs.ndim != 3:
        raise ValueError(
            f"epochs must be 3-D (n_epochs, n_channels, n_times), got {epochs.shape!r}."
        )
    if epochs.shape[0] == 0:
        return np.empty((0, len(feature_names), epochs.shape[2]), dtype=np.float64)

    info = mne.create_info(
        ch_names=list(channel_names),
        sfreq=float(sfreq),
        ch_types=["seeg"] * len(channel_names),
    )
    events = np.column_stack(
        [
            np.arange(epochs.shape[0], dtype=np.int64),
            np.zeros(epochs.shape[0], dtype=np.int64),
            np.ones(epochs.shape[0], dtype=np.int64),
        ]
    )
    epochs_obj = mne.EpochsArray(
        np.asarray(epochs, dtype=np.float64),
        info=info,
        events=events,
        event_id={"anchor": 1},
        tmin=float(tmin_s),
        verbose=False,
    )
    groups = {
        str(name): np.asarray(indices, dtype=np.int64).tolist()
        for name, indices in zip(feature_names, feature_channel_indices)
    }
    combined = mne.channels.combine_channels(
        epochs_obj,
        groups=groups,
        method="mean",
        keep_stim=False,
        drop_bad=False,
    )
    return np.asarray(combined.get_data(copy=True), dtype=np.float64)


def _stack_epochs(
    epochs: Sequence[np.ndarray],
    n_channels: int,
    n_times: int,
) -> np.ndarray:
    """Stack a list of (n_channels, n_times) epoch arrays into a (n_epochs, n_channels, n_times) array.

    Returns an empty array of the correct shape when *epochs* is empty.
    """
    if not epochs:
        return np.empty((0, n_channels, n_times), dtype=np.float64)
    return np.stack(epochs, axis=0).astype(np.float64)


def _compute_condition_sem(
    epochs: np.ndarray,
    *,
    n_channels: int,
    n_times: int,
) -> np.ndarray:
    """Return per-feature SEM across trials, or NaN when fewer than 2 trials."""
    if epochs.shape[0] < 2:
        return np.full((n_channels, n_times), np.nan, dtype=np.float64)

    std = np.nanstd(
        epochs,
        axis=0,
        ddof=1,
        dtype=np.float64,
    )
    return std / np.sqrt(float(epochs.shape[0]))


def _window_samples(sfreq: float, window_ms: float) -> int:
    """Convert a temporal window duration (ms) to the nearest sample count (minimum 1)."""
    return max(1, int(round((window_ms / 1000.0) * sfreq)))


def _temporal_bin_epochs(
    epochs: np.ndarray,
    time_axis_s: np.ndarray,
    window_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average non-overlapping temporal windows, returning downsampled epochs and time axis.

    Input shape:  (n_epochs, n_channels, n_times)
    Output shape: (n_epochs, n_channels, n_bins)
    """
    if window_samples <= 1:
        return epochs, time_axis_s

    n_epochs, n_channels, n_times = epochs.shape
    starts = list(range(0, n_times, window_samples))
    # Epoch extraction is inclusive on both bounds, which commonly leaves a
    # trailing 1-sample tail for otherwise exact-duration windows. Merge that
    # sample into the previous bin to avoid a visually confusing tiny final bin.
    if len(starts) >= 2 and (n_times - starts[-1]) == 1:
        starts = starts[:-1]

    stops = [min(start + window_samples, n_times) for start in starts]
    if stops and stops[-1] < n_times:
        stops[-1] = n_times

    return _aggregate_time_bins(epochs, time_axis_s, starts, stops)


def _temporal_bin_epochs_by_n_bins(
    epochs: np.ndarray,
    time_axis_s: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average into exactly ``n_bins`` contiguous non-overlapping temporal bins."""
    if n_bins <= 0:
        return epochs, time_axis_s

    n_times = int(epochs.shape[2])
    if n_bins == n_times:
        return epochs, time_axis_s
    if n_bins > n_times:
        raise ValueError(
            f"n_bins={n_bins} cannot exceed n_times={n_times}."
        )

    starts = [
        (idx * n_times) // n_bins
        for idx in range(n_bins)
    ]
    stops = [
        ((idx + 1) * n_times) // n_bins
        for idx in range(n_bins)
    ]
    return _aggregate_time_bins(epochs, time_axis_s, starts, stops)


def _aggregate_time_bins(
    epochs: np.ndarray,
    time_axis_s: np.ndarray,
    starts: Sequence[int],
    stops: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate epochs over explicit ``[start, stop)`` temporal slices."""
    if len(starts) != len(stops):
        raise ValueError("starts and stops must have the same length.")
    if not starts:
        n_epochs, n_channels, _ = epochs.shape
        return np.empty((n_epochs, n_channels, 0), dtype=np.float64), np.empty((0,), dtype=np.float64)

    for start, stop in zip(starts, stops):
        if start < 0 or stop <= start or stop > len(time_axis_s):
            raise ValueError(
                f"Invalid temporal slice [{start}, {stop}) for n_times={len(time_axis_s)}."
            )

    n_epochs, n_channels, _ = epochs.shape
    binned_time_axis = np.array(
        [float(np.nanmean(time_axis_s[start:stop])) for start, stop in zip(starts, stops)],
        dtype=np.float64,
    )
    if n_epochs == 0:
        return np.empty((0, n_channels, len(starts)), dtype=np.float64), binned_time_axis

    binned = np.empty((n_epochs, n_channels, len(starts)), dtype=np.float64)
    for bin_idx, (start, stop) in enumerate(zip(starts, stops)):
        binned[:, :, bin_idx] = np.nanmean(
            epochs[:, :, start:stop],
            axis=2,
            dtype=np.float64,
        ).astype(np.float64)
    return binned, binned_time_axis


def _is_na_like_region_label(label: str) -> bool:
    """Return True for common ``n/a`` placeholder variants."""
    normalized = "".join(ch for ch in label.strip().casefold() if ch.isalnum())
    return normalized in {"na", "notavailable", "notapplicable"}


def _resolve_atlas_grouping(
    *,
    ieeg_file: BIDSFile,
    electrodes_files: Sequence[BIDSFile],
    channel_names: Sequence[str],
    atlas_name: str,
    atlas_regions: Sequence[str],
) -> _AtlasGrouping:
    """Build an ``_AtlasGrouping`` for *ieeg_file* from the best-matching electrodes table.

    Steps:
    1. Find the electrodes file whose entities best match the iEEG file.
    2. Parse the table to build a channel→region mapping using the *atlas_name* column.
    3. Group iEEG channel indices by region; optionally filter to *atlas_regions*.
    """
    # Step 1: find the best-matching electrodes file.
    electrodes_file = find_best_entity_match(
        ieeg_file,
        electrodes_files,
        ambiguity_label="electrodes table",
        ambiguity_hint=(
            "Please disambiguate entities (e.g. run/task) in *_electrodes.tsv."
        ),
    )
    if electrodes_file is None:
        raise ValueError(
            f"atlas_name={atlas_name!r} requires a matching *_electrodes.tsv/csv file "
            f"for {ieeg_file.path.name}."
        )

    # Step 2: parse the electrodes table and build the channel→region mapping.
    with electrodes_file.ensure_loaded() as rows:
        if not rows:
            raise ValueError(f"Electrodes table {electrodes_file.path.name} is empty.")

        columns = list(rows[0].keys())
        channel_col = select_column(columns, preferred=["name", "channel", "label"])
        atlas_col = select_column(columns, preferred=[atlas_name])
        if channel_col is None:
            raise ValueError(
                f"Electrodes table {electrodes_file.path.name} must contain a channel name "
                "column (e.g. 'name')."
            )
        if atlas_col is None:
            raise ValueError(
                f"Electrodes table {electrodes_file.path.name} has no column matching "
                f"atlas_name={atlas_name!r}."
            )

        channel_to_region: dict[str, str] = {}
        for row in rows:
            raw_channel = (row.get(channel_col) or "").strip()
            raw_region = (row.get(atlas_col) or "").strip()
            if not raw_channel or not raw_region:
                continue
            channel_key = normalize_channel_name(raw_channel)
            previous = channel_to_region.get(channel_key)
            if previous is not None and previous != raw_region:
                raise ValueError(
                    f"Channel {raw_channel!r} has multiple atlas labels in "
                    f"{electrodes_file.path.name}: {previous!r} and {raw_region!r}."
                )
            channel_to_region[channel_key] = raw_region

        if not channel_to_region:
            raise ValueError(
                f"No non-empty atlas labels found in column {atlas_col!r} of "
                f"{electrodes_file.path.name}."
            )

    # Step 3: group iEEG channel row indices by atlas region.
    region_to_indices: dict[str, list[int]] = {}
    for idx, channel_name in enumerate(channel_names):
        channel_key = normalize_channel_name(channel_name)
        region = channel_to_region.get(channel_key)
        if region is None:
            continue
        region_to_indices.setdefault(region, []).append(idx)

    if not region_to_indices:
        raise ValueError(
            f"No ieeg channels from {ieeg_file.path.name} matched atlas labels in "
            f"{electrodes_file.path.name}."
        )

    # Filter to the requested subset of regions (if any), collecting missing ones.
    missing_regions: list[str] = []
    if atlas_regions:
        available_by_norm = {
            region.casefold(): region for region in region_to_indices
        }
        selected_regions: list[str] = []
        for requested in atlas_regions:
            region = available_by_norm.get(requested.casefold())
            if region is None:
                missing_regions.append(requested)
                continue
            if region not in selected_regions:
                selected_regions.append(region)
        if not selected_regions:
            available = ", ".join(sorted(region_to_indices.keys()))
            raise ValueError(
                f"None of atlas_regions={list(atlas_regions)!r} were found in "
                f"{electrodes_file.path.name}. Available regions: {available}"
            )
    else:
        selected_regions = [
            region
            for region in region_to_indices
            if not _is_na_like_region_label(region)
        ]
        if not selected_regions:
            available = ", ".join(sorted(region_to_indices.keys()))
            raise ValueError(
                "No usable atlas regions remained after removing n/a-like labels in "
                f"{electrodes_file.path.name}. Available regions: {available}"
            )

    feature_indices = [
        np.asarray(region_to_indices[region], dtype=np.int64)
        for region in selected_regions
    ]

    return _AtlasGrouping(
        source_file=electrodes_file,
        feature_names=selected_regions,
        feature_channel_indices=feature_indices,
        missing_regions=missing_regions,
    )
