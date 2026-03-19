"""Processor for per-subject condition-A vs condition-B statistics on iEEG derivatives.

Orchestrates file loading, trial resolution, epoch extraction, optional atlas-region
aggregation, temporal binning, and statistical testing across all iEEG files in one
``BIDSFileGroup``.  Heavy numerical work is delegated to ``stats.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.bids.matching import (
    files_matching_entities,
    find_best_entity_match,
    shared_entities,
)
from gin_bids_py_analysis.data.loader import load_ieeg
from gin_bids_py_analysis.processing.base import BaseProcessing
from gin_bids_py_analysis.processing.utils.channels import normalize_channel_name
from gin_bids_py_analysis.processing.utils.events import coerce_annotation_events
from gin_bids_py_analysis.processing.utils.tables import read_table_rows, select_column

from .params import TrialStatsParams
from .resolver import ResolvedTrial, TrialLabelResolver
from .result import TrialStatsProcessingResult
from .stats import (
    build_time_axis_s,
    compute_condition_statistics,
    correct_p_values,
    extract_epochs,
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
            raw = load_ieeg(ieeg_file)
            sfreq = float(raw.info["sfreq"])
            data = raw.get_data().astype(np.float32)
            channel_names = list(raw.ch_names)

            # Capture reference values from the first file; validate consistency for the rest.
            if sfreq_ref is None:
                sfreq_ref = sfreq
                channel_names_ref = channel_names
                time_axis_ref = build_time_axis_s(
                    sfreq,
                    self.params.tmin_s,
                    self.params.tmax_s,
                )
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
            anchor_events = [
                event
                for event in coerce_annotation_events(raw.annotations)
                if event.code in anchor_codes
            ]
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
            extraction = extract_epochs(
                data,
                sfreq,
                self._normalize_trial_labels(resolved_trials),
                self.params.tmin_s,
                self.params.tmax_s,
                drop_partial_epochs=self.params.drop_partial_epochs,
            )
            all_resolved_trials.extend(extraction.updated_trials)

            # Route each kept epoch to the appropriate condition list.
            for epoch, trial in zip(extraction.epochs, extraction.kept_trials):
                epoch_for_stats = epoch
                if atlas_mode:
                    # Collapse channel dimension into atlas-region means.
                    assert feature_indices_ref is not None
                    epoch_for_stats = _aggregate_channels(epoch, feature_indices_ref)

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
        temporal_window_samples = 1
        if self.params.temporal_window_ms > 0:
            # Reduce temporal resolution by averaging consecutive sample bins.
            temporal_window_samples = _window_samples(
                sfreq_ref,
                self.params.temporal_window_ms,
            )
            epochs_a_array, time_axis_eval = _temporal_bin_epochs(
                epochs_a_array,
                time_axis_ref,
                temporal_window_samples,
            )
            epochs_b_array, _ = _temporal_bin_epochs(
                epochs_b_array,
                time_axis_ref,
                temporal_window_samples,
            )

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

        p_values = correct_p_values(
            p_values_raw,
            method=self.params.p_value_correction_method,
        )
        significant_mask = np.isfinite(p_values) & (p_values < self.params.significance_alpha)

        source_electrodes_files = sorted(used_electrodes_paths)
        if not source_electrodes_files and electrodes_files:
            source_electrodes_files = sorted({str(file.path) for file in electrodes_files})

        return TrialStatsProcessingResult(
            source_group=group,
            output_entities=shared_entities(ieeg_files),
            metadata={
                "anchor_event_codes": list(self.params.anchor_event_codes),
                "tmin_s": self.params.tmin_s,
                "tmax_s": self.params.tmax_s,
                "min_trials_per_condition": self.params.min_trials_per_condition,
                "drop_partial_epochs": self.params.drop_partial_epochs,
                "equal_var": self.params.equal_var,
                "p_value_correction_method": self.params.p_value_correction_method,
                "significance_alpha": self.params.significance_alpha,
                "analysis_level": "roi" if atlas_mode else "channel",
                "atlas_name": self.params.atlas_name,
                "atlas_regions": feature_names if atlas_mode else [],
                "atlas_regions_requested": list(self.params.atlas_regions),
                "atlas_regions_missing": sorted(missing_atlas_regions),
                "temporal_window_ms": self.params.temporal_window_ms,
                "temporal_window_samples": temporal_window_samples,
            },
            t_values=t_values,
            p_values=p_values,
            p_values_uncorrected=p_values_raw,
            condition_a_mean=mean_a,
            condition_b_mean=mean_b,
            mean_difference=mean_difference,
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
            temporal_window_ms=self.params.temporal_window_ms,
            p_value_correction_method=self.params.p_value_correction_method,
            significance_alpha=self.params.significance_alpha,
            stats_valid=stats_valid,
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

def _stack_epochs(
    epochs: Sequence[np.ndarray],
    n_channels: int,
    n_times: int,
) -> np.ndarray:
    """Stack a list of (n_channels, n_times) epoch arrays into a (n_epochs, n_channels, n_times) array.

    Returns an empty array of the correct shape when *epochs* is empty.
    """
    if not epochs:
        return np.empty((0, n_channels, n_times), dtype=np.float32)
    return np.stack(epochs, axis=0).astype(np.float32)


def _window_samples(sfreq: float, temporal_window_ms: float) -> int:
    """Convert a temporal window duration (ms) to the nearest sample count (minimum 1)."""
    return max(1, int(round((temporal_window_ms / 1000.0) * sfreq)))


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
    binned_time_axis = np.array(
        [
            float(np.nanmean(time_axis_s[start:min(start + window_samples, n_times)]))
            for start in starts
        ],
        dtype=np.float64,
    )

    if n_epochs == 0:
        return np.empty((0, n_channels, len(starts)), dtype=np.float32), binned_time_axis

    binned = np.empty((n_epochs, n_channels, len(starts)), dtype=np.float32)
    for bin_idx, start in enumerate(starts):
        stop = min(start + window_samples, n_times)
        binned[:, :, bin_idx] = np.nanmean(
            epochs[:, :, start:stop],
            axis=2,
            dtype=np.float64,
        ).astype(np.float32)

    return binned, binned_time_axis


def _aggregate_channels(
    epoch: np.ndarray,
    channel_indices_by_feature: Sequence[np.ndarray],
) -> np.ndarray:
    """Replace channel rows with per-region means.

    Input shape:  (n_channels, n_times)
    Output shape: (n_regions, n_times)
    """
    aggregated = [
        np.nanmean(epoch[channel_indices, :], axis=0, dtype=np.float64)
        for channel_indices in channel_indices_by_feature
    ]
    return np.stack(aggregated, axis=0).astype(np.float32)


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
    rows = read_table_rows(electrodes_file)
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
        selected_regions = list(region_to_indices.keys())

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
