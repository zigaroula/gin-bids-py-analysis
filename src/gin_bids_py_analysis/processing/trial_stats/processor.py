from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.data.loader import load_ieeg
from gin_bids_py_analysis.processing.base import BaseProcessing
from gin_bids_py_analysis.processing.utils.events import coerce_annotation_events

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
    source_file: BIDSFile
    feature_names: list[str]
    feature_channel_indices: list[np.ndarray]
    missing_regions: list[str]


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
        del progress_tracking_position

        ieeg_files = _ieeg_files(group)
        if not ieeg_files:
            raise ValueError(
                "TrialStatsProcessing requires at least one ieeg BrainVision file."
            )

        source_table_files = sorted(
            {
                str(file.path)
                for file in group.all_files
                if file.extension in {".tsv", ".csv"} and file.suffix != "electrodes"
            }
        )
        electrodes_files = _electrodes_files(group)

        atlas_mode = bool(self.params.atlas_name)
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

        for ieeg_file in ieeg_files:
            raw = load_ieeg(ieeg_file)
            sfreq = float(raw.info["sfreq"])
            data = raw.get_data().astype(np.float32)
            channel_names = list(raw.ch_names)

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

            extraction = extract_epochs(
                data,
                sfreq,
                self._normalize_trial_labels(resolved_trials),
                self.params.tmin_s,
                self.params.tmax_s,
                drop_partial_epochs=self.params.drop_partial_epochs,
            )
            all_resolved_trials.extend(extraction.updated_trials)

            for epoch, trial in zip(extraction.epochs, extraction.kept_trials):
                epoch_for_stats = epoch
                if atlas_mode:
                    assert feature_indices_ref is not None
                    epoch_for_stats = _aggregate_channels(epoch, feature_indices_ref)

                if trial.label == self.params.condition_a:
                    epochs_a.append(epoch_for_stats)
                elif trial.label == self.params.condition_b:
                    epochs_b.append(epoch_for_stats)

        assert sfreq_ref is not None
        assert channel_names_ref is not None
        assert time_axis_ref is not None

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
            output_entities=_shared_entities(ieeg_files),
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


def _ieeg_files(group: BIDSFileGroup) -> list[BIDSFile]:
    return sorted(
        [
            file
            for file in group.all_files
            if file.extension == ".vhdr" and file.suffix == "ieeg"
        ],
        key=lambda file: str(file.path),
    )


def _electrodes_files(group: BIDSFileGroup) -> list[BIDSFile]:
    return sorted(
        [
            file
            for file in group.all_files
            if file.extension in {".tsv", ".csv"} and file.suffix == "electrodes"
        ],
        key=lambda file: str(file.path),
    )


def _shared_entities(files: Sequence[BIDSFile]) -> dict[str, str]:
    shared = dict(files[0].entities)
    for file in files[1:]:
        shared = {
            key: value
            for key, value in shared.items()
            if file.get(key) == value
        }
    for removable in ("suffix", "extension", "datatype", "desc", "description"):
        shared.pop(removable, None)
    return {str(key): str(value) for key, value in shared.items()}


def _stack_epochs(
    epochs: Sequence[np.ndarray],
    n_channels: int,
    n_times: int,
) -> np.ndarray:
    if not epochs:
        return np.empty((0, n_channels, n_times), dtype=np.float32)
    return np.stack(epochs, axis=0).astype(np.float32)


def _window_samples(sfreq: float, temporal_window_ms: float) -> int:
    return max(1, int(round((temporal_window_ms / 1000.0) * sfreq)))


def _temporal_bin_epochs(
    epochs: np.ndarray,
    time_axis_s: np.ndarray,
    window_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    electrodes_file = _find_matching_electrodes_file(ieeg_file, electrodes_files)
    if electrodes_file is None:
        raise ValueError(
            f"atlas_name={atlas_name!r} requires a matching *_electrodes.tsv/csv file "
            f"for {ieeg_file.path.name}."
        )

    rows = _read_table_rows(electrodes_file)
    if not rows:
        raise ValueError(f"Electrodes table {electrodes_file.path.name} is empty.")

    columns = list(rows[0].keys())
    channel_col = _select_column(columns, preferred=["name", "channel", "label"])
    atlas_col = _select_column(columns, preferred=[atlas_name])
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
        channel_key = _normalize_channel_name(raw_channel)
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

    region_to_indices: dict[str, list[int]] = {}
    for idx, channel_name in enumerate(channel_names):
        channel_key = _normalize_channel_name(channel_name)
        region = channel_to_region.get(channel_key)
        if region is None:
            continue
        region_to_indices.setdefault(region, []).append(idx)

    if not region_to_indices:
        raise ValueError(
            f"No ieeg channels from {ieeg_file.path.name} matched atlas labels in "
            f"{electrodes_file.path.name}."
        )

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


def _find_matching_electrodes_file(
    ieeg_file: BIDSFile,
    electrodes_files: Sequence[BIDSFile],
) -> BIDSFile | None:
    scored: list[tuple[int, int, str, BIDSFile]] = []
    for electrodes_file in electrodes_files:
        score = _entity_match_score(ieeg_file.entities, electrodes_file.entities)
        if score is None:
            continue
        specificity = _entity_specificity(electrodes_file.entities)
        scored.append((score, specificity, str(electrodes_file.path), electrodes_file))

    if not scored:
        return None

    best_score = max(score for score, _, _, _ in scored)
    best = [
        (specificity, path, file)
        for score, specificity, path, file in scored
        if score == best_score
    ]
    if len(best) > 1:
        min_specificity = min(specificity for specificity, _, _ in best)
        best = [
            (specificity, path, file)
            for specificity, path, file in best
            if specificity == min_specificity
        ]
    if len(best) > 1:
        names = ", ".join(sorted(file.path.name for _, _, file in best))
        raise ValueError(
            f"Ambiguous electrodes table match for {ieeg_file.path.name}: {names}. "
            "Please disambiguate entities (e.g. run/task) in *_electrodes.tsv."
        )
    return best[0][2]


def _entity_match_score(
    target_entities: dict[str, Any],
    candidate_entities: dict[str, Any],
) -> int | None:
    score = 0
    for aliases in (
        ("subject", "sub"),
        ("session", "ses"),
        ("task",),
        ("run",),
    ):
        target = _entity_value(target_entities, aliases)
        if target is None:
            continue

        candidate = _entity_value(candidate_entities, aliases)
        if candidate is None:
            continue
        if candidate != target:
            return None
        score += 1

    return score


def _entity_value(entities: dict[str, Any], aliases: Sequence[str]) -> str | None:
    for alias in aliases:
        value = _normalize_entity_value(entities.get(alias))
        if value is not None:
            return value
    return None


def _normalize_entity_value(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.startswith("sub-"):
        return text[4:]
    if text.startswith("ses-"):
        return text[4:]
    return text


def _normalize_channel_name(name: str) -> str:
    return str(name).strip().casefold()


def _entity_specificity(entities: dict[str, Any]) -> int:
    """
    Return an entity specificity score for tie-breaking file matches.

    Lower values mean the file is more generic (fewer explicit BIDS entities).
    """
    ignored = {"suffix", "extension", "datatype", "scope"}
    return sum(
        1
        for key, value in entities.items()
        if key not in ignored and _normalize_entity_value(value) is not None
    )


def _read_table_rows(file: BIDSFile) -> list[dict[str, str]]:
    delimiter = "\t" if file.extension == ".tsv" else ","
    with open(file.path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        return [
            {
                str(key).strip(): "" if value is None else str(value).strip()
                for key, value in row.items()
            }
            for row in reader
        ]


def _select_column(columns: Sequence[str], preferred: Sequence[str]) -> str | None:
    by_norm = {column.casefold(): column for column in columns}
    for wanted in preferred:
        match = by_norm.get(wanted.casefold())
        if match is not None:
            return match
    return None
