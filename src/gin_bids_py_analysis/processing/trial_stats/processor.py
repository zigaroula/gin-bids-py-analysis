"""Shared processor base for subject-level trial statistics pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from typing import Any, Sequence

import numpy as np
from mne import Annotations
from mne.io import BaseRaw

from gin_bids_py_analysis.bids.file import BIDSFile
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
from gin_bids_py_analysis.processing.utils.events import (
    AnnotationEvent,
    parse_annotation_description,
)
from gin_bids_py_analysis.processing.utils.statistics import zscore_activity_by_baseline
from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial, TrialResolver

from .params import BaseTrialStatsParams
from .result import BaseTrialStatsProcessingResult


@dataclass
class TrialStatsProcessingContext:
    """Shared execution context passed from the base processor to subclasses."""

    group: BIDSFileGroup
    ieeg_files: list[BIDSFile]
    feature_names: list[str]
    reference_channel_names: list[str]
    feature_indices: list[np.ndarray] | None
    time_axis_ref: np.ndarray
    time_axis_eval: np.ndarray
    epochs_a: np.ndarray
    epochs_b: np.ndarray
    kept_trials_a: list[ResolvedTrial]
    kept_trials_b: list[ResolvedTrial]
    all_resolved_trials: list[ResolvedTrial]
    source_table_files: list[str]
    source_electrodes_files: list[str]
    region_channels: dict[str, list[str]]
    atlas_mode: bool
    metadata: dict[str, Any]
    sfreq: float
    state: dict[str, Any]

    @property
    def analysis_level(self) -> str:
        return "roi" if self.atlas_mode else "channel"


class BaseTrialStatsProcessing(BaseProcessing, ABC):
    """Template method for subject-level trial statistics pipelines."""

    def __init__(
        self,
        params: BaseTrialStatsParams,
        resolver: TrialResolver,
    ) -> None:
        self.params = params
        self.resolver = resolver
        self._validate_resolver_labels()

    def process_group(
        self,
        group: BIDSFileGroup,
        progress_tracking_position: int = 0,
    ) -> BaseTrialStatsProcessingResult:
        del progress_tracking_position

        ieeg_files = files_matching_entities(
            group.all_files,
            extension=".vhdr",
            suffix="ieeg",
        )
        if not ieeg_files:
            raise ValueError(
                f"{self.__class__.__name__} requires at least one ieeg BrainVision file."
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

        state = self._initialize_pipeline_state()
        epochs_a: list[np.ndarray] = []
        epochs_b: list[np.ndarray] = []
        kept_trials_a: list[ResolvedTrial] = []
        kept_trials_b: list[ResolvedTrial] = []
        all_resolved_trials: list[ResolvedTrial] = []

        anchor_codes = set(self.params.anchor_event_codes)
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
                            "Atlas region grouping must be consistent across all ieeg files "
                            f"in a subject group. Got {grouping.feature_names} for "
                            f"{ieeg_file.path.name} but expected {feature_names_ref}."
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

                normalized_trials = self._normalize_trials(
                    group=group,
                    ieeg_file=ieeg_file,
                    raw=raw,
                    anchor_events=anchor_events,
                    trials=resolved_trials,
                )
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
                    if trial.label == self.params.condition_a:
                        epochs_a.append(epoch_for_stats)
                        kept_trials_a.append(trial)
                        self._on_kept_trial_epoch(
                            epoch_for_stats=epoch_for_stats,
                            trial=trial,
                            condition="a",
                            state=state,
                        )
                    elif trial.label == self.params.condition_b:
                        epochs_b.append(epoch_for_stats)
                        kept_trials_b.append(trial)
                        self._on_kept_trial_epoch(
                            epoch_for_stats=epoch_for_stats,
                            trial=trial,
                            condition="b",
                            state=state,
                        )

        assert sfreq_ref is not None
        assert channel_names_ref is not None
        assert time_axis_ref is not None

        feature_names = list(feature_names_ref if atlas_mode else channel_names_ref)
        feature_indices = list(feature_indices_ref) if feature_indices_ref is not None else None

        epochs_a_array = stack_epochs(epochs_a, len(feature_names), len(time_axis_ref))
        epochs_b_array = stack_epochs(epochs_b, len(feature_names), len(time_axis_ref))
        (
            epochs_a_array,
            epochs_b_array,
            feature_names,
            feature_indices,
        ) = self._prepare_epochs_before_activity_zscore(
            epochs_a=epochs_a_array,
            epochs_b=epochs_b_array,
            feature_names=feature_names,
            feature_indices=feature_indices,
            time_axis_s=time_axis_ref,
            state=state,
            atlas_mode=atlas_mode,
        )

        epochs_a_array, epochs_b_array = self._apply_activity_zscore(
            epochs_a=epochs_a_array,
            epochs_b=epochs_b_array,
            time_axis_s=time_axis_ref,
            state=state,
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

        source_electrodes_files = sorted(used_electrodes_paths)
        if not source_electrodes_files and electrodes_files:
            source_electrodes_files = sorted({str(file.path) for file in electrodes_files})

        region_channels: dict[str, list[str]] = {}
        if atlas_mode:
            assert feature_indices is not None
            region_channels = {
                region: [channel_names_ref[int(index)] for index in indices]
                for region, indices in zip(feature_names, feature_indices)
            }

        metadata = self._build_shared_metadata(
            feature_names=feature_names,
            atlas_mode=atlas_mode,
            missing_atlas_regions=sorted(missing_atlas_regions),
            binning_mode=binning_mode,
            window_sample_count=window_sample_count,
            effective_n_bins=effective_n_bins,
            state=state,
        )
        metadata.update(self._build_pipeline_metadata(state=state))

        context = TrialStatsProcessingContext(
            group=group,
            ieeg_files=ieeg_files,
            feature_names=feature_names,
            reference_channel_names=channel_names_ref,
            feature_indices=feature_indices,
            time_axis_ref=time_axis_ref,
            time_axis_eval=time_axis_eval,
            epochs_a=epochs_a_array,
            epochs_b=epochs_b_array,
            kept_trials_a=kept_trials_a,
            kept_trials_b=kept_trials_b,
            all_resolved_trials=all_resolved_trials,
            source_table_files=source_table_files,
            source_electrodes_files=source_electrodes_files,
            region_channels=region_channels,
            atlas_mode=atlas_mode,
            metadata=metadata,
            sfreq=sfreq_ref,
            state=state,
        )
        return self._compute_and_build_result(context)

    def _initialize_pipeline_state(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def _normalize_trials(
        self,
        *,
        group: BIDSFileGroup,
        ieeg_file: BIDSFile,
        raw: BaseRaw,
        anchor_events: Sequence[Any],
        trials: list[ResolvedTrial],
    ) -> list[ResolvedTrial]:
        """Normalize resolver trials before epoch extraction."""

    def _on_kept_trial_epoch(
        self,
        *,
        epoch_for_stats: np.ndarray,
        trial: ResolvedTrial,
        condition: str,
        state: dict[str, Any],
    ) -> None:
        del epoch_for_stats, trial, condition, state

    def _prepare_epochs_before_activity_zscore(
        self,
        *,
        epochs_a: np.ndarray,
        epochs_b: np.ndarray,
        feature_names: list[str],
        feature_indices: list[np.ndarray] | None,
        time_axis_s: np.ndarray,
        state: dict[str, Any],
        atlas_mode: bool,
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[np.ndarray] | None]:
        del time_axis_s, state, atlas_mode
        return epochs_a, epochs_b, feature_names, feature_indices

    def _apply_activity_zscore(
        self,
        *,
        epochs_a: np.ndarray,
        epochs_b: np.ndarray,
        time_axis_s: np.ndarray,
        state: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        del state
        if self.params.activity_zscore != "baseline":
            return epochs_a, epochs_b
        return zscore_activity_by_baseline(
            epochs_a,
            epochs_b,
            time_axis_s,
            baseline_tmin_s=self.params.activity_baseline_tmin_s,
            baseline_tmax_s=self.params.activity_baseline_tmax_s,
            baseline_scope=self.params.activity_baseline_scope,
            remove_outlier_trial_means=self.params.activity_baseline_remove_outlier_trial_means,
        )

    def _attach_trial_activity_summary_metadata(
        self,
        *,
        raw_annotations: Annotations,
        anchor_events: Sequence[Any],
        trials: list[ResolvedTrial],
    ) -> list[ResolvedTrial]:
        if self.params.trial_activity_summary.kind != "anchor_to_response_mean":
            return trials

        response_source = self.params.trial_activity_summary.response
        if response_source is None:
            return trials

        anchor_event_list = [event for event in anchor_events if isinstance(event, AnnotationEvent)]
        if len(anchor_event_list) != len(trials):
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
                anchor_event_list,
                event_code=response_source.event_code,
            )

        updated_trials: list[ResolvedTrial] = []
        for trial, response_time_s in zip(trials, response_times_s):
            metadata = dict(trial.metadata)
            metadata["trial_activity_summary_response_time_s"] = (
                float(response_time_s) if np.isfinite(response_time_s) else np.nan
            )
            updated_trials.append(
                ResolvedTrial(
                    source_file=trial.source_file,
                    anchor_event_index=trial.anchor_event_index,
                    anchor_event_code=trial.anchor_event_code,
                    anchor_onset_s=trial.anchor_onset_s,
                    anchor_duration_s=trial.anchor_duration_s,
                    label=trial.label,
                    trial_id=trial.trial_id,
                    keep=trial.keep,
                    exclusion_reason=trial.exclusion_reason,
                    metadata=metadata,
                )
            )
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
            return np.nanmean(epochs, axis=2, dtype=np.float64).T.astype(np.float64)
        return np.empty((n_features, 0), dtype=np.float64)

    def _compute_trial_activity_summary_values(
        self,
        *,
        epochs: np.ndarray,
        time_axis_s: np.ndarray,
        trials: list[ResolvedTrial],
        n_features: int,
    ) -> np.ndarray:
        if self.params.trial_activity_summary.kind == "epoch_mean":
            return self._compute_epoch_means(epochs, n_features)

        if epochs.ndim != 3 or epochs.size == 0:
            return np.empty((n_features, 0), dtype=np.float64)

        summary = np.full((epochs.shape[0], n_features), np.nan, dtype=np.float64)
        time_axis = np.asarray(time_axis_s, dtype=np.float64).ravel()
        summary_window_start_s = 0.0
        summary_window_end_s = float(self.params.tmax_s)
        boundary_tol_s = 1e-12
        missing_response_policy = (
            str(self.params.trial_activity_summary.missing_response_policy).strip().lower()
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

    def _build_shared_metadata(
        self,
        *,
        feature_names: list[str],
        atlas_mode: bool,
        missing_atlas_regions: list[str],
        binning_mode: str,
        window_sample_count: int,
        effective_n_bins: int,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        del state
        return {
            "anchor_event_codes": list(self.params.anchor_event_codes),
            "experiment_start_event_code": self.params.experiment_start_event_code,
            "experiment_end_event_code": self.params.experiment_end_event_code,
            "tmin_s": self.params.tmin_s,
            "tmax_s": self.params.tmax_s,
            "min_trials_per_condition": self.params.min_trials_per_condition,
            "drop_partial_epochs": self.params.drop_partial_epochs,
            "p_value_correction_method": self.params.p_value_correction_method,
            "significance_alpha": self.params.significance_alpha,
            "analysis_level": "roi" if atlas_mode else "channel",
            "atlas_name": self.params.atlas_name,
            "atlas_regions": feature_names if atlas_mode else [],
            "atlas_regions_requested": list(self.params.atlas_regions),
            "atlas_regions_missing": missing_atlas_regions,
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
            "trial_activity_summary": json.loads(
                self.params.trial_activity_summary.model_dump_json()
            ),
            "trial_activity_summary_kind": self.params.trial_activity_summary.kind,
            "trial_activity_summary_missing_response_policy": (
                self.params.trial_activity_summary.missing_response_policy
            ),
            "trial_activity_summary_source": self._serialize_trial_activity_summary_source(),
            "trial_activity_summary_label": self._trial_activity_summary_label(),
        }

    def _build_pipeline_metadata(self, *, state: dict[str, Any]) -> dict[str, Any]:
        del state
        return {}

    def _build_common_result_kwargs(
        self,
        context: TrialStatsProcessingContext,
    ) -> dict[str, Any]:
        n_features = len(context.feature_names)
        return {
            "source_group": context.group,
            "output_entities": shared_entities(
                context.ieeg_files,
                excluded_entities=frozenset(
                    {"suffix", "extension", "datatype", "desc", "description", "run"}
                ),
            ),
            "metadata": context.metadata,
            "time_axis_s": context.time_axis_eval,
            "channel_names": context.feature_names,
            "condition_a": self.params.condition_a,
            "condition_b": self.params.condition_b,
            "condition_a_trial_count": int(context.epochs_a.shape[0]),
            "condition_b_trial_count": int(context.epochs_b.shape[0]),
            "sfreq": context.sfreq,
            "resolved_trials": context.all_resolved_trials,
            "source_ieeg_files": [str(file.path) for file in context.ieeg_files],
            "source_table_files": context.source_table_files,
            "source_electrodes_files": context.source_electrodes_files,
            "analysis_level": context.analysis_level,
            "atlas_name": self.params.atlas_name,
            "atlas_regions": context.feature_names if context.atlas_mode else [],
            "region_channels": context.region_channels,
            "window_ms": self.params.window_ms,
            "n_bins": self.params.n_bins,
            "activity_zscore": self.params.activity_zscore,
            "activity_baseline_tmin_s": self.params.activity_baseline_tmin_s,
            "activity_baseline_tmax_s": self.params.activity_baseline_tmax_s,
            "activity_baseline_scope": self.params.activity_baseline_scope,
            "activity_baseline_remove_outlier_trial_means": self.params.activity_baseline_remove_outlier_trial_means,
            "p_value_correction_method": self.params.p_value_correction_method,
            "significance_alpha": self.params.significance_alpha,
            "trial_activity_summary_kind": self.params.trial_activity_summary.kind,
            "trial_activity_summary_missing_response_policy": (
                self.params.trial_activity_summary.missing_response_policy
            ),
            "trial_activity_summary_source": self._serialize_trial_activity_summary_source(),
            "trial_activity_summary_label": self._trial_activity_summary_label(),
            "condition_a_epochs": context.epochs_a,
            "condition_b_epochs": context.epochs_b,
            "condition_a_trial_activity_summary_values": self._compute_trial_activity_summary_values(
                epochs=context.epochs_a,
                time_axis_s=context.time_axis_eval,
                trials=context.kept_trials_a,
                n_features=n_features,
            ),
            "condition_b_trial_activity_summary_values": self._compute_trial_activity_summary_values(
                epochs=context.epochs_b,
                time_axis_s=context.time_axis_eval,
                trials=context.kept_trials_b,
                n_features=n_features,
            ),
        }

    @abstractmethod
    def _compute_and_build_result(
        self,
        context: TrialStatsProcessingContext,
    ) -> BaseTrialStatsProcessingResult:
        """Compute pipeline-specific outputs and build the final result."""

    def _validate_resolver_labels(self) -> None:
        resolver_labels = getattr(self.resolver, "condition_labels", None)
        if resolver_labels is None:
            return

        labels = tuple(str(label) for label in resolver_labels)
        expected = (self.params.condition_a, self.params.condition_b)
        if len(labels) != 2 or set(labels) != set(expected):
            raise ValueError(
                "Resolver condition labels must match "
                f"{self.params.__class__.__name__}.condition_a/condition_b exactly. "
                f"Expected {expected!r}, got {labels!r}."
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
