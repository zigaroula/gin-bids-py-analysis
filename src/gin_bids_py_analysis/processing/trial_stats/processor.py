from __future__ import annotations

from typing import Sequence

import numpy as np

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.data.loader import load_ieeg
from gin_bids_py_analysis.processing.base import BaseProcessing
from gin_bids_py_analysis.processing.utils.events import coerce_annotation_events

from .params import TrialStatsParams
from .resolver import ResolvedTrial, TrialLabelResolver
from .result import TrialStatsProcessingResult
from .stats import build_time_axis_s, compute_condition_statistics, extract_epochs


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
                if file.extension in {".tsv", ".csv"}
            }
        )

        anchor_codes = set(self.params.anchor_event_codes)
        all_resolved_trials: list[ResolvedTrial] = []
        epochs_a: list[np.ndarray] = []
        epochs_b: list[np.ndarray] = []

        sfreq_ref: float | None = None
        channel_names_ref: list[str] | None = None
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
                if trial.label == self.params.condition_a:
                    epochs_a.append(epoch)
                elif trial.label == self.params.condition_b:
                    epochs_b.append(epoch)

        assert sfreq_ref is not None
        assert channel_names_ref is not None
        assert time_axis_ref is not None

        epochs_a_array = _stack_epochs(
            epochs_a,
            len(channel_names_ref),
            len(time_axis_ref),
        )
        epochs_b_array = _stack_epochs(
            epochs_b,
            len(channel_names_ref),
            len(time_axis_ref),
        )

        stats_valid = (
            epochs_a_array.shape[0] >= self.params.min_trials_per_condition
            and epochs_b_array.shape[0] >= self.params.min_trials_per_condition
        )
        t_values, p_values, mean_a, mean_b, mean_difference = (
            compute_condition_statistics(
                epochs_a_array if stats_valid else np.empty_like(epochs_a_array[:0]),
                epochs_b_array if stats_valid else np.empty_like(epochs_b_array[:0]),
                n_channels=len(channel_names_ref),
                n_times=len(time_axis_ref),
                equal_var=self.params.equal_var,
            )
        )
        if epochs_a_array.size:
            mean_a = np.nanmean(epochs_a_array, axis=0, dtype=np.float64)
        if epochs_b_array.size:
            mean_b = np.nanmean(epochs_b_array, axis=0, dtype=np.float64)
        mean_difference = mean_a - mean_b

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
            },
            t_values=t_values,
            p_values=p_values,
            condition_a_mean=mean_a,
            condition_b_mean=mean_b,
            mean_difference=mean_difference,
            time_axis_s=time_axis_ref,
            channel_names=channel_names_ref,
            condition_a=self.params.condition_a,
            condition_b=self.params.condition_b,
            condition_a_trial_count=int(epochs_a_array.shape[0]),
            condition_b_trial_count=int(epochs_b_array.shape[0]),
            sfreq=sfreq_ref,
            resolved_trials=all_resolved_trials,
            source_ieeg_files=[str(file.path) for file in ieeg_files],
            source_table_files=source_table_files,
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
