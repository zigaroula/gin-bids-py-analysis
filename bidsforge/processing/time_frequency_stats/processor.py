from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.bids.matching import files_matching_entities, shared_entities
from bidsforge.processing.base import BaseProcessing
from bidsforge.processing.time_frequency.result_loader import load_time_frequency_result
from bidsforge.processing.utils.events import AnnotationEvent, coerce_annotation_events
from bidsforge.processing.utils.trial_resolver import ResolvedTrial, TrialResolver

from .params import BaseTimeFrequencyStatsParams
from .result import BaseTimeFrequencyStatsResult


@dataclass
class TimeFrequencyStatsContext:
    group: BIDSFileGroup
    tfr_file: BIDSFile
    power: np.ndarray
    time_axis_s: np.ndarray
    frequency_hz: np.ndarray
    channel_names: list[str]
    epochs_a: np.ndarray
    epochs_b: np.ndarray
    kept_trials_a: list[ResolvedTrial]
    kept_trials_b: list[ResolvedTrial]
    resolved_trials: list[ResolvedTrial]
    source_table_files: list[str]
    source_ieeg_files: list[str]
    source_electrodes_files: list[str]
    metadata: dict[str, Any]


class BaseTimeFrequencyStatsProcessing(BaseProcessing, ABC):
    """Template for statistics computed from stored time-frequency derivatives."""

    def __init__(self, params: BaseTimeFrequencyStatsParams, resolver: TrialResolver) -> None:
        self.params = params
        self.resolver = resolver
        self._validate_resolver_labels()

    def process_group(
        self,
        group: BIDSFileGroup,
        progress_tracking_position: int = 0,
    ) -> BaseTimeFrequencyStatsResult:
        del progress_tracking_position

        tfr_file = _find_tfr_file(group)
        result = load_time_frequency_result(tfr_file.path)
        power = _select_power(np.asarray(result.power_db, dtype=np.float64), result, self.params.power_mode)
        if self.params.baseline_grand_average:
            power = power - np.nanmean(power, axis=3, keepdims=True, dtype=np.float64)

        time_axis_s = np.asarray(result.time_s, dtype=np.float64).ravel()
        power, time_axis_s = _select_time_window(
            power,
            time_axis_s,
            self.params.time_window_s,
            self.params.time_selection,
        )

        anchors = self._build_anchor_events(result.events, n_trials=int(power.shape[0]))
        resolved = self.resolver.resolve_trials(group, tfr_file, anchors)
        if len(resolved) != power.shape[0]:
            raise ValueError(
                f"Resolver returned {len(resolved)} trial(s) for {power.shape[0]} TFR trial(s) "
                f"in {tfr_file.path.name}."
            )
        resolved = self._normalize_trials(resolved)

        kept_a_idx: list[int] = []
        kept_b_idx: list[int] = []
        kept_a: list[ResolvedTrial] = []
        kept_b: list[ResolvedTrial] = []
        for idx, trial in enumerate(resolved):
            if not trial.keep:
                continue
            if trial.label == self.params.condition_a:
                kept_a_idx.append(idx)
                kept_a.append(trial)
            elif trial.label == self.params.condition_b:
                kept_b_idx.append(idx)
                kept_b.append(trial)

        epochs_a = power[np.asarray(kept_a_idx, dtype=np.int64)] if kept_a_idx else _empty_epochs(power)
        epochs_b = power[np.asarray(kept_b_idx, dtype=np.int64)] if kept_b_idx else _empty_epochs(power)

        table_files = files_matching_entities(group.all_files, extension={".tsv", ".csv"})
        source_table_files = sorted(
            {str(file.path) for file in table_files if file.suffix != "electrodes"}
        )
        source_electrodes_files = sorted(
            {str(file.path) for file in table_files if file.suffix == "electrodes"}
        )
        source_ieeg_files = [str(result.source_group.primary.path)]
        metadata = {
            "time_window_s": list(self.params.time_window_s or []),
            "time_selection": self.params.time_selection,
            "power_mode": self.params.power_mode,
            "baseline_grand_average": bool(self.params.baseline_grand_average),
            "min_trials_per_condition": int(self.params.min_trials_per_condition),
            "p_value_correction_method": self.params.p_value_correction_method,
            "significance_alpha": float(self.params.significance_alpha),
            "n_permutations": int(self.params.n_permutations),
            **self._build_pipeline_metadata(),
        }
        context = TimeFrequencyStatsContext(
            group=group,
            tfr_file=tfr_file,
            power=power,
            time_axis_s=time_axis_s,
            frequency_hz=np.asarray(result.frequency_hz, dtype=np.float64).ravel(),
            channel_names=list(result.channel_names),
            epochs_a=epochs_a,
            epochs_b=epochs_b,
            kept_trials_a=kept_a,
            kept_trials_b=kept_b,
            resolved_trials=resolved,
            source_table_files=source_table_files,
            source_ieeg_files=source_ieeg_files,
            source_electrodes_files=source_electrodes_files,
            metadata=metadata,
        )
        return self._compute_and_build_result(context)

    def _build_anchor_events(
        self,
        events: Sequence[dict[str, Any]] | None,
        *,
        n_trials: int,
    ) -> list[AnnotationEvent]:
        codes = {str(code) for code in self.params.anchor_event_codes}
        if events and codes:
            matched = [event for event in coerce_annotation_events(events) if str(event.code) in codes]
            if len(matched) >= n_trials:
                return matched[:n_trials]
        code = self.params.anchor_event_codes[0] if self.params.anchor_event_codes else ""
        return [
            AnnotationEvent(
                onset_s=float(idx),
                duration_s=0.0,
                event_type="Stimulus",
                description=str(code),
                code=str(code) if code else None,
            )
            for idx in range(n_trials)
        ]

    def _normalize_trials(self, trials: list[ResolvedTrial]) -> list[ResolvedTrial]:
        supported = {self.params.condition_a, self.params.condition_b}
        normalized: list[ResolvedTrial] = []
        for trial in trials:
            if not trial.keep:
                normalized.append(trial)
                continue
            if trial.label not in supported:
                normalized.append(
                    replace(
                        trial,
                        keep=False,
                        exclusion_reason=trial.exclusion_reason or "unsupported_label",
                    )
                )
            else:
                normalized.append(trial)
        return normalized

    def _build_pipeline_metadata(self) -> dict[str, Any]:
        return {}

    def _common_result_kwargs(self, context: TimeFrequencyStatsContext) -> dict[str, Any]:
        return {
            "source_group": context.group,
            "output_entities": shared_entities(
                [context.tfr_file],
                excluded_entities=frozenset({"suffix", "extension", "datatype", "desc", "description"}),
            ),
            "metadata": context.metadata,
            "channel_names": context.channel_names,
            "frequency_hz": context.frequency_hz,
            "time_axis_s": context.time_axis_s,
            "condition_a": self.params.condition_a,
            "condition_b": self.params.condition_b,
            "condition_a_trial_count": int(context.epochs_a.shape[0]),
            "condition_b_trial_count": int(context.epochs_b.shape[0]),
            "resolved_trials": context.resolved_trials,
            "source_tfr_file": str(context.tfr_file.path),
            "source_table_files": context.source_table_files,
            "source_ieeg_files": context.source_ieeg_files,
            "source_electrodes_files": context.source_electrodes_files,
            "epochs": None,
            "power_mode": self.params.power_mode,
            "p_value_correction_method": self.params.p_value_correction_method,
            "significance_alpha": self.params.significance_alpha,
        }

    @abstractmethod
    def _compute_and_build_result(
        self,
        context: TimeFrequencyStatsContext,
    ) -> BaseTimeFrequencyStatsResult:
        """Build a concrete result from the prepared TF trial context."""

    def _validate_resolver_labels(self) -> None:
        labels = getattr(self.resolver, "condition_labels", None)
        if labels is None:
            return
        labels = tuple(str(label) for label in labels)
        expected = (self.params.condition_a, self.params.condition_b)
        if len(labels) == 2 and set(labels) == set(expected):
            return
        raise ValueError(
            "Resolver condition labels must match condition_a/condition_b exactly. "
            f"Expected {expected!r}, got {labels!r}."
        )


def flatten_tf_epochs(epochs: np.ndarray) -> np.ndarray:
    """Return epochs as (trial, channel*frequency, time)."""
    arr = np.asarray(epochs, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError("TF epochs must be 4-D (trial, channel, frequency, time).")
    return arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2], arr.shape[3])


def unflatten_tf_map(values: np.ndarray, *, n_channels: int, n_freqs: int) -> np.ndarray:
    """Return a feature x time map as channel x frequency x time."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("values must be 2-D (channel*frequency, time).")
    return arr.reshape(n_channels, n_freqs, arr.shape[1])


def _find_tfr_file(group: BIDSFileGroup) -> BIDSFile:
    candidates = files_matching_entities(
        group.all_files,
        extension={".h5", ".hdf5", ".mat"},
        suffix="tfr",
    )
    if not candidates:
        candidates = files_matching_entities(
            group.all_files,
            extension={".h5", ".hdf5", ".mat"},
        )
    if not candidates:
        raise ValueError("Time-frequency stats require a time_frequency .h5/.mat derivative.")
    return sorted(candidates, key=lambda file: str(file.path))[0]


def _select_power(power: np.ndarray, result: Any, mode: str) -> np.ndarray:
    baseline = np.asarray(result.baseline_db, dtype=np.float64)
    apply_baseline = bool(result.metadata.get("apply_baseline", False))
    if mode == "stored":
        return power.copy()
    if mode == "raw":
        return power + baseline[:, :, :, np.newaxis] if apply_baseline else power.copy()
    if mode == "baseline_corrected":
        return power.copy() if apply_baseline else power - baseline[:, :, :, np.newaxis]
    raise ValueError(f"Unsupported power_mode={mode!r}.")


def _select_time_window(
    power: np.ndarray,
    time_axis_s: np.ndarray,
    time_window_s: tuple[float, float] | None,
    time_selection: str,
) -> tuple[np.ndarray, np.ndarray]:
    if time_window_s is None:
        return power, time_axis_s
    start, stop = time_window_s
    if time_selection == "strict":
        mask = (time_axis_s > start) & (time_axis_s < stop)
    elif time_selection == "inclusive":
        mask = (time_axis_s >= start) & (time_axis_s <= stop)
    else:
        raise ValueError(f"Unsupported time_selection={time_selection!r}.")
    if not np.any(mask):
        raise ValueError(
            f"time_window_s={time_window_s!r} selects no TFR time samples."
        )
    return power[:, :, :, mask], time_axis_s[mask]


def _empty_epochs(power: np.ndarray) -> np.ndarray:
    return np.empty((0, power.shape[1], power.shape[2], power.shape[3]), dtype=np.float64)
