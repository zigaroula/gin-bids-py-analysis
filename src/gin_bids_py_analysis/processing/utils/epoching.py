"""Epoch extraction and temporal binning utilities shared across processing pipelines.

Provides MNE-based epoch extraction, temporal binning, and related helpers used by
trial_stats and trial_slope_stats processors.  Atlas-based channel grouping lives in
``gin_bids_py_analysis.processing.utils.atlas``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import mne
import numpy as np

from gin_bids_py_analysis.processing.utils.events import (
    AnnotationEvent,
    parse_annotation_description,
)
from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial


@dataclass
class EpochExtractionResult:
    """Epoch extraction outputs for one recording."""

    epochs: np.ndarray
    kept_trials: list[ResolvedTrial]
    updated_trials: list[ResolvedTrial]
    time_axis_s: np.ndarray


def build_time_axis_s(sfreq: float, tmin_s: float, tmax_s: float) -> np.ndarray:
    """Return the inclusive epoch time axis for the given sampling rate and window."""
    sample_offsets = _sample_offsets(sfreq, tmin_s, tmax_s)
    return sample_offsets.astype(np.float64) / sfreq


def extract_anchor_events_with_mne(
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


def extract_epochs_with_mne(
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


def stack_epochs(
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


def window_samples(sfreq: float, window_ms: float) -> int:
    """Convert a temporal window duration (ms) to the nearest sample count (minimum 1)."""
    return max(1, int(round((window_ms / 1000.0) * sfreq)))


def temporal_bin_epochs(
    epochs: np.ndarray,
    time_axis_s: np.ndarray,
    window_samples_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average non-overlapping temporal windows, returning downsampled epochs and time axis.

    Input shape:  (n_epochs, n_channels, n_times)
    Output shape: (n_epochs, n_channels, n_bins)
    """
    if window_samples_count <= 1:
        return epochs, time_axis_s

    n_epochs, n_channels, n_times = epochs.shape
    starts = list(range(0, n_times, window_samples_count))
    # Epoch extraction is inclusive on both bounds, which commonly leaves a
    # trailing 1-sample tail for otherwise exact-duration windows. Merge that
    # sample into the previous bin to avoid a visually confusing tiny final bin.
    if len(starts) >= 2 and (n_times - starts[-1]) == 1:
        starts = starts[:-1]

    stops = [min(start + window_samples_count, n_times) for start in starts]
    if stops and stops[-1] < n_times:
        stops[-1] = n_times

    return aggregate_time_bins(epochs, time_axis_s, starts, stops)


def temporal_bin_epochs_by_n_bins(
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
    return aggregate_time_bins(epochs, time_axis_s, starts, stops)


def aggregate_time_bins(
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


def _annotation_code_parser(description: str) -> int | None:
    """Return integer event code parsed from an annotation description."""
    _, _, code = parse_annotation_description(description)
    if code is None:
        return None
    try:
        return int(code)
    except ValueError:
        return None


def _sample_offsets(sfreq: float, tmin_s: float, tmax_s: float) -> np.ndarray:
    """Return integer sample offsets for a fixed epoch window."""
    n_samples = int(round((tmax_s - tmin_s) * sfreq)) + 1
    start_offset = int(round(tmin_s * sfreq))
    return np.arange(start_offset, start_offset + n_samples, dtype=np.int64)



