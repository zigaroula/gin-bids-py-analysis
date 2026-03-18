from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.stats import ttest_ind

from .resolver import ResolvedTrial


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


def extract_epochs(
    data: np.ndarray,
    sfreq: float,
    trials: list[ResolvedTrial],
    tmin_s: float,
    tmax_s: float,
    *,
    drop_partial_epochs: bool = True,
) -> EpochExtractionResult:
    """Extract fixed-width epochs around kept trial anchors."""
    offsets = _sample_offsets(sfreq, tmin_s, tmax_s)
    time_axis_s = offsets.astype(np.float64) / sfreq

    epochs: list[np.ndarray] = []
    kept_trials: list[ResolvedTrial] = []
    updated_trials: list[ResolvedTrial] = []

    for trial in trials:
        if not trial.keep:
            updated_trials.append(trial)
            continue

        anchor_sample = int(round(trial.anchor_onset_s * sfreq))
        start_sample = anchor_sample + int(offsets[0])
        stop_sample = anchor_sample + int(offsets[-1])

        if start_sample < 0 or stop_sample >= data.shape[1]:
            if not drop_partial_epochs:
                raise ValueError(
                    f"Trial at {trial.anchor_onset_s:.6f}s would create a partial epoch."
                )
            updated_trials.append(
                replace(
                    trial,
                    keep=False,
                    exclusion_reason=trial.exclusion_reason or "partial_epoch",
                )
            )
            continue

        epoch = data[:, start_sample : stop_sample + 1]
        epochs.append(epoch)
        kept_trials.append(trial)
        updated_trials.append(trial)

    epoch_array = (
        np.stack(epochs, axis=0).astype(np.float32)
        if epochs
        else np.empty((0, data.shape[0], len(time_axis_s)), dtype=np.float32)
    )

    return EpochExtractionResult(
        epochs=epoch_array,
        kept_trials=kept_trials,
        updated_trials=updated_trials,
        time_axis_s=time_axis_s,
    )


def compute_condition_statistics(
    epochs_a: np.ndarray,
    epochs_b: np.ndarray,
    *,
    n_channels: int,
    n_times: int,
    equal_var: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-channel, per-time condition summaries and t-test outputs."""
    empty = np.full((n_channels, n_times), np.nan, dtype=np.float64)

    mean_a = (
        np.nanmean(epochs_a, axis=0, dtype=np.float64)
        if epochs_a.size
        else empty.copy()
    )
    mean_b = (
        np.nanmean(epochs_b, axis=0, dtype=np.float64)
        if epochs_b.size
        else empty.copy()
    )
    mean_difference = mean_a - mean_b

    if not epochs_a.size or not epochs_b.size:
        return empty.copy(), empty.copy(), mean_a, mean_b, mean_difference

    stats = ttest_ind(
        epochs_a,
        epochs_b,
        axis=0,
        equal_var=equal_var,
        nan_policy="omit",
    )
    t_values = np.asarray(stats.statistic, dtype=np.float64)
    p_values = np.asarray(stats.pvalue, dtype=np.float64)
    return t_values, p_values, mean_a, mean_b, mean_difference


def correct_p_values(
    p_values: np.ndarray,
    *,
    method: Literal["none", "fdr_bh", "bonferroni"] = "fdr_bh",
) -> np.ndarray:
    """Apply a multiple-comparisons correction to p-values."""
    corrected = np.asarray(p_values, dtype=np.float64).copy()
    finite_mask = np.isfinite(corrected)
    if not finite_mask.any() or method == "none":
        return corrected

    flat = corrected[finite_mask]
    n_tests = flat.size

    if method == "bonferroni":
        corrected[finite_mask] = np.minimum(flat * n_tests, 1.0)
        return corrected

    if method == "fdr_bh":
        order = np.argsort(flat)
        ranked = flat[order]
        ranks = np.arange(1, n_tests + 1, dtype=np.float64)

        adjusted = ranked * (n_tests / ranks)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0.0, 1.0)

        flat_corrected = np.empty_like(flat)
        flat_corrected[order] = adjusted
        corrected[finite_mask] = flat_corrected
        return corrected

    raise ValueError(f"Unsupported p-value correction method: {method!r}")


def _sample_offsets(sfreq: float, tmin_s: float, tmax_s: float) -> np.ndarray:
    start = int(round(tmin_s * sfreq))
    stop = int(round(tmax_s * sfreq))
    return np.arange(start, stop + 1, dtype=np.int64)
