from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Literal

import numpy as np
from scipy.stats import t, ttest_ind

from bidsforge.processing.utils.epoching import (
    EpochExtractionResult,
    _sample_offsets,
)
from bidsforge.processing.utils.statistics import correct_p_values
from bidsforge.processing.utils.trial_resolver import ResolvedTrial


def compute_permuted_statistics(
    epochs_a: np.ndarray,
    epochs_b: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    *,
    equal_var: bool = False,
) -> np.ndarray:
    """Compute permuted t-values by randomly shuffling condition labels.

    All trials are pooled and randomly split into two pseudo-groups (preserving
    the source trial counts) on each iteration.  The resulting t-values form a
    null distribution against which the real statistic can be compared.

    Parameters
    ----------
    epochs_a, epochs_b:
        Arrays of shape ``(n_trials_a, n_channels, n_times)`` and
        ``(n_trials_b, n_channels, n_times)`` respectively.
    n_perm:
        Number of permutation iterations.
    rng:
        NumPy random Generator used for reproducible label shuffling.
    equal_var:
        Forwarded to ``scipy.stats.ttest_ind``; ``False`` selects Welch's t-test.

    Returns
    -------
    permuted_t_values : float32 array, shape ``(n_perm, n_channels, n_times)``
        NaN-filled when either condition is empty.
    """
    if epochs_a.ndim != 3 or epochs_b.ndim != 3:
        raise ValueError("epochs_a and epochs_b must be 3-D (n_trials, n_channels, n_times).")

    n_channels = epochs_a.shape[1]
    n_times = epochs_a.shape[2]
    n_a = epochs_a.shape[0]
    n_b = epochs_b.shape[0]

    if n_a == 0 or n_b == 0:
        return np.full((n_perm, n_channels, n_times), np.nan, dtype=np.float32)

    pooled = np.concatenate([epochs_a, epochs_b], axis=0)  # (n_total, n_channels, n_times)
    n_total = n_a + n_b
    out = np.empty((n_perm, n_channels, n_times), dtype=np.float32)

    for i in range(n_perm):
        perm_idx = rng.permutation(n_total)
        perm_a = pooled[perm_idx[:n_a]]
        perm_b = pooled[perm_idx[n_a:]]
        stats = ttest_ind(perm_a, perm_b, axis=0, equal_var=equal_var, nan_policy="omit")
        out[i] = np.asarray(stats.statistic, dtype=np.float32)

    return out


def compute_bootstrap_difference_ci95(
    epochs_a: np.ndarray,
    epochs_b: np.ndarray,
    *,
    n_bootstraps: int = 2000,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate 95% bootstrap CIs for mean(A) - mean(B) per channel and time."""
    arr_a = np.asarray(epochs_a, dtype=np.float64)
    arr_b = np.asarray(epochs_b, dtype=np.float64)
    if arr_a.ndim != 3 or arr_b.ndim != 3:
        raise ValueError("epochs_a and epochs_b must be 3-D (n_trials, n_channels, n_times).")
    if arr_a.shape[1:] != arr_b.shape[1:]:
        raise ValueError(
            "epochs_a and epochs_b must share channel/time dimensions, got "
            f"{arr_a.shape[1:]!r} and {arr_b.shape[1:]!r}."
        )

    n_a, n_channels, n_times = arr_a.shape
    n_b = arr_b.shape[0]
    if n_a < 2 or n_b < 2:
        nan = np.full((n_channels, n_times), np.nan, dtype=np.float64)
        return nan.copy(), nan.copy()
    if n_bootstraps < 1:
        raise ValueError(f"n_bootstraps must be >= 1, got {n_bootstraps}.")

    rng = np.random.default_rng(random_state)
    boot_diff = np.empty((n_bootstraps, n_channels, n_times), dtype=np.float64)

    for idx in range(n_bootstraps):
        sample_a = arr_a[rng.integers(0, n_a, size=n_a)]
        sample_b = arr_b[rng.integers(0, n_b, size=n_b)]
        boot_diff[idx] = (
            np.nanmean(sample_a, axis=0, dtype=np.float64)
            - np.nanmean(sample_b, axis=0, dtype=np.float64)
        )

    low = np.nanpercentile(boot_diff, 2.5, axis=0)
    high = np.nanpercentile(boot_diff, 97.5, axis=0)
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)


def compute_analytic_difference_ci95(
    epochs_a: np.ndarray,
    epochs_b: np.ndarray,
    *,
    equal_var: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate fast parametric 95% CIs for mean(A) - mean(B)."""
    arr_a = np.asarray(epochs_a, dtype=np.float64)
    arr_b = np.asarray(epochs_b, dtype=np.float64)
    if arr_a.ndim != 3 or arr_b.ndim != 3:
        raise ValueError("epochs_a and epochs_b must be 3-D (n_trials, n_channels, n_times).")
    if arr_a.shape[1:] != arr_b.shape[1:]:
        raise ValueError(
            "epochs_a and epochs_b must share channel/time dimensions, got "
            f"{arr_a.shape[1:]!r} and {arr_b.shape[1:]!r}."
        )

    n_channels, n_times = arr_a.shape[1:]
    count_a = np.sum(np.isfinite(arr_a), axis=0).astype(np.float64)
    count_b = np.sum(np.isfinite(arr_b), axis=0).astype(np.float64)
    valid = (count_a >= 2) & (count_b >= 2)
    if not np.any(valid):
        nan = np.full((n_channels, n_times), np.nan, dtype=np.float64)
        return nan.copy(), nan.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_a = np.nanmean(arr_a, axis=0, dtype=np.float64)
        mean_b = np.nanmean(arr_b, axis=0, dtype=np.float64)
        var_a = np.nanvar(arr_a, axis=0, ddof=1, dtype=np.float64)
        var_b = np.nanvar(arr_b, axis=0, ddof=1, dtype=np.float64)
    diff = mean_a - mean_b

    if equal_var:
        df = count_a + count_b - 2.0
        pooled_var = ((count_a - 1.0) * var_a + (count_b - 1.0) * var_b) / df
        se = np.sqrt(pooled_var * ((1.0 / count_a) + (1.0 / count_b)))
    else:
        term_a = var_a / count_a
        term_b = var_b / count_b
        se = np.sqrt(term_a + term_b)
        with np.errstate(invalid="ignore", divide="ignore"):
            df = np.square(term_a + term_b) / (
                np.square(term_a) / (count_a - 1.0)
                + np.square(term_b) / (count_b - 1.0)
            )

    finite = valid & np.isfinite(diff) & np.isfinite(se) & np.isfinite(df) & (df > 0)
    low = np.full((n_channels, n_times), np.nan, dtype=np.float64)
    high = low.copy()
    if np.any(finite):
        margin = t.ppf(0.975, df[finite]) * se[finite]
        low[finite] = diff[finite] - margin
        high[finite] = diff[finite] + margin
    return low, high


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


def compute_single_bin_channel_significance(
    epochs_a: np.ndarray,
    epochs_b: np.ndarray,
    *,
    equal_var: bool = False,
    p_value_correction_method: Literal["none", "fdr_bh", "bonferroni"] = "fdr_bh",
    significance_alpha: float = 0.05,
) -> np.ndarray:
    """Derive a per-channel significance flag by collapsing epochs to their temporal mean.

    Each trial epoch is reduced to a single value per channel by averaging over the time
    axis.  A two-sample t-test is then run across trials for each channel (equivalent to
    running the full pipeline with ``n_bins=1``).  The same ``p_value_correction_method``
    that controls the main analysis is applied, but only across the channel dimension.

    Parameters
    ----------
    epochs_a, epochs_b:
        Shape ``(n_trials, n_channels, n_times)``.  Both arrays must have ``n_times >= 1``.
    equal_var:
        Forwarded to ``scipy.stats.ttest_ind``; ``False`` selects Welch's t-test.
    p_value_correction_method:
        Multiple-comparisons correction applied across channels.
    significance_alpha:
        Threshold applied to corrected p-values.

    Returns
    -------
    channel_significant_mask : bool array, shape ``(n_channels,)``
        ``True`` for channels where the corrected p-value is below ``significance_alpha``.
        All ``False`` when either condition is empty.
    """
    if epochs_a.ndim != 3 or epochs_b.ndim != 3:
        raise ValueError("epochs_a and epochs_b must be 3-D (n_trials, n_channels, n_times).")

    n_channels = epochs_a.shape[1]

    if epochs_a.shape[0] == 0 or epochs_b.shape[0] == 0:
        return np.zeros(n_channels, dtype=bool)

    # Collapse time axis: shape (n_trials, n_channels, 1) so existing helpers are reusable.
    means_a = epochs_a.mean(axis=2, keepdims=True)  # (n_trials_a, n_channels, 1)
    means_b = epochs_b.mean(axis=2, keepdims=True)  # (n_trials_b, n_channels, 1)

    _, p_raw_2d, _, _, _ = compute_condition_statistics(
        means_a,
        means_b,
        n_channels=n_channels,
        n_times=1,
        equal_var=equal_var,
    )
    p_raw_1d = p_raw_2d[:, 0]  # (n_channels,)

    p_corrected_1d = correct_p_values(
        p_raw_1d.reshape(1, n_channels),
        method=p_value_correction_method,
    )[0]

    return np.isfinite(p_corrected_1d) & (p_corrected_1d < significance_alpha)


def compute_duration_channel_significance(
    significant_mask: np.ndarray,
    time_axis_s: np.ndarray,
    *,
    threshold_ms: float = 100.0,
) -> np.ndarray:
    """Flag channels whose total significant-bin duration meets a threshold.

    Every significant bin contributes its full duration (derived from the uniform time
    axis) to the channel total.  A channel is flagged when that total reaches
    ``threshold_ms``.

    Parameters
    ----------
    significant_mask : bool array, shape ``(n_channels, n_times)``
        Per-channel, per-bin significance flags (e.g. from ``TrialStatsProcessingResult``).
    time_axis_s : float array, shape ``(n_times,)``
        Time axis in seconds.  Used to derive the per-bin duration in milliseconds.
        Must contain at least 2 elements when ``n_times >= 2``.
    threshold_ms:
        Minimum total significant duration (in milliseconds) required to flag a channel.
        Default is 100 ms.

    Returns
    -------
    channel_significant_mask : bool array, shape ``(n_channels,)``
        ``True`` for channels whose total significant duration is ``>= threshold_ms``.
    """
    mask = np.asarray(significant_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("significant_mask must be 2-D (n_channels, n_times).")

    n_channels, n_times = mask.shape

    if n_times == 0:
        return np.zeros(n_channels, dtype=bool)

    if n_times == 1:
        # Cannot derive a meaningful bin duration from a single-element time axis;
        # treat total duration as 0 so no channel can reach any positive threshold.
        return np.zeros(n_channels, dtype=bool)

    dt_ms = float(time_axis_s[1] - time_axis_s[0]) * 1000.0

    significant_count = mask.sum(axis=1)  # (n_channels,)
    duration_ms = significant_count * dt_ms

    return duration_ms >= threshold_ms
