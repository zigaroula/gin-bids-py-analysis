"""Group-level statistical helpers shared across group-analysis pipelines.

Provides the ROI channel contribution type and the three core one-sample
statistical functions used by ``trial_stats_group`` and
``trial_slope_stats_group``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel


@dataclass(frozen=True)
class ROIChannelContribution:
    """One channel contribution used for one ROI in a group-level analysis."""

    roi: str
    subject: str
    channel: str
    source_stats_file: str


def compute_one_sample_timecourse(
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute one-sample (vs 0) statistics for a ``[n_samples, n_times]`` matrix.

    Parameters
    ----------
    samples:
        2D array of shape ``(n_samples, n_times)``.

    Returns
    -------
    t_values, p_values, mean_values, sem_values
        All shape ``(n_times,)``.  All-NaN arrays are returned when
        ``n_samples == 0``.
    """
    sample_arr = np.asarray(samples, dtype=np.float64)
    if sample_arr.ndim != 2:
        raise ValueError(
            f"samples must be a 2D array of shape [n_samples, n_times], got {sample_arr.shape!r}."
        )
    n_times = int(sample_arr.shape[1])
    if sample_arr.shape[0] == 0:
        empty = np.full((n_times,), np.nan, dtype=np.float64)
        return empty.copy(), empty.copy(), empty.copy(), empty.copy()

    stats = ttest_1samp(sample_arr, popmean=0.0, axis=0, nan_policy="omit")
    t_values = np.asarray(stats.statistic, dtype=np.float64)
    p_values = np.asarray(stats.pvalue, dtype=np.float64)

    mean_values = np.nanmean(sample_arr, axis=0, dtype=np.float64)
    sem_values = _nansem(sample_arr, axis=0)
    return t_values, p_values, mean_values, sem_values


def compute_one_sample_epoch_summary(
    samples: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Summarize one ROI on the full epoch by averaging each channel over time then testing vs 0.

    Parameters
    ----------
    samples:
        2D array of shape ``(n_samples, n_times)``.

    Returns
    -------
    t_value, p_value, df, mean, sem
        All scalars.  NaN when there are insufficient finite values.
    """
    sample_arr = np.asarray(samples, dtype=np.float64)
    if sample_arr.ndim != 2:
        raise ValueError(
            f"samples must be a 2D array of shape [n_samples, n_times], got {sample_arr.shape!r}."
        )
    if sample_arr.shape[0] == 0 or sample_arr.shape[1] == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    epoch_means = np.nanmean(sample_arr, axis=1, dtype=np.float64)
    valid = np.isfinite(epoch_means)
    if not valid.any():
        return np.nan, np.nan, np.nan, np.nan, np.nan

    valid_values = epoch_means[valid]
    summary_mean = float(np.nanmean(valid_values))
    summary_sem = float(_nansem(valid_values, axis=0))
    if valid_values.size < 2:
        return np.nan, np.nan, np.nan, summary_mean, summary_sem

    stats = ttest_1samp(valid_values, popmean=0.0, nan_policy="omit")
    df = float(valid_values.size - 1)
    return (
        float(np.asarray(stats.statistic, dtype=np.float64)),
        float(np.asarray(stats.pvalue, dtype=np.float64)),
        df,
        summary_mean,
        summary_sem,
    )


def compute_condition_group_stats(
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return group-level (mean, SEM) for a ``[n_samples, n_times]`` matrix.

    Parameters
    ----------
    samples:
        2D array of shape ``(n_samples, n_times)`` — each row is one channel's
        timecourse (already channel-mean from the subject-level file).

    Returns
    -------
    mean_values : shape ``(n_times,)``
    sem_values  : shape ``(n_times,)``
    """
    sample_arr = np.asarray(samples, dtype=np.float64)
    if sample_arr.ndim != 2:
        raise ValueError(
            f"samples must be a 2D array of shape [n_samples, n_times], got {sample_arr.shape!r}."
        )
    n_times = int(sample_arr.shape[1])
    if sample_arr.shape[0] == 0:
        empty = np.full((n_times,), np.nan, dtype=np.float64)
        return empty.copy(), empty.copy()
    mean_values = np.nanmean(sample_arr, axis=0, dtype=np.float64)
    sem_values = _nansem(sample_arr, axis=0)
    return mean_values, sem_values


def _nansem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return NaN-aware SEM, with NaN when fewer than 2 finite values are available."""
    arr = np.asarray(values, dtype=np.float64)
    count = np.sum(np.isfinite(arr), axis=axis)
    std = np.nanstd(arr, axis=axis, ddof=1, dtype=np.float64)
    sem = std / np.sqrt(np.maximum(count, 1))
    return np.where(count >= 2, sem, np.nan)


def compute_two_sample_timecourse(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute two-sample (a vs b) t-test statistics for two ``[n_samples, n_times]`` matrices.

    Parameters
    ----------
    samples_a, samples_b:
        2D arrays of shape ``(n_samples_a, n_times)`` and ``(n_samples_b, n_times)``.

    Returns
    -------
    t_values, p_values
        Both shape ``(n_times,)``.  All-NaN arrays are returned when either
        group has fewer than 2 samples.
    """
    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("samples_a and samples_b must be 2D arrays of shape [n_samples, n_times].")
    n_times = int(a.shape[1])
    if a.shape[0] < 2 or b.shape[0] < 2:
        empty = np.full((n_times,), np.nan, dtype=np.float64)
        return empty.copy(), empty.copy()

    stats = ttest_ind(a, b, axis=0, equal_var=False, nan_policy="omit")
    t_values = np.asarray(stats.statistic, dtype=np.float64)
    p_values = np.asarray(stats.pvalue, dtype=np.float64)
    return t_values, p_values


def compute_paired_timecourse(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute paired t-test statistics for two matched ``[n_samples, n_times]`` matrices."""
    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("samples_a and samples_b must be 2D arrays of shape [n_samples, n_times].")
    if a.shape != b.shape:
        raise ValueError(
            "Paired statistics require samples_a and samples_b to share the same shape."
        )
    n_times = int(a.shape[1])
    if a.shape[0] < 2:
        empty = np.full((n_times,), np.nan, dtype=np.float64)
        return empty.copy(), empty.copy()

    t_values = np.full((n_times,), np.nan, dtype=np.float64)
    p_values = np.full((n_times,), np.nan, dtype=np.float64)
    for time_idx in range(n_times):
        a_col = a[:, time_idx]
        b_col = b[:, time_idx]
        valid = np.isfinite(a_col) & np.isfinite(b_col)
        if np.sum(valid) < 2:
            continue
        stats = ttest_rel(a_col[valid], b_col[valid], nan_policy="omit")
        t_values[time_idx] = float(np.asarray(stats.statistic, dtype=np.float64))
        p_values[time_idx] = float(np.asarray(stats.pvalue, dtype=np.float64))
    return t_values, p_values


def compute_two_sample_epoch_summary(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> tuple[float, float, float]:
    """Summarize one ROI on the full epoch by testing group A vs group B over the mean time.

    Each channel timecourse is first averaged across time, then the two groups are
    compared with a two-sample t-test.

    Parameters
    ----------
    samples_a, samples_b:
        2D arrays of shape ``(n_samples_a, n_times)`` and ``(n_samples_b, n_times)``.

    Returns
    -------
    t_value, p_value, df
        All scalars.  NaN when there are insufficient finite values.
    """
    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("samples_a and samples_b must be 2D arrays of shape [n_samples, n_times].")
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.nan, np.nan, np.nan

    means_a = np.nanmean(a, axis=1, dtype=np.float64)
    means_b = np.nanmean(b, axis=1, dtype=np.float64)
    valid_a = means_a[np.isfinite(means_a)]
    valid_b = means_b[np.isfinite(means_b)]
    if valid_a.size < 2 or valid_b.size < 2:
        return np.nan, np.nan, np.nan

    stats = ttest_ind(valid_a, valid_b, equal_var=False, nan_policy="omit")
    df = float(valid_a.size + valid_b.size - 2)
    return (
        float(np.asarray(stats.statistic, dtype=np.float64)),
        float(np.asarray(stats.pvalue, dtype=np.float64)),
        df,
    )


def compute_paired_epoch_summary(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> tuple[float, float, float]:
    """Summarize one ROI on the full epoch with a paired t-test on channel means."""
    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("samples_a and samples_b must be 2D arrays of shape [n_samples, n_times].")
    if a.shape != b.shape:
        raise ValueError(
            "Paired statistics require samples_a and samples_b to share the same shape."
        )
    if a.shape[0] == 0 or a.shape[1] == 0:
        return np.nan, np.nan, np.nan

    means_a = np.nanmean(a, axis=1, dtype=np.float64)
    means_b = np.nanmean(b, axis=1, dtype=np.float64)
    valid = np.isfinite(means_a) & np.isfinite(means_b)
    if np.sum(valid) < 2:
        return np.nan, np.nan, np.nan

    stats = ttest_rel(means_a[valid], means_b[valid], nan_policy="omit")
    df = float(np.sum(valid) - 1)
    return (
        float(np.asarray(stats.statistic, dtype=np.float64)),
        float(np.asarray(stats.pvalue, dtype=np.float64)),
        df,
    )
