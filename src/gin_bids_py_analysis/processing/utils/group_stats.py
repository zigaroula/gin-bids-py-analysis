"""Group-level statistical helpers shared across group-analysis pipelines.

Provides the ROI channel contribution type and the three core one-sample
statistical functions used by ``trial_stats_group`` and
``trial_slope_stats_group``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ttest_1samp


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
