"""Pure functions for epoch-level and channel-level quality control.

This module provides two levels of cleaning suitable for trial-epoched data:

*Level A – trial-channel NaN masking.*
    Per channel, trials whose temporal mean or maximum deviates more than
    ``threshold_factor`` standard deviations from the channel's across-trial
    mean are set to NaN.  This replicates the Matlab ``rmoutliers(..., 'mean',
    ...)``  call applied per channel on ``mean_alldata_per_trial`` and
    ``max_alldata_per_trial`` (``std_thresh = 3`` in the Matlab initialisation).

*Level B – channel exclusion.*
    Channels are dropped when the across-trial spread of their per-trial means
    (or maxima) is an outlier in the channel population
    (``ThresholdFactor = 1`` in the Matlab code), or when more than
    ``max_nan_trial_ratio`` of their trials are NaN.

Level A must be applied before Level B so that the NaN trial counts used in
the NaN-ratio check include Level A rejects.
"""

from __future__ import annotations

import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mean_outlier_mask_1d(values: np.ndarray, threshold_factor: float) -> np.ndarray:
    """Return a bool mask where ``|x − mean| > threshold_factor × std`` (ddof=1).

    Parameters
    ----------
    values:
        1-D float array.  Non-finite values are ignored when computing the
        reference mean and standard deviation, but retain their original
        (non-outlier) classification.
    threshold_factor:
        Number of standard deviations defining the outlier boundary.

    Returns
    -------
    np.ndarray
        Bool array with the same shape as *values*; ``True`` = outlier.
        All-False when fewer than two finite values are present or when the
        standard deviation is zero.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(arr)
    if int(np.sum(finite)) < 2:
        return np.zeros(arr.shape, dtype=bool)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=1))
    if not (std > 0.0):
        return np.zeros(arr.shape, dtype=bool)
    return np.abs(arr - mean) > threshold_factor * std


# ---------------------------------------------------------------------------
# Level A: trial-channel NaN masking
# ---------------------------------------------------------------------------


def detect_outlier_trial_channel_pairs_by_mean(
    epochs: np.ndarray,
    threshold_factor: float = 3.0,
) -> np.ndarray:
    """Detect per-channel trial outliers using the epoch temporal mean.

    For each channel independently, trials whose mean activity deviates by more
    than ``threshold_factor × std`` from the channel's across-trial mean are
    flagged.  The detection is equivalent to Matlab's
    ``rmoutliers(mean_alldata_per_trial(:, ichan), 'mean', 'ThresholdFactor',
    std_thresh)``.

    Parameters
    ----------
    epochs:
        Array of shape ``(n_trials, n_channels, n_times)``.  Must not contain
        NaN (call before :func:`apply_trial_nan_mask`).
    threshold_factor:
        Number of standard deviations defining the outlier threshold.

    Returns
    -------
    np.ndarray
        Bool array of shape ``(n_trials, n_channels)``.  ``True`` at
        ``(t, c)`` means trial *t* is an outlier for channel *c*.
    """
    epochs_arr = np.asarray(epochs, dtype=np.float64)
    trial_means = np.mean(epochs_arr, axis=2)  # [n_trials, n_channels]
    chan_means = np.mean(trial_means, axis=0)  # [n_channels]
    chan_stds = np.std(trial_means, axis=0, ddof=1)  # [n_channels]
    deviation = np.abs(trial_means - chan_means[np.newaxis, :])
    valid_std = chan_stds > 0.0
    mask = np.zeros(trial_means.shape, dtype=bool)
    mask[:, valid_std] = deviation[:, valid_std] > threshold_factor * chan_stds[np.newaxis, valid_std]
    return mask


def detect_outlier_trial_channel_pairs_by_max(
    epochs: np.ndarray,
    threshold_factor: float = 3.0,
) -> np.ndarray:
    """Detect per-channel trial outliers using the epoch temporal absolute maximum.

    Same as :func:`detect_outlier_trial_channel_pairs_by_mean` but uses the
    per-trial temporal absolute maximum instead of the mean.  This captures
    large-amplitude artefacts regardless of polarity.  Inspired by Matlab's
    ``rmoutliers(max_alldata_pertrial(:, ichan), 'mean', 'ThresholdFactor',
    std_thresh)`` but applied to ``abs(epochs)`` for correctness on biphasic
    signals.

    Parameters
    ----------
    epochs:
        Array of shape ``(n_trials, n_channels, n_times)``.
    threshold_factor:
        Number of standard deviations defining the outlier threshold.

    Returns
    -------
    np.ndarray
        Bool array of shape ``(n_trials, n_channels)``.
    """
    epochs_arr = np.asarray(epochs, dtype=np.float64)
    trial_maxes = np.max(np.abs(epochs_arr), axis=2)  # [n_trials, n_channels]
    chan_means = np.mean(trial_maxes, axis=0)  # [n_channels]
    chan_stds = np.std(trial_maxes, axis=0, ddof=1)  # [n_channels]
    deviation = np.abs(trial_maxes - chan_means[np.newaxis, :])
    valid_std = chan_stds > 0.0
    mask = np.zeros(trial_maxes.shape, dtype=bool)
    mask[:, valid_std] = deviation[:, valid_std] > threshold_factor * chan_stds[np.newaxis, valid_std]
    return mask


def apply_trial_nan_mask(
    epochs: np.ndarray,
    nan_mask: np.ndarray,
) -> np.ndarray:
    """Return a float64 copy of *epochs* with flagged (trial, channel) pairs set to NaN.

    Parameters
    ----------
    epochs:
        Array of shape ``(n_trials, n_channels, n_times)``.
    nan_mask:
        Bool array of shape ``(n_trials, n_channels)``.  Each ``True`` position
        causes the entire time slice ``epochs[t, c, :]`` to become NaN.

    Returns
    -------
    np.ndarray
        Float64 copy of *epochs* with NaN where *nan_mask* is ``True``.
    """
    out = np.array(epochs, dtype=np.float64)
    # Boolean indexing over the first two dimensions: selects [n_selected, n_times]
    # slices where nan_mask is True and fills them with NaN.
    out[nan_mask] = np.nan
    return out


# ---------------------------------------------------------------------------
# Level B: channel exclusion
# ---------------------------------------------------------------------------


def reject_channels_by_trial_mean_spread(
    epochs: np.ndarray,
    threshold_factor: float = 1.0,
) -> np.ndarray:
    """Flag channels whose across-trial spread (from trial means) is an outlier.

    For each channel the standard deviation of per-trial temporal means is
    computed.  Channels whose spread deviates more than ``threshold_factor``
    standard deviations from the population mean of spreads are flagged.
    Equivalent to Matlab's ``rmoutliers(sd_perchan, 'mean',
    'ThresholdFactor', 1)`` applied to the std-of-trial-means per channel.

    Parameters
    ----------
    epochs:
        Array of shape ``(n_trials, n_channels, n_times)``.  May contain NaN
        (after Level A).
    threshold_factor:
        Outlier threshold; defaults to ``1.0`` to match the Matlab pipeline.

    Returns
    -------
    np.ndarray
        Bool array of shape ``(n_channels,)``.  ``True`` = exclude.
    """
    epochs_arr = np.asarray(epochs, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        trial_means = np.nanmean(epochs_arr, axis=2)  # [n_trials, n_channels]
    chan_spread = np.nanstd(trial_means, axis=0, ddof=1)  # [n_channels]
    return _mean_outlier_mask_1d(chan_spread, threshold_factor)


def reject_channels_by_trial_max_spread(
    epochs: np.ndarray,
    threshold_factor: float = 1.0,
) -> np.ndarray:
    """Flag channels whose across-trial spread (from trial absolute maxima) is an outlier.

    Same as :func:`reject_channels_by_trial_mean_spread` but uses the
    per-trial temporal absolute maximum instead of the mean.

    Parameters
    ----------
    epochs:
        Array of shape ``(n_trials, n_channels, n_times)``.  May contain NaN.
    threshold_factor:
        Outlier threshold.

    Returns
    -------
    np.ndarray
        Bool array of shape ``(n_channels,)``.  ``True`` = exclude.
    """
    epochs_arr = np.asarray(epochs, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        trial_maxes = np.nanmax(np.abs(epochs_arr), axis=2)  # [n_trials, n_channels]
    chan_spread = np.nanstd(trial_maxes, axis=0, ddof=1)  # [n_channels]
    return _mean_outlier_mask_1d(chan_spread, threshold_factor)


def reject_channels_by_nan_trial_ratio(
    epochs: np.ndarray,
    max_ratio: float,
) -> np.ndarray:
    """Flag channels where the fraction of NaN trials meets or exceeds *max_ratio*.

    A trial is counted as NaN for channel *c* when any of its time samples is
    non-finite (as set by :func:`apply_trial_nan_mask`).

    Parameters
    ----------
    epochs:
        Array of shape ``(n_trials, n_channels, n_times)``.
    max_ratio:
        Exclusion threshold in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Bool array of shape ``(n_channels,)``.  ``True`` = exclude.
    """
    epochs_arr = np.asarray(epochs, dtype=np.float64)
    nan_per_trial_chan = np.any(~np.isfinite(epochs_arr), axis=2)  # [n_trials, n_channels]
    nan_ratios = np.mean(nan_per_trial_chan.astype(np.float64), axis=0)  # [n_channels]
    return nan_ratios >= max_ratio


def apply_channel_exclusions(
    epochs: np.ndarray,
    channel_names: list[str],
    exclusion_mask: np.ndarray,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Remove flagged channels from *epochs* and return the updated names.

    Parameters
    ----------
    epochs:
        Array of shape ``(n_trials, n_channels, n_times)``.
    channel_names:
        List of channel names matching the channel axis (length ``n_channels``).
    exclusion_mask:
        Bool array of shape ``(n_channels,)``.  ``True`` = exclude the channel.

    Returns
    -------
    epochs_clean, names_clean, excluded_names
        ``epochs_clean`` has shape ``(n_trials, n_kept, n_times)``.
        ``names_clean`` lists the retained channel names.
        ``excluded_names`` lists the removed channel names.
    """
    exc = np.asarray(exclusion_mask, dtype=bool)
    keep = ~exc
    kept_names = [name for name, k in zip(channel_names, keep) if k]
    excluded_names = [name for name, e in zip(channel_names, exc) if e]
    return epochs[:, keep, :], kept_names, excluded_names
