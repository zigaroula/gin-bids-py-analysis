"""Statistical helper utilities shared across processing pipelines.

Provides p-value correction and per-condition summary statistics (mean, SEM)
used by trial_stats and trial_slope_stats processors.
"""
from __future__ import annotations

import warnings
from typing import Literal

from mne.stats import bonferroni_correction, fdr_correction
import numpy as np


def correct_p_values(
    p_values: np.ndarray,
    *,
    method: Literal["none", "fdr_bh", "bonferroni", "permutation", "cluster_permutation"] = "fdr_bh",
) -> np.ndarray:
    """Apply multiple-comparisons correction to p-values.

    Parameters
    ----------
    p_values:
        Array of p-values of any shape. NaN values are ignored.
    method:
        Correction method. ``"none"`` returns a copy unchanged.
        ``"permutation"`` and ``"cluster_permutation"`` are treated as ``"none"``
        here because permutation p-values are computed externally; the caller
        is responsible for passing the already-corrected values or handling
        this case separately.

    Returns
    -------
    corrected : float64 array, same shape as *p_values*.
    """
    corrected = np.asarray(p_values, dtype=np.float64).copy()
    finite_mask = np.isfinite(corrected)
    if not finite_mask.any() or method in ("none", "permutation", "cluster_permutation"):
        return corrected

    flat = corrected[finite_mask]
    if method == "bonferroni":
        _, corrected_flat = bonferroni_correction(flat, alpha=0.05)
        corrected[finite_mask] = np.asarray(corrected_flat, dtype=np.float64)
        return corrected

    if method == "fdr_bh":
        _, corrected_flat = fdr_correction(flat, alpha=0.05, method="indep")
        corrected[finite_mask] = np.asarray(corrected_flat, dtype=np.float64)
        return corrected

    raise ValueError(
        f"Unsupported p-value correction method: {method!r}. "
        "Valid methods are 'none', 'fdr_bh', 'bonferroni', 'permutation', 'cluster_permutation'."
    )


def compute_condition_mean(
    epochs: np.ndarray,
    *,
    n_features: int,
    n_times: int,
) -> np.ndarray:
    """Return per-feature mean across trials, or NaN when no trials are present."""
    if epochs.shape[0] == 0:
        return np.full((n_features, n_times), np.nan, dtype=np.float64)
    return np.nanmean(epochs, axis=0, dtype=np.float64)


def compute_condition_sem(
    epochs: np.ndarray,
    *,
    n_features: int,
    n_times: int,
) -> np.ndarray:
    """Return per-feature SEM across trials, or NaN when fewer than 2 trials."""
    if epochs.shape[0] < 2:
        return np.full((n_features, n_times), np.nan, dtype=np.float64)

    std = np.nanstd(
        epochs,
        axis=0,
        ddof=1,
        dtype=np.float64,
    )
    return std / np.sqrt(float(epochs.shape[0]))


def zscore_activity_by_baseline(
    epochs_a: np.ndarray,
    epochs_b: np.ndarray,
    time_axis_s: np.ndarray,
    *,
    baseline_tmin_s: float,
    baseline_tmax_s: float,
    baseline_scope: Literal["trial", "condition", "global"] = "global",
    remove_outlier_trial_means: bool = False,
    outlier_method: Literal["median_mad", "mean"] = "median_mad",
) -> tuple[np.ndarray, np.ndarray]:
    """Z-score activity relative to a baseline window.

    ``baseline_scope="trial"`` uses each trial's own baseline samples.
    ``"condition"`` computes a separate reference for each condition from the
    per-trial baseline means of that condition. ``"global"`` computes one shared
    reference from the per-trial baseline means pooled across both conditions,
    which matches the MATLAB direction used for the current alignment work.
    """
    arr_a = np.asarray(epochs_a, dtype=np.float64)
    arr_b = np.asarray(epochs_b, dtype=np.float64)
    time_axis = np.asarray(time_axis_s, dtype=np.float64).ravel()
    if arr_a.ndim != 3 or arr_b.ndim != 3:
        raise ValueError("epochs_a and epochs_b must be 3-D (n_trials, n_features, n_times).")
    if arr_a.shape[1:] != arr_b.shape[1:]:
        raise ValueError(
            "epochs_a and epochs_b must share feature/time dimensions, got "
            f"{arr_a.shape[1:]!r} and {arr_b.shape[1:]!r}."
        )
    if arr_a.shape[2] != time_axis.shape[0]:
        raise ValueError(
            "time_axis_s must match the epoch time dimension, got "
            f"{time_axis.shape[0]!r} and {arr_a.shape[2]!r}."
        )
    if baseline_tmax_s <= baseline_tmin_s:
        raise ValueError(
            "baseline_tmax_s must be greater than baseline_tmin_s."
        )

    baseline_mask = (time_axis >= baseline_tmin_s) & (time_axis <= baseline_tmax_s)
    if not np.any(baseline_mask):
        raise ValueError(
            "Baseline window does not overlap the epoch time axis: "
            f"[{baseline_tmin_s}, {baseline_tmax_s}] vs "
            f"[{time_axis[0]}, {time_axis[-1]}]."
        )

    scope = str(baseline_scope).strip().lower()
    if scope == "trial":
        return (
            _zscore_activity_by_trial_baseline(arr_a, baseline_mask),
            _zscore_activity_by_trial_baseline(arr_b, baseline_mask),
        )

    if scope == "condition":
        return (
            _zscore_activity_by_trial_mean_reference(
                arr_a,
                baseline_mask,
                remove_outlier_trial_means=remove_outlier_trial_means,
                outlier_method=outlier_method,
            ),
            _zscore_activity_by_trial_mean_reference(
                arr_b,
                baseline_mask,
                remove_outlier_trial_means=remove_outlier_trial_means,
                outlier_method=outlier_method,
            ),
        )

    if scope != "global":
        raise ValueError(
            f"Unsupported baseline_scope: {baseline_scope!r}. "
            "Valid values are 'trial', 'condition', and 'global'."
        )

    trial_means_parts = [
        _trial_baseline_means(arr, baseline_mask)
        for arr in (arr_a, arr_b)
        if arr.shape[0] > 0
    ]
    if not trial_means_parts:
        return arr_a.copy(), arr_b.copy()

    reference_mean, reference_scale = _baseline_reference_from_trial_means(
        np.concatenate(trial_means_parts, axis=0),
        remove_outlier_trial_means=remove_outlier_trial_means,
        outlier_method=outlier_method,
    )
    return (
        _apply_feature_reference(arr_a, reference_mean, reference_scale),
        _apply_feature_reference(arr_b, reference_mean, reference_scale),
    )


def _zscore_activity_by_trial_baseline(
    epochs: np.ndarray,
    baseline_mask: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(epochs, dtype=np.float64)
    if arr.shape[0] == 0:
        return arr.copy()
    baseline = arr[:, :, baseline_mask]
    mean = np.nanmean(baseline, axis=2, dtype=np.float64)
    std = np.nanstd(baseline, axis=2, ddof=1, dtype=np.float64)
    scale = np.where(np.isfinite(std) & (std > 0.0), std, 1.0)
    centered = arr - mean[:, :, np.newaxis]
    out = centered / scale[:, :, np.newaxis]
    out[~np.isfinite(arr)] = np.nan
    return out


def _zscore_activity_by_trial_mean_reference(
    epochs: np.ndarray,
    baseline_mask: np.ndarray,
    *,
    remove_outlier_trial_means: bool,
    outlier_method: Literal["median_mad", "mean"] = "median_mad",
) -> np.ndarray:
    arr = np.asarray(epochs, dtype=np.float64)
    if arr.shape[0] == 0:
        return arr.copy()
    trial_means = _trial_baseline_means(arr, baseline_mask)
    reference_mean, reference_scale = _baseline_reference_from_trial_means(
        trial_means,
        remove_outlier_trial_means=remove_outlier_trial_means,
        outlier_method=outlier_method,
    )
    return _apply_feature_reference(arr, reference_mean, reference_scale)


def _trial_baseline_means(
    epochs: np.ndarray,
    baseline_mask: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(epochs, dtype=np.float64)
    if arr.shape[0] == 0:
        return np.empty((0, arr.shape[1]), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(arr[:, :, baseline_mask], axis=2, dtype=np.float64)


def _baseline_reference_from_trial_means(
    trial_means: np.ndarray,
    *,
    remove_outlier_trial_means: bool,
    outlier_method: Literal["median_mad", "mean"] = "median_mad",
) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(trial_means, dtype=np.float64)
    if means.ndim != 2:
        raise ValueError(
            "trial_means must be 2-D with shape (n_trials, n_features)."
        )
    if means.shape[0] == 0:
        empty = np.empty((means.shape[1],), dtype=np.float64)
        return empty, empty

    if remove_outlier_trial_means:
        cleaned = _remove_outlier_trial_means(means, method=outlier_method)
    else:
        cleaned = means

    reference_mean = np.nanmean(cleaned, axis=0, dtype=np.float64)
    reference_std = np.nanstd(cleaned, axis=0, ddof=1, dtype=np.float64)
    reference_scale = np.where(
        np.isfinite(reference_std) & (reference_std > 0.0),
        reference_std,
        1.0,
    )
    return reference_mean, reference_scale


def _apply_feature_reference(
    epochs: np.ndarray,
    reference_mean: np.ndarray,
    reference_scale: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(epochs, dtype=np.float64)
    if arr.shape[0] == 0:
        return arr.copy()
    centered = arr - reference_mean[np.newaxis, :, np.newaxis]
    out = centered / reference_scale[np.newaxis, :, np.newaxis]
    out[~np.isfinite(arr)] = np.nan
    return out


def _outlier_trial_means_mask(
    trial_means: np.ndarray,
    method: Literal["median_mad", "mean"] = "median_mad",
) -> np.ndarray:
    """Return a bool mask ``(n_trials, n_features)`` flagging outlier baseline means.

    ``method='median_mad'`` uses the median ± 3 × 1.4826 × MAD criterion
    (matching MATLAB ``rmoutliers`` default).
    ``method='mean'`` uses the mean ± 3σ criterion (matching MATLAB
    ``rmoutliers(..., 'mean', 'ThresholdFactor', 3)`` as used in ``b2``
    HGA-trial cleaning).
    """
    arr = np.asarray(trial_means, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros(arr.shape, dtype=bool)

    mask = np.zeros(arr.shape, dtype=bool)
    for feature_idx in range(arr.shape[1]):
        values = arr[:, feature_idx]
        finite_positions = np.isfinite(values)
        finite_values = values[finite_positions]
        if finite_values.size < 3:
            continue
        if method == "mean":
            center = float(np.nanmean(finite_values))
            spread = float(np.nanstd(finite_values, ddof=1))
            if not np.isfinite(spread) or spread <= 0.0:
                continue
            threshold = 3.0 * spread
            outlier_positions = np.abs(finite_values - center) > threshold
        else:  # median_mad
            center = float(np.nanmedian(finite_values))
            mad = float(np.nanmedian(np.abs(finite_values - center)))
            if not np.isfinite(mad) or mad <= 0.0:
                continue
            threshold = 3.0 * 1.4826 * mad
            outlier_positions = np.abs(finite_values - center) > threshold
        if np.any(outlier_positions):
            finite_indices = np.flatnonzero(finite_positions)
            mask[finite_indices[outlier_positions], feature_idx] = True
    return mask


def _remove_outlier_trial_means(
    trial_means: np.ndarray,
    method: Literal["median_mad", "mean"] = "median_mad",
) -> np.ndarray:
    """Approximate MATLAB ``rmoutliers`` on per-feature trial means.

    ``method='median_mad'`` mirrors the MATLAB default (median/MAD).
    ``method='mean'`` mirrors ``rmoutliers(..., 'mean', 'ThresholdFactor', 3)``.
    """
    arr = np.asarray(trial_means, dtype=np.float64).copy()
    if arr.ndim != 2 or arr.shape[0] == 0:
        return arr
    arr[_outlier_trial_means_mask(arr, method=method)] = np.nan
    return arr


def compute_baseline_outlier_mask(
    epochs_a: np.ndarray,
    epochs_b: np.ndarray,
    time_axis_s: np.ndarray,
    *,
    baseline_tmin_s: float,
    baseline_tmax_s: float,
    baseline_scope: Literal["trial", "condition", "global"] = "global",
    remove_outlier_trial_means: bool = False,
    outlier_method: Literal["median_mad", "mean"] = "median_mad",
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-trial outlier masks matching the detection in ``zscore_activity_by_baseline``.

    Parameters
    ----------
    epochs_a, epochs_b:
        3-D epoch arrays ``(n_trials, n_features, n_times)``.
    time_axis_s:
        1-D time axis aligned with the last dimension of the epoch arrays.
    baseline_tmin_s, baseline_tmax_s:
        Baseline window boundaries (same values as passed to
        ``zscore_activity_by_baseline``).
    baseline_scope:
        Must match the value passed to ``zscore_activity_by_baseline``.
    remove_outlier_trial_means:
        Must match the value passed to ``zscore_activity_by_baseline``.

    Returns
    -------
    mask_a, mask_b : ndarray[bool]
        Shape ``(n_trials_a, n_features)`` and ``(n_trials_b, n_features)``.
        ``True`` = that trial's baseline mean was flagged as an outlier for
        that feature and was NaN'd before computing the zscore reference.
        All-False arrays are returned when ``remove_outlier_trial_means`` is
        ``False`` or ``baseline_scope`` is ``"trial"`` (which has no outlier
        removal step).
    """
    arr_a = np.asarray(epochs_a, dtype=np.float64)
    arr_b = np.asarray(epochs_b, dtype=np.float64)
    n_features = arr_a.shape[1] if arr_a.ndim == 3 else 0
    n_a = arr_a.shape[0] if arr_a.ndim == 3 else 0
    n_b = arr_b.shape[0] if arr_b.ndim == 3 else 0
    empty_a = np.zeros((n_a, n_features), dtype=bool)
    empty_b = np.zeros((n_b, n_features), dtype=bool)

    scope = str(baseline_scope).strip().lower()
    if not remove_outlier_trial_means or scope == "trial":
        return empty_a, empty_b
    if arr_a.ndim != 3 or arr_b.ndim != 3:
        return empty_a, empty_b

    time_axis = np.asarray(time_axis_s, dtype=np.float64).ravel()
    baseline_mask = (time_axis >= baseline_tmin_s) & (time_axis <= baseline_tmax_s)
    if not np.any(baseline_mask):
        return empty_a, empty_b

    if scope == "condition":
        mask_a = (
            _outlier_trial_means_mask(
                _trial_baseline_means(arr_a, baseline_mask), method=outlier_method
            )
            if n_a > 0
            else empty_a
        )
        mask_b = (
            _outlier_trial_means_mask(
                _trial_baseline_means(arr_b, baseline_mask), method=outlier_method
            )
            if n_b > 0
            else empty_b
        )
        return mask_a, mask_b

    if scope == "global":
        if n_a == 0 and n_b == 0:
            return empty_a, empty_b
        parts = [
            _trial_baseline_means(arr, baseline_mask)
            for arr, n in ((arr_a, n_a), (arr_b, n_b))
            if n > 0
        ]
        pooled = np.concatenate(parts, axis=0)
        pooled_mask = _outlier_trial_means_mask(pooled, method=outlier_method)
        return pooled_mask[:n_a], pooled_mask[n_a:]

    return empty_a, empty_b
