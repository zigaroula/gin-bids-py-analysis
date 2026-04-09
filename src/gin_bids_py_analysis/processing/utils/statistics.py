"""Statistical helper utilities shared across processing pipelines.

Provides p-value correction and per-condition summary statistics (mean, SEM)
used by trial_stats and trial_slope_stats processors.
"""
from __future__ import annotations

from typing import Literal

from mne.stats import bonferroni_correction, fdr_correction
import numpy as np


def correct_p_values(
    p_values: np.ndarray,
    *,
    method: Literal["none", "fdr_bh", "bonferroni", "permutation"] = "fdr_bh",
) -> np.ndarray:
    """Apply multiple-comparisons correction to p-values.

    Parameters
    ----------
    p_values:
        Array of p-values of any shape. NaN values are ignored.
    method:
        Correction method. ``"none"`` returns a copy unchanged.
        ``"permutation"`` is treated as ``"none"`` here because permutation
        p-values are computed externally; the caller is responsible for
        passing the already-corrected values or handling this case separately.

    Returns
    -------
    corrected : float64 array, same shape as *p_values*.
    """
    corrected = np.asarray(p_values, dtype=np.float64).copy()
    finite_mask = np.isfinite(corrected)
    if not finite_mask.any() or method in ("none", "permutation"):
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
        "Valid methods are 'none', 'fdr_bh', 'bonferroni', 'permutation'."
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
            ),
            _zscore_activity_by_trial_mean_reference(
                arr_b,
                baseline_mask,
                remove_outlier_trial_means=remove_outlier_trial_means,
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
) -> np.ndarray:
    arr = np.asarray(epochs, dtype=np.float64)
    if arr.shape[0] == 0:
        return arr.copy()
    trial_means = _trial_baseline_means(arr, baseline_mask)
    reference_mean, reference_scale = _baseline_reference_from_trial_means(
        trial_means,
        remove_outlier_trial_means=remove_outlier_trial_means,
    )
    return _apply_feature_reference(arr, reference_mean, reference_scale)


def _trial_baseline_means(
    epochs: np.ndarray,
    baseline_mask: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(epochs, dtype=np.float64)
    if arr.shape[0] == 0:
        return np.empty((0, arr.shape[1]), dtype=np.float64)
    return np.nanmean(arr[:, :, baseline_mask], axis=2, dtype=np.float64)


def _baseline_reference_from_trial_means(
    trial_means: np.ndarray,
    *,
    remove_outlier_trial_means: bool,
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
        cleaned = _remove_outlier_trial_means(means)
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


def _remove_outlier_trial_means(trial_means: np.ndarray) -> np.ndarray:
    """Approximate MATLAB ``rmoutliers`` on per-feature trial means.

    The default MATLAB method is median/MAD-based. We mirror that here so the
    boolean option stays simple while remaining close to the intended behavior.
    """
    arr = np.asarray(trial_means, dtype=np.float64).copy()
    if arr.ndim != 2 or arr.shape[0] == 0:
        return arr

    for feature_idx in range(arr.shape[1]):
        values = arr[:, feature_idx]
        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask]
        if finite_values.size < 3:
            continue
        median = float(np.nanmedian(finite_values))
        mad = float(np.nanmedian(np.abs(finite_values - median)))
        if not np.isfinite(mad) or mad <= 0.0:
            continue
        scaled_mad = 1.4826 * mad
        threshold = 3.0 * scaled_mad
        outlier_mask = np.abs(finite_values - median) > threshold
        if np.any(outlier_mask):
            finite_indices = np.flatnonzero(finite_mask)
            arr[finite_indices[outlier_mask], feature_idx] = np.nan
    return arr
