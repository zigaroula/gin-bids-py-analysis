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
) -> tuple[np.ndarray, np.ndarray]:
    """Z-score pooled condition activity relative to a baseline window.

    The baseline reference is computed per feature by pooling all baseline
    samples from both conditions across the trial and time dimensions. The same
    feature-wise affine transform is then applied to the full epoch. This makes
    the transform independent of the post-event effect of interest and keeps
    condition-comparison statistics invariant.

    When the pooled baseline standard deviation is non-finite or zero for a
    feature, the helper falls back to baseline-centering only for that feature
    (division by 1 instead of 0) to avoid infs while preserving deviations from
    baseline.
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

    pooled_parts = [arr[:, :, baseline_mask] for arr in (arr_a, arr_b) if arr.shape[0] > 0]
    if not pooled_parts:
        return arr_a.copy(), arr_b.copy()

    pooled_baseline = np.concatenate(pooled_parts, axis=0)
    mean = np.nanmean(pooled_baseline, axis=(0, 2), dtype=np.float64)
    std = np.nanstd(pooled_baseline, axis=(0, 2), ddof=1, dtype=np.float64)
    scale = np.where(np.isfinite(std) & (std > 0.0), std, 1.0)

    def _scale(arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] == 0:
            return arr.copy()
        centered = arr - mean[np.newaxis, :, np.newaxis]
        out = centered / scale[np.newaxis, :, np.newaxis]
        out[~np.isfinite(arr)] = np.nan
        return out

    return _scale(arr_a), _scale(arr_b)
