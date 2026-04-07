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
