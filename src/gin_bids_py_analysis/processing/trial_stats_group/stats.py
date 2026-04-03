from __future__ import annotations

from typing import Literal

from mne.stats import (
    bonferroni_correction,
    fdr_correction,
    permutation_cluster_1samp_test,
)
import numpy as np
from scipy.ndimage import label as _ndimage_label
from scipy.stats import t as t_dist
from scipy.stats import ttest_1samp


def compute_one_sample_timecourse(
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute one-sample (vs 0) statistics for a ``[n_samples, n_times]`` matrix."""
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
    """Summarize one ROI on the full epoch by averaging each channel over time then testing vs 0."""
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


def correct_p_values(
    p_values: np.ndarray,
    *,
    method: Literal["none", "fdr_bh", "bonferroni", "cluster_permutation"] = "none",
) -> np.ndarray:
    """Apply a multiple-comparisons correction to p-values.

    The ``"cluster_permutation"`` method is handled in the processor; passing
    it here acts like ``"none"`` (the corrected p-values live in
    ``cluster_p_values``, not in ``p_values``).
    """
    corrected = np.asarray(p_values, dtype=np.float64).copy()
    finite_mask = np.isfinite(corrected)
    if not finite_mask.any() or method in ("none", "cluster_permutation"):
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

    raise ValueError(f"Unsupported p-value correction method: {method!r}. "
                     "Valid methods are 'none', 'fdr_bh', 'bonferroni', 'cluster_permutation'.")


def find_temporal_clusters(
    h_mask: np.ndarray,
    t_values: np.ndarray,
) -> list[tuple[int, int, float]]:
    """Find connected temporal clusters in a 1-D significance mask.

    Parameters
    ----------
    h_mask : bool array, shape ``(n_times,)``
        Significance mask (True = significant time point).
    t_values : float array, shape ``(n_times,)``
        Group-level t-values at each time point.

    Returns
    -------
    clusters : list of ``(start_idx, end_idx, cluster_t_sum)``
        Sorted by ``|cluster_t_sum|`` descending; indices are inclusive.
        Empty list when no significant time points exist.
    """
    mask = np.asarray(h_mask, dtype=bool)
    if not mask.any():
        return []

    t = np.asarray(t_values, dtype=np.float64)
    clusters: list[tuple[int, int, float]] = []
    for sign_mask in [mask & (t > 0), mask & (t < 0)]:
        if not sign_mask.any():
            continue
        labeled, n_clusters = _ndimage_label(sign_mask)
        for k in range(1, n_clusters + 1):
            indices = np.where(labeled == k)[0]
            t_sum = float(np.sum(t[indices]))
            clusters.append((int(indices[0]), int(indices[-1]), t_sum))

    clusters.sort(key=lambda c: abs(c[2]), reverse=True)
    return clusters


def compute_cluster_null_distribution(
    contributions_perm_t: list[np.ndarray],    *,
    cluster_threshold_alpha: float,
    n_group_perm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the null distribution of maximum cluster t-sums via random permutations.

    For each of ``n_group_perm`` iterations:
    1. For every contribution, draw one permuted t-map at random from its stored pool.
    2. Stack the sampled maps into ``(n_contributions, n_times)`` and run a
       one-sample t-test (vs 0) across contributions at each time point.
    3. Apply ``cluster_threshold_alpha`` to obtain a significance mask, detect
       temporal clusters, and record the maximum ``|cluster_t_sum|``.

    Parameters
    ----------
    contributions_perm_t : list of arrays, each shape ``(n_perm_subj, n_times)``
        Per-contribution permuted t-value pools (one per channel contributing to the ROI).
    cluster_threshold_alpha : float
        Threshold for calling a time-point significant within each null iteration.
    n_group_perm : int
        Number of group-level permutation iterations.
    rng : numpy Generator

    Returns
    -------
    null_distribution : float64 array, shape ``(n_group_perm,)``
        Maximum absolute cluster t-sum from each null iteration (0 when no cluster found).
    """
    if not contributions_perm_t:
        return np.zeros(n_group_perm, dtype=np.float64)

    n_contribs = len(contributions_perm_t)
    n_times = contributions_perm_t[0].shape[1]
    perm_arrays = [np.asarray(c, dtype=np.float64) for c in contributions_perm_t]
    n_perms_each = [int(a.shape[0]) for a in perm_arrays]
    null = np.zeros(n_group_perm, dtype=np.float64)
    sampled = np.empty((n_contribs, n_times), dtype=np.float64)

    for i in range(n_group_perm):
        for j, (arr, n_p) in enumerate(zip(perm_arrays, n_perms_each)):
            perm_idx = int(rng.integers(0, n_p))
            sampled[j] = arr[perm_idx]

        group_t, group_p = _one_sample_ttest(sampled)
        h_mask = np.isfinite(group_p) & (group_p < cluster_threshold_alpha)
        clusters = find_temporal_clusters(h_mask, group_t)
        if clusters:
            null[i] = abs(clusters[0][2])

    return null


def compute_cluster_null_distribution_sign_flip(
    contributions_observed: list[np.ndarray],
    *,
    cluster_threshold_alpha: float,
    n_group_perm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the null distribution via group-level sign-flipping (Maris & Oostenveld, 2007).

    For each of ``n_group_perm`` iterations:
    1. Randomly flip the sign of each contribution's observed timecourse.
    2. Stack into ``(n_contributions, n_times)`` and run a one-sample t-test vs 0.
    3. Apply ``cluster_threshold_alpha``, detect temporal clusters, and record the
       maximum ``|cluster_t_sum|``.

    Parameters
    ----------
    contributions_observed : list of arrays, each shape ``(n_times,)``
        Per-contribution observed timecourses (one per channel contributing to the ROI).
    cluster_threshold_alpha : float
        Threshold for calling a time-point significant within each null iteration.
    n_group_perm : int
        Number of group-level permutation iterations.
    rng : numpy Generator

    Returns
    -------
    null_distribution : float64 array, shape ``(n_group_perm,)``
        Maximum absolute cluster t-sum from each null iteration (0 when no cluster found).
    """
    if not contributions_observed:
        return np.zeros(n_group_perm, dtype=np.float64)

    n_contribs = len(contributions_observed)
    obs = np.stack([np.asarray(c, dtype=np.float64) for c in contributions_observed], axis=0)
    null = np.zeros(n_group_perm, dtype=np.float64)

    for i in range(n_group_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_contribs)
        flipped = signs[:, np.newaxis] * obs
        group_t, group_p = _one_sample_ttest(flipped)
        h_mask = np.isfinite(group_p) & (group_p < cluster_threshold_alpha)
        clusters = find_temporal_clusters(h_mask, group_t)
        if clusters:
            null[i] = abs(clusters[0][2])

    return null


def compute_cluster_permutation_pvalue(
    observed_cluster_tsum: float,
    null_distribution: np.ndarray,
) -> float:
    """Compute a p-value for the observed cluster t-sum against its null distribution.

    Parameters
    ----------
    observed_cluster_tsum : float
        Sum of t-values over the best observed temporal cluster.
    null_distribution : shape ``(n_group_perm,)``
        Null distribution of maximum absolute cluster t-sums.

    Returns
    -------
    p_value : float
        Conservative permutation p-value ``(count + 1) / (n_perm + 1)`` where
        ``count`` is the number of null values ``>= |observed_cluster_tsum|``.
        Returns ``1.0`` when ``null_distribution`` is empty.
    """
    null = np.asarray(null_distribution, dtype=np.float64).ravel()
    if null.size == 0:
        return 1.0
    count = int(np.sum(null >= abs(float(observed_cluster_tsum))))
    return float((count + 1) / (null.size + 1))


def compute_mne_cluster_permutation(
    samples_observed: np.ndarray,
    *,
    cluster_threshold_alpha: float,
    n_group_perm: int,
    seed: int | None = None,
) -> tuple[float, tuple[int, int] | None, np.ndarray]:
    """Run MNE one-sample temporal cluster permutation and return best-cluster stats."""
    samples = np.asarray(samples_observed, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(
            f"samples_observed must be 2-D [n_samples, n_times], got {samples.shape!r}."
        )
    n_samples, n_times = samples.shape
    if n_samples < 2 or n_times == 0:
        return 1.0, None, np.zeros(0, dtype=np.float64)

    threshold = float(t_dist.ppf(1.0 - (cluster_threshold_alpha / 2.0), df=n_samples - 1))
    if not np.isfinite(threshold) or threshold <= 0:
        threshold = None

    t_obs, clusters, cluster_p_values, h0 = permutation_cluster_1samp_test(
        samples,
        n_permutations=n_group_perm,
        threshold=threshold,
        tail=0,
        adjacency=None,
        out_type="mask",
        seed=seed,
        verbose=False,
    )
    t_values = np.asarray(t_obs, dtype=np.float64).ravel()
    null_distribution = np.asarray(h0, dtype=np.float64).ravel()
    if not clusters:
        return 1.0, None, null_distribution

    cluster_p = np.asarray(cluster_p_values, dtype=np.float64).ravel()
    best_idx = -1
    best_abs_tsum = -np.inf
    for idx, cluster_mask in enumerate(clusters):
        mask = np.asarray(cluster_mask, dtype=bool).ravel()
        if mask.size != n_times or not mask.any():
            continue
        tsum = float(np.sum(t_values[mask]))
        abs_tsum = abs(tsum)
        if abs_tsum > best_abs_tsum:
            best_abs_tsum = abs_tsum
            best_idx = idx

    if best_idx < 0:
        return 1.0, None, null_distribution

    best_mask = np.asarray(clusters[best_idx], dtype=bool).ravel()
    indices = np.where(best_mask)[0]
    if indices.size == 0:
        return 1.0, None, null_distribution
    best_window = (int(indices[0]), int(indices[-1]))
    p_value = float(cluster_p[best_idx]) if best_idx < cluster_p.size else 1.0
    return p_value, best_window, null_distribution


def _one_sample_ttest(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-sample t-test (vs 0) for shape ``(n_samples, n_times)`` — no NaN handling."""
    n = samples.shape[0]
    n_times = samples.shape[1]
    if n < 2:
        empty = np.full(n_times, np.nan, dtype=np.float64)
        return empty.copy(), empty.copy()
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(std > 0, mean / (std / np.sqrt(n)), np.nan)
    p = np.where(np.isfinite(t), 2.0 * t_dist.sf(np.abs(t), df=n - 1), np.nan)
    return t, p

def _nansem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return NaN-aware SEM, with NaN when fewer than 2 finite values are available."""
    arr = np.asarray(values, dtype=np.float64)
    count = np.sum(np.isfinite(arr), axis=axis)
    std = np.nanstd(arr, axis=axis, ddof=1, dtype=np.float64)
    sem = std / np.sqrt(np.maximum(count, 1))
    return np.where(count >= 2, sem, np.nan)
