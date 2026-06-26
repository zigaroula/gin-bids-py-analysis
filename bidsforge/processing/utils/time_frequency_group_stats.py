from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import label as label_connected_components
from scipy.stats import ttest_1samp, ttest_rel

from bidsforge.processing.utils.statistics import correct_p_values


def nansem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    count = np.sum(np.isfinite(arr), axis=axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        std = np.nanstd(arr, axis=axis, ddof=1, dtype=np.float64)
    sem = std / np.sqrt(np.maximum(count, 1))
    return np.where(count >= 2, sem, np.nan)


def compute_one_sample_tf_maps(
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One-sample stats for ``(n_samples, n_freqs, n_times)`` TF samples."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(
            f"samples must be 3-D (n_samples, n_freqs, n_times), got {arr.shape!r}."
        )
    n_freqs, n_times = int(arr.shape[1]), int(arr.shape[2])
    if arr.shape[0] == 0:
        empty = np.full((n_freqs, n_times), np.nan, dtype=np.float64)
        return empty.copy(), empty.copy(), empty.copy(), empty.copy()

    stats = ttest_1samp(arr, popmean=0.0, axis=0, nan_policy="omit")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(arr, axis=0, dtype=np.float64)
    return (
        np.asarray(stats.statistic, dtype=np.float64),
        np.asarray(stats.pvalue, dtype=np.float64),
        mean,
        nansem(arr, axis=0),
    )


def compute_condition_tf_stats(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean/SEM for ``(n_samples, n_freqs, n_times)`` TF samples."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(
            f"samples must be 3-D (n_samples, n_freqs, n_times), got {arr.shape!r}."
        )
    n_freqs, n_times = int(arr.shape[1]), int(arr.shape[2])
    if arr.shape[0] == 0:
        empty = np.full((n_freqs, n_times), np.nan, dtype=np.float64)
        return empty.copy(), empty.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(arr, axis=0, dtype=np.float64)
    return mean, nansem(arr, axis=0)


def compute_paired_tf_maps(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Paired t-test for two ``(n_samples, n_freqs, n_times)`` TF sample stacks."""
    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("samples_a and samples_b must be 3-D TF sample stacks.")
    if a.shape != b.shape:
        raise ValueError("Paired TF statistics require matching sample shapes.")
    if a.shape[0] < 2:
        empty = np.full(a.shape[1:], np.nan, dtype=np.float64)
        return empty.copy(), empty.copy()
    stats = ttest_rel(a, b, axis=0, nan_policy="omit")
    return np.asarray(stats.statistic, dtype=np.float64), np.asarray(stats.pvalue, dtype=np.float64)


def compute_tf_epoch_summary(samples: np.ndarray) -> tuple[float, float, float, float, float]:
    """Average each contribution over frequency/time, then test against zero."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[0] == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        values = np.nanmean(arr, axis=(1, 2), dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    mean = float(np.nanmean(finite))
    sem = float(nansem(finite, axis=0))
    if finite.size < 2:
        return np.nan, np.nan, np.nan, mean, sem
    stats = ttest_1samp(finite, popmean=0.0, nan_policy="omit")
    return (
        float(np.asarray(stats.statistic, dtype=np.float64)),
        float(np.asarray(stats.pvalue, dtype=np.float64)),
        float(finite.size - 1),
        mean,
        sem,
    )


def correct_tf_p_values(p_values: np.ndarray, *, method: str) -> np.ndarray:
    arr = np.asarray(p_values, dtype=np.float64)
    if method in {"none", "cluster_permutation"} or arr.size == 0:
        return arr.copy()
    return correct_p_values(arr.ravel(), method=method).reshape(arr.shape)


def label_tf_clusters(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 2-D TF clusters with 8-connectivity."""
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError(f"cluster mask must be 2-D, got {arr.shape!r}.")
    structure = np.ones((3, 3), dtype=np.int8)
    labels, n_labels = label_connected_components(arr, structure=structure)
    return labels.astype(np.int64), int(n_labels)


def signed_percentile_cluster_mask(
    *,
    source_map: np.ndarray,
    p_values: np.ndarray,
    null_distribution: np.ndarray,
    cluster_threshold_alpha: float,
    cluster_percentile_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a significant cluster mask and per-cluster sums.

    Clusters are detected from ``p_values <= cluster_threshold_alpha``.  Each
    observed cluster is retained when the sum of ``source_map`` inside it falls
    outside the signed percentile interval of ``null_distribution``.
    """
    source = np.asarray(source_map, dtype=np.float64)
    p = np.asarray(p_values, dtype=np.float64)
    if source.shape != p.shape:
        raise ValueError("source_map and p_values must have the same TF shape.")
    null = np.asarray(null_distribution, dtype=np.float64).ravel()
    if null.size == 0 or not np.any(np.isfinite(null)):
        raise ValueError("cluster_permutation requires a non-empty finite null distribution.")

    labels, n_labels = label_tf_clusters(np.isfinite(p) & (p <= cluster_threshold_alpha))
    mask = np.zeros_like(labels, dtype=bool)
    cluster_sums = np.full((n_labels,), np.nan, dtype=np.float64)
    lower = float(np.nanpercentile(null, cluster_percentile_alpha * 100.0))
    upper = float(np.nanpercentile(null, (1.0 - cluster_percentile_alpha) * 100.0))
    for idx in range(1, n_labels + 1):
        in_cluster = labels == idx
        cluster_sum = float(np.nansum(source[in_cluster]))
        cluster_sums[idx - 1] = cluster_sum
        if cluster_sum < lower or cluster_sum > upper:
            mask[in_cluster] = True
    return mask, labels, cluster_sums


def build_tf_cluster_null_from_permutations(
    permuted_samples: list[np.ndarray],
    *,
    cluster_threshold_alpha: float,
) -> np.ndarray:
    """Build signed max-cluster sums from per-contribution permutation maps.

    Each list item is shaped ``(n_perm, n_freqs, n_times)``.  For each
    permutation, samples are stacked across contributions, a one-sample TF map
    is computed, clusters are thresholded by the permutation p-map, and the
    largest absolute source-map cluster sum is stored with its sign.
    """
    if not permuted_samples:
        return np.zeros(0, dtype=np.float64)
    arrays = [np.asarray(item, dtype=np.float64) for item in permuted_samples]
    n_perm = min((arr.shape[0] for arr in arrays if arr.ndim == 3), default=0)
    if n_perm == 0:
        return np.zeros(0, dtype=np.float64)

    out = np.zeros((n_perm,), dtype=np.float64)
    for perm_idx in range(n_perm):
        samples = np.stack([arr[perm_idx] for arr in arrays], axis=0)
        _t, p, source_mean, _sem = compute_one_sample_tf_maps(samples)
        labels, n_labels = label_tf_clusters(np.isfinite(p) & (p <= cluster_threshold_alpha))
        best = 0.0
        for label_idx in range(1, n_labels + 1):
            cluster_sum = float(np.nansum(source_mean[labels == label_idx]))
            if abs(cluster_sum) > abs(best):
                best = cluster_sum
        out[perm_idx] = best
    return out
