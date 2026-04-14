"""Cluster-based permutation testing utilities shared across group-level pipelines.

Provides temporal-cluster detection and null-distribution building functions used
by any group-level analysis that supports ``p_value_correction_method='cluster_permutation'``.
"""
from __future__ import annotations

import numpy as np
from mne.stats import permutation_cluster_1samp_test
from scipy.ndimage import label as _ndimage_label
from scipy.stats import t as t_dist


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
    contributions_perm: list[np.ndarray],
    *,
    cluster_threshold_alpha: float,
    n_group_perm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the null distribution of maximum cluster t-sums via random permutations.

    For each of ``n_group_perm`` iterations:

    1. For every contribution, draw one permuted map at random from its stored pool.
    2. Stack the sampled maps into ``(n_contributions, n_times)`` and run a
       one-sample t-test (vs 0) across contributions at each time point.
    3. Apply ``cluster_threshold_alpha`` to obtain a significance mask, detect
       temporal clusters, and record the maximum ``|cluster_t_sum|``.

    Parameters
    ----------
    contributions_perm : list of arrays, each shape ``(n_perm_subj, n_times)``
        Per-contribution permuted value pools (one array per channel contributing
        to the ROI).
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
    if not contributions_perm:
        return np.zeros(n_group_perm, dtype=np.float64)

    n_contribs = len(contributions_perm)
    n_times = contributions_perm[0].shape[1]
    perm_arrays = [np.asarray(c, dtype=np.float64) for c in contributions_perm]
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


def compute_cluster_null_distribution_paired(
    contributions_perm_a: list[np.ndarray],
    contributions_perm_b: list[np.ndarray],
    *,
    cluster_threshold_alpha: float,
    n_group_perm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the null distribution for a paired two-condition contrast via random permutations.

    Equivalent to :func:`compute_cluster_null_distribution` but for paired designs where
    each contribution provides two permuted pools (one per condition).  For each null
    iteration, an independent random permutation index is drawn for condition A and
    condition B, the difference ``perm_a[k_a] - perm_b[k_b]`` is computed per
    contribution, and a one-sample t-test is run on the differences.

    Parameters
    ----------
    contributions_perm_a : list of arrays, each shape ``(n_perm_subj, n_times)``
        Per-contribution permuted slope pools for condition A.
    contributions_perm_b : list of arrays, each shape ``(n_perm_subj, n_times)``
        Per-contribution permuted slope pools for condition B.  Must have the same
        length as *contributions_perm_a*.
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
    if not contributions_perm_a or not contributions_perm_b:
        return np.zeros(n_group_perm, dtype=np.float64)
    if len(contributions_perm_a) != len(contributions_perm_b):
        raise ValueError(
            "contributions_perm_a and contributions_perm_b must have the same length."
        )

    n_contribs = len(contributions_perm_a)
    n_times = contributions_perm_a[0].shape[1]
    perm_a = [np.asarray(c, dtype=np.float64) for c in contributions_perm_a]
    perm_b = [np.asarray(c, dtype=np.float64) for c in contributions_perm_b]
    n_perms_a = [int(a.shape[0]) for a in perm_a]
    n_perms_b = [int(b.shape[0]) for b in perm_b]
    null = np.zeros(n_group_perm, dtype=np.float64)
    sampled_diff = np.empty((n_contribs, n_times), dtype=np.float64)

    for i in range(n_group_perm):
        for j in range(n_contribs):
            k_a = int(rng.integers(0, n_perms_a[j]))
            k_b = int(rng.integers(0, n_perms_b[j]))
            sampled_diff[j] = perm_a[j][k_a] - perm_b[j][k_b]

        group_t, group_p = _one_sample_ttest(sampled_diff)
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
    """Run MNE one-sample temporal cluster permutation and return best-cluster stats.

    Parameters
    ----------
    samples_observed : float64 array, shape ``(n_samples, n_times)``
        Per-contribution observed timecourses to test against zero.
    cluster_threshold_alpha : float
        Two-tailed alpha used to derive the t-threshold for cluster detection.
    n_group_perm : int
        Number of permutation iterations passed to MNE.
    seed : int or None
        Random seed forwarded to MNE.

    Returns
    -------
    p_value : float
        P-value of the best (strongest) observed cluster.
    best_window : tuple ``(start_idx, end_idx)`` or None
        Sample indices of the best cluster window, or None when no cluster is found.
    null_distribution : float64 array, shape ``(n_group_perm,)``
        Maximum cluster statistic from MNE's permutation distribution (``h0``).
    """
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
