from __future__ import annotations

import numpy as np
import pytest

from gin_bids_py_analysis.processing.utils.cluster_permutation import (
    compute_cluster_null_distribution,
    compute_cluster_null_distribution_paired,
    compute_cluster_permutation_pvalue,
    find_temporal_clusters,
)


# ---------------------------------------------------------------------------
# find_temporal_clusters
# ---------------------------------------------------------------------------

def test_find_temporal_clusters_returns_empty_for_no_above_threshold() -> None:
    h_mask = np.array([False, False, False])
    t_values = np.array([0.5, 0.3, 0.2])
    clusters = find_temporal_clusters(h_mask, t_values)
    assert clusters == []


def test_find_temporal_clusters_detects_single_cluster() -> None:
    h_mask = np.array([False, True, True, True, False])
    t_values = np.array([0.1, 2.0, 3.0, 1.5, 0.1])
    clusters = find_temporal_clusters(h_mask, t_values)
    assert len(clusters) == 1
    start, end, tsum = clusters[0]
    assert start == 1
    assert end == 3
    np.testing.assert_allclose(tsum, 2.0 + 3.0 + 1.5)


def test_find_temporal_clusters_sorts_by_abs_tsum_descending() -> None:
    h_mask = np.array([True, True, False, True, False])
    t_values = np.array([5.0, 4.0, 0.0, 1.0, 0.0])
    # cluster 0: [0,1] tsum=9, cluster 1: [3,3] tsum=1
    clusters = find_temporal_clusters(h_mask, t_values)
    assert len(clusters) == 2
    assert clusters[0][2] > clusters[1][2]


# ---------------------------------------------------------------------------
# compute_cluster_null_distribution
# ---------------------------------------------------------------------------

def test_compute_cluster_null_distribution_shape() -> None:
    rng = np.random.default_rng(0)
    n_perm = 5
    n_times = 10
    contributions_perm = [
        rng.standard_normal((n_perm, n_times)).astype(np.float64)
        for _ in range(3)
    ]
    null = compute_cluster_null_distribution(
        contributions_perm,
        cluster_threshold_alpha=0.05,
        n_group_perm=50,
        rng=rng,
    )
    assert null.shape == (50,)
    assert null.dtype == np.float64


def test_compute_cluster_null_distribution_reproducible() -> None:
    n_perm = 20
    n_times = 8
    contributions_perm = [
        np.random.default_rng(i).standard_normal((n_perm, n_times))
        for i in range(4)
    ]
    null_a = compute_cluster_null_distribution(
        contributions_perm,
        cluster_threshold_alpha=0.05,
        n_group_perm=100,
        rng=np.random.default_rng(42),
    )
    null_b = compute_cluster_null_distribution(
        contributions_perm,
        cluster_threshold_alpha=0.05,
        n_group_perm=100,
        rng=np.random.default_rng(42),
    )
    np.testing.assert_array_equal(null_a, null_b)


# ---------------------------------------------------------------------------
# compute_cluster_null_distribution_paired  (new function)
# ---------------------------------------------------------------------------

def test_compute_cluster_null_distribution_paired_shape() -> None:
    rng = np.random.default_rng(7)
    n_perm = 10
    n_times = 6
    perm_a = [rng.standard_normal((n_perm, n_times)) for _ in range(3)]
    perm_b = [rng.standard_normal((n_perm, n_times)) for _ in range(3)]
    null = compute_cluster_null_distribution_paired(
        perm_a,
        perm_b,
        cluster_threshold_alpha=0.05,
        n_group_perm=30,
        rng=rng,
    )
    assert null.shape == (30,)
    assert null.dtype == np.float64


def test_compute_cluster_null_distribution_paired_reproducible() -> None:
    n_perm = 15
    n_times = 8
    perm_a = [
        np.random.default_rng(i).standard_normal((n_perm, n_times))
        for i in range(4)
    ]
    perm_b = [
        np.random.default_rng(i + 10).standard_normal((n_perm, n_times))
        for i in range(4)
    ]
    null_a = compute_cluster_null_distribution_paired(
        perm_a, perm_b,
        cluster_threshold_alpha=0.05,
        n_group_perm=50,
        rng=np.random.default_rng(1),
    )
    null_b = compute_cluster_null_distribution_paired(
        perm_a, perm_b,
        cluster_threshold_alpha=0.05,
        n_group_perm=50,
        rng=np.random.default_rng(1),
    )
    np.testing.assert_array_equal(null_a, null_b)


def test_compute_cluster_null_distribution_paired_differs_from_unpaired() -> None:
    """Paired and unpaired null distributions drawn from the same RNG seed differ."""
    rng_seed = 99
    n_perm = 20
    n_times = 10
    perm_a = [
        np.random.default_rng(i).standard_normal((n_perm, n_times))
        for i in range(5)
    ]
    perm_b = [
        np.random.default_rng(i + 20).standard_normal((n_perm, n_times))
        for i in range(5)
    ]
    null_paired = compute_cluster_null_distribution_paired(
        perm_a, perm_b,
        cluster_threshold_alpha=0.05,
        n_group_perm=100,
        rng=np.random.default_rng(rng_seed),
    )
    null_unpaired = compute_cluster_null_distribution(
        perm_a,
        cluster_threshold_alpha=0.05,
        n_group_perm=100,
        rng=np.random.default_rng(rng_seed),
    )
    assert not np.array_equal(null_paired, null_unpaired)


# ---------------------------------------------------------------------------
# compute_cluster_permutation_pvalue
# ---------------------------------------------------------------------------

def test_compute_cluster_permutation_pvalue_conservative_formula() -> None:
    # With observed_tsum = 5 and all null values < 5, p = (0+1)/(100+1) = 1/101
    null = np.zeros(100, dtype=np.float64)
    p = compute_cluster_permutation_pvalue(5.0, null)
    np.testing.assert_allclose(p, 1.0 / 101.0)


def test_compute_cluster_permutation_pvalue_max_is_one_when_all_null_exceed() -> None:
    null = np.full(100, 10.0, dtype=np.float64)
    p = compute_cluster_permutation_pvalue(1.0, null)
    np.testing.assert_allclose(p, 1.0)



