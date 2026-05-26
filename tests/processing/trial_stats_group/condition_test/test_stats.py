from __future__ import annotations

import numpy as np

from gin_bids_py_analysis.processing.utils.group_stats import (
    compute_one_sample_epoch_summary,
    compute_one_sample_timecourse,
)
from gin_bids_py_analysis.processing.utils.cluster_permutation import (
    compute_cluster_null_distribution,
    compute_cluster_null_distribution_sign_flip,
    compute_mne_cluster_permutation,
    compute_cluster_permutation_pvalue,
    find_temporal_clusters,
)
from gin_bids_py_analysis.processing.utils.statistics import correct_p_values


def test_compute_one_sample_timecourse_detects_positive_effect() -> None:
    samples = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.5, 2.5, 3.5],
            [0.5, 1.5, 2.5],
            [1.2, 2.2, 3.2],
        ],
        dtype=np.float64,
    )

    t_values, p_values, mean_values, sem_values = compute_one_sample_timecourse(samples)

    assert t_values.shape == (3,)
    assert p_values.shape == (3,)
    assert mean_values.shape == (3,)
    assert sem_values.shape == (3,)
    assert np.all(mean_values > 0.0)
    assert np.all(t_values > 0.0)


def test_compute_one_sample_epoch_summary_returns_expected_fields() -> None:
    samples = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ],
        dtype=np.float64,
    )

    t_value, p_value, df, mean_value, sem_value = compute_one_sample_epoch_summary(samples)

    assert np.isfinite(t_value)
    assert np.isfinite(p_value)
    assert df == 2.0
    assert mean_value == 2.0
    assert np.isfinite(sem_value)


def test_correct_p_values_fdr_bh_bonferroni_and_none() -> None:
    raw = np.array([[0.001, 0.01, 0.02, 0.2, np.nan]], dtype=np.float64)

    fdr = correct_p_values(raw, method="fdr_bh")
    bonf = correct_p_values(raw, method="bonferroni")
    none = correct_p_values(raw, method="none")

    np.testing.assert_allclose(
        fdr[0, :4],
        np.array([0.004, 0.02, 0.02666666666666667, 0.2], dtype=np.float64),
    )
    np.testing.assert_allclose(
        bonf[0, :4],
        np.array([0.004, 0.04, 0.08, 0.8], dtype=np.float64),
    )
    np.testing.assert_allclose(none[0, :4], raw[0, :4])
    assert np.isnan(fdr[0, 4])


# --- cluster permutation tests ---


def test_find_temporal_clusters_known_mask() -> None:
    # mask: [F, T, T, F, T, F]  → two clusters: [1,2] and [4,4]
    h_mask = np.array([False, True, True, False, True, False])
    t_values = np.array([0.0, 3.0, 5.0, 0.0, 2.0, 0.0], dtype=np.float64)

    clusters = find_temporal_clusters(h_mask, t_values)

    assert len(clusters) == 2
    # Best cluster has |t_sum| = 8 (indices 1-2)
    best = clusters[0]
    assert best[0] == 1
    assert best[1] == 2
    assert np.isclose(best[2], 8.0)
    # Second cluster: index 4, t_sum = 2
    second = clusters[1]
    assert second[0] == 4
    assert second[1] == 4
    assert np.isclose(second[2], 2.0)


def test_find_temporal_clusters_empty_mask_returns_empty() -> None:
    h_mask = np.zeros(5, dtype=bool)
    t_values = np.ones(5, dtype=np.float64)

    assert find_temporal_clusters(h_mask, t_values) == []


def test_find_temporal_clusters_separates_positive_and_negative() -> None:
    # mask covers indices 1-3; t_values change sign at index 2.
    # Without sign-split these would be one cluster (t_sum = 4 - 3 - 2 = -1).
    # With sign-split: positive cluster at index 1 (t_sum=4), negative cluster
    # at indices 2-3 (t_sum=-5). Best is the negative cluster (|−5| > |4|).
    h_mask = np.array([False, True, True, True, False])
    t_values = np.array([0.0, 4.0, -3.0, -2.0, 0.0], dtype=np.float64)

    clusters = find_temporal_clusters(h_mask, t_values)

    assert len(clusters) == 2
    t_sums = sorted([c[2] for c in clusters], key=abs, reverse=True)
    assert np.isclose(t_sums[0], -5.0)
    assert np.isclose(t_sums[1], 4.0)
    # Best cluster (index 0) is the strongest by |t_sum|
    assert np.isclose(clusters[0][2], -5.0)


def test_compute_cluster_null_distribution_shape() -> None:
    rng = np.random.default_rng(99)
    n_perm_subj, n_times = 50, 20
    contributions = [
        rng.standard_normal((n_perm_subj, n_times)).astype(np.float32)
        for _ in range(4)
    ]

    null = compute_cluster_null_distribution(
        contributions,
        cluster_threshold_alpha=0.05,
        n_group_perm=200,
        rng=rng,
    )

    assert null.shape == (200,)
    assert null.dtype == np.float64
    assert np.all(null >= 0.0)


def test_compute_cluster_null_distribution_sign_flip_shape() -> None:
    rng = np.random.default_rng(42)
    n_times = 20
    contributions = [
        rng.standard_normal(n_times).astype(np.float64) * 5.0
        for _ in range(4)
    ]

    null = compute_cluster_null_distribution_sign_flip(
        contributions,
        cluster_threshold_alpha=0.05,
        n_group_perm=200,
        rng=rng,
    )

    assert null.shape == (200,)
    assert null.dtype == np.float64
    assert np.all(null >= 0.0)


def test_compute_cluster_null_distribution_sign_flip_empty_returns_zeros() -> None:
    rng = np.random.default_rng(0)
    null = compute_cluster_null_distribution_sign_flip(
        [],
        cluster_threshold_alpha=0.05,
        n_group_perm=100,
        rng=rng,
    )
    assert null.shape == (100,)
    assert np.all(null == 0.0)


def test_compute_cluster_permutation_pvalue_empty_null() -> None:
    assert compute_cluster_permutation_pvalue(5.0, np.zeros(0, dtype=np.float64)) == 1.0


def test_compute_cluster_permutation_pvalue_extreme() -> None:
    null = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    # observed > all null values → smallest possible non-zero p
    assert compute_cluster_permutation_pvalue(100.0, null) == 0.2
    # observed < all null values → p = 1
    assert compute_cluster_permutation_pvalue(0.5, null) == 1.0


def test_compute_mne_cluster_permutation_returns_expected_shapes() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=1.5, scale=0.5, size=(6, 20)).astype(np.float64)

    p_value, top_windows, top_p_values, null = compute_mne_cluster_permutation(
        samples,
        cluster_threshold_alpha=0.05,
        n_group_perm=50,
        seed=42,
    )

    assert 0.0 <= p_value <= 1.0
    assert isinstance(top_windows, list)
    assert isinstance(top_p_values, list)
    assert len(top_windows) == len(top_p_values)
    for w in top_windows:
        assert 0 <= w[0] <= w[1] < samples.shape[1]
    assert null.ndim == 1





