from __future__ import annotations

import numpy as np
import pytest

from gin_bids_py_analysis.processing.utils.group_stats import (
    compute_condition_group_stats,
    compute_one_sample_epoch_summary,
    compute_one_sample_timecourse,
)
from gin_bids_py_analysis.processing.utils.statistics import correct_p_values


def test_compute_one_sample_timecourse_shape() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(2.0, 0.5, size=(5, 10))
    t, p, mean, sem = compute_one_sample_timecourse(samples)
    assert t.shape == (10,)
    assert p.shape == (10,)
    assert mean.shape == (10,)
    assert sem.shape == (10,)


def test_compute_one_sample_timecourse_known_mean() -> None:
    samples = np.ones((4, 3)) * 3.0
    _, _, mean, _ = compute_one_sample_timecourse(samples)
    np.testing.assert_allclose(mean, [3.0, 3.0, 3.0])


def test_compute_one_sample_timecourse_single_sample_returns_nan_t() -> None:
    samples = np.array([[1.0, 2.0, 3.0]])
    t, p, _, _ = compute_one_sample_timecourse(samples)
    assert np.all(np.isnan(t))
    assert np.all(np.isnan(p))


def test_compute_one_sample_epoch_summary_shape() -> None:
    samples = np.random.default_rng(1).normal(0.0, 1.0, (6, 20))
    t, p, df, mean, sem = compute_one_sample_epoch_summary(samples)
    assert np.ndim(t) == 0 or t.size == 1
    assert np.ndim(p) == 0 or p.size == 1


def test_compute_condition_group_stats_shape() -> None:
    samples = np.ones((4, 8)) * 2.5
    mean, sem = compute_condition_group_stats(samples)
    assert mean.shape == (8,)
    assert sem.shape == (8,)
    np.testing.assert_allclose(mean, np.full(8, 2.5))
    np.testing.assert_allclose(sem, np.zeros(8))


def test_correct_p_values_none_returns_unchanged() -> None:
    p = np.array([0.01, 0.05, 0.10])
    result = correct_p_values(p, method="none")
    np.testing.assert_array_equal(result, p)


def test_correct_p_values_bonferroni() -> None:
    p = np.array([0.01, 0.05])
    result = correct_p_values(p, method="bonferroni")
    assert result[0] == pytest.approx(0.02)
    assert result[1] == pytest.approx(0.10)


def test_correct_p_values_fdr_bh_monotone() -> None:
    p = np.array([0.001, 0.01, 0.05, 0.20])
    result = correct_p_values(p, method="fdr_bh")
    assert np.all(result >= p)
    assert np.all(result <= 1.0)
