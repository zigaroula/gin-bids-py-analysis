from __future__ import annotations

import numpy as np

from gin_bids_py_analysis.processing.trial_stats_group.stats import (
    compute_one_sample_epoch_summary,
    compute_one_sample_timecourse,
    correct_p_values,
)


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
