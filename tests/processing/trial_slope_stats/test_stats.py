from __future__ import annotations

import numpy as np

from gin_bids_py_analysis.processing.utils.statistics import correct_p_values
from gin_bids_py_analysis.processing.trial_slope_stats.stats import (
    compute_linear_regression_maps,
)


def test_compute_linear_regression_maps_known_solution() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    y = (2.0 * x[:, None, None]) + 1.0
    y = np.broadcast_to(y, (4, 2, 3)).astype(np.float64)

    slope, intercept, r_value, p_value, valid = compute_linear_regression_maps(
        x,
        y,
        n_features=2,
        n_times=3,
    )

    assert valid is True
    np.testing.assert_allclose(slope, np.full((2, 3), 2.0), atol=1e-12)
    np.testing.assert_allclose(intercept, np.full((2, 3), 1.0), atol=1e-12)
    np.testing.assert_allclose(r_value, np.ones((2, 3)), atol=1e-12)
    np.testing.assert_allclose(p_value, np.zeros((2, 3)), atol=1e-12)


def test_compute_linear_regression_maps_invalid_when_predictor_variance_zero() -> None:
    x = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    y = np.ones((4, 2, 3), dtype=np.float64)

    slope, intercept, r_value, p_value, valid = compute_linear_regression_maps(
        x,
        y,
        n_features=2,
        n_times=3,
    )

    assert valid is False
    assert np.isnan(slope).all()
    assert np.isnan(intercept).all()
    assert np.isnan(r_value).all()
    assert np.isnan(p_value).all()


def test_correct_p_values_fdr_and_bonferroni() -> None:
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
