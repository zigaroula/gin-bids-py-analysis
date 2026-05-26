from __future__ import annotations

import numpy as np

from gin_bids_py_analysis.processing.trial_stats.regression.stats import (
    compute_linear_regression_maps,
    compute_permuted_regression_maps,
)
from gin_bids_py_analysis.processing.utils.statistics import correct_p_values


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


def test_compute_linear_regression_maps_nan_aware_matches_dense() -> None:
    """NaN-aware path and dense path agree when there are no NaN values."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(10)
    y = rng.standard_normal((10, 3, 5))

    slope_d, intercept_d, r_d, p_d, valid_d = compute_linear_regression_maps(
        x, y, n_features=3, n_times=5
    )
    # Force the NaN-aware path by injecting a NaN into a trial that is then
    # immediately finite for every real channel — add a dummy 4th channel whose
    # single trial will generate the NaN branch but still yield the same stats
    # for the other channels.
    y_with_nan = np.concatenate(
        [y, np.full((10, 1, 5), np.nan)], axis=1
    )  # last channel all NaN → triggers NaN-aware path
    x_padded = x

    slope_n, intercept_n, r_n, p_n, valid_n = compute_linear_regression_maps(
        x_padded, y_with_nan, n_features=4, n_times=5
    )

    assert valid_n is True
    np.testing.assert_allclose(slope_n[:3], slope_d, atol=1e-12)
    np.testing.assert_allclose(intercept_n[:3], intercept_d, atol=1e-12)
    np.testing.assert_allclose(r_n[:3], r_d, atol=1e-12)
    assert np.all(np.isnan(slope_n[3])), "All-NaN channel should yield NaN slope."


def test_compute_linear_regression_maps_nan_per_channel_independent() -> None:
    """Channels with different valid-trial subsets produce independent regression results."""
    # Channel 0: trials 0–5 valid; channel 1: trials 0–3 valid (2 NaN-masked).
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    # Channel 0: y = 2x + 1 over trials 0–5.
    y = np.empty((6, 2, 1), dtype=np.float64)
    y[:, 0, 0] = 2.0 * x + 1.0
    # Channel 1: y = -x + 5 over trials 0–3; trials 4–5 NaN.
    y[:4, 1, 0] = -x[:4] + 5.0
    y[4:, 1, 0] = np.nan

    slope, intercept, r, p, valid = compute_linear_regression_maps(
        x, y, n_features=2, n_times=1
    )

    assert valid is True
    np.testing.assert_allclose(slope[0, 0], 2.0, atol=1e-10)
    np.testing.assert_allclose(intercept[0, 0], 1.0, atol=1e-10)
    np.testing.assert_allclose(slope[1, 0], -1.0, atol=1e-10)
    np.testing.assert_allclose(intercept[1, 0], 5.0, atol=1e-10)


def test_compute_linear_regression_maps_nan_aware_returns_false_when_all_channels_invalid() -> None:
    """stats_valid should be False when every channel has fewer than 3 valid trials."""
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    y = np.ones((4, 2, 3), dtype=np.float64)
    # NaN out all but 2 trials for both channels → < 3 valid trials each.
    y[2:, :, :] = np.nan

    slope, intercept, r, p, valid = compute_linear_regression_maps(
        x, y, n_features=2, n_times=3
    )

    assert valid is False
    assert np.all(np.isnan(slope))


# ---------------------------------------------------------------------------
# compute_permuted_regression_maps
# ---------------------------------------------------------------------------

def test_compute_permuted_regression_maps_shape_and_dtype() -> None:
    rng = np.random.default_rng(0)
    n_trials = 8
    n_features = 3
    n_times = 5
    n_perm = 10
    predictor = rng.standard_normal(n_trials)
    epochs = rng.standard_normal((n_trials, n_features, n_times)).astype(np.float32)

    result = compute_permuted_regression_maps(
        predictor, epochs,
        n_perm=n_perm, rng=rng, n_features=n_features, n_times=n_times,
    )

    assert result.shape == (n_perm, n_features, n_times)
    assert result.dtype == np.float32


def test_compute_permuted_regression_maps_zero_perm_returns_empty() -> None:
    rng = np.random.default_rng(1)
    predictor = rng.standard_normal(7)
    epochs = rng.standard_normal((7, 2, 4)).astype(np.float32)

    result = compute_permuted_regression_maps(
        predictor, epochs,
        n_perm=0, rng=rng, n_features=2, n_times=4,
    )

    assert result.shape == (0, 2, 4)
    assert result.dtype == np.float32


def test_compute_permuted_regression_maps_reproducible() -> None:
    n_trials = 12
    n_features = 2
    n_times = 6
    n_perm = 20
    predictor = np.random.default_rng(7).standard_normal(n_trials)
    epochs = np.random.default_rng(8).standard_normal((n_trials, n_features, n_times)).astype(np.float32)

    result_a = compute_permuted_regression_maps(
        predictor, epochs,
        n_perm=n_perm, rng=np.random.default_rng(42), n_features=n_features, n_times=n_times,
    )
    result_b = compute_permuted_regression_maps(
        predictor, epochs,
        n_perm=n_perm, rng=np.random.default_rng(42), n_features=n_features, n_times=n_times,
    )

    np.testing.assert_array_equal(result_a, result_b)


def test_compute_permuted_regression_maps_nan_for_too_few_trials() -> None:
    rng = np.random.default_rng(3)
    predictor = rng.standard_normal(3)
    epochs = rng.standard_normal((3, 2, 4)).astype(np.float32)
    # Set all but 2 trials to NaN so valid count < 3 for all channels.
    epochs[2:, :, :] = np.nan

    result = compute_permuted_regression_maps(
        predictor, epochs,
        n_perm=5, rng=rng, n_features=2, n_times=4,
    )

    assert result.shape == (5, 2, 4)
    assert np.all(np.isnan(result))





