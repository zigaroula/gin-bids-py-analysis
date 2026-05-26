from __future__ import annotations

import numpy as np
import pytest

from gin_bids_py_analysis.processing.utils.epoch_quality import (
    apply_channel_exclusions,
    apply_trial_nan_mask,
    detect_outlier_trial_channel_pairs_by_max,
    detect_outlier_trial_channel_pairs_by_mean,
    reject_channels_by_nan_trial_ratio,
    reject_channels_by_trial_max_spread,
    reject_channels_by_trial_mean_spread,
)


# ---------------------------------------------------------------------------
# detect_outlier_trial_channel_pairs_by_mean
# ---------------------------------------------------------------------------


def test_detect_outlier_by_mean_flags_clear_outlier() -> None:
    """A single trial with an epoch mean far from the others should be flagged."""
    # 21 trials, 2 channels, 4 time points.
    # Channel 0: 20 trials with epoch mean = 0.0 and 1 clear outlier with mean = 30.0.
    # With 20 baseline trials the outlier exceeds mean + 3σ.
    # Channel 1: uniform zeros → std = 0 → no outlier possible.
    epochs = np.zeros((21, 2, 4), dtype=np.float64)
    epochs[20, 0, :] = 30.0  # trial 20, channel 0: clear outlier

    mask = detect_outlier_trial_channel_pairs_by_mean(epochs, threshold_factor=3.0)

    assert mask.shape == (21, 2)
    assert mask[20, 0], "Trial 20, channel 0 should be flagged."
    assert not np.any(mask[:20, 0]), "Trials 0–19 on channel 0 should not be flagged."
    assert not np.any(mask[:, 1]), "Channel 1 (uniform zeros) should have no outlier trials."


def test_detect_outlier_by_mean_no_flags_when_uniform() -> None:
    """Identical trial means per channel should never produce outliers."""
    epochs = np.ones((6, 3, 5), dtype=np.float64)
    mask = detect_outlier_trial_channel_pairs_by_mean(epochs, threshold_factor=3.0)
    assert not np.any(mask)


def test_detect_outlier_by_mean_shape() -> None:
    epochs = np.zeros((8, 4, 10), dtype=np.float64)
    mask = detect_outlier_trial_channel_pairs_by_mean(epochs)
    assert mask.shape == (8, 4)
    assert mask.dtype == bool


# ---------------------------------------------------------------------------
# detect_outlier_trial_channel_pairs_by_max
# ---------------------------------------------------------------------------


def test_detect_outlier_by_max_flags_clear_outlier() -> None:
    """A single trial with a very high maximum should be flagged."""
    # 20 trials, 2 channels, 8 time points.
    # Channel 1: 19 trials with small constant value (max = 0.1) and trial 2
    # injected with a large maximum (20.0).  With 19 baseline trials the
    # outlier trial's max exceeds mean + 3σ of the per-trial max distribution.
    epochs = np.full((20, 2, 8), 0.1, dtype=np.float64)
    epochs[2, 1, 5] = 20.0  # inject a clear maximum outlier in trial 2, channel 1

    mask = detect_outlier_trial_channel_pairs_by_max(epochs, threshold_factor=3.0)

    assert mask.shape == (20, 2)
    assert mask[2, 1], "Trial 2, channel 1 should be flagged."
    assert not np.any(mask[:, 0]), "Channel 0 should have no outlier trials."


def test_detect_outlier_by_max_no_flags_when_uniform() -> None:
    epochs = np.ones((5, 3, 6), dtype=np.float64)
    mask = detect_outlier_trial_channel_pairs_by_max(epochs)
    assert not np.any(mask)


def test_detect_outlier_by_max_ignores_pre_nanmasked_trials() -> None:
    """A pre-NaN'd trial must not inflate the max distribution.

    Scenario mirrors the real bug: a large-spike trial in the pool inflates std
    enough to hide a moderate outlier.  When that trial has already been NaN'd
    (via apply_trial_nan_mask), the remaining moderate outlier must be detected.
    """
    # 21 trials, 1 channel, 4 time points.
    # Trials 0-19 (normal): constant 1.0  → per-trial max = 1.0.
    # Trial 15: pre-NaN'd (spike already removed by a prior mean pass).
    # Trial 3: one sample = 6.0  → per-trial max = 6.0 (moderate outlier).
    #
    # With trial 15 NaN'd, valid maxes are {1.0}×19 + {6.0}:
    #   nanmean ≈ 1.25, nanstd ≈ 1.12 → z_trial3 ≈ 4.25 > 3.0  → detected ✓
    # Without NaN-ing trial 15 first (spike max >> 50), the std would be huge
    # and trial 3 would fall below threshold — that was the pre-fix behaviour.
    epochs = np.ones((21, 1, 4), dtype=np.float64)
    epochs[3, 0, 2] = 6.0    # moderate max outlier
    epochs[15, 0, :] = np.nan  # spike trial already NaN'd

    mask = detect_outlier_trial_channel_pairs_by_max(epochs, threshold_factor=3.0)

    assert mask[3, 0], "Trial 3 (moderate outlier) should be detected after NaN'd spike."
    assert not mask[15, 0], "Already-NaN'd trial 15 should not be re-flagged."
    assert not np.any(mask[[i for i in range(21) if i not in (3, 15)], 0])


def test_detect_outlier_by_mean_ignores_pre_nanmasked_trials() -> None:
    """Same two-pass NaN-awareness test for the mean-based detector."""
    # 21 trials, 1 channel, 4 time points.
    # Trial 3: all samples = 6.0  → per-trial mean = 6.0 (moderate outlier).
    # Trial 15: pre-NaN'd → nanmean returns NaN, excluded from distribution.
    #
    # With trial 15 NaN'd, valid means are {1.0}×19 + {6.0}:
    #   nanmean ≈ 1.25, nanstd ≈ 1.12 → z_trial3 ≈ 4.25 > 3.0  → detected ✓
    epochs = np.ones((21, 1, 4), dtype=np.float64)
    epochs[3, 0, :] = 6.0    # moderate mean outlier
    epochs[15, 0, :] = np.nan  # spike trial already NaN'd

    mask = detect_outlier_trial_channel_pairs_by_mean(epochs, threshold_factor=3.0)

    assert mask[3, 0], "Trial 3 (moderate mean outlier) should be detected after NaN'd spike."
    assert not mask[15, 0], "Already-NaN'd trial 15 should not be re-flagged."
    assert not np.any(mask[[i for i in range(21) if i not in (3, 15)], 0])


# ---------------------------------------------------------------------------
# apply_trial_nan_mask
# ---------------------------------------------------------------------------


def test_apply_trial_nan_mask_fills_entire_time_slice() -> None:
    """NaN should fill the full time axis for every flagged (trial, channel) pair."""
    epochs = np.ones((4, 3, 5), dtype=np.float32)
    nan_mask = np.zeros((4, 3), dtype=bool)
    nan_mask[1, 2] = True  # only trial 1, channel 2

    result = apply_trial_nan_mask(epochs, nan_mask)

    assert result.dtype == np.float64
    assert np.all(np.isnan(result[1, 2, :])), "All time points of trial 1 / channel 2 should be NaN."
    assert np.all(np.isfinite(result[0, :, :])), "Untouched trials should remain finite."
    assert np.all(np.isfinite(result[:, :2, :])), "Untouched channels should remain finite."


def test_apply_trial_nan_mask_returns_copy() -> None:
    epochs = np.ones((3, 2, 4), dtype=np.float64)
    nan_mask = np.array([[True, False], [False, False], [False, True]])
    result = apply_trial_nan_mask(epochs, nan_mask)
    # Original should be unchanged.
    assert np.all(epochs == 1.0)
    assert np.isnan(result[0, 0, 0])
    assert np.isnan(result[2, 1, 0])


# ---------------------------------------------------------------------------
# reject_channels_by_trial_mean_spread
# ---------------------------------------------------------------------------


def test_reject_channels_by_trial_mean_spread_flags_noisy_channel() -> None:
    """A channel with much higher across-trial variability should be flagged."""
    rng = np.random.default_rng(2)
    # 3 channels, 20 trials, 10 time points.
    epochs = np.zeros((20, 3, 10), dtype=np.float64)
    # Channel 0 and 1: small, stable trial means (low spread).
    epochs[:, 0, :] = rng.normal(0, 0.1, size=(20, 10))
    epochs[:, 1, :] = rng.normal(0, 0.1, size=(20, 10))
    # Channel 2: large variability across trials (high spread).
    epochs[:, 2, :] = rng.normal(0, 50.0, size=(20, 10))

    mask = reject_channels_by_trial_mean_spread(epochs, threshold_factor=1.0)

    assert mask.shape == (3,)
    assert mask[2], "Channel 2 (high spread) should be flagged."


def test_reject_channels_by_trial_mean_spread_no_flags_when_uniform() -> None:
    epochs = np.ones((10, 4, 6), dtype=np.float64)
    mask = reject_channels_by_trial_mean_spread(epochs)
    assert not np.any(mask)


# ---------------------------------------------------------------------------
# reject_channels_by_nan_trial_ratio
# ---------------------------------------------------------------------------


def test_reject_channels_by_nan_trial_ratio_threshold() -> None:
    """Channel 1 has 50 % NaN trials — should be flagged when threshold ≤ 0.5."""
    epochs = np.ones((4, 2, 5), dtype=np.float64)
    epochs[0, 1, :] = np.nan  # trial 0, channel 1
    epochs[2, 1, :] = np.nan  # trial 2, channel 1  → 50 % NaN for channel 1

    mask_25 = reject_channels_by_nan_trial_ratio(epochs, max_ratio=0.25)
    mask_50 = reject_channels_by_nan_trial_ratio(epochs, max_ratio=0.50)
    mask_75 = reject_channels_by_nan_trial_ratio(epochs, max_ratio=0.75)

    # Channel 1 NaN ratio = 0.5; channel 0 NaN ratio = 0.0.
    assert not mask_25[0], "Channel 0 has no NaN trials."
    assert mask_25[1], "Channel 1 NaN ratio = 0.5 should be flagged at threshold 0.25."
    assert mask_50[1], "Channel 1 NaN ratio = 0.5 should be flagged at threshold 0.50."
    assert not mask_50[0], "Channel 0 has no NaN trials."
    assert not mask_75[1], "Channel 1 NaN ratio = 0.5 should NOT be flagged at threshold 0.75."
    assert not mask_75[0], "Channel 0 has no NaN trials."


# ---------------------------------------------------------------------------
# apply_channel_exclusions
# ---------------------------------------------------------------------------


def test_apply_channel_exclusions_removes_correct_channels() -> None:
    epochs = np.arange(24, dtype=np.float64).reshape(2, 4, 3)
    names = ["A", "B", "C", "D"]
    excl = np.array([False, True, False, True])

    clean, kept, removed = apply_channel_exclusions(epochs, names, excl)

    assert clean.shape == (2, 2, 3)
    assert kept == ["A", "C"]
    assert removed == ["B", "D"]
    np.testing.assert_array_equal(clean[:, 0, :], epochs[:, 0, :])
    np.testing.assert_array_equal(clean[:, 1, :], epochs[:, 2, :])


def test_apply_channel_exclusions_no_exclusion() -> None:
    epochs = np.ones((3, 5, 7), dtype=np.float64)
    names = [f"ch{i}" for i in range(5)]
    excl = np.zeros(5, dtype=bool)

    clean, kept, removed = apply_channel_exclusions(epochs, names, excl)

    assert clean.shape == epochs.shape
    assert kept == names
    assert removed == []


# ---------------------------------------------------------------------------
# Integration: Level A → Level B
# ---------------------------------------------------------------------------


def test_level_a_nan_then_level_b_nan_ratio() -> None:
    """NaN-masked trials from Level A should count towards Level B's NaN-trial ratio."""
    # 4 trials, 2 channels, 5 time points.
    # After Level A marks 2 out of 4 trials as NaN for channel 1,
    # Level B should reject channel 1 when max_nan_trial_ratio ≤ 0.5.
    epochs = np.ones((4, 2, 5), dtype=np.float64)
    # Build a fake nan_mask as Level A output would produce.
    nan_mask = np.array([[False, True], [False, True], [False, False], [False, False]])
    epochs_with_nan = apply_trial_nan_mask(epochs, nan_mask)

    rejection = reject_channels_by_nan_trial_ratio(epochs_with_nan, max_ratio=0.50)

    assert not rejection[0], "Channel 0 has no NaN trials."
    assert rejection[1], "Channel 1 has 50 % NaN trials and should be excluded."



