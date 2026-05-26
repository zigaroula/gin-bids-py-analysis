from __future__ import annotations

import numpy as np
import pytest

from gin_bids_py_analysis.processing.trial_stats import ConditionTestParams
from gin_bids_py_analysis.processing.trial_stats.condition_test.stats import (
    compute_bootstrap_difference_ci95,
    compute_condition_statistics,
    compute_duration_channel_significance,
    compute_permuted_statistics,
    compute_single_bin_channel_significance,
    extract_epochs,
)
from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial
from gin_bids_py_analysis.processing.utils.statistics import (
    correct_p_values,
    zscore_activity_by_baseline,
)


def _trial(onset_s: float, *, keep: bool = True) -> ResolvedTrial:
    class _DummyFile:
        path = "dummy.vhdr"

    return ResolvedTrial(
        source_file=_DummyFile(),  # type: ignore[arg-type]
        anchor_event_index=0,
        anchor_event_code="10",
        anchor_onset_s=onset_s,
        anchor_duration_s=0.0,
        label="accepted",
        keep=keep,
    )


def test_extract_epochs_drops_partial_trials() -> None:
    data = np.arange(20, dtype=np.float32)[None, :]
    trials = [_trial(0.0), _trial(0.5)]

    extraction = extract_epochs(
        data,
        sfreq=10.0,
        trials=trials,
        tmin_s=-0.1,
        tmax_s=0.1,
        drop_partial_epochs=True,
    )

    assert extraction.epochs.shape == (1, 1, 3)
    assert len(extraction.kept_trials) == 1
    assert extraction.updated_trials[0].keep is False
    assert extraction.updated_trials[0].exclusion_reason == "partial_epoch"


def test_extract_epochs_raises_when_partial_epochs_are_not_dropped() -> None:
    data = np.arange(20, dtype=np.float32)[None, :]

    with pytest.raises(ValueError, match="partial epoch"):
        extract_epochs(
            data,
            sfreq=10.0,
            trials=[_trial(0.0)],
            tmin_s=-0.1,
            tmax_s=0.1,
            drop_partial_epochs=False,
        )


def test_compute_condition_statistics_detects_known_difference() -> None:
    epochs_a = np.full((4, 2, 3), 5.0, dtype=np.float32)
    epochs_b = np.full((4, 2, 3), 1.0, dtype=np.float32)

    t_values, p_values, mean_a, mean_b, mean_difference = compute_condition_statistics(
        epochs_a,
        epochs_b,
        n_channels=2,
        n_times=3,
        equal_var=False,
    )

    assert t_values.shape == (2, 3)
    assert p_values.shape == (2, 3)
    assert np.all(mean_a == 5.0)
    assert np.all(mean_b == 1.0)
    assert np.all(mean_difference == 4.0)
    assert np.all(t_values > 0)


def test_correct_p_values_fdr_bh_and_bonferroni() -> None:
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


def test_trial_stats_params_rejects_atlas_regions_without_atlas_name() -> None:
    with pytest.raises(ValueError, match="atlas_regions requires atlas_name"):
        ConditionTestParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.1,
            atlas_regions=["R1"],
        )

def test_trial_stats_params_rejects_window_ms_and_n_bins_together() -> None:
    with pytest.raises(ValueError, match="window_ms and n_bins are mutually exclusive"):
        ConditionTestParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.1,
            window_ms=100.0,
            n_bins=2,
        )


# --- permutation-derived statistics ---


def test_compute_permuted_statistics_shape_and_dtype() -> None:
    rng = np.random.default_rng(42)
    n_a, n_b, n_ch, n_t = 5, 4, 3, 10
    epochs_a = rng.standard_normal((n_a, n_ch, n_t)).astype(np.float32)
    epochs_b = rng.standard_normal((n_b, n_ch, n_t)).astype(np.float32)

    out = compute_permuted_statistics(epochs_a, epochs_b, n_perm=20, rng=rng)

    assert out.shape == (20, n_ch, n_t)
    assert out.dtype == np.float32


def test_compute_permuted_statistics_empty_condition_returns_nan() -> None:
    rng = np.random.default_rng(0)
    epochs_a = np.zeros((0, 2, 5), dtype=np.float32)
    epochs_b = np.ones((3, 2, 5), dtype=np.float32)

    out = compute_permuted_statistics(epochs_a, epochs_b, n_perm=10, rng=rng)

    assert out.shape == (10, 2, 5)
    assert np.all(np.isnan(out))


def test_bootstrap_difference_ci95_is_deterministic_with_seed() -> None:
    rng = np.random.default_rng(123)
    epochs_a = rng.normal(loc=2.0, scale=0.5, size=(12, 2, 4)).astype(np.float64)
    epochs_b = rng.normal(loc=1.0, scale=0.5, size=(12, 2, 4)).astype(np.float64)

    low_1, high_1 = compute_bootstrap_difference_ci95(
        epochs_a,
        epochs_b,
        n_bootstraps=200,
        random_state=7,
    )
    low_2, high_2 = compute_bootstrap_difference_ci95(
        epochs_a,
        epochs_b,
        n_bootstraps=200,
        random_state=7,
    )

    np.testing.assert_allclose(low_1, low_2)
    np.testing.assert_allclose(high_1, high_2)
    assert low_1.shape == (2, 4)
    assert high_1.shape == (2, 4)
    assert np.all(low_1 <= high_1)


def test_bootstrap_difference_ci95_returns_nan_when_trials_insufficient() -> None:
    epochs_a = np.ones((1, 2, 3), dtype=np.float64)
    epochs_b = np.ones((4, 2, 3), dtype=np.float64)

    low, high = compute_bootstrap_difference_ci95(
        epochs_a,
        epochs_b,
        n_bootstraps=50,
        random_state=0,
    )

    assert np.all(np.isnan(low))
    assert np.all(np.isnan(high))


def test_trial_stats_params_rejects_permutation_correction_method() -> None:
    with pytest.raises(ValueError, match="p_value_correction_method"):
        ConditionTestParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.1,
            p_value_correction_method="permutation",
        )


def test_trial_stats_params_rejects_baseline_outside_epoch_when_baseline_scaling_enabled() -> None:
    with pytest.raises(ValueError, match="activity_baseline_tmin_s"):
        ConditionTestParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.1,
            activity_zscore="baseline",
        )


def test_zscore_activity_by_baseline_global_uses_trial_mean_reference() -> None:
    epochs_a = np.array(
        [
            [[1.0, 2.0, 11.0]],
            [[2.0, 3.0, 12.0]],
        ],
        dtype=np.float64,
    )
    epochs_b = np.array(
        [
            [[3.0, 4.0, 13.0]],
            [[4.0, 5.0, 14.0]],
        ],
        dtype=np.float64,
    )
    time_axis_s = np.array([-0.2, 0.0, 0.2], dtype=np.float64)

    z_a, z_b = zscore_activity_by_baseline(
        epochs_a,
        epochs_b,
        time_axis_s,
        baseline_tmin_s=-0.2,
        baseline_tmax_s=0.0,
    )
    pooled_trial_means = np.concatenate(
        [
            np.nanmean(z_a[:, :, :2], axis=2),
            np.nanmean(z_b[:, :, :2], axis=2),
        ],
        axis=0,
    )

    assert z_a.shape == epochs_a.shape
    assert z_b.shape == epochs_b.shape
    np.testing.assert_allclose(
        np.nanmean(pooled_trial_means, axis=0),
        np.zeros((1,)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.nanstd(pooled_trial_means, axis=0, ddof=1),
        np.ones((1,)),
        atol=1e-12,
    )


def test_zscore_activity_by_baseline_falls_back_to_centering_when_variance_is_zero() -> None:
    epochs_a = np.array([[[5.0, 5.0, 7.0]]], dtype=np.float64)
    epochs_b = np.array([[[5.0, 5.0, 3.0]]], dtype=np.float64)
    time_axis_s = np.array([-0.2, 0.0, 0.2], dtype=np.float64)

    z_a, z_b = zscore_activity_by_baseline(
        epochs_a,
        epochs_b,
        time_axis_s,
        baseline_tmin_s=-0.2,
        baseline_tmax_s=0.0,
    )

    np.testing.assert_allclose(z_a, [[[0.0, 0.0, 2.0]]])
    np.testing.assert_allclose(z_b, [[[0.0, 0.0, -2.0]]])


def test_zscore_activity_by_baseline_trial_uses_each_trial_reference() -> None:
    epochs_a = np.array(
        [
            [[1.0, 3.0, 11.0]],
            [[2.0, 4.0, 12.0]],
        ],
        dtype=np.float64,
    )
    epochs_b = np.empty((0, 1, 3), dtype=np.float64)
    time_axis_s = np.array([-0.2, 0.0, 0.2], dtype=np.float64)

    z_a, z_b = zscore_activity_by_baseline(
        epochs_a,
        epochs_b,
        time_axis_s,
        baseline_tmin_s=-0.2,
        baseline_tmax_s=0.0,
        baseline_scope="trial",
    )

    np.testing.assert_allclose(np.nanmean(z_a[:, :, :2], axis=2), np.zeros((2, 1)))
    np.testing.assert_allclose(np.nanstd(z_a[:, :, :2], axis=2, ddof=1), np.ones((2, 1)))
    assert z_b.shape == epochs_b.shape


def test_zscore_activity_by_baseline_condition_centers_each_condition_separately() -> None:
    epochs_a = np.array(
        [
            [[1.0, 1.0, 10.0]],
            [[2.0, 2.0, 11.0]],
            [[3.0, 3.0, 12.0]],
        ],
        dtype=np.float64,
    )
    epochs_b = np.array(
        [
            [[101.0, 101.0, 20.0]],
            [[102.0, 102.0, 21.0]],
            [[103.0, 103.0, 22.0]],
        ],
        dtype=np.float64,
    )
    time_axis_s = np.array([-0.2, 0.0, 0.2], dtype=np.float64)

    z_a, z_b = zscore_activity_by_baseline(
        epochs_a,
        epochs_b,
        time_axis_s,
        baseline_tmin_s=-0.2,
        baseline_tmax_s=0.0,
        baseline_scope="condition",
    )

    np.testing.assert_allclose(np.nanmean(np.nanmean(z_a[:, :, :2], axis=2), axis=0), np.zeros((1,)))
    np.testing.assert_allclose(np.nanmean(np.nanmean(z_b[:, :, :2], axis=2), axis=0), np.zeros((1,)))


def test_zscore_activity_by_baseline_global_outlier_removal_changes_reference() -> None:
    epochs_a = np.array(
        [
            [[1.0, 1.0, 10.0]],
            [[2.0, 2.0, 11.0]],
            [[100.0, 100.0, 12.0]],
        ],
        dtype=np.float64,
    )
    epochs_b = np.empty((0, 1, 3), dtype=np.float64)
    time_axis_s = np.array([-0.2, 0.0, 0.2], dtype=np.float64)

    z_without, _ = zscore_activity_by_baseline(
        epochs_a,
        epochs_b,
        time_axis_s,
        baseline_tmin_s=-0.2,
        baseline_tmax_s=0.0,
        baseline_scope="global",
        remove_outlier_trial_means=False,
    )
    z_with, _ = zscore_activity_by_baseline(
        epochs_a,
        epochs_b,
        time_axis_s,
        baseline_tmin_s=-0.2,
        baseline_tmax_s=0.0,
        baseline_scope="global",
        remove_outlier_trial_means=True,
    )

    assert not np.allclose(z_with, z_without, equal_nan=True)


# ---------------------------------------------------------------------------
# compute_single_bin_channel_significance
# ---------------------------------------------------------------------------


def test_compute_single_bin_channel_significance_detects_known_difference() -> None:
    # Channel 0: clearly different means; Channel 1: identical means (no difference)
    rng = np.random.default_rng(0)
    n_trials, n_channels, n_times = 10, 2, 5
    epochs_a = np.zeros((n_trials, n_channels, n_times), dtype=np.float32)
    epochs_b = np.zeros((n_trials, n_channels, n_times), dtype=np.float32)
    # Channel 0: condition A = 10, condition B = 0 → large t-stat
    epochs_a[:, 0, :] = 10.0
    epochs_b[:, 0, :] = 0.0
    # Channel 1: no difference
    epochs_a[:, 1, :] = 5.0
    epochs_b[:, 1, :] = 5.0

    mask = compute_single_bin_channel_significance(
        epochs_a,
        epochs_b,
        equal_var=False,
        p_value_correction_method="none",
        significance_alpha=0.05,
    )

    assert mask.shape == (n_channels,)
    assert mask.dtype == bool
    assert mask[0] is np.bool_(True)
    assert mask[1] is np.bool_(False)


def test_compute_single_bin_channel_significance_fdr_bh_correction() -> None:
    # Two channels: one with strong signal, one null. FDR-BH should still flag the strong one.
    n_trials, n_channels, n_times = 12, 2, 8
    epochs_a = np.zeros((n_trials, n_channels, n_times), dtype=np.float32)
    epochs_b = np.zeros((n_trials, n_channels, n_times), dtype=np.float32)
    epochs_a[:, 0, :] = 20.0
    epochs_b[:, 0, :] = 0.0

    mask = compute_single_bin_channel_significance(
        epochs_a,
        epochs_b,
        p_value_correction_method="fdr_bh",
        significance_alpha=0.05,
    )

    assert mask.shape == (n_channels,)
    assert mask[0] is np.bool_(True)
    assert mask[1] is np.bool_(False)


def test_compute_single_bin_channel_significance_bonferroni_correction() -> None:
    # Same strong signal as above; Bonferroni should still flag the significant channel.
    n_trials, n_channels, n_times = 12, 2, 8
    epochs_a = np.zeros((n_trials, n_channels, n_times), dtype=np.float32)
    epochs_b = np.zeros((n_trials, n_channels, n_times), dtype=np.float32)
    epochs_a[:, 0, :] = 20.0
    epochs_b[:, 0, :] = 0.0

    mask = compute_single_bin_channel_significance(
        epochs_a,
        epochs_b,
        p_value_correction_method="bonferroni",
        significance_alpha=0.05,
    )

    assert mask.shape == (n_channels,)
    assert mask[0] is np.bool_(True)


def test_compute_single_bin_channel_significance_empty_condition_returns_all_false() -> None:
    epochs_a = np.zeros((0, 3, 5), dtype=np.float32)
    epochs_b = np.ones((4, 3, 5), dtype=np.float32)

    mask = compute_single_bin_channel_significance(epochs_a, epochs_b)

    assert mask.shape == (3,)
    assert not mask.any()


# ---------------------------------------------------------------------------
# compute_duration_channel_significance
# ---------------------------------------------------------------------------


def test_compute_duration_channel_significance_counts_all_significant_bins() -> None:
    # Channel 0: 1 significant bin at 100 ms/bin → 100 ms < 250 ms threshold
    # Channel 1: 3 significant bins at 100 ms/bin → 300 ms ≥ 250 ms threshold
    dt = 0.1  # 100 ms per bin
    n_times = 10
    time_axis_s = np.arange(n_times) * dt

    sig_mask = np.zeros((2, n_times), dtype=bool)
    sig_mask[0, 5] = True     # single bin → 100 ms total
    sig_mask[1, 4:7] = True   # 3 bins → 300 ms total

    result = compute_duration_channel_significance(
        sig_mask,
        time_axis_s,
        threshold_ms=250.0,
    )

    assert result.shape == (2,)
    assert result.dtype == bool
    assert result[0] is np.bool_(False)  # 1 bin × 100 ms = 100 ms < 250 ms
    assert result[1] is np.bool_(True)   # 3 bins × 100 ms = 300 ms ≥ 250 ms


def test_compute_duration_channel_significance_threshold_boundary() -> None:
    # 2 consecutive bins at 100 ms each → 200 ms total
    dt = 0.1
    n_times = 5
    time_axis_s = np.arange(n_times) * dt

    sig_mask = np.zeros((1, n_times), dtype=bool)
    sig_mask[0, 1:3] = True  # bins 1 and 2 are significant (neighbours of each other)

    # Just below threshold: 199 ms → False
    result_below = compute_duration_channel_significance(
        sig_mask, time_axis_s, threshold_ms=201.0
    )
    # At threshold: 200 ms → True
    result_at = compute_duration_channel_significance(
        sig_mask, time_axis_s, threshold_ms=200.0
    )

    assert result_below[0] is np.bool_(False)
    assert result_at[0] is np.bool_(True)


def test_compute_duration_channel_significance_edge_bins_counted() -> None:
    # Significant bins at the start and end of the time axis are counted like any other bin.
    dt = 0.1
    n_times = 5
    time_axis_s = np.arange(n_times) * dt

    sig_mask = np.zeros((2, n_times), dtype=bool)
    sig_mask[0, 0:2] = True   # 2 bins at the start
    sig_mask[1, -2:] = True   # 2 bins at the end

    result = compute_duration_channel_significance(sig_mask, time_axis_s, threshold_ms=150.0)

    # 2 bins × 100 ms = 200 ms > 150 ms
    assert result[0] is np.bool_(True)
    assert result[1] is np.bool_(True)


def test_compute_duration_channel_significance_single_time_bin_returns_all_false() -> None:
    # With a single-element time axis the bin duration cannot be derived; all channels → False.
    time_axis_s = np.array([0.0])
    sig_mask = np.ones((3, 1), dtype=bool)

    result = compute_duration_channel_significance(sig_mask, time_axis_s, threshold_ms=1.0)

    assert result.shape == (3,)
    assert not result.any()




