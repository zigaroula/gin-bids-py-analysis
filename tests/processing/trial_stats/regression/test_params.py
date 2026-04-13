from __future__ import annotations

import pytest

from gin_bids_py_analysis.processing.trial_stats import RegressionParams


def test_params_validate_core_constraints() -> None:
    params = RegressionParams(
        anchor_event_codes=["10"],
        tmin_s=-1.0,
        tmax_s=1.0,
        condition_a="accepted",
        condition_b="rejected",
    )
    assert params.min_trials_per_condition == 3
    assert params.predictor_zscore == "none"
    assert params.activity_zscore == "none"
    assert params.activity_baseline_tmin_s == pytest.approx(-0.2)
    assert params.activity_baseline_tmax_s == pytest.approx(0.0)
    assert params.activity_baseline_scope == "global"
    assert params.activity_baseline_remove_outlier_trial_means is False
    assert params.trial_activity_summary.missing_response_policy == "clamp_to_epoch"


def test_params_reject_equal_conditions() -> None:
    with pytest.raises(ValueError, match="condition_a and condition_b"):
        RegressionParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.2,
            condition_a="same",
            condition_b="same",
        )


def test_params_rejects_window_ms_and_n_bins_together() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        RegressionParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=1.0,
            window_ms=100.0,
            n_bins=2,
        )


def test_params_rejects_empty_predictor_key() -> None:
    with pytest.raises(ValueError, match="predictor"):
        RegressionParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=1.0,
            predictor="   ",
        )


def test_params_rejects_baseline_outside_epoch_when_baseline_zscore_enabled() -> None:
    with pytest.raises(ValueError, match="activity_baseline_tmin_s"):
        RegressionParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=1.0,
            activity_zscore="baseline",
        )


def test_params_rejects_legacy_within_condition_predictor_zscore() -> None:
    with pytest.raises(ValueError, match="condition"):
        RegressionParams(
            anchor_event_codes=["10"],
            tmin_s=-1.0,
            tmax_s=1.0,
            predictor_zscore="within_condition",
        )


