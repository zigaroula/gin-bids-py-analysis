from __future__ import annotations

import pytest

from gin_bids_py_analysis.processing.trial_slope_stats import TrialSlopeStatsParams


def test_params_validate_core_constraints() -> None:
    params = TrialSlopeStatsParams(
        anchor_event_codes=["10"],
        tmin_s=-1.0,
        tmax_s=1.0,
        condition_a="accepted",
        condition_b="rejected",
    )
    assert params.min_trials_per_condition == 3
    assert params.predictor_scaling == "none"


def test_params_reject_equal_conditions() -> None:
    with pytest.raises(ValueError, match="condition_a and condition_b"):
        TrialSlopeStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.2,
            condition_a="same",
            condition_b="same",
        )


def test_params_rejects_window_ms_and_n_bins_together() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrialSlopeStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=1.0,
            window_ms=100.0,
            n_bins=2,
        )


def test_params_rejects_empty_predictor_key() -> None:
    with pytest.raises(ValueError, match="predictor"):
        TrialSlopeStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=1.0,
            predictor="   ",
        )


