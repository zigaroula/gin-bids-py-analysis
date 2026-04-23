from __future__ import annotations

import pytest

from gin_bids_py_analysis.processing.trial_stats import (
    BaseTrialStatsParams,
    ConditionTestParams,
    RegressionParams,
)


def test_base_params_normalize_common_fields() -> None:
    params = BaseTrialStatsParams(
        anchor_event_codes=10,
        tmin_s=-1.0,
        tmax_s=2.0,
        atlas_name="  dk  ",
        atlas_regions="insula",
        n_bins=-1,
    )

    assert params.anchor_event_codes == ["10"]
    assert params.atlas_name == "dk"
    assert params.atlas_regions == ["insula"]
    assert params.n_bins == 0
    assert params.events_source == "annotations"
    assert params.trial_activity_summary.kind == "epoch_mean"


def test_base_params_accepts_trial_activity_summary_payload() -> None:
    params = BaseTrialStatsParams(
        anchor_event_codes=["10"],
        tmin_s=-1.0,
        tmax_s=2.0,
        trial_activity_summary={
            "kind": "anchor_to_response_mean",
            "missing_response_policy": "nan_if_missing",
            "response": {
                "source": "table_column",
                "column": "rt_ms",
                "units": "ms",
            },
        },
    )

    assert params.trial_activity_summary.kind == "anchor_to_response_mean"
    assert params.trial_activity_summary.missing_response_policy == "nan_if_missing"
    assert params.trial_activity_summary.response is not None
    assert params.trial_activity_summary.response.source == "table_column"


def test_base_params_reject_mutually_exclusive_binning() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        BaseTrialStatsParams(
            anchor_event_codes=["10"],
            tmin_s=-1.0,
            tmax_s=2.0,
            window_ms=50.0,
            n_bins=4,
        )


def test_condition_test_restricts_activity_zscore_and_permutation_correction() -> None:
    with pytest.raises(ValueError, match="activity_zscore"):
        ConditionTestParams(
            anchor_event_codes=["10"],
            tmin_s=-1.0,
            tmax_s=2.0,
            activity_zscore="across_trials",
        )

    with pytest.raises(ValueError, match="p_value_correction_method"):
        ConditionTestParams(
            anchor_event_codes=["10"],
            tmin_s=-1.0,
            tmax_s=2.0,
            p_value_correction_method="permutation",
        )

    params = ConditionTestParams(
        anchor_event_codes=["10"],
        tmin_s=-1.0,
        tmax_s=2.0,
        p_value_correction_method="fdr_bh",
        n_permutations=250,
    )

    assert params.n_permutations == 250


def test_regression_accepts_across_trials_but_rejects_permutation_correction() -> None:
    params = RegressionParams(
        anchor_event_codes=["10"],
        tmin_s=-1.0,
        tmax_s=2.0,
        predictor="predictor_value",
        activity_zscore="across_trials",
    )

    assert params.activity_zscore == "across_trials"

    with pytest.raises(ValueError, match="p_value_correction_method"):
        RegressionParams(
            anchor_event_codes=["10"],
            tmin_s=-1.0,
            tmax_s=2.0,
            predictor="predictor_value",
            p_value_correction_method="permutation",
        )
