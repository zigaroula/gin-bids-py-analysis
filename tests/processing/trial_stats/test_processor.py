from __future__ import annotations

import numpy as np

from gin_bids_py_analysis.processing.trial_stats import (
    BaseTrialStatsParams,
    BaseTrialStatsProcessing,
    BaseTrialStatsProcessingResult,
)
from gin_bids_py_analysis.processing.utils.statistics import zscore_activity_by_baseline


class _DummyResolver:
    condition_labels = ("condition_a", "condition_b")


class _DummyProcessor(BaseTrialStatsProcessing):
    def _normalize_trials(self, **kwargs):
        return kwargs["trials"]

    def _compute_and_build_result(self, context):
        return BaseTrialStatsProcessingResult(
            source_group=context.group,
            metadata=context.metadata,
        )


def test_base_processor_leaves_epochs_unchanged_without_activity_zscore() -> None:
    processor = _DummyProcessor(
        BaseTrialStatsParams(
            anchor_event_codes=["10"],
            tmin_s=-0.2,
            tmax_s=0.4,
            activity_zscore="none",
        ),
        resolver=_DummyResolver(),
    )
    epochs_a = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float64)
    epochs_b = np.array([[[4.0, 5.0, 6.0]]], dtype=np.float64)
    time_axis_s = np.array([-0.2, 0.0, 0.2], dtype=np.float64)

    actual_a, actual_b = processor._apply_activity_zscore(
        epochs_a=epochs_a,
        epochs_b=epochs_b,
        time_axis_s=time_axis_s,
        state={},
    )

    np.testing.assert_array_equal(actual_a, epochs_a)
    np.testing.assert_array_equal(actual_b, epochs_b)


def test_base_processor_applies_shared_baseline_zscore() -> None:
    params = BaseTrialStatsParams(
        anchor_event_codes=["10"],
        tmin_s=-0.2,
        tmax_s=0.4,
        activity_zscore="baseline",
        activity_baseline_tmin_s=-0.2,
        activity_baseline_tmax_s=0.0,
        activity_baseline_scope="global",
    )
    processor = _DummyProcessor(params, resolver=_DummyResolver())
    epochs_a = np.array(
        [
            [[1.0, 2.0, 4.0, 6.0]],
            [[2.0, 3.0, 5.0, 7.0]],
        ],
        dtype=np.float64,
    )
    epochs_b = np.array(
        [
            [[1.5, 2.5, 3.5, 4.5]],
            [[2.5, 3.5, 4.5, 5.5]],
        ],
        dtype=np.float64,
    )
    time_axis_s = np.array([-0.2, 0.0, 0.2, 0.4], dtype=np.float64)

    expected_a, expected_b = zscore_activity_by_baseline(
        epochs_a,
        epochs_b,
        time_axis_s,
        baseline_tmin_s=params.activity_baseline_tmin_s,
        baseline_tmax_s=params.activity_baseline_tmax_s,
        baseline_scope=params.activity_baseline_scope,
        remove_outlier_trial_means=params.activity_baseline_remove_outlier_trial_means,
    )
    actual_a, actual_b = processor._apply_activity_zscore(
        epochs_a=epochs_a,
        epochs_b=epochs_b,
        time_axis_s=time_axis_s,
        state={},
    )

    np.testing.assert_allclose(actual_a, expected_a)
    np.testing.assert_allclose(actual_b, expected_b)
