from __future__ import annotations

from types import SimpleNamespace

from scripts.visualize_trial_slope_stats import (
    VM_PFC_SPIKE_EXCLUSION_REASON,
    _count_excluded_trials_by_reason,
    _format_subject_trial_exclusion_summary,
)


def test_count_excluded_trials_by_reason_counts_only_matching_exclusions() -> None:
    trials = [
        SimpleNamespace(keep=False, exclusion_reason=VM_PFC_SPIKE_EXCLUSION_REASON),
        SimpleNamespace(keep=False, exclusion_reason=VM_PFC_SPIKE_EXCLUSION_REASON),
        SimpleNamespace(keep=False, exclusion_reason="invalid_predictor_value"),
        SimpleNamespace(keep=True, exclusion_reason=None),
    ]

    assert _count_excluded_trials_by_reason(
        trials,
        VM_PFC_SPIKE_EXCLUSION_REASON,
    ) == 2


def test_format_subject_trial_exclusion_summary_reports_reason_and_totals() -> None:
    result = SimpleNamespace(
        resolved_trials=[
            SimpleNamespace(keep=False, exclusion_reason=VM_PFC_SPIKE_EXCLUSION_REASON),
            SimpleNamespace(keep=False, exclusion_reason=VM_PFC_SPIKE_EXCLUSION_REASON),
            SimpleNamespace(keep=False, exclusion_reason="invalid_predictor_value"),
            SimpleNamespace(keep=True, exclusion_reason=None),
        ]
    )

    message = _format_subject_trial_exclusion_summary("01", result)

    assert message == (
        "Subject 01: removed 2 trial(s) by vmPFC_spike_0_3s "
        "(3 excluded total / 4 resolved)."
    )
