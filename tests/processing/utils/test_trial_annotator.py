from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.utils.condition_rules import ConditionExpr
from gin_bids_py_analysis.processing.utils.trial_annotator import (
    EventAnnotationInvalidationRule,
    EventFileWindowAnnotator,
    TrialMetadataInvalidationRule,
)
from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict[str, Any]) -> None:
        self.path = path
        self.entities = entities


def _make_bids_file(path: Path, entities: dict[str, Any]) -> BIDSFile:
    return BIDSFile(_MockPyBIDSFile(str(path), entities))


def _make_trial(anchor_onset_s: float = 10.0, keep: bool = True) -> ResolvedTrial:
    return ResolvedTrial(
        source_file=_make_bids_file(Path("/fake/sub-01_ieeg.vhdr"), {}),
        anchor_event_index=0,
        anchor_event_code="10",
        anchor_onset_s=anchor_onset_s,
        anchor_duration_s=0.0,
        keep=keep,
    )


def _make_event_file(tmp_path: Path, rows: list[dict[str, str]], entities: dict[str, Any]) -> BIDSFile:
    """Write a TSV with given rows and return a BIDSFile pointing to it."""
    filename = "sub-01_" + "_".join(f"{k}-{v}" for k, v in entities.items()) + ".tsv"
    tsv_path = tmp_path / filename
    if not rows:
        tsv_path.write_text("onset\tduration\n", encoding="utf-8")
        return _make_bids_file(tsv_path, {"extension": ".tsv", **entities})
    header = "\t".join(rows[0].keys())
    body = "\n".join("\t".join(row.values()) for row in rows)
    tsv_path.write_text(f"{header}\n{body}\n", encoding="utf-8")
    return _make_bids_file(tsv_path, {"extension": ".tsv", **entities})


def _make_group(secondaries: list[BIDSFile]) -> BIDSFileGroup:
    primary = _make_bids_file(Path("/fake/sub-01_ieeg.vhdr"), {"subject": "01"})
    return BIDSFileGroup(primary=primary, secondaries=secondaries)


# ---------------------------------------------------------------------------
# EventFileWindowAnnotator tests
# ---------------------------------------------------------------------------


def test_events_in_window_collected(tmp_path: Path) -> None:
    event_file = _make_event_file(
        tmp_path,
        [
            {"onset": "10.5", "event_type": "Spk", "channel": "A1"},  # inside [-1, +3] of anchor 10
            {"onset": "14.0", "event_type": "Spk", "channel": "A2"},  # outside
        ],
        {"suffix": "events", "desc": "hfospikes"},
    )
    group = _make_group([event_file])
    trial = _make_trial(anchor_onset_s=10.0)

    annotator = EventFileWindowAnnotator(
        filter={"suffix": "events", "desc": "hfospikes"},
        metadata_events_key="hfo_spike_events",
    )
    annotator.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=3.0, ieeg_channel_names=[])

    events = trial.metadata["hfo_spike_events"]
    assert len(events) == 1
    assert events[0]["onset"] == "10.5"


def test_events_outside_window_not_collected(tmp_path: Path) -> None:
    event_file = _make_event_file(
        tmp_path,
        [
            {"onset": "5.0", "event_type": "Spk", "channel": "A1"},   # before window
            {"onset": "15.0", "event_type": "Spk", "channel": "A2"},  # after window
        ],
        {"suffix": "events", "desc": "hfospikes"},
    )
    group = _make_group([event_file])
    trial = _make_trial(anchor_onset_s=10.0)

    annotator = EventFileWindowAnnotator(
        filter={"suffix": "events", "desc": "hfospikes"},
        metadata_events_key="hfo_spike_events",
    )
    annotator.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=3.0, ieeg_channel_names=[])

    assert trial.metadata["hfo_spike_events"] == []


def test_metadata_events_key_always_set_even_when_no_events(tmp_path: Path) -> None:
    event_file = _make_event_file(tmp_path, [], {"suffix": "events", "desc": "hfospikes"})
    group = _make_group([event_file])
    trial = _make_trial(anchor_onset_s=10.0)

    annotator = EventFileWindowAnnotator(
        filter={"suffix": "events", "desc": "hfospikes"},
        metadata_events_key="hfo_spike_events",
    )
    annotator.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=3.0, ieeg_channel_names=[])

    assert "hfo_spike_events" in trial.metadata
    assert trial.metadata["hfo_spike_events"] == []


def test_filter_selects_correct_files(tmp_path: Path) -> None:
    """Only files matching the annotator filter should be loaded."""
    hfo_spike_file = _make_event_file(
        tmp_path,
        [{"onset": "10.5", "event_type": "Spk", "channel": "A1"}],
        {"suffix": "events", "desc": "hfospikes"},
    )
    beh_file = _make_event_file(
        tmp_path,
        [{"onset": "10.5", "event_type": "Osc", "channel": "A2"}],
        {"suffix": "beh"},
    )
    group = _make_group([hfo_spike_file, beh_file])
    trial = _make_trial(anchor_onset_s=10.0)

    annotator = EventFileWindowAnnotator(
        filter={"suffix": "events", "desc": "hfospikes"},
        metadata_events_key="hfo_spike_events",
    )
    annotator.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=3.0, ieeg_channel_names=[])

    events = trial.metadata["hfo_spike_events"]
    # Only the HFO/spike file contributes; its event_type is "Spk", not "Osc"
    assert all(e["event_type"] == "Spk" for e in events)


def test_window_override(tmp_path: Path) -> None:
    """window_tmin_s / window_tmax_s override the processor defaults."""
    event_file = _make_event_file(
        tmp_path,
        [
            {"onset": "9.5", "event_type": "Spk", "channel": "A1"},   # inside tight window [-0.5, +1]
            {"onset": "11.5", "event_type": "Spk", "channel": "A2"},  # outside tight window
        ],
        {"suffix": "events", "desc": "hfospikes"},
    )
    group = _make_group([event_file])
    trial = _make_trial(anchor_onset_s=10.0)

    annotator = EventFileWindowAnnotator(
        filter={"suffix": "events", "desc": "hfospikes"},
        metadata_events_key="hfo_spike_events",
        window_tmin_s=-0.5,
        window_tmax_s=1.0,
    )
    # Processor defaults are wider but should be ignored
    annotator.annotate_trials(group, group.primary, [trial], tmin_s=-5.0, tmax_s=5.0, ieeg_channel_names=[])

    events = trial.metadata["hfo_spike_events"]
    assert len(events) == 1
    assert events[0]["onset"] == "9.5"


def test_rows_with_invalid_onset_are_skipped(tmp_path: Path) -> None:
    event_file = _make_event_file(
        tmp_path,
        [
            {"onset": "n/a", "event_type": "Spk", "channel": "A1"},
            {"onset": "10.1", "event_type": "Spk", "channel": "A2"},
        ],
        {"suffix": "events", "desc": "hfospikes"},
    )
    group = _make_group([event_file])
    trial = _make_trial(anchor_onset_s=10.0)

    annotator = EventFileWindowAnnotator(
        filter={"suffix": "events", "desc": "hfospikes"},
        metadata_events_key="hfo_spike_events",
    )
    annotator.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=1.0, ieeg_channel_names=[])

    events = trial.metadata["hfo_spike_events"]
    assert len(events) == 1
    assert events[0]["channel"] == "A2"


def test_only_events_compatible_with_current_ieeg_file_are_collected(tmp_path: Path) -> None:
    run1_file = _make_event_file(
        tmp_path,
        [{"onset": "10.1", "event_type": "Spk", "channel": "A1"}],
        {"subject": "01", "task": "decid", "run": "1", "suffix": "events", "desc": "hfospikes"},
    )
    run2_file = _make_event_file(
        tmp_path,
        [{"onset": "10.2", "event_type": "Spk", "channel": "A2"}],
        {"subject": "01", "task": "decid", "run": "2", "suffix": "events", "desc": "hfospikes"},
    )
    primary = _make_bids_file(
        tmp_path / "sub-01_task-decid_run-1_ieeg.vhdr",
        {
            "subject": "01",
            "task": "decid",
            "run": "1",
            "suffix": "ieeg",
            "extension": ".vhdr",
            "datatype": "ieeg",
        },
    )
    group = BIDSFileGroup(primary=primary, secondaries=[run1_file, run2_file])
    trial = _make_trial(anchor_onset_s=10.0)

    annotator = EventFileWindowAnnotator(
        filter={"suffix": "events", "desc": "hfospikes"},
        metadata_events_key="hfo_spike_events",
    )
    annotator.annotate_trials(group, primary, [trial], tmin_s=-1.0, tmax_s=1.0, ieeg_channel_names=[])

    events = trial.metadata["hfo_spike_events"]
    assert len(events) == 1
    assert events[0]["channel"] == "A1"
    assert events[0]["subject"] == "01"
    assert events[0]["run"] == "1"
    assert str(events[0]["source_path"]).endswith("desc-hfospikes.tsv")


# ---------------------------------------------------------------------------
# EventAnnotationInvalidationRule tests
# ---------------------------------------------------------------------------


def test_no_filter_counts_all_events() -> None:
    trial = _make_trial()
    trial.metadata["hfo_spike_events"] = [
        {"onset": "10.1", "event_type": "Spk", "channel": "A1"},
        {"onset": "10.2", "event_type": "Osc", "channel": "A2"},
    ]
    group = _make_group([])
    rule = EventAnnotationInvalidationRule(
        metadata_events_key="hfo_spike_events",
        invalidate_if_count_gte=2,
    )
    rule.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=3.0, ieeg_channel_names=[])

    assert trial.keep is False


def test_event_filter_condition_expr_filters_before_count() -> None:
    trial = _make_trial()
    trial.metadata["hfo_spike_events"] = [
        {"onset": "10.1", "event_type": "Spk", "channel": "A1"},
        {"onset": "10.2", "event_type": "Osc", "channel": "A2"},
    ]
    group = _make_group([])
    rule = EventAnnotationInvalidationRule(
        metadata_events_key="hfo_spike_events",
        event_filter=ConditionExpr(column="event_type", op="==", value="Spk"),
        invalidate_if_count_gte=1,
    )
    rule.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=3.0, ieeg_channel_names=[])

    assert trial.keep is False


def test_event_filter_channel_in_list() -> None:
    """event_filter with 'in' operator on channel column."""
    trial = _make_trial()
    trial.metadata["hfo_spike_events"] = [
        {"onset": "10.1", "event_type": "Spk", "channel": "vmPFC-1"},
        {"onset": "10.2", "event_type": "Spk", "channel": "A2"},  # not in vmPFC
    ]
    group = _make_group([])
    rule = EventAnnotationInvalidationRule(
        metadata_events_key="hfo_spike_events",
        event_filter=ConditionExpr(column="channel", op="in", values=["vmPFC-1", "vmPFC-2"]),
        invalidate_if_count_gte=1,
    )
    rule.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=3.0, ieeg_channel_names=[])

    assert trial.keep is False


def test_event_filter_no_matching_events_keeps_trial() -> None:
    trial = _make_trial()
    trial.metadata["hfo_spike_events"] = [
        {"onset": "10.1", "event_type": "Osc", "channel": "A1"},
    ]
    group = _make_group([])
    rule = EventAnnotationInvalidationRule(
        metadata_events_key="hfo_spike_events",
        event_filter=ConditionExpr(column="event_type", op="==", value="Spk"),
        invalidate_if_count_gte=1,
    )
    rule.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=3.0, ieeg_channel_names=[])

    assert trial.keep is True


def test_threshold_not_met_keeps_trial() -> None:
    trial = _make_trial()
    trial.metadata["hfo_spike_events"] = [
        {"onset": "10.1", "event_type": "Spk", "channel": "A1"},
    ]
    group = _make_group([])
    rule = EventAnnotationInvalidationRule(
        metadata_events_key="hfo_spike_events",
        invalidate_if_count_gte=2,
    )
    rule.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=3.0, ieeg_channel_names=[])

    assert trial.keep is True


def test_threshold_met_invalidates_trial() -> None:
    trial = _make_trial()
    trial.metadata["hfo_spike_events"] = [
        {"onset": "10.1", "event_type": "Spk", "channel": "A1"},
        {"onset": "10.2", "event_type": "Spk", "channel": "A2"},
    ]
    group = _make_group([])
    rule = EventAnnotationInvalidationRule(
        metadata_events_key="hfo_spike_events",
        invalidate_if_count_gte=2,
        exclusion_reason="spike_in_window",
    )
    rule.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=3.0, ieeg_channel_names=[])

    assert trial.keep is False
    assert trial.exclusion_reason == "spike_in_window"


def test_exclusion_reason_not_overwritten_when_already_set() -> None:
    trial = _make_trial(keep=False)
    trial.exclusion_reason = "missing_label"
    trial.metadata["hfo_spike_events"] = [
        {"onset": "10.1", "event_type": "Spk", "channel": "A1"},
    ]
    group = _make_group([])
    rule = EventAnnotationInvalidationRule(
        metadata_events_key="hfo_spike_events",
        invalidate_if_count_gte=1,
        exclusion_reason="spike_in_window",
    )
    rule.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=3.0, ieeg_channel_names=[])

    assert trial.exclusion_reason == "missing_label"


def test_missing_metadata_key_does_not_invalidate() -> None:
    """If the annotator key is absent, count = 0, trial stays valid."""
    trial = _make_trial()
    group = _make_group([])
    rule = EventAnnotationInvalidationRule(
        metadata_events_key="hfo_spike_events",
        invalidate_if_count_gte=1,
    )
    rule.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=3.0, ieeg_channel_names=[])

    assert trial.keep is True
    assert trial.exclusion_reason is None


def test_combined_annotator_then_invalidation_rule(tmp_path: Path) -> None:
    """End-to-end: annotator collects events, rule invalidates trial."""
    event_file = _make_event_file(
        tmp_path,
        [
            {"onset": "10.2", "event_type": "Spk", "channel": "vmPFC-1"},
            {"onset": "10.8", "event_type": "Osc", "channel": "A2"},
        ],
        {"suffix": "events", "desc": "hfospikes"},
    )
    group = _make_group([event_file])
    trial = _make_trial(anchor_onset_s=10.0)

    annotator = EventFileWindowAnnotator(
        filter={"suffix": "events", "desc": "hfospikes"},
        metadata_events_key="hfo_spike_events",
    )
    rule = EventAnnotationInvalidationRule(
        metadata_events_key="hfo_spike_events",
        event_filter=ConditionExpr(
            all=[
                ConditionExpr(column="event_type", op="==", value="Spk"),
                ConditionExpr(column="channel", op="in", values=["vmPFC-1", "vmPFC-2"]),
            ]
        ),
        invalidate_if_count_gte=1,
        exclusion_reason="vmPFC_spike",
    )

    annotator.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=3.0, ieeg_channel_names=[])
    rule.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=3.0, ieeg_channel_names=[])

    assert trial.keep is False
    assert trial.exclusion_reason == "vmPFC_spike"


# ---------------------------------------------------------------------------
# TrialMetadataInvalidationRule tests
# ---------------------------------------------------------------------------


def _make_trial_with_metadata(metadata: dict[str, Any], keep: bool = True) -> ResolvedTrial:
    trial = _make_trial(keep=keep)
    trial.metadata.update(metadata)
    return trial


def test_metadata_condition_satisfied_keeps_trial() -> None:
    trial = _make_trial_with_metadata({"RT": 2.5, "rating": 1})
    group = _make_group([])
    rule = TrialMetadataInvalidationRule(
        condition={"column": "RT", "op": "<=", "value": 20.0},
        exclusion_reason="rt_too_long",
    )
    rule.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=3.0, ieeg_channel_names=[])

    assert trial.keep is True
    assert trial.exclusion_reason is None


def test_metadata_condition_not_satisfied_invalidates_trial() -> None:
    trial = _make_trial_with_metadata({"RT": 25.0, "rating": 1})
    group = _make_group([])
    rule = TrialMetadataInvalidationRule(
        condition={"column": "RT", "op": "<=", "value": 20.0},
        exclusion_reason="rt_too_long",
    )
    rule.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=3.0, ieeg_channel_names=[])

    assert trial.keep is False
    assert trial.exclusion_reason == "rt_too_long"


def test_metadata_already_excluded_reason_not_overwritten() -> None:
    trial = _make_trial_with_metadata({"RT": 25.0}, keep=False)
    trial.exclusion_reason = "vmPFC_spike"
    group = _make_group([])
    rule = TrialMetadataInvalidationRule(
        condition={"column": "RT", "op": "<=", "value": 20.0},
        exclusion_reason="rt_too_long",
    )
    rule.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=3.0, ieeg_channel_names=[])

    assert trial.exclusion_reason == "vmPFC_spike"


def test_metadata_missing_key_invalidates_trial() -> None:
    """Missing metadata key → condition not satisfiable → trial excluded."""
    trial = _make_trial_with_metadata({})
    group = _make_group([])
    rule = TrialMetadataInvalidationRule(
        condition={"column": "RT", "op": "<=", "value": 20.0},
        exclusion_reason="rt_too_long",
    )
    rule.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=3.0, ieeg_channel_names=[])

    assert trial.keep is False


def test_metadata_compound_all_condition() -> None:
    """Compound 'all' condition: both RT and rating must pass."""
    trial_pass = _make_trial_with_metadata({"RT": 5.0, "rating": 2})
    trial_fail_rt = _make_trial_with_metadata({"RT": 25.0, "rating": 2})
    trial_fail_rating = _make_trial_with_metadata({"RT": 5.0, "rating": -1})
    group = _make_group([])
    rule = TrialMetadataInvalidationRule(
        condition={"all": [
            {"column": "RT", "op": "<=", "value": 20.0},
            {"column": "rating", "op": ">=", "value": 0},
        ]},
        exclusion_reason="behavioral_threshold",
    )
    for trial in [trial_pass, trial_fail_rt, trial_fail_rating]:
        rule.annotate_trials(group, group.primary, [trial], tmin_s=-1.0, tmax_s=3.0, ieeg_channel_names=[])

    assert trial_pass.keep is True
    assert trial_fail_rt.keep is False
    assert trial_fail_rating.keep is False



