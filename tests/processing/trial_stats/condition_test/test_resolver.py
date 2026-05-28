from __future__ import annotations

from pathlib import Path

import pytest

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.processing.utils.events import AnnotationEvent
from bidsforge.processing.utils.trial_resolver import TableTrialResolver


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


def _make_bids_file(path: Path, entities: dict[str, str]) -> BIDSFile:
    return BIDSFile(_MockPyBIDSFile(str(path), entities))


def _decision_conditions() -> list[dict]:
    return [
        {"label": "accepted", "when": {"column": "decision", "op": "==", "value": "accept"}},
        {"label": "rejected", "when": {"column": "decision", "op": "==", "value": "reject"}},
    ]


def test_table_trial_resolver_joins_event_and_condition_tables_by_trial_id(
    tmp_path: Path,
) -> None:
    ieeg_file = _make_bids_file(
        tmp_path / "sub-01_ses-01_task-decid_run-1_ieeg.vhdr",
        {
            "subject": "01",
            "session": "01",
            "task": "decid",
            "run": "1",
            "suffix": "ieeg",
            "extension": ".vhdr",
            "datatype": "ieeg",
        },
    )
    events_path = tmp_path / "sub-01_ses-01_task-decid_run-1_events.tsv"
    events_path.write_text(
        "trial_id\tonset\tanchor_event_code\n"
        "trial-2\t2.0\t10\n"
        "trial-1\t1.0\t10\n",
        encoding="utf-8",
    )
    events_file = _make_bids_file(
        events_path,
        {
            "subject": "01",
            "session": "01",
            "task": "decid",
            "run": "1",
            "suffix": "events",
            "extension": ".tsv",
            "datatype": "ieeg",
        },
    )
    beh_path = tmp_path / "sub-01_ses-01_task-decid_run-1_beh.tsv"
    beh_path.write_text(
        "trial_id\tdecision\n"
        "trial-1\taccept\n"
        "trial-2\treject\n",
        encoding="utf-8",
    )
    beh_file = _make_bids_file(
        beh_path,
        {
            "subject": "01",
            "session": "01",
            "task": "decid",
            "run": "1",
            "suffix": "beh",
            "extension": ".tsv",
            "datatype": "beh",
        },
    )

    resolver = TableTrialResolver(
        conditions=_decision_conditions(),
        trial_id_column="trial_id",
        anchor_onset_column="onset",
        anchor_event_code_column="anchor_event_code",
    )
    anchor_events = [
        AnnotationEvent(1.0, 0.0, "Stimulus", "S  10", "10"),
        AnnotationEvent(2.0, 0.0, "Stimulus", "S  10", "10"),
    ]

    resolved = resolver.resolve_trials(
        BIDSFileGroup(primary=ieeg_file, secondaries=[events_file, beh_file]),
        ieeg_file,
        anchor_events,
    )

    assert [trial.label for trial in resolved] == ["accepted", "rejected"]
    assert [trial.trial_id for trial in resolved] == ["trial-1", "trial-2"]
    assert all(trial.keep for trial in resolved)
    assert resolved[0].metadata["condition_resolution_reason"] == "matched_condition"


def test_table_trial_resolver_falls_back_to_anchor_order(
    tmp_path: Path,
) -> None:
    ieeg_file = _make_bids_file(
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
    trial_table_path = tmp_path / "sub-01_task-decid_run-1_trials.tsv"
    trial_table_path.write_text(
        "decision\n"
        "accept\n"
        "reject\n",
        encoding="utf-8",
    )
    trial_table_file = _make_bids_file(
        trial_table_path,
        {
            "subject": "01",
            "task": "decid",
            "run": "1",
            "suffix": "events",
            "extension": ".tsv",
            "datatype": "ieeg",
        },
    )

    resolver = TableTrialResolver(conditions=_decision_conditions())
    anchor_events = [
        AnnotationEvent(1.0, 0.0, "Stimulus", "S  10", "10"),
        AnnotationEvent(2.0, 0.0, "Stimulus", "S  10", "10"),
    ]

    resolved = resolver.resolve_trials(
        BIDSFileGroup(primary=ieeg_file, secondaries=[trial_table_file]),
        ieeg_file,
        anchor_events,
    )

    assert [trial.label for trial in resolved] == ["accepted", "rejected"]
    assert all(trial.keep for trial in resolved)


def test_table_trial_resolver_rejects_rows_with_task_mismatch(
    tmp_path: Path,
) -> None:
    ieeg_file = _make_bids_file(
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
    trial_table_path = tmp_path / "sub-01_task-other_run-1_trials.tsv"
    trial_table_path.write_text(
        "subject\ttask\trun\tdecision\n"
        "01\tother\t1\taccept\n",
        encoding="utf-8",
    )
    trial_table_file = _make_bids_file(
        trial_table_path,
        {
            "subject": "01",
            "task": "other",
            "run": "1",
            "suffix": "events",
            "extension": ".tsv",
            "datatype": "ieeg",
        },
    )
    resolver = TableTrialResolver(conditions=_decision_conditions())
    anchor_events = [AnnotationEvent(1.0, 0.0, "Stimulus", "S  10", "10")]

    resolved = resolver.resolve_trials(
        BIDSFileGroup(primary=ieeg_file, secondaries=[trial_table_file]),
        ieeg_file,
        anchor_events,
    )

    assert len(resolved) == 1
    assert resolved[0].keep is False
    assert resolved[0].exclusion_reason == "no_matching_table_row"


def test_table_trial_resolver_copies_extract_columns(
    tmp_path: Path,
) -> None:
    ieeg_file = _make_bids_file(
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
    trial_table_path = tmp_path / "sub-01_task-decid_run-1_trials.tsv"
    trial_table_path.write_text(
        "decision\tvalue_for_slope\n"
        "accept\t1.5\n"
        "reject\t2.75\n",
        encoding="utf-8",
    )
    trial_table_file = _make_bids_file(
        trial_table_path,
        {
            "subject": "01",
            "task": "decid",
            "run": "1",
            "suffix": "events",
            "extension": ".tsv",
            "datatype": "ieeg",
        },
    )

    resolver = TableTrialResolver(
        conditions=_decision_conditions(),
        extract_columns=["value_for_slope"],
    )
    anchor_events = [
        AnnotationEvent(1.0, 0.0, "Stimulus", "S  10", "10"),
        AnnotationEvent(2.0, 0.0, "Stimulus", "S  10", "10"),
    ]

    resolved = resolver.resolve_trials(
        BIDSFileGroup(primary=ieeg_file, secondaries=[trial_table_file]),
        ieeg_file,
        anchor_events,
    )

    assert [trial.label for trial in resolved] == ["accepted", "rejected"]
    assert resolved[0].metadata["value_for_slope"] == "1.5"
    assert resolved[1].metadata["value_for_slope"] == "2.75"


def test_table_trial_resolver_rejects_non_list_extract_columns() -> None:
    with pytest.raises(TypeError, match="extract_columns must be a list\\[str\\]"):
        TableTrialResolver(
            conditions=_decision_conditions(),
            extract_columns={"rating": "rating"},  # type: ignore[arg-type]
        )


def test_table_trial_resolver_without_conditions_uses_anchor_rows(
    tmp_path: Path,
) -> None:
    ieeg_file = _make_bids_file(
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
    events_path = tmp_path / "sub-01_task-decid_run-1_events.tsv"
    events_path.write_text(
        "onset\tanchor_event_code\trating\n"
        "1.0\t10\t1.5\n"
        "2.0\t10\t2.5\n",
        encoding="utf-8",
    )
    events_file = _make_bids_file(
        events_path,
        {
            "subject": "01",
            "task": "decid",
            "run": "1",
            "suffix": "events",
            "extension": ".tsv",
            "datatype": "ieeg",
        },
    )
    resolver = TableTrialResolver(
        conditions=[],
        extract_columns=["rating"],
    )
    anchor_events = [
        AnnotationEvent(1.0, 0.0, "Stimulus", "S  10", "10"),
        AnnotationEvent(2.0, 0.0, "Stimulus", "S  10", "10"),
    ]

    resolved = resolver.resolve_trials(
        BIDSFileGroup(primary=ieeg_file, secondaries=[events_file]),
        ieeg_file,
        anchor_events,
    )

    assert [trial.label for trial in resolved] == [None, None]
    assert [trial.keep for trial in resolved] == [False, False]
    assert [trial.exclusion_reason for trial in resolved] == ["missing_label", "missing_label"]
    assert [trial.metadata["rating"] for trial in resolved] == ["1.5", "2.5"]
    assert [trial.metadata["condition_resolution_reason"] for trial in resolved] == [
        "conditions_not_configured",
        "conditions_not_configured",
    ]


def test_table_trial_resolver_supports_numeric_positive_negative_split(
    tmp_path: Path,
) -> None:
    ieeg_file = _make_bids_file(
        tmp_path / "sub-01_task-rate_run-1_ieeg.vhdr",
        {
            "subject": "01",
            "task": "rate",
            "run": "1",
            "suffix": "ieeg",
            "extension": ".vhdr",
            "datatype": "ieeg",
        },
    )
    trial_table_path = tmp_path / "sub-01_task-rate_run-1_trials.tsv"
    trial_table_path.write_text(
        "rating\n"
        "-2.5\n"
        "3.25\n",
        encoding="utf-8",
    )
    trial_table_file = _make_bids_file(
        trial_table_path,
        {
            "subject": "01",
            "task": "rate",
            "run": "1",
            "suffix": "events",
            "extension": ".tsv",
            "datatype": "ieeg",
        },
    )
    resolver = TableTrialResolver(
        conditions=[
            {"label": "negative", "when": {"column": "rating", "op": "<", "value": 0}},
            {"label": "positive", "when": {"column": "rating", "op": ">", "value": 0}},
        ]
    )
    anchor_events = [
        AnnotationEvent(1.0, 0.0, "Stimulus", "S  10", "10"),
        AnnotationEvent(2.0, 0.0, "Stimulus", "S  10", "10"),
    ]

    resolved = resolver.resolve_trials(
        BIDSFileGroup(primary=ieeg_file, secondaries=[trial_table_file]),
        ieeg_file,
        anchor_events,
    )

    assert [trial.label for trial in resolved] == ["negative", "positive"]
    assert resolved[0].metadata["condition_inputs"] == {"rating": "-2.5"}
    assert resolved[1].metadata["condition_inputs"] == {"rating": "3.25"}


def test_table_trial_resolver_supports_multi_column_compound_conditions(
    tmp_path: Path,
) -> None:
    ieeg_file = _make_bids_file(
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
    trial_table_path = tmp_path / "sub-01_task-decid_run-1_trials.tsv"
    trial_table_path.write_text(
        "choice\tconfidence\n"
        "1\t4\n"
        "1\t2\n",
        encoding="utf-8",
    )
    trial_table_file = _make_bids_file(
        trial_table_path,
        {
            "subject": "01",
            "task": "decid",
            "run": "1",
            "suffix": "events",
            "extension": ".tsv",
            "datatype": "ieeg",
        },
    )
    resolver = TableTrialResolver(
        conditions=[
            {
                "label": "accepted",
                "when": {
                    "all": [
                        {"column": "choice", "op": "==", "value": "1"},
                        {"column": "confidence", "op": ">=", "value": 4},
                    ]
                },
            },
            {
                "label": "rejected",
                "when": {
                    "any": [
                        {"column": "choice", "op": "==", "value": "0"},
                        {"column": "confidence", "op": "<", "value": 4},
                    ]
                },
            },
        ]
    )
    anchor_events = [
        AnnotationEvent(1.0, 0.0, "Stimulus", "S  10", "10"),
        AnnotationEvent(2.0, 0.0, "Stimulus", "S  10", "10"),
    ]

    resolved = resolver.resolve_trials(
        BIDSFileGroup(primary=ieeg_file, secondaries=[trial_table_file]),
        ieeg_file,
        anchor_events,
    )

    assert [trial.label for trial in resolved] == ["accepted", "rejected"]


def test_table_trial_resolver_marks_ambiguous_condition_matches(
    tmp_path: Path,
) -> None:
    ieeg_file = _make_bids_file(
        tmp_path / "sub-01_task-rate_run-1_ieeg.vhdr",
        {
            "subject": "01",
            "task": "rate",
            "run": "1",
            "suffix": "ieeg",
            "extension": ".vhdr",
            "datatype": "ieeg",
        },
    )
    trial_table_path = tmp_path / "sub-01_task-rate_run-1_trials.tsv"
    trial_table_path.write_text("rating\n0\n", encoding="utf-8")
    trial_table_file = _make_bids_file(
        trial_table_path,
        {
            "subject": "01",
            "task": "rate",
            "run": "1",
            "suffix": "events",
            "extension": ".tsv",
            "datatype": "ieeg",
        },
    )
    resolver = TableTrialResolver(
        conditions=[
            {"label": "negative", "when": {"column": "rating", "op": "<=", "value": 0}},
            {"label": "positive", "when": {"column": "rating", "op": ">=", "value": 0}},
        ]
    )
    anchor_events = [AnnotationEvent(1.0, 0.0, "Stimulus", "S  10", "10")]

    resolved = resolver.resolve_trials(
        BIDSFileGroup(primary=ieeg_file, secondaries=[trial_table_file]),
        ieeg_file,
        anchor_events,
    )

    assert resolved[0].label is None
    assert resolved[0].keep is False
    assert resolved[0].exclusion_reason == "ambiguous_condition_match"
    assert resolved[0].metadata["condition_resolution_reason"] == "ambiguous_condition_match"


def test_table_trial_resolver_marks_cast_errors_when_numeric_rule_cannot_parse_value(
    tmp_path: Path,
) -> None:
    ieeg_file = _make_bids_file(
        tmp_path / "sub-01_task-rate_run-1_ieeg.vhdr",
        {
            "subject": "01",
            "task": "rate",
            "run": "1",
            "suffix": "ieeg",
            "extension": ".vhdr",
            "datatype": "ieeg",
        },
    )
    trial_table_path = tmp_path / "sub-01_task-rate_run-1_trials.tsv"
    trial_table_path.write_text("rating\nBAD\n", encoding="utf-8")
    trial_table_file = _make_bids_file(
        trial_table_path,
        {
            "subject": "01",
            "task": "rate",
            "run": "1",
            "suffix": "events",
            "extension": ".tsv",
            "datatype": "ieeg",
        },
    )
    resolver = TableTrialResolver(
        conditions=[
            {"label": "negative", "when": {"column": "rating", "op": "<", "value": 0}},
            {"label": "positive", "when": {"column": "rating", "op": ">", "value": 0}},
        ]
    )
    anchor_events = [AnnotationEvent(1.0, 0.0, "Stimulus", "S  10", "10")]

    resolved = resolver.resolve_trials(
        BIDSFileGroup(primary=ieeg_file, secondaries=[trial_table_file]),
        ieeg_file,
        anchor_events,
    )

    assert resolved[0].label is None
    assert resolved[0].keep is False
    assert resolved[0].exclusion_reason == "condition_value_cast_error"
    assert resolved[0].metadata["condition_resolution_reason"] == "condition_value_cast_error"



