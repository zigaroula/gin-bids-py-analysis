from __future__ import annotations

from pathlib import Path

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats import TableTrialLabelResolver
from gin_bids_py_analysis.processing.utils.events import AnnotationEvent


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


def _make_bids_file(path: Path, entities: dict[str, str]) -> BIDSFile:
    return BIDSFile(_MockPyBIDSFile(str(path), entities))


def test_table_trial_label_resolver_joins_event_and_label_tables_by_trial_id(
    tmp_path: Path,
) -> None:
    hilbert_file = _make_bids_file(
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

    resolver = TableTrialLabelResolver(
        label_column="decision",
        label_map={"accept": "accepted", "reject": "rejected"},
        trial_id_column="trial_id",
        anchor_onset_column="onset",
        anchor_event_code_column="anchor_event_code",
    )
    anchor_events = [
        AnnotationEvent(1.0, 0.0, "Stimulus", "S  10", "10"),
        AnnotationEvent(2.0, 0.0, "Stimulus", "S  10", "10"),
    ]

    resolved = resolver.resolve_trials(
        BIDSFileGroup(primary=hilbert_file, secondaries=[events_file, beh_file]),
        hilbert_file,
        anchor_events,
    )

    assert [trial.label for trial in resolved] == ["accepted", "rejected"]
    assert [trial.trial_id for trial in resolved] == ["trial-1", "trial-2"]
    assert all(trial.keep for trial in resolved)


def test_table_trial_label_resolver_falls_back_to_anchor_order(
    tmp_path: Path,
) -> None:
    hilbert_file = _make_bids_file(
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

    resolver = TableTrialLabelResolver(
        label_column="decision",
        label_map={"accept": "accepted", "reject": "rejected"},
    )
    anchor_events = [
        AnnotationEvent(1.0, 0.0, "Stimulus", "S  10", "10"),
        AnnotationEvent(2.0, 0.0, "Stimulus", "S  10", "10"),
    ]

    resolved = resolver.resolve_trials(
        BIDSFileGroup(primary=hilbert_file, secondaries=[trial_table_file]),
        hilbert_file,
        anchor_events,
    )

    assert [trial.label for trial in resolved] == ["accepted", "rejected"]
    assert all(trial.keep for trial in resolved)
