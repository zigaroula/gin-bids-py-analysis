from __future__ import annotations

from pathlib import Path

import pytest

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.matching import (
    entities_compatible,
    entity_match_score,
    entity_specificity,
    entity_value,
    files_matching_entities,
    find_best_entity_match,
    normalize_entity_value,
    shared_entities,
)


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


def _make_bids_file(path: Path, entities: dict[str, str]) -> BIDSFile:
    return BIDSFile(_MockPyBIDSFile(str(path), entities))


def test_normalize_entity_value_strips_subject_and_session_prefixes() -> None:
    assert normalize_entity_value(" sub-01 ") == "01"
    assert normalize_entity_value("ses-02") == "02"
    assert normalize_entity_value("run-1") == "run-1"
    assert normalize_entity_value("") is None
    assert normalize_entity_value(None) is None


def test_entity_value_uses_aliases() -> None:
    assert entity_value({"sub": "01"}, ("subject", "sub")) == "01"
    assert entity_value({"session": "ses-03"}, ("ses", "session")) == "03"
    assert entity_value({"run": "1"}, ("task",)) is None


def test_entity_match_score_allows_partial_candidate_matches() -> None:
    target = {"subject": "01", "session": "01", "task": "decid", "run": "1"}
    candidate = {"sub": "sub-01", "task": "decid"}

    assert entity_match_score(target, candidate) == 2


def test_entity_match_score_returns_none_on_conflict() -> None:
    target = {"subject": "01", "task": "decid", "run": "1"}
    candidate = {"subject": "01", "task": "other", "run": "1"}

    assert entity_match_score(target, candidate) is None


def test_entities_compatible_prefers_preferred_entities() -> None:
    target = {"subject": "01", "task": "decid", "run": "1"}
    candidate = {"subject": "01", "task": "other", "run": "1"}
    preferred = {"task": "decid"}

    assert entities_compatible(
        target,
        candidate,
        preferred_entities=preferred,
    )


def test_entities_compatible_detects_mismatch() -> None:
    target = {"subject": "01", "task": "decid", "run": "1"}
    candidate = {"subject": "01", "task": "other", "run": "1"}

    assert not entities_compatible(target, candidate)


def test_entity_specificity_ignores_provenance_entities() -> None:
    entities = {
        "subject": "01",
        "task": "decid",
        "space": "MNI305",
        "suffix": "electrodes",
        "extension": ".tsv",
        "datatype": "ieeg",
    }

    assert entity_specificity(entities) == 3


def test_files_matching_entities_filters_and_sorts() -> None:
    f1 = _make_bids_file(
        Path("/tmp/sub-01_task-decid_run-1_ieeg.vhdr"),
        {"suffix": "ieeg", "extension": ".vhdr", "run": "1"},
    )
    f2 = _make_bids_file(
        Path("/tmp/sub-01_task-decid_run-2_ieeg.vhdr"),
        {"suffix": "ieeg", "extension": ".vhdr", "run": "2"},
    )
    f3 = _make_bids_file(
        Path("/tmp/sub-01_task-decid_run-1_events.tsv"),
        {"suffix": "events", "extension": ".tsv", "run": "1"},
    )

    matched = files_matching_entities(
        [f2, f3, f1],
        suffix="ieeg",
        extension=".vhdr",
        run={"1", "2"},
    )

    assert [str(file.path) for file in matched] == [str(f1.path), str(f2.path)]


def test_shared_entities_returns_intersection_without_provenance_keys() -> None:
    f1 = _make_bids_file(
        Path("/tmp/sub-01_task-decid_run-1_ieeg.vhdr"),
        {
            "subject": "01",
            "task": "decid",
            "run": "1",
            "suffix": "ieeg",
            "extension": ".vhdr",
        },
    )
    f2 = _make_bids_file(
        Path("/tmp/sub-01_task-decid_run-2_ieeg.vhdr"),
        {
            "subject": "01",
            "task": "decid",
            "run": "2",
            "suffix": "ieeg",
            "extension": ".vhdr",
        },
    )

    assert shared_entities([f1, f2]) == {"subject": "01", "task": "decid"}


def test_find_best_entity_match_prefers_more_generic_file() -> None:
    target = _make_bids_file(
        Path("/tmp/sub-01_task-decid_run-1_ieeg.vhdr"),
        {"subject": "01", "task": "decid", "run": "1", "suffix": "ieeg"},
    )
    generic = _make_bids_file(
        Path("/tmp/sub-01_task-decid_run-1_electrodes.tsv"),
        {"subject": "01", "task": "decid", "run": "1", "suffix": "electrodes"},
    )
    specific = _make_bids_file(
        Path("/tmp/sub-01_task-decid_run-1_space-MNI_electrodes.tsv"),
        {
            "subject": "01",
            "task": "decid",
            "run": "1",
            "space": "MNI",
            "suffix": "electrodes",
        },
    )

    assert find_best_entity_match(target, [specific, generic]) == generic


def test_find_best_entity_match_raises_on_persistent_ambiguity() -> None:
    target = _make_bids_file(
        Path("/tmp/sub-01_task-decid_run-1_ieeg.vhdr"),
        {"subject": "01", "task": "decid", "run": "1", "suffix": "ieeg"},
    )
    a = _make_bids_file(
        Path("/tmp/sub-01_task-decid_run-1_desc-a_electrodes.tsv"),
        {
            "subject": "01",
            "task": "decid",
            "run": "1",
            "desc": "a",
            "suffix": "electrodes",
        },
    )
    b = _make_bids_file(
        Path("/tmp/sub-01_task-decid_run-1_desc-b_electrodes.tsv"),
        {
            "subject": "01",
            "task": "decid",
            "run": "1",
            "desc": "b",
            "suffix": "electrodes",
        },
    )

    with pytest.raises(ValueError, match="Ambiguous electrodes table match"):
        find_best_entity_match(
            target,
            [a, b],
            ambiguity_label="electrodes table",
        )
