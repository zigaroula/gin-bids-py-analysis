"""
Tests for BIDSSubject filtering.
"""

from __future__ import annotations

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.subject import BIDSSubject


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


def _make_file(entities: dict) -> BIDSFile:
    stem = "_".join(f"{k}-{v}" for k, v in entities.items() if k not in ("suffix", "extension"))
    return BIDSFile(_MockPyBIDSFile(path=f"/fake/{stem}.vhdr", entities=entities))


def test_get_files_no_filter_returns_all() -> None:
    f1 = _make_file({"subject": "01", "run": "1", "suffix": "ieeg"})
    f2 = _make_file({"subject": "01", "run": "2", "suffix": "ieeg"})
    subj = BIDSSubject("01", [f1, f2])
    assert subj.get_files() == [f1, f2]


def test_get_files_single_entity_filter() -> None:
    f1 = _make_file({"subject": "01", "run": "1", "suffix": "ieeg"})
    f2 = _make_file({"subject": "01", "run": "2", "suffix": "ieeg"})
    subj = BIDSSubject("01", [f1, f2])
    assert subj.get_files(run="1") == [f1]


def test_get_files_multi_entity_filter() -> None:
    f1 = _make_file({"subject": "01", "task": "rest", "run": "1", "suffix": "ieeg"})
    f2 = _make_file({"subject": "01", "task": "active", "run": "1", "suffix": "ieeg"})
    subj = BIDSSubject("01", [f1, f2])
    assert subj.get_files(task="rest", run="1") == [f1]


def test_get_files_no_match_returns_empty() -> None:
    f1 = _make_file({"subject": "01", "run": "1"})
    subj = BIDSSubject("01", [f1])
    assert subj.get_files(run="99") == []


def test_files_property_returns_copy() -> None:
    f1 = _make_file({"subject": "01"})
    subj = BIDSSubject("01", [f1])
    copy = subj.files
    copy.clear()
    assert len(subj.files) == 1



