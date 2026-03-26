"""
Tests for BIDS path building and entity parsing helpers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.helpers import (
    build_bids_path,
    build_subject_groups,
    normalize_subject_value,
    parse_entities,
)


# ---------------------------------------------------------------------------
# Helpers for build_subject_groups tests
# ---------------------------------------------------------------------------


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


def _make_file(tmp_path: Path, name: str, entities: dict) -> BIDSFile:
    """Create a BIDSFile whose underlying path file exists on disk."""
    p = tmp_path / name
    p.touch()
    return BIDSFile(_MockPyBIDSFile(str(p), entities))


def _mock_dataset(
    tmp_path: Path, files_by_filter: dict[tuple, list[BIDSFile]]
) -> MagicMock:
    """Return a mock BIDSDataset where get_files(**kw) returns the list keyed by tuple(sorted(kw.items()))."""

    def _get_files(**kw: object) -> list[BIDSFile]:
        key = tuple(sorted(kw.items()))
        return files_by_filter.get(key, [])

    ds = MagicMock()
    ds.get_files.side_effect = _get_files
    return ds


def test_build_bids_path_filename_order(tmp_path: Path) -> None:
    """Entities must appear in canonical BIDS order in the filename."""
    entities = {"sub": "01", "ses": "01", "task": "rest", "run": "1"}
    path = build_bids_path(entities, tmp_path, suffix="ieeg", extension=".vhdr")
    assert path.name == "sub-01_ses-01_task-rest_run-1_ieeg.vhdr"


def test_build_bids_path_folder_structure(tmp_path: Path) -> None:
    """Output must be nested under sub-<id>/ses-<id>/<datatype>."""
    entities = {"sub": "01", "ses": "01"}
    path = build_bids_path(entities, tmp_path, suffix="ieeg", extension=".vhdr")
    assert "sub-01" in path.parts
    assert "ses-01" in path.parts


def test_build_bids_path_no_session(tmp_path: Path) -> None:
    """When no session is present, the ses- folder must be omitted."""
    entities = {"sub": "01", "task": "rest"}
    path = build_bids_path(entities, tmp_path, suffix="ieeg", extension=".vhdr")
    assert not any(p.startswith("ses-") for p in path.parts)


def test_build_bids_path_desc_entity(tmp_path: Path) -> None:
    entities = {"sub": "01", "desc": "hilbert"}
    path = build_bids_path(entities, tmp_path, suffix="hilbert", extension=".npy")
    assert "desc-hilbert" in path.name


def test_build_bids_path_subject_alias(tmp_path: Path) -> None:
    """Both 'sub' and 'subject' should be accepted as the subject entity key."""
    p1 = build_bids_path({"sub": "01"}, tmp_path, suffix="ieeg", extension=".vhdr")
    p2 = build_bids_path({"subject": "01"}, tmp_path, suffix="ieeg", extension=".vhdr")
    assert p1 == p2


def test_build_bids_path_custom_datatype(tmp_path: Path) -> None:
    entities = {"sub": "01"}
    path = build_bids_path(
        entities, tmp_path, suffix="hilbert", extension=".npy", datatype="ieeg"
    )
    assert "ieeg" in path.parts


def test_parse_entities_basic() -> None:
    p = Path("sub-01_ses-01_task-rest_run-1_ieeg.vhdr")
    entities = parse_entities(p)
    assert entities == {"sub": "01", "ses": "01", "task": "rest", "run": "1"}


def test_parse_entities_ignores_suffix_component() -> None:
    """Bare components without a '-' (the BIDS suffix) must not appear as keys."""
    p = Path("sub-02_desc-hilbert_hilbert.npy")
    entities = parse_entities(p)
    assert "sub" in entities
    assert "desc" in entities
    # 'hilbert' (suffix, no dash) must not appear as a key
    assert "hilbert" not in entities


def test_parse_entities_empty_for_no_entities() -> None:
    assert parse_entities(Path("ieeg.vhdr")) == {}


def test_normalize_subject_value_strips_sub_prefix() -> None:
    assert normalize_subject_value("sub-01") == "01"


# ---------------------------------------------------------------------------
# build_subject_groups
# ---------------------------------------------------------------------------

IEEG_FILTERS = {"suffix": "ieeg", "extension": ".vhdr"}
BEHAVIOUR_FILTERS = [{"suffix": "beh", "extension": ".tsv"}]


class TestBuildSubjectGroupsAggregate:
    """aggregate_runs=True (default): one group per subject."""

    def test_single_subject_single_run_no_secondaries(self, tmp_path: Path) -> None:
        ieeg = _make_file(tmp_path, "sub-01_run-1_ieeg.vhdr", {"subject": "01", "run": "1"})
        ds = _mock_dataset(
            tmp_path,
            {
                tuple(sorted(IEEG_FILTERS.items())): [ieeg],
                tuple(sorted(BEHAVIOUR_FILTERS[0].items())): [],
            },
        )
        groups = build_subject_groups(ds, IEEG_FILTERS, BEHAVIOUR_FILTERS)
        assert len(groups) == 1
        assert groups[0].primary is ieeg
        assert groups[0].secondaries == []

    def test_two_runs_pooled_into_one_group(self, tmp_path: Path) -> None:
        run1 = _make_file(tmp_path, "sub-01_run-1_ieeg.vhdr", {"subject": "01", "run": "1"})
        run2 = _make_file(tmp_path, "sub-01_run-2_ieeg.vhdr", {"subject": "01", "run": "2"})
        ds = _mock_dataset(
            tmp_path,
            {
                tuple(sorted(IEEG_FILTERS.items())): [run1, run2],
                tuple(sorted(BEHAVIOUR_FILTERS[0].items())): [],
            },
        )
        groups = build_subject_groups(ds, IEEG_FILTERS, BEHAVIOUR_FILTERS)
        assert len(groups) == 1
        # The first (path-sorted) run must be primary; the other must be a secondary
        assert groups[0].primary is run1
        assert run2 in groups[0].secondaries

    def test_two_subjects_yield_two_groups(self, tmp_path: Path) -> None:
        s1 = _make_file(tmp_path, "sub-01_run-1_ieeg.vhdr", {"subject": "01", "run": "1"})
        s2 = _make_file(tmp_path, "sub-02_run-1_ieeg.vhdr", {"subject": "02", "run": "1"})
        ds = _mock_dataset(
            tmp_path,
            {
                tuple(sorted(IEEG_FILTERS.items())): [s1, s2],
                tuple(sorted(BEHAVIOUR_FILTERS[0].items())): [],
            },
        )
        groups = build_subject_groups(ds, IEEG_FILTERS, BEHAVIOUR_FILTERS)
        assert len(groups) == 2
        primaries = {g.primary for g in groups}
        assert primaries == {s1, s2}

    def test_secondary_files_attached_to_correct_subject(self, tmp_path: Path) -> None:
        ieeg01 = _make_file(tmp_path, "sub-01_run-1_ieeg.vhdr", {"subject": "01", "run": "1"})
        ieeg02 = _make_file(tmp_path, "sub-02_run-1_ieeg.vhdr", {"subject": "02", "run": "1"})
        beh01 = _make_file(tmp_path, "sub-01_beh.tsv", {"subject": "01"})
        beh02 = _make_file(tmp_path, "sub-02_beh.tsv", {"subject": "02"})
        ds = _mock_dataset(
            tmp_path,
            {
                tuple(sorted(IEEG_FILTERS.items())): [ieeg01, ieeg02],
                tuple(sorted(BEHAVIOUR_FILTERS[0].items())): [beh01, beh02],
            },
        )
        groups = build_subject_groups(ds, IEEG_FILTERS, BEHAVIOUR_FILTERS)
        grp01 = next(g for g in groups if g.primary is ieeg01)
        grp02 = next(g for g in groups if g.primary is ieeg02)
        assert beh01 in grp01.secondaries
        assert beh02 not in grp01.secondaries
        assert beh02 in grp02.secondaries

    def test_secondary_deduplicated_across_filters(self, tmp_path: Path) -> None:
        """The same secondary file returned by two filter dicts appears only once."""
        ieeg = _make_file(tmp_path, "sub-01_run-1_ieeg.vhdr", {"subject": "01", "run": "1"})
        beh = _make_file(tmp_path, "sub-01_beh.tsv", {"subject": "01"})
        extra_filter = {"suffix": "beh", "extension": ".tsv", "desc": "extra"}
        ds = _mock_dataset(
            tmp_path,
            {
                tuple(sorted(IEEG_FILTERS.items())): [ieeg],
                tuple(sorted(BEHAVIOUR_FILTERS[0].items())): [beh],
                tuple(sorted(extra_filter.items())): [beh],  # same file, second filter
            },
        )
        groups = build_subject_groups(ds, IEEG_FILTERS, [BEHAVIOUR_FILTERS[0], extra_filter])
        assert groups[0].secondaries.count(beh) == 1


class TestBuildSubjectGroupsPerRun:
    """aggregate_runs=False: one group per iEEG file."""

    def test_two_runs_yield_two_groups(self, tmp_path: Path) -> None:
        run1 = _make_file(tmp_path, "sub-01_run-1_ieeg.vhdr", {"subject": "01", "run": "1"})
        run2 = _make_file(tmp_path, "sub-01_run-2_ieeg.vhdr", {"subject": "01", "run": "2"})
        ds = _mock_dataset(
            tmp_path,
            {
                tuple(sorted(IEEG_FILTERS.items())): [run1, run2],
                tuple(sorted(BEHAVIOUR_FILTERS[0].items())): [],
            },
        )
        groups = build_subject_groups(ds, IEEG_FILTERS, BEHAVIOUR_FILTERS, aggregate_runs=False)
        assert len(groups) == 2
        primaries = {g.primary for g in groups}
        assert primaries == {run1, run2}

    def test_secondary_by_subject_and_session(self, tmp_path: Path) -> None:
        run1 = _make_file(
            tmp_path, "sub-01_ses-01_run-1_ieeg.vhdr", {"subject": "01", "session": "01", "run": "1"}
        )
        run2 = _make_file(
            tmp_path, "sub-01_ses-02_run-1_ieeg.vhdr", {"subject": "01", "session": "02", "run": "1"}
        )
        beh_ses01 = _make_file(tmp_path, "sub-01_ses-01_beh.tsv", {"subject": "01", "session": "01"})
        beh_ses02 = _make_file(tmp_path, "sub-01_ses-02_beh.tsv", {"subject": "01", "session": "02"})
        ds = _mock_dataset(
            tmp_path,
            {
                tuple(sorted(IEEG_FILTERS.items())): [run1, run2],
                tuple(sorted(BEHAVIOUR_FILTERS[0].items())): [beh_ses01, beh_ses02],
            },
        )
        groups = build_subject_groups(ds, IEEG_FILTERS, BEHAVIOUR_FILTERS, aggregate_runs=False)
        grp1 = next(g for g in groups if g.primary is run1)
        grp2 = next(g for g in groups if g.primary is run2)
        assert beh_ses01 in grp1.secondaries
        assert beh_ses02 not in grp1.secondaries
        assert beh_ses02 in grp2.secondaries
        assert beh_ses01 not in grp2.secondaries


def test_normalize_subject_value_handles_whitespace_and_case() -> None:
    assert normalize_subject_value("  SuB-XYZ  ") == "XYZ"
