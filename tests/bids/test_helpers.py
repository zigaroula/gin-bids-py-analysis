"""
Tests for BIDS path building and entity parsing helpers.
"""

from __future__ import annotations

from pathlib import Path

from gin_bids_py_analysis.bids.helpers import build_bids_path, parse_entities


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
