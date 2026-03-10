"""
Tests for BIDSFile entity access.

These tests use the ``mock_bids_file`` fixture (no pybids indexing required).
"""

from __future__ import annotations

import pytest

from gin_bids_py_analysis.bids.file import BIDSFile


def test_entity_access_via_getitem(mock_bids_file: BIDSFile) -> None:
    assert mock_bids_file["subject"] == "01"
    assert mock_bids_file["task"] == "rest"


def test_entity_access_via_get(mock_bids_file: BIDSFile) -> None:
    assert mock_bids_file.get("run") == "1"
    assert mock_bids_file.get("nonexistent", "default") == "default"
    assert mock_bids_file.get("nonexistent") is None


def test_entities_returns_dict(mock_bids_file: BIDSFile) -> None:
    ents = mock_bids_file.entities
    assert isinstance(ents, dict)
    assert "subject" in ents


def test_entities_is_a_copy(mock_bids_file: BIDSFile) -> None:
    ents = mock_bids_file.entities
    ents["subject"] = "mutated"
    assert mock_bids_file["subject"] == "01"


def test_suffix_property(mock_bids_file: BIDSFile) -> None:
    assert mock_bids_file.suffix == "ieeg"


def test_extension_property(mock_bids_file: BIDSFile) -> None:
    assert mock_bids_file.extension == ".vhdr"


def test_datatype_property(mock_bids_file: BIDSFile) -> None:
    assert mock_bids_file.datatype == "ieeg"


def test_missing_entity_raises_key_error(mock_bids_file: BIDSFile) -> None:
    with pytest.raises(KeyError, match="nonexistent_entity"):
        _ = mock_bids_file["nonexistent_entity"]


def test_repr(mock_bids_file: BIDSFile) -> None:
    assert "BIDSFile" in repr(mock_bids_file)
