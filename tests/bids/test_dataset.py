"""
Contract tests for BIDSDataset.

Tests that require a real pybids-indexed dataset are marked ``skip`` until
pybids indexing is verified in the environment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gin_bids_py_analysis.bids import BIDSDataset, BIDSFile, BIDSSubject


@pytest.mark.skip(reason="Requires pybids indexing — enable once pybids is installed")
def test_get_files_returns_bids_file_instances(bids_root: Path) -> None:
    ds = BIDSDataset(bids_root, derivatives=False)
    files = ds.get_files(suffix="ieeg")
    assert len(files) > 0
    assert all(isinstance(f, BIDSFile) for f in files)


@pytest.mark.skip(reason="Requires pybids indexing — enable once pybids is installed")
def test_get_subjects_returns_bids_subjects(bids_root: Path) -> None:
    ds = BIDSDataset(bids_root, derivatives=False)
    subjects = ds.get_subjects()
    assert all(isinstance(s, BIDSSubject) for s in subjects)
    assert {s.subject_id for s in subjects} == {"01", "02"}


@pytest.mark.skip(reason="Requires pybids indexing — enable once pybids is installed")
def test_get_files_entity_filter(bids_root: Path) -> None:
    ds = BIDSDataset(bids_root, derivatives=False)
    files = ds.get_files(subject="01", suffix="ieeg")
    assert all(f["subject"] == "01" for f in files)


@pytest.mark.skip(reason="Requires pybids indexing — enable once pybids is installed")
def test_get_subject_by_id(bids_root: Path) -> None:
    ds = BIDSDataset(bids_root, derivatives=False)
    subject = ds.get_subject("01")
    assert isinstance(subject, BIDSSubject)
    assert subject.subject_id == "01"


@pytest.mark.skip(reason="Requires pybids indexing — enable once pybids is installed")
def test_root_property(bids_root: Path) -> None:
    ds = BIDSDataset(bids_root, derivatives=False)
    assert ds.root == Path(bids_root)
