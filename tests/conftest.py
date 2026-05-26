"""
Shared pytest fixtures for gin-bids-py-analysis tests.

Fixtures defined here are available to all test modules without explicit import.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gin_bids_py_analysis.bids.file import BIDSFile


# ---------------------------------------------------------------------------
# Synthetic BIDS tree
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@pytest.fixture()
def bids_root(tmp_path: Path) -> Path:
    """
    Create a minimal synthetic BIDS dataset on a temporary path.

    Structure::

        bids_dataset/
        ├── dataset_description.json
        ├── participants.tsv
        ├── sub-01/ses-01/ieeg/sub-01_ses-01_task-rest_run-1_ieeg.{vhdr,json}
        └── sub-02/ses-01/ieeg/sub-02_ses-01_task-rest_run-1_ieeg.{vhdr,json}
    """
    root = tmp_path / "bids_dataset"
    root.mkdir()

    _write_json(root / "dataset_description.json", {
        "Name": "Test Dataset",
        "BIDSVersion": "1.11.1",
    })
    (root / "participants.tsv").write_text(
        "participant_id\n" + "\n".join(["sub-01", "sub-02"]),
        encoding="utf-8",
    )

    for sub in ("01", "02"):
        ieeg_dir = root / f"sub-{sub}" / "ses-01" / "ieeg"
        ieeg_dir.mkdir(parents=True)
        stem = f"sub-{sub}_ses-01_task-rest_run-1_ieeg"
        (ieeg_dir / f"{stem}.vhdr").write_text("", encoding="utf-8")
        _write_json(ieeg_dir / f"{stem}.json", {"SamplingFrequency": 1000})

    return root


# ---------------------------------------------------------------------------
# Mock BIDSFile (no real pybids required)
# ---------------------------------------------------------------------------

class _MockPyBIDSFile:
    """Minimal stand-in for a pybids BIDSFile — used to avoid pybids indexing in unit tests."""

    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


@pytest.fixture()
def mock_bids_file(tmp_path: Path) -> BIDSFile:
    """
    Return a :class:`BIDSFile` wrapping a lightweight mock pybids object.

    Entities present: ``subject``, ``session``, ``task``, ``run``,
    ``suffix``, ``extension``, ``datatype``.
    """
    fake_path = str(tmp_path / "sub-01_ses-01_task-rest_run-1_ieeg.vhdr")
    mock = _MockPyBIDSFile(
        path=fake_path,
        entities={
            "subject": "01",
            "session": "01",
            "task": "rest",
            "run": "1",
            "suffix": "ieeg",
            "extension": ".vhdr",
            "datatype": "ieeg",
        },
    )
    return BIDSFile(mock)



