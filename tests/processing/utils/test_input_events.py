from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from mne import Annotations

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.utils.input_events import resolve_input_events


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


def _make_file(
    path: Path,
    *,
    suffix: str,
    extension: str,
    data: object | None = None,
    task: str = "rest",
    run: str = "1",
) -> BIDSFile:
    file = BIDSFile(
        _MockPyBIDSFile(
            str(path),
            {
                "subject": "01",
                "session": "01",
                "task": task,
                "run": run,
                "suffix": suffix,
                "extension": extension,
                "datatype": "ieeg",
            },
        )
    )
    if data is not None:
        file.attach_data(data)
    return file


def _raw_with_annotations() -> SimpleNamespace:
    return SimpleNamespace(
        annotations=Annotations(
            onset=[1.0],
            duration=[0.0],
            description=["Stimulus/S 1"],
        )
    )


def _events_rows() -> list[dict[str, str]]:
    return [
        {
            "onset": "1.234567",
            "duration": "0",
            "trial_type": "Trigger",
            "code": "11",
        }
    ]


def test_annotations_mode_ignores_secondary_events_tsv(tmp_path: Path) -> None:
    primary = _make_file(tmp_path / "sub-01_ses-01_task-rest_run-1_ieeg.vhdr", suffix="ieeg", extension=".vhdr")
    events = _make_file(
        tmp_path / "sub-01_ses-01_task-rest_run-1_events.tsv",
        suffix="events",
        extension=".tsv",
        data=_events_rows(),
    )
    raw = _raw_with_annotations()

    resolved = resolve_input_events(
        BIDSFileGroup(primary=primary, secondaries=[events]),
        raw,
        "annotations",
    )

    assert resolved.events is raw.annotations
    assert resolved.source_resolved == "annotations"
    assert resolved.onset_precision == "sample_quantized"
    assert resolved.event_file is None


def test_events_tsv_mode_loads_matching_secondary(tmp_path: Path) -> None:
    primary = _make_file(tmp_path / "sub-01_ses-01_task-rest_run-1_ieeg.vhdr", suffix="ieeg", extension=".vhdr")
    events = _make_file(
        tmp_path / "sub-01_ses-01_task-rest_run-1_events.tsv",
        suffix="events",
        extension=".tsv",
        data=_events_rows(),
    )

    resolved = resolve_input_events(
        BIDSFileGroup(primary=primary, secondaries=[events]),
        _raw_with_annotations(),
        "events_tsv",
    )

    assert resolved.source_resolved == "events_tsv"
    assert resolved.onset_precision == "exact_time"
    assert resolved.event_file == events.path
    assert resolved.events == [
        {
            "onset": 1.234567,
            "duration": 0.0,
            "description": "Stimulus/S 11",
        }
    ]


def test_events_tsv_mode_requires_secondary(tmp_path: Path) -> None:
    primary = _make_file(tmp_path / "sub-01_ses-01_task-rest_run-1_ieeg.vhdr", suffix="ieeg", extension=".vhdr")

    with pytest.raises(FileNotFoundError, match="no matching _events.tsv"):
        resolve_input_events(
            BIDSFileGroup(primary=primary),
            _raw_with_annotations(),
            "events_tsv",
        )


def test_auto_prefers_events_tsv_when_present(tmp_path: Path) -> None:
    primary = _make_file(tmp_path / "sub-01_ses-01_task-rest_run-1_ieeg.vhdr", suffix="ieeg", extension=".vhdr")
    events = _make_file(
        tmp_path / "sub-01_ses-01_task-rest_run-1_events.tsv",
        suffix="events",
        extension=".tsv",
        data=_events_rows(),
    )

    resolved = resolve_input_events(
        BIDSFileGroup(primary=primary, secondaries=[events]),
        _raw_with_annotations(),
        "auto",
    )

    assert resolved.source_resolved == "events_tsv"
    assert resolved.events[0]["onset"] == 1.234567


def test_auto_falls_back_to_annotations_without_secondary(tmp_path: Path) -> None:
    primary = _make_file(tmp_path / "sub-01_ses-01_task-rest_run-1_ieeg.vhdr", suffix="ieeg", extension=".vhdr")
    raw = _raw_with_annotations()

    resolved = resolve_input_events(
        BIDSFileGroup(primary=primary),
        raw,
        "auto",
    )

    assert resolved.events is raw.annotations
    assert resolved.source_resolved == "annotations"


def test_multiple_matching_events_tsv_files_are_ambiguous(tmp_path: Path) -> None:
    primary = _make_file(tmp_path / "sub-01_ses-01_task-rest_run-1_ieeg.vhdr", suffix="ieeg", extension=".vhdr")
    events_a = _make_file(
        tmp_path / "sub-01_ses-01_task-rest_run-1_events.tsv",
        suffix="events",
        extension=".tsv",
        data=_events_rows(),
    )
    events_b = _make_file(
        tmp_path / "sub-01_ses-01_task-rest_run-1_desc-copy_events.tsv",
        suffix="events",
        extension=".tsv",
        data=_events_rows(),
    )

    with pytest.raises(ValueError, match="Ambiguous _events.tsv secondary"):
        resolve_input_events(
            BIDSFileGroup(primary=primary, secondaries=[events_a, events_b]),
            _raw_with_annotations(),
            "events_tsv",
        )
