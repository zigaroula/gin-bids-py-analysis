"""
Contract tests for BaseProcessing, BaseProcessingResult, BaseProcessingWriter.

These tests use lightweight concrete stubs to verify that:
  - the abstract base classes enforce their contracts
  - the intended usage pattern works end-to-end
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.base import (
    BaseProcessing,
    BaseProcessingParams,
    BaseProcessingResult,
    BaseProcessingWriter,
    BaseWriterParams,
)


# ---------------------------------------------------------------------------
# Concrete stubs
# ---------------------------------------------------------------------------

@dataclass
class _DummyResult(BaseProcessingResult):
    value: Any = None


class _DummyProcessor(BaseProcessing):
    def process_group(self, group: BIDSFileGroup, progress_tracking_position: int = 0) -> _DummyResult:
        return _DummyResult(source_group=group, metadata={"n_files": len(group.all_files)})


class _ParamsProcessor(BaseProcessing):
    """Stub processor that carries a BaseProcessingParams instance."""

    def __init__(self, params: BaseProcessingParams) -> None:
        self.params = params

    def process_group(self, group: BIDSFileGroup, progress_tracking_position: int = 0) -> _DummyResult:
        return _DummyResult(source_group=group)


class _DummyWriter(BaseProcessingWriter):
    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        pass  # no-op for tests; output_path is already created by the base


def _dummy_writer(bids_root: Path) -> _DummyWriter:
    """Helper: build a _DummyWriter with minimal BaseWriterParams."""
    params = BaseWriterParams(
        bids_root=bids_root,
        pipeline_label="dummy",
        output_modality="dummy",
        output_description="dummy",
        output_suffix="dummy",
        output_extension=".npy",
    )
    return _DummyWriter(params)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_processor_returns_result(mock_bids_file: BIDSFile) -> None:
    group = BIDSFileGroup(primary=mock_bids_file)
    results = _DummyProcessor().execute([group])
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], BaseProcessingResult)
    assert results[0].source_group.primary == mock_bids_file
    assert results[0].metadata["n_files"] == 1


def test_result_metadata_defaults_to_empty(mock_bids_file: BIDSFile) -> None:
    result = _DummyResult(source_group=BIDSFileGroup(primary=mock_bids_file))
    assert result.metadata == {}


def test_result_value_field(mock_bids_file: BIDSFile) -> None:
    result = _DummyResult(source_group=BIDSFileGroup(primary=mock_bids_file), value=42)
    assert result.value == 42


def test_writer_returns_path(mock_bids_file: BIDSFile, tmp_path: Path) -> None:
    writer = _dummy_writer(tmp_path)
    result = _DummyResult(source_group=BIDSFileGroup(primary=mock_bids_file))
    out = writer.write(result)
    assert isinstance(out, Path)


def test_run_writes_and_returns_paths(mock_bids_file: BIDSFile, tmp_path: Path) -> None:
    groups = [BIDSFileGroup(primary=mock_bids_file), BIDSFileGroup(primary=mock_bids_file)]
    paths = _DummyProcessor().run(groups, _dummy_writer(tmp_path))
    assert len(paths) == 2
    assert all(isinstance(p, Path) for p in paths)


def test_run_n_jobs_parallel(mock_bids_file: BIDSFile, tmp_path: Path) -> None:
    groups = [BIDSFileGroup(primary=mock_bids_file), BIDSFileGroup(primary=mock_bids_file)]
    paths = _DummyProcessor().run(groups, _dummy_writer(tmp_path), n_jobs=2)
    assert len(paths) == 2


def test_cannot_instantiate_abstract_processor() -> None:
    with pytest.raises(TypeError):
        BaseProcessing()  # type: ignore[abstract]


def test_cannot_instantiate_abstract_writer() -> None:
    with pytest.raises(TypeError):
        BaseProcessingWriter()  # type: ignore[abstract]


def test_processor_accepts_multiple_files(mock_bids_file: BIDSFile) -> None:
    groups = [BIDSFileGroup(primary=mock_bids_file), BIDSFileGroup(primary=mock_bids_file)]
    results = _DummyProcessor().execute(groups)
    assert len(results) == 2
    assert all(r.metadata["n_files"] == 1 for r in results)


def test_execute_n_jobs_parallel(mock_bids_file: BIDSFile) -> None:
    groups = [BIDSFileGroup(primary=mock_bids_file), BIDSFileGroup(primary=mock_bids_file)]
    results = _DummyProcessor().execute(groups, n_jobs=2)
    assert len(results) == 2


def test_execute_accepts_bare_bids_files(mock_bids_file: BIDSFile) -> None:
    # Bare BIDSFile objects should be auto-wrapped into single-file groups
    results = _DummyProcessor().execute([mock_bids_file, mock_bids_file])
    assert len(results) == 2
    assert all(isinstance(r, BaseProcessingResult) for r in results)


def test_run_accepts_bare_bids_files(mock_bids_file: BIDSFile, tmp_path: Path) -> None:
    paths = _DummyProcessor().run([mock_bids_file], _dummy_writer(tmp_path))
    assert len(paths) == 1
    assert isinstance(paths[0], Path)


def test_group_with_secondaries(mock_bids_file: BIDSFile) -> None:
    group = BIDSFileGroup(primary=mock_bids_file, secondaries=[mock_bids_file])
    results = _DummyProcessor().execute([group])
    assert results[0].metadata["n_files"] == 2
    assert results[0].source_group.primary == mock_bids_file
    assert len(results[0].source_group.secondaries) == 1


# ---------------------------------------------------------------------------
# dataset_description.json tests
# ---------------------------------------------------------------------------

import json


def test_run_creates_dataset_description(mock_bids_file: BIDSFile, tmp_path: Path) -> None:
    """run() must write dataset_description.json before the processing loop."""
    _DummyProcessor().run([mock_bids_file], _dummy_writer(tmp_path))

    desc_path = tmp_path / "derivatives" / "dummy" / "dataset_description.json"
    assert desc_path.exists(), "dataset_description.json was not created"

    desc = json.loads(desc_path.read_text(encoding="utf-8"))
    assert desc["Name"] == "dummy"
    assert desc["BIDSVersion"] == "1.7.0"
    assert desc["DatasetType"] == "derivative"
    assert desc["GeneratedBy"][0]["Name"] == "gin-bids-py-analysis"
    assert "Version" in desc["GeneratedBy"][0]


def test_run_dataset_description_overwrites(mock_bids_file: BIDSFile, tmp_path: Path) -> None:
    """Calling run() a second time must overwrite the existing file."""
    writer = _dummy_writer(tmp_path)
    _DummyProcessor().run([mock_bids_file], writer)
    _DummyProcessor().run([mock_bids_file], writer)  # second call must not raise

    desc_path = tmp_path / "derivatives" / "dummy" / "dataset_description.json"
    assert desc_path.exists()


def test_run_dataset_description_includes_params(mock_bids_file: BIDSFile, tmp_path: Path) -> None:
    """When the processor has a .params attribute, Parameters key is written."""

    class _MinimalParams(BaseProcessingParams):
        my_field: int = 42

    processor = _ParamsProcessor(params=_MinimalParams())
    processor.run([mock_bids_file], _dummy_writer(tmp_path))

    desc_path = tmp_path / "derivatives" / "dummy" / "dataset_description.json"
    desc = json.loads(desc_path.read_text(encoding="utf-8"))
    params_in_file = desc["GeneratedBy"][0]["Parameters"]
    assert params_in_file["my_field"] == 42


def test_run_dataset_description_no_params_key_when_none(mock_bids_file: BIDSFile, tmp_path: Path) -> None:
    """When processor has no .params, the Parameters key must be absent."""
    _DummyProcessor().run([mock_bids_file], _dummy_writer(tmp_path))

    desc_path = tmp_path / "derivatives" / "dummy" / "dataset_description.json"
    desc = json.loads(desc_path.read_text(encoding="utf-8"))
    assert "Parameters" not in desc["GeneratedBy"][0]
