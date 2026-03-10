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
    BaseProcessingResult,
    BaseProcessingWriter,
)


# ---------------------------------------------------------------------------
# Concrete stubs
# ---------------------------------------------------------------------------

@dataclass
class _DummyResult(BaseProcessingResult):
    value: Any = None


class _DummyProcessor(BaseProcessing):
    def process_group(self, group: BIDSFileGroup) -> _DummyResult:
        return _DummyResult(source_group=group, metadata={"n_files": len(group.all_files)})


class _DummyWriter(BaseProcessingWriter):
    PIPELINE_LABEL = "dummy"
    OUTPUT_SUFFIX = "dummy"
    OUTPUT_EXTENSION = ".npy"

    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        pass  # no-op for tests; output_path is already created by the base


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
    writer = _DummyWriter(tmp_path)
    result = _DummyResult(source_group=BIDSFileGroup(primary=mock_bids_file))
    out = writer.write(result)
    assert isinstance(out, Path)


def test_run_writes_and_returns_paths(mock_bids_file: BIDSFile, tmp_path: Path) -> None:
    groups = [BIDSFileGroup(primary=mock_bids_file), BIDSFileGroup(primary=mock_bids_file)]
    paths = _DummyProcessor().run(groups, _DummyWriter(tmp_path))
    assert len(paths) == 2
    assert all(isinstance(p, Path) for p in paths)


def test_run_n_jobs_parallel(mock_bids_file: BIDSFile, tmp_path: Path) -> None:
    groups = [BIDSFileGroup(primary=mock_bids_file), BIDSFileGroup(primary=mock_bids_file)]
    paths = _DummyProcessor().run(groups, _DummyWriter(tmp_path), n_jobs=2)
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
    paths = _DummyProcessor().run([mock_bids_file], _DummyWriter(tmp_path))
    assert len(paths) == 1
    assert isinstance(paths[0], Path)


def test_group_with_secondaries(mock_bids_file: BIDSFile) -> None:
    group = BIDSFileGroup(primary=mock_bids_file, secondaries=[mock_bids_file])
    results = _DummyProcessor().execute([group])
    assert results[0].metadata["n_files"] == 2
    assert results[0].source_group.primary == mock_bids_file
    assert len(results[0].source_group.secondaries) == 1
