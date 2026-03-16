"""
Unit tests for HilbertProcessingWriter — BrainVision output path.

Covers:
- Correct number of files written per smoothing window (.vhdr/.vmrk/.eeg)
- Returned path is a .vhdr header
- Channel names are embedded in the .vhdr header
- File naming encodes the per-window desc entity (desc-hilbertsm{N})
- Marker import: MNE Annotations carried in the result are written to .vmrk
- HDF5 path is unaffected (regression)
"""

from __future__ import annotations

import numpy as np
import pytest

from mne import Annotations
from pathlib import Path

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.hilbert.params import HilbertWriterParams
from gin_bids_py_analysis.processing.hilbert.result import HilbertProcessingResult
from gin_bids_py_analysis.processing.hilbert.writer import HilbertProcessingWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockPyBIDSFile:
    """Minimal pybids-compatible stand-in."""

    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


def _make_bids_file(path: str) -> BIDSFile:
    mock = _MockPyBIDSFile(
        path=path,
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


def _make_result(
    source_path: str,
    n_channels: int = 3,
    n_samples: int = 100,
    windows: list[int] | None = None,
    original_events: Annotations | None = None,
) -> HilbertProcessingResult:
    """Build a minimal HilbertProcessingResult with synthetic data."""
    if windows is None:
        windows = [0, 250]
    rng = np.random.default_rng(seed=42)
    smoothed = {
        w: rng.standard_normal((n_channels, n_samples)).astype(np.float32)
        for w in windows
    }
    ch_names = [f"A{i + 1}" for i in range(n_channels)]
    source_file = _make_bids_file(source_path)
    return HilbertProcessingResult(
        source_group=BIDSFileGroup(primary=source_file),
        smoothed=smoothed,
        channel_names=ch_names,
        bins=[50.0, 60.0, 70.0],
        downsampled_fs=64.0,
        original_fs=1000.0,
        original_events=original_events,
        metadata={"unit": "percent", "montage_mode": "mono", "centered": False},
    )


def _bv_writer(bids_root: Path) -> HilbertProcessingWriter:
    return HilbertProcessingWriter(
        HilbertWriterParams(bids_root=bids_root, output_format="brainvision")
    )


# ---------------------------------------------------------------------------
# BrainVision file-writing tests
# ---------------------------------------------------------------------------


class TestBrainVisionWriter:
    def test_writes_three_files_per_window(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.nii"), windows=[0, 250])
        _bv_writer(tmp_path).write(result)

        written = list((tmp_path / "derivatives" / "hilbert").rglob("*"))
        extensions = [f.suffix for f in written if f.is_file()]
        assert extensions.count(".vhdr") == 2
        assert extensions.count(".vmrk") == 2
        assert extensions.count(".eeg") == 2

    def test_returns_vhdr_path(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.nii"), windows=[250, 0, 500])
        out = _bv_writer(tmp_path).write(result)

        assert out.suffix == ".vhdr"

    def test_file_names_contain_per_window_desc_label(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.nii"), windows=[0, 1000])
        _bv_writer(tmp_path).write(result)

        vhdr_files = list((tmp_path / "derivatives" / "hilbert").rglob("*.vhdr"))
        names = {f.name for f in vhdr_files}
        assert any("desc-hilbertsm0" in n for n in names)
        assert any("desc-hilbertsm1000" in n for n in names)

    def test_channel_names_in_vhdr_header(self, tmp_path: Path) -> None:
        n_ch = 3
        result = _make_result(str(tmp_path / "dummy.nii"), n_channels=n_ch, windows=[0])
        _bv_writer(tmp_path).write(result)

        vhdr = next((tmp_path / "derivatives" / "hilbert").rglob("*.vhdr"))
        text = vhdr.read_text(encoding="utf-8")
        for name in result.channel_names:
            assert name in text, f"Channel {name!r} not found in .vhdr header"

    def test_eeg_extension_alias_also_writes_bv_files(self, tmp_path: Path) -> None:
        """output_format='brainvision' should trigger BrainVision output."""
        result = _make_result(str(tmp_path / "dummy.nii"), windows=[0])
        writer = HilbertProcessingWriter(
            HilbertWriterParams(bids_root=tmp_path, output_format="brainvision")
        )
        writer.write(result)

        vhdr_files = list((tmp_path / "derivatives" / "hilbert").rglob("*.vhdr"))
        assert len(vhdr_files) == 1

    def test_single_window_writes_one_triplet(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.nii"), windows=[0])
        _bv_writer(tmp_path).write(result)

        vhdr_files = list((tmp_path / "derivatives" / "hilbert").rglob("*.vhdr"))
        assert len(vhdr_files) == 1

    def test_output_directory_is_created(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.nii"), windows=[0])
        _bv_writer(tmp_path).write(result)

        expected_dir = (
            tmp_path / "derivatives" / "hilbert" / "sub-01" / "ses-01" / "ieeg"
        )
        assert expected_dir.is_dir()

    def test_written_bv_file_contains_markers(self, tmp_path: Path) -> None:
        """Annotations carried in the result are written to the output .vmrk."""
        events = Annotations(
            onset=[0.512], duration=[0.0], description=["Stimulus/S  1"]
        )
        result = _make_result(
            str(tmp_path / "dummy.nii"), windows=[0], original_events=events
        )
        _bv_writer(tmp_path).write(result)

        out_vmrk = next((tmp_path / "derivatives" / "hilbert").rglob("*.vmrk"))
        vmrk_text = out_vmrk.read_text(encoding="utf-8")
        assert "Stimulus" in vmrk_text


# ---------------------------------------------------------------------------
# HDF5 regression test
# ---------------------------------------------------------------------------


class TestHdf5Regression:
    def test_default_extension_still_writes_h5(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.nii"), windows=[0, 250])
        writer = HilbertProcessingWriter(HilbertWriterParams(bids_root=tmp_path))
        out = writer.write(result)
        assert out.suffix == ".h5"
        assert out.is_file()
