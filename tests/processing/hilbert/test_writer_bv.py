"""
Unit tests for HilbertProcessingWriter — BrainVision output path.

Covers:
- Correct number of files written per smoothing window (.vhdr/.vmrk/.eeg)
- Returned path is a .vhdr header of the smallest (first) window
- Channel names are embedded in the .vhdr header
- File naming includes the _smwin-{N}ms label
- _read_bv_events returns None for non-BrainVision source files
- Marker import: annotations from a real .vhdr source are remapped and written
- HDF5 path is unaffected (regression)
"""

from __future__ import annotations

import numpy as np
import pytest

from pathlib import Path

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.hilbert.params import HilbertWriterParams
from gin_bids_py_analysis.processing.hilbert.result import HilbertProcessingResult
from gin_bids_py_analysis.processing.hilbert.writer import (
    HilbertProcessingWriter,
    _read_bv_events,
)


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
        metadata={"unit": "percent", "montage_mode": "mono", "centered": False},
    )


def _bv_writer(bids_root: Path) -> HilbertProcessingWriter:
    return HilbertProcessingWriter(
        HilbertWriterParams(bids_root=bids_root, output_extension=".vhdr")
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

    def test_returns_smallest_window_vhdr_path(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.nii"), windows=[250, 0, 500])
        out = _bv_writer(tmp_path).write(result)

        assert out.suffix == ".vhdr"
        assert "smwin-0ms" in out.name

    def test_file_names_contain_smwin_label(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.nii"), windows=[0, 1000])
        _bv_writer(tmp_path).write(result)

        vhdr_files = list((tmp_path / "derivatives" / "hilbert").rglob("*.vhdr"))
        names = {f.name for f in vhdr_files}
        assert any("smwin-0ms" in n for n in names)
        assert any("smwin-1000ms" in n for n in names)

    def test_channel_names_in_vhdr_header(self, tmp_path: Path) -> None:
        n_ch = 3
        result = _make_result(str(tmp_path / "dummy.nii"), n_channels=n_ch, windows=[0])
        _bv_writer(tmp_path).write(result)

        vhdr = next((tmp_path / "derivatives" / "hilbert").rglob("*.vhdr"))
        text = vhdr.read_text(encoding="utf-8")
        for name in result.channel_names:
            assert name in text, f"Channel {name!r} not found in .vhdr header"

    def test_eeg_extension_alias_also_produces_vhdr(self, tmp_path: Path) -> None:
        """.eeg extension should still produce .vhdr output."""
        result = _make_result(str(tmp_path / "dummy.nii"), windows=[0])
        writer = HilbertProcessingWriter(
            HilbertWriterParams(bids_root=tmp_path, output_extension=".eeg")
        )
        out = writer.write(result)
        assert out.suffix == ".vhdr"

    def test_single_window_writes_one_triplet(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.nii"), windows=[0])
        _bv_writer(tmp_path).write(result)

        vhdr_files = list((tmp_path / "derivatives" / "hilbert").rglob("*.vhdr"))
        assert len(vhdr_files) == 1

    def test_output_directory_is_created(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.nii"), windows=[0])
        _bv_writer(tmp_path).write(result)

        expected_dir = (
            tmp_path / "derivatives" / "hilbert" / "sub-01" / "ses-01" / "hilbert"
        )
        assert expected_dir.is_dir()


# ---------------------------------------------------------------------------
# _read_bv_events tests
# ---------------------------------------------------------------------------


class TestReadBvEvents:
    def test_returns_none_for_non_vhdr(self, tmp_path: Path) -> None:
        assert _read_bv_events(tmp_path / "data.h5", 1000.0, 64.0) is None
        assert _read_bv_events(tmp_path / "data.nii", 1000.0, 64.0) is None
        assert _read_bv_events(tmp_path / "data.edf", 1000.0, 64.0) is None

    def test_returns_none_for_empty_vmrk(self, tmp_path: Path) -> None:
        """A .vhdr file with no markers should return None."""
        vhdr, _vmrk, _eeg = _write_minimal_bv(tmp_path, "empty", n_samples=100, markers=[])
        result = _read_bv_events(vhdr, original_fs=1000.0, downsampled_fs=64.0)
        assert result is None

    def test_remaps_marker_onset_to_downsampled_rate(self, tmp_path: Path) -> None:
        """Marker onset at sample 500 @ 1000 Hz → sample 32 @ 64 Hz."""
        markers = [("Stimulus/S  1", 500, 1)]  # (description, onset_samples, duration)
        vhdr, _vmrk, _eeg = _write_minimal_bv(tmp_path, "marked", n_samples=1000, markers=markers)
        events = _read_bv_events(vhdr, original_fs=1000.0, downsampled_fs=64.0)

        assert events is not None
        assert len(events) == 1
        # onset_samples = round(500/1000 * 64) = round(32.0) = 32
        assert events[0]["onset"] == 32

    def test_event_description_preserved(self, tmp_path: Path) -> None:
        markers = [("Stimulus/S  1", 100, 0), ("Response/R  2", 200, 0)]
        vhdr, _vmrk, _eeg = _write_minimal_bv(tmp_path, "multi", n_samples=500, markers=markers)
        events = _read_bv_events(vhdr, original_fs=1000.0, downsampled_fs=64.0)

        assert events is not None
        assert len(events) == 2
        # _read_bv_events converts "S  1" / "R  2" to integers for pybv compatibility.
        descriptions = {e["description"] for e in events}
        assert 1 in descriptions
        assert 2 in descriptions

    def test_written_bv_file_contains_markers(self, tmp_path: Path) -> None:
        """End-to-end: markers from source .vhdr appear in the output .vmrk."""
        markers = [("Stimulus/S  1", 512, 0)]
        src_vhdr, _vmrk, _eeg = _write_minimal_bv(
            tmp_path / "src", "source", n_samples=1000, markers=markers
        )

        result = _make_result(str(src_vhdr), windows=[0])
        _bv_writer(tmp_path).write(result)

        out_vmrk = next(
            (tmp_path / "derivatives" / "hilbert").rglob("*.vmrk")
        )
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


# ---------------------------------------------------------------------------
# Minimal BrainVision file builder (for marker tests)
# ---------------------------------------------------------------------------


def _write_minimal_bv(
    folder: Path,
    stem: str,
    n_samples: int,
    markers: list[tuple[str, int, int]],
    n_ch: int = 2,
    sfreq: int = 1000,
) -> tuple[Path, Path, Path]:
    """Write a minimal valid BrainVision file triplet in *folder*.

    Args:
        folder:   Directory to write into (created if missing).
        stem:     Base filename without extension.
        n_samples: Number of time points in the binary data.
        markers:  List of ``("Type/Description", onset_sample, duration_sample)``
                  tuples.  The description is split on the first ``"/"`` to
                  produce the BrainVision ``Type,Description`` fields, e.g.
                  ``"Stimulus/S  1"`` → ``Mk1=Stimulus,S  1,onset,dur,0``.
        n_ch:     Number of channels.
        sfreq:    Sampling frequency in Hz.

    Returns:
        ``(vhdr_path, vmrk_path, eeg_path)``
    """
    folder.mkdir(parents=True, exist_ok=True)
    vhdr = folder / f"{stem}.vhdr"
    vmrk = folder / f"{stem}.vmrk"
    eeg = folder / f"{stem}.eeg"

    # Build header — avoid textwrap.dedent with multi-line interpolated values.
    ch_lines = "\n".join(f"Ch{i + 1}=C{i + 1},,1,µV" for i in range(n_ch))
    vhdr.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "; Generated by test helper\n\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        f"DataFile={stem}.eeg\n"
        f"MarkerFile={stem}.vmrk\n"
        "DataFormat=BINARY\n"
        "DataOrientation=MULTIPLEXED\n"
        f"NumberOfChannels={n_ch}\n"
        f"DataPoints={n_samples}\n"
        f"SamplingInterval={int(1e6 / sfreq)}\n\n"
        "[Binary Infos]\n"
        "BinaryFormat=IEEE_FLOAT_32\n\n"
        "[Channel Infos]\n"
        + ch_lines + "\n",
        encoding="utf-8",
    )

    # Build marker file — write proper BrainVision "Type,Description" format.
    # Each marker is "Mk{N}=Type,Description,onset,duration,0".
    mk_lines = []
    for i, (full_desc, onset, dur) in enumerate(markers):
        if "/" in full_desc:
            mk_type, mk_desc = full_desc.split("/", 1)
        else:
            mk_type, mk_desc = "Stimulus", full_desc
        mk_lines.append(f"Mk{i + 1}={mk_type},{mk_desc},{onset},{dur},0")

    vmrk.write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        f"DataFile={stem}.eeg\n\n"
        "[Marker Infos]\n"
        + "\n".join(mk_lines) + "\n",
        encoding="utf-8",
    )

    # Binary data: n_ch × n_samples float32 samples (multiplexed layout)
    data = np.zeros((n_ch, n_samples), dtype=np.float32)
    eeg.write_bytes(data.T.tobytes())  # MULTIPLEXED = time-major in BrainVision

    return vhdr, vmrk, eeg
