"""
Tests for HfoSpikeDetectorProcessingResult.to_output_tree() and the
HDF5/MATLAB/TSV writer paths.

Covers:
- to_output_tree() produces the expected top-level keys
- to_output_tree() handles empty markers list
- to_output_tree() handles optional arrays (n_spk, n_osc, freq_band)
- HDF5 writer round-trip: schema keys, event count, channel names
- MATLAB writer: writes .mat, loadable, envelope structure present
- TSV writer: events.tsv and rates.tsv both written
- output_format="matlab" produces a .mat extension
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.io

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.hfo_spike_detection.params import (
    HfoSpikeDetectorWriterParams,
)
from gin_bids_py_analysis.processing.hfo_spike_detection.result import (
    HfoSpikeDetectorProcessingResult,
)
from gin_bids_py_analysis.processing.hfo_spike_detection.result_loader import (
    load_hfo_spike_detection_result,
)
from gin_bids_py_analysis.processing.hfo_spike_detection.writer import (
    HfoSpikeDetectorProcessingWriter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


def _make_bids_file(path: str) -> BIDSFile:
    return BIDSFile(
        _MockPyBIDSFile(
            path=path,
            entities={
                "subject": "01",
                "session": "01",
                "task": "rest",
                "suffix": "events",
                "extension": ".tsv",
                "datatype": "ieeg",
            },
        )
    )


def _make_marker(ch: str = "A1", event_type: str = "Ripple") -> dict:
    return {
        "onset_time_seconds": 1.25,
        "duration": 0.05,
        "channel_label": ch,
        "event_type": event_type,
        "peak_frequency_hz": 120.0,
        "sample_index": 640,
        "frequency_index": 3,
        "detection_strength": 2.5,
        "visualization_color": "#FF0000",
    }


def _make_result(
    source_path: str,
    n_channels: int = 3,
    n_events: int = 5,
    include_counts: bool = True,
    include_freq_band: bool = True,
    n_samples: int = 1024,
) -> HfoSpikeDetectorProcessingResult:
    ch_names = [f"A{i + 1}" for i in range(n_channels)]
    markers = [_make_marker(ch_names[i % n_channels]) for i in range(n_events)]
    return HfoSpikeDetectorProcessingResult(
        source_group=BIDSFileGroup(primary=_make_bids_file(source_path)),
        metadata={"montage_mode": "bipolar", "n_samples": n_samples},
        markers=markers,
        channel_names=ch_names,
        freq_band=np.array([[80, 250], [250, 500]], dtype=np.float32) if include_freq_band else np.array([]),
        n_spk=np.zeros(n_channels, dtype=np.float64) if include_counts else np.array([]),
        n_osc=np.zeros((n_channels, 2), dtype=np.float64) if include_counts else np.array([]),
        detection_charac=np.array([]),
        original_fs=512.0,
    )


def _hdf5_writer(bids_root: Path) -> HfoSpikeDetectorProcessingWriter:
    return HfoSpikeDetectorProcessingWriter(
        HfoSpikeDetectorWriterParams(bids_root=bids_root)
    )


def _matlab_writer(bids_root: Path) -> HfoSpikeDetectorProcessingWriter:
    return HfoSpikeDetectorProcessingWriter(
        HfoSpikeDetectorWriterParams(bids_root=bids_root, output_format="matlab")
    )


def _tsv_writer(bids_root: Path) -> HfoSpikeDetectorProcessingWriter:
    return HfoSpikeDetectorProcessingWriter(
        HfoSpikeDetectorWriterParams(bids_root=bids_root, output_format="tsv")
    )


# ---------------------------------------------------------------------------
# to_output_tree tests
# ---------------------------------------------------------------------------


class TestToOutputTree:
    def test_top_level_keys(self) -> None:
        result = _make_result("dummy.vhdr")
        tree = result.to_output_tree()
        assert set(tree.keys()) == {"events", "counts", "axes", "meta", "provenance"}

    def test_events_subkeys(self) -> None:
        result = _make_result("dummy.vhdr", n_events=2)
        tree = result.to_output_tree()
        events = tree["events"]  # type: ignore[index]
        for key in ("onset", "duration", "channel", "event_type", "peak_frequency",
                    "sample_index", "frequency_index", "detection_strength", "color"):
            assert key in events, f"Missing events key: {key!r}"

    def test_events_length_matches_markers(self) -> None:
        n = 7
        result = _make_result("dummy.vhdr", n_events=n)
        tree = result.to_output_tree()
        assert len(tree["events"]["onset"]) == n  # type: ignore[index]

    def test_empty_markers_gives_empty_arrays(self) -> None:
        result = _make_result("dummy.vhdr", n_events=0)
        tree = result.to_output_tree()
        assert len(tree["events"]["onset"]) == 0  # type: ignore[index]

    def test_counts_present_when_arrays_nonempty(self) -> None:
        result = _make_result("dummy.vhdr", include_counts=True)
        tree = result.to_output_tree()
        counts = tree["counts"]  # type: ignore[index]
        assert "n_spk" in counts
        assert "n_osc" in counts

    def test_counts_none_when_arrays_empty(self) -> None:
        result = _make_result("dummy.vhdr", include_counts=False)
        tree = result.to_output_tree()
        counts = tree["counts"]  # type: ignore[index]
        # None values are skip-if-none; actual value stored as None
        assert counts.get("n_spk") is None
        assert counts.get("n_osc") is None

    def test_freq_band_none_when_empty(self) -> None:
        result = _make_result("dummy.vhdr", include_freq_band=False)
        tree = result.to_output_tree()
        assert tree["axes"]["freq_band"] is None  # type: ignore[index]

    def test_meta_duration_computed_from_n_samples(self) -> None:
        result = _make_result("dummy.vhdr", n_samples=512)
        result.original_fs = 512.0
        tree = result.to_output_tree()
        assert tree["meta"]["duration_seconds"] == pytest.approx(1.0)  # type: ignore[index]

    def test_provenance_pipeline_version_passthrough(self) -> None:
        result = _make_result("dummy.vhdr")
        tree = result.to_output_tree(pipeline_version="9.9.9")
        assert tree["provenance"]["pipeline_version"] == "9.9.9"  # type: ignore[index]


# ---------------------------------------------------------------------------
# HDF5 writer tests
# ---------------------------------------------------------------------------


class TestHdf5Writer:
    def test_writes_h5_file(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), n_events=3)
        out = _hdf5_writer(tmp_path).write(result)
        assert out.suffix == ".h5"
        assert out.is_file()

    def test_hdf5_top_level_groups(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), n_events=2)
        out = _hdf5_writer(tmp_path).write(result)
        with h5py.File(out, "r") as fh:
            assert set(fh.keys()) == {"events", "counts", "axes", "meta", "provenance"}

    def test_hdf5_n_events_matches_markers(self, tmp_path: Path) -> None:
        n = 4
        result = _make_result(str(tmp_path / "dummy.vhdr"), n_events=n)
        out = _hdf5_writer(tmp_path).write(result)
        with h5py.File(out, "r") as fh:
            assert len(fh["events/onset"]) == n

    def test_hdf5_channel_names_in_axes(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), n_channels=3)
        out = _hdf5_writer(tmp_path).write(result)
        with h5py.File(out, "r") as fh:
            stored = [
                c.decode() if isinstance(c, bytes) else c
                for c in fh["axes/channel_names"][:]
            ]
        assert stored == result.channel_names

    def test_hdf5_pipeline_name_in_provenance(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"))
        out = _hdf5_writer(tmp_path).write(result)
        with h5py.File(out, "r") as fh:
            name = fh["provenance/pipeline_name"][()]
            if isinstance(name, bytes):
                name = name.decode()
            assert name == "hfo_spike_detection"

    def test_hdf5_counts_absent_when_empty(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), include_counts=False)
        out = _hdf5_writer(tmp_path).write(result)
        with h5py.File(out, "r") as fh:
            counts_grp = fh["counts"]
            assert "n_spk" not in counts_grp
            assert "n_osc" not in counts_grp

    def test_hdf5_roundtrip_via_loader(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), n_events=3)
        out = _hdf5_writer(tmp_path).write(result)

        loaded = load_hfo_spike_detection_result(out)

        assert loaded.channel_names == result.channel_names
        assert loaded.original_fs == result.original_fs
        np.testing.assert_array_equal(loaded.freq_band, result.freq_band)
        np.testing.assert_array_equal(loaded.n_spk, result.n_spk.astype(np.int64))
        assert loaded.markers == result.markers


# ---------------------------------------------------------------------------
# MATLAB writer tests
# ---------------------------------------------------------------------------


class TestMatlabWriter:
    def test_writes_mat_file(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), n_events=2)
        out = _matlab_writer(tmp_path).write(result)
        assert out.suffix == ".mat"
        assert out.is_file()

    def test_mat_file_is_loadable(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), n_events=2)
        out = _matlab_writer(tmp_path).write(result)
        mat = scipy.io.loadmat(str(out))
        assert "data" in mat

    def test_mat_n_events_matches(self, tmp_path: Path) -> None:
        n = 3
        result = _make_result(str(tmp_path / "dummy.vhdr"), n_events=n)
        out = _matlab_writer(tmp_path).write(result)
        mat = scipy.io.loadmat(str(out), squeeze_me=False)
        # onset lives at data.events.onset; stored as a row-vector (1, n_events)
        onset = mat["data"]["events"][0, 0]["onset"][0, 0]
        assert onset.size == n

    def test_mat_roundtrip_via_loader(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), n_events=2)
        out = _matlab_writer(tmp_path).write(result)

        loaded = load_hfo_spike_detection_result(out)

        assert loaded.channel_names == result.channel_names
        assert loaded.markers == result.markers


# ---------------------------------------------------------------------------
# TSV writer tests (unchanged paths should still work)
# ---------------------------------------------------------------------------


class TestTsvWriter:
    def test_writes_events_tsv(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), n_events=2)
        out = _tsv_writer(tmp_path).write(result)
        assert out.suffix == ".tsv"
        assert out.is_file()

    def test_writes_rates_tsv_alongside_events(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), n_events=2)
        _tsv_writer(tmp_path).write(result)
        tsv_files = list((tmp_path / "derivatives" / "hfo_spike_detection").rglob("*.tsv"))
        names = {f.name for f in tsv_files}
        assert any("events" in n for n in names)
        assert any("rates" in n for n in names)

    def test_tsv_roundtrip_via_loader(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), n_events=2)
        out = _tsv_writer(tmp_path).write(result)

        loaded = load_hfo_spike_detection_result(out)

        assert loaded.channel_names == result.channel_names[:2]
        assert loaded.markers == result.markers



