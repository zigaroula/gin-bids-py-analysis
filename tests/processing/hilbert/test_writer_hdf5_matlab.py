"""
Tests for HilbertProcessingResult.to_output_tree() and the HDF5/MATLAB writer paths.

Covers:
- to_output_tree() produces the expected top-level keys and shapes
- to_output_tree() sorts smoothing windows ascending
- to_output_tree() embeds optional metadata keys only when present
- HDF5 writer uses the output tree (schema round-trip)
- MATLAB writer writes a .mat file with the expected structure
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
from gin_bids_py_analysis.processing.hilbert.params import (
    HilbertWriterParams,
    NormalizationMode,
)
from gin_bids_py_analysis.processing.hilbert.result_loader import load_hilbert_result
from gin_bids_py_analysis.processing.hilbert.result import HilbertProcessingResult
from gin_bids_py_analysis.processing.hilbert.writer import HilbertProcessingWriter


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
                "run": "1",
                "suffix": "ieeg",
                "extension": ".vhdr",
                "datatype": "ieeg",
            },
        )
    )


def _make_result(
    source_path: str,
    n_channels: int = 3,
    n_samples: int = 100,
    windows: list[int] | None = None,
    extra_metadata: dict | None = None,
    events: list[dict] | None = None,
) -> HilbertProcessingResult:
    if windows is None:
        windows = [0, 250]
    rng = np.random.default_rng(seed=0)
    smoothed = {
        w: rng.standard_normal((n_channels, n_samples)).astype(np.float32)
        for w in windows
    }
    ch_names = [f"A{i + 1}" for i in range(n_channels)]
    meta = {
        "unit": NormalizationMode.PERCENT.unit,
        "scale_factor": NormalizationMode.PERCENT.scale_factor,
        "montage_mode": "mono",
        "centered": False,
    }
    if extra_metadata:
        meta.update(extra_metadata)
    return HilbertProcessingResult(
        source_group=BIDSFileGroup(primary=_make_bids_file(source_path)),
        smoothed=smoothed,
        channel_names=ch_names,
        bins=[50.0, 60.0, 70.0],
        downsampled_fs=64.0,
        original_fs=1000.0,
        metadata=meta,
        events=events,
    )


# ---------------------------------------------------------------------------
# to_output_tree tests
# ---------------------------------------------------------------------------


class TestToOutputTree:
    def test_top_level_keys(self) -> None:
        result = _make_result("dummy.vhdr", windows=[0])
        tree = result.to_output_tree()
        assert set(tree.keys()) == {"data", "axes", "meta", "provenance"}

    def test_events_are_serialized_when_present(self) -> None:
        result = _make_result(
            "dummy.vhdr",
            windows=[0],
            events=[
                {
                    "onset": 15214,
                    "duration": 0,
                    "type": "Stimulus",
                    "description": 11,
                }
            ],
        )
        tree = result.to_output_tree()

        assert "events" in tree
        events = tree["events"]  # type: ignore[index]
        assert list(events["onset"]) == [15214]
        assert list(events["duration"]) == [0]
        assert list(events["type"]) == ["Stimulus"]
        assert list(events["description"]) == ["11"]

    def test_data_envelope_shape(self) -> None:
        n_ch, n_samp = 4, 80
        result = _make_result("dummy.vhdr", n_channels=n_ch, n_samples=n_samp, windows=[0, 250])
        tree = result.to_output_tree()
        # Unwrap OutputNode
        envelope_node = tree["data"]["envelope"]  # type: ignore[index]
        envelope = envelope_node.value if hasattr(envelope_node, "value") else envelope_node
        assert envelope.shape == (2, n_ch, n_samp)
        assert envelope.dtype == np.float32

    def test_smoothing_windows_sorted_ascending(self) -> None:
        result = _make_result("dummy.vhdr", windows=[1000, 0, 250])
        tree = result.to_output_tree()
        sw = tree["axes"]["smoothing_window_ms"]  # type: ignore[index]
        assert list(sw) == [0, 250, 1000]

    def test_axes_shapes(self) -> None:
        n_ch, n_samp = 3, 100
        result = _make_result("dummy.vhdr", n_channels=n_ch, n_samples=n_samp, windows=[0])
        tree = result.to_output_tree()
        axes = tree["axes"]  # type: ignore[index]
        assert len(axes["channel"]) == n_ch
        assert "band_limits_hz" not in axes
        assert len(axes["time_s"]) == n_samp

    def test_band_limits_are_metadata(self) -> None:
        result = _make_result("dummy.vhdr", windows=[0])
        meta = result.to_output_tree()["meta"]  # type: ignore[index]
        np.testing.assert_allclose(meta["band_limits_hz"], result.bins)

    def test_time_axis_spacing(self) -> None:
        n_samp = 64
        result = _make_result("dummy.vhdr", n_samples=n_samp, windows=[0])
        result.downsampled_fs = 64.0
        tree = result.to_output_tree()
        time_s = tree["axes"]["time_s"]  # type: ignore[index]
        np.testing.assert_allclose(time_s[1] - time_s[0], 1.0 / 64.0, rtol=1e-6)

    def test_meta_required_fields(self) -> None:
        result = _make_result("dummy.vhdr", windows=[0])
        meta = result.to_output_tree()["meta"]  # type: ignore[index]
        for key in (
            "schema_name",
            "schema_version",
            "sampling_frequency_hz",
            "downsampled_frequency_hz",
            "band_limits_hz",
            "montage_mode",
            "unit",
            "dimension_order",
            "centered",
        ):
            assert key in meta, f"Missing meta key: {key!r}"

    def test_meta_optional_events_keys_absent_when_not_in_metadata(self) -> None:
        result = _make_result("dummy.vhdr", windows=[0])
        meta = result.to_output_tree()["meta"]  # type: ignore[index]
        for key in (
            "events_source_requested",
            "events_source_resolved",
            "events_onset_precision",
        ):
            assert key not in meta

    def test_meta_optional_events_keys_present_when_in_metadata(self) -> None:
        result = _make_result(
            "dummy.vhdr",
            windows=[0],
            extra_metadata={"events_onset_precision": "sample_quantized"},
        )
        meta = result.to_output_tree()["meta"]  # type: ignore[index]
        assert meta["events_onset_precision"] == "sample_quantized"

    def test_provenance_pipeline_version_passthrough(self) -> None:
        result = _make_result("dummy.vhdr", windows=[0])
        tree = result.to_output_tree(pipeline_version="1.2.3")
        assert tree["provenance"]["pipeline_version"] == "1.2.3"  # type: ignore[index]

    def test_provenance_raw_bids_path(self) -> None:
        path = "/data/study/sub-01/ieeg/sub-01_task-rest_ieeg.vhdr"
        result = _make_result(path, windows=[0])
        tree = result.to_output_tree()
        stored = tree["provenance"]["raw_bids_path"]  # type: ignore[index]
        # Compare normalised paths to handle platform-specific separators.
        assert Path(stored) == Path(path)


# ---------------------------------------------------------------------------
# HDF5 writer round-trip tests
# ---------------------------------------------------------------------------


class TestHdf5Writer:
    def test_writes_h5_file(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), windows=[0, 250])
        writer = HilbertProcessingWriter(HilbertWriterParams(bids_root=tmp_path))
        out = writer.write(result)
        assert out.suffix == ".h5"
        assert out.is_file()

    def test_hdf5_top_level_groups(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), windows=[0])
        writer = HilbertProcessingWriter(HilbertWriterParams(bids_root=tmp_path))
        out = writer.write(result)
        with h5py.File(out, "r") as fh:
            assert set(fh.keys()) == {"data", "axes", "meta", "provenance"}

    def test_hdf5_envelope_shape(self, tmp_path: Path) -> None:
        n_ch, n_samp = 3, 100
        result = _make_result(
            str(tmp_path / "dummy.vhdr"), n_channels=n_ch, n_samples=n_samp, windows=[0, 250]
        )
        writer = HilbertProcessingWriter(HilbertWriterParams(bids_root=tmp_path))
        out = writer.write(result)
        with h5py.File(out, "r") as fh:
            assert fh["data/envelope"].shape == (2, n_ch, n_samp)

    def test_hdf5_channel_names_round_trip(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), n_channels=4, windows=[0])
        writer = HilbertProcessingWriter(HilbertWriterParams(bids_root=tmp_path))
        out = writer.write(result)
        with h5py.File(out, "r") as fh:
            channels = [c.decode() if isinstance(c, bytes) else c for c in fh["axes/channel"][:]]
        assert channels == result.channel_names

    def test_hdf5_smoothing_windows_sorted(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), windows=[1000, 0, 250])
        writer = HilbertProcessingWriter(HilbertWriterParams(bids_root=tmp_path))
        out = writer.write(result)
        with h5py.File(out, "r") as fh:
            assert list(fh["axes/smoothing_window_ms"][:]) == [0, 250, 1000]

    def test_hdf5_band_limits_are_metadata(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), windows=[0])
        writer = HilbertProcessingWriter(HilbertWriterParams(bids_root=tmp_path))
        out = writer.write(result)
        with h5py.File(out, "r") as fh:
            assert "band_limits_hz" not in fh["axes"]
            np.testing.assert_allclose(fh["meta/band_limits_hz"][:], result.bins)

    def test_hdf5_pipeline_name_in_provenance(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), windows=[0])
        writer = HilbertProcessingWriter(HilbertWriterParams(bids_root=tmp_path))
        out = writer.write(result)
        with h5py.File(out, "r") as fh:
            name = fh["provenance/pipeline_name"][()]
            if isinstance(name, bytes):
                name = name.decode()
            assert name == "hilbert"

    def test_hdf5_roundtrip_via_loader(self, tmp_path: Path) -> None:
        result = _make_result(
            str(tmp_path / "dummy.vhdr"),
            windows=[250, 0],
            events=[
                {
                    "onset": 10,
                    "duration": 2,
                    "type": "Stimulus",
                    "description": "S  1",
                }
            ],
        )
        out = HilbertProcessingWriter(HilbertWriterParams(bids_root=tmp_path)).write(result)

        loaded = load_hilbert_result(out)

        assert loaded.channel_names == result.channel_names
        assert loaded.bins == result.bins
        assert loaded.downsampled_fs == result.downsampled_fs
        assert loaded.original_fs == result.original_fs
        assert sorted(loaded.smoothed) == [0, 250]
        np.testing.assert_allclose(loaded.smoothed[0], result.smoothed[0])
        assert loaded.events == result.events


# ---------------------------------------------------------------------------
# MATLAB writer tests
# ---------------------------------------------------------------------------


class TestMatlabWriter:
    def _matlab_writer(self, bids_root: Path) -> HilbertProcessingWriter:
        return HilbertProcessingWriter(
            HilbertWriterParams(bids_root=bids_root, output_format="matlab")
        )

    def test_writes_mat_file(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), windows=[0, 250])
        out = self._matlab_writer(tmp_path).write(result)
        assert out.suffix == ".mat"
        assert out.is_file()

    def test_mat_file_is_loadable(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), windows=[0])
        out = self._matlab_writer(tmp_path).write(result)
        mat = scipy.io.loadmat(str(out))
        assert "hilbert" in mat

    def test_mat_envelope_shape(self, tmp_path: Path) -> None:
        n_ch, n_samp = 3, 100
        result = _make_result(
            str(tmp_path / "dummy.vhdr"),
            n_channels=n_ch,
            n_samples=n_samp,
            windows=[0, 250],
        )
        out = self._matlab_writer(tmp_path).write(result)
        mat = scipy.io.loadmat(str(out), squeeze_me=False)
        envelope = mat["hilbert"]["data"][0, 0]["envelope"][0, 0]
        assert envelope.shape == (2, n_ch, n_samp)

    def test_mat_band_limits_are_metadata(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), windows=[0])
        out = self._matlab_writer(tmp_path).write(result)
        mat = scipy.io.loadmat(str(out), squeeze_me=False)

        axes = mat["hilbert"]["axes"][0, 0]
        meta = mat["hilbert"]["meta"][0, 0]
        assert "band_limits_hz" not in axes.dtype.names
        np.testing.assert_allclose(meta["band_limits_hz"][0, 0].ravel(), result.bins)

    def test_mat_roundtrip_via_loader(self, tmp_path: Path) -> None:
        result = _make_result(str(tmp_path / "dummy.vhdr"), windows=[0, 250])
        out = self._matlab_writer(tmp_path).write(result)

        loaded = load_hilbert_result(out)

        assert loaded.channel_names == result.channel_names
        assert loaded.bins == result.bins
        assert sorted(loaded.smoothed) == [0, 250]
        np.testing.assert_allclose(loaded.smoothed[250], result.smoothed[250])



