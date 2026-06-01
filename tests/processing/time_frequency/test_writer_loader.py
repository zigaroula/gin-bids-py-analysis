from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scipy.io

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.processing.time_frequency import (
    TimeFrequencyProcessingResult,
    TimeFrequencyProcessingWriter,
    TimeFrequencyWriterParams,
    load_time_frequency_result,
)


def _result(path: Path) -> TimeFrequencyProcessingResult:
    power = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    baseline = power[..., :2].mean(axis=-1)
    return TimeFrequencyProcessingResult(
        source_group=BIDSFileGroup(primary=BIDSFile.from_path(path)),
        power_db=power,
        baseline_db=baseline,
        trial_ids=["0", "1"],
        channel_names=["A1", "A2", "A3"],
        frequency_hz=np.array([4, 8, 16, 32], dtype=np.float64),
        time_s=np.linspace(-1, 1, 5),
        original_fs=100.0,
        metadata={
            "apply_baseline": True,
            "montage_mode": "mono",
            "time_frequency_method": "fieldtrip",
            "fieldtrip_polyorder": 0,
            "fieldtrip_pad_s": 2.0,
            "fieldtrip_padtype": "zero",
            "fieldtrip_scaling": True,
        },
        events=[{"onset": 10, "duration": 0, "type": "Stimulus", "description": "S 1"}],
    )


def test_hdf5_writer_schema_and_roundtrip(tmp_path: Path) -> None:
    result = _result(tmp_path / "sub-01_task-test_ieeg.vhdr")
    writer = TimeFrequencyProcessingWriter(TimeFrequencyWriterParams(bids_root=tmp_path))

    out = writer.write(result)

    assert out.suffix == ".h5"
    with h5py.File(out, "r") as fh:
        assert fh["data/power_db"].shape == (2, 3, 4, 5)
        assert fh["data/baseline_db"].shape == (2, 3, 4)
        assert fh["meta/dimension_order"].asstr()[()] == "trial x channel x frequency x time"
        assert bool(fh["meta/apply_baseline"][()]) is True
        assert fh["meta/time_frequency_method"].asstr()[()] == "fieldtrip"
        assert int(fh["meta/fieldtrip_polyorder"][()]) == 0
        assert float(fh["meta/fieldtrip_pad_s"][()]) == 2.0
        assert fh["meta/fieldtrip_padtype"].asstr()[()] == "zero"
        assert bool(fh["meta/fieldtrip_scaling"][()]) is True

    loaded = load_time_frequency_result(out)
    np.testing.assert_allclose(loaded.power_db, result.power_db)
    np.testing.assert_allclose(loaded.baseline_db, result.baseline_db)
    assert loaded.channel_names == result.channel_names
    assert loaded.trial_ids == result.trial_ids
    assert loaded.events == result.events


def test_matlab_writer_is_loadable(tmp_path: Path) -> None:
    result = _result(tmp_path / "sub-01_task-test_ieeg.vhdr")
    writer = TimeFrequencyProcessingWriter(
        TimeFrequencyWriterParams(bids_root=tmp_path, output_format="matlab")
    )

    out = writer.write(result)

    assert out.suffix == ".mat"
    mat = scipy.io.loadmat(str(out), squeeze_me=False)
    assert "time_frequency" in mat
    power = mat["time_frequency"]["data"][0, 0]["power_db"][0, 0]
    assert power.shape == result.power_db.shape


def test_writer_accepts_events_tsv_style_events_without_type(tmp_path: Path) -> None:
    result = _result(tmp_path / "sub-01_task-test_ieeg.vhdr")
    result.events = [
        {"onset": 1.25, "duration": 0.0, "description": "Stimulus/S 11"},
        {"onset": 2.50, "duration": 0.1, "description": "Response/R 2"},
    ]
    writer = TimeFrequencyProcessingWriter(TimeFrequencyWriterParams(bids_root=tmp_path))

    out = writer.write(result)

    loaded = load_time_frequency_result(out)
    assert loaded.events == [
        {"onset": 1.25, "duration": 0.0, "type": "Stimulus", "description": "S 11"},
        {"onset": 2.5, "duration": 0.1, "type": "Response", "description": "R 2"},
    ]
