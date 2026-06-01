from __future__ import annotations

import numpy as np
import pytest

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.processing.time_frequency.dsp import TimeFrequencyGrid
from bidsforge.processing.time_frequency.params import TimeFrequencyParams
import bidsforge.processing.time_frequency.processor as processor_module


def _raw_with_annotation():
    import mne

    fs = 100.0
    n_samples = 500
    data = np.vstack([
        np.sin(2 * np.pi * 10 * np.arange(n_samples) / fs),
        np.cos(2 * np.pi * 10 * np.arange(n_samples) / fs),
    ]).astype(np.float32)
    info = mne.create_info(["A1", "A2"], sfreq=fs, ch_types=["seeg", "seeg"])
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(mne.Annotations(onset=[2.0], duration=[0.0], description=["Stimulus/S 1"]))
    return raw


def test_process_group_extracts_epochs_and_applies_baseline(monkeypatch, tmp_path) -> None:
    raw = _raw_with_annotation()
    bids_file = BIDSFile.from_path(tmp_path / "sub-01_task-test_ieeg.vhdr").attach_data(raw)
    captured: dict[str, object] = {}

    def fake_compute_power_db(
        epochs,
        epoch_time_s,
        sampling_frequency_hz,
        params,
        **_kwargs,
    ):
        captured["epochs_shape"] = epochs.shape
        captured["time_axis"] = np.asarray(epoch_time_s)
        captured["fs"] = sampling_frequency_hz
        power = np.ones((1, 2, 1, 4), dtype=np.float32) * np.array([10, 12, 14, 16], dtype=np.float32)
        grid = TimeFrequencyGrid(
            time_s=np.array([-1.0, -0.5, 0.0, 0.5]),
            frequency_hz=np.array([10.0]),
            time_window_s=np.array([0.6]),
            smoothing_hz=np.array([10 / 3]),
            n_tapers=np.array([3]),
        )
        return power, grid

    monkeypatch.setattr(processor_module, "compute_power_db", fake_compute_power_db)

    params = TimeFrequencyParams(
        anchor_event_codes=["1"],
        tmin_s=-1.0,
        tmax_s=1.0,
        baseline_window_s=(-1.0, -0.5),
        apply_baseline=True,
    )
    result = processor_module.TimeFrequencyProcessing(params).process_group(
        BIDSFileGroup(primary=bids_file)
    )

    assert captured["epochs_shape"] == (1, 2, 201)
    assert captured["fs"] == 100.0
    assert result.power_db.shape == (1, 2, 1, 4)
    np.testing.assert_allclose(result.baseline_db, 11.0)
    np.testing.assert_allclose(result.power_db[0, 0, 0], [-1, 1, 3, 5])
    assert result.metadata["apply_baseline"] is True
    assert result.metadata["events_source_resolved"] == "annotations"


def test_process_group_can_leave_power_uncorrected(monkeypatch, tmp_path) -> None:
    raw = _raw_with_annotation()
    bids_file = BIDSFile.from_path(tmp_path / "sub-01_task-test_ieeg.vhdr").attach_data(raw)

    def fake_compute_power_db(*_args, **_kwargs):
        power = np.ones((1, 2, 1, 4), dtype=np.float32) * np.array([10, 12, 14, 16], dtype=np.float32)
        return power, TimeFrequencyGrid(
            time_s=np.array([-1.0, -0.5, 0.0, 0.5]),
            frequency_hz=np.array([10.0]),
            time_window_s=np.array([0.6]),
            smoothing_hz=np.array([10 / 3]),
            n_tapers=np.array([3]),
        )

    monkeypatch.setattr(processor_module, "compute_power_db", fake_compute_power_db)
    params = TimeFrequencyParams(
        anchor_event_codes=["1"],
        tmin_s=-1.0,
        tmax_s=1.0,
        baseline_window_s=(-1.0, -0.5),
        apply_baseline=False,
    )

    result = processor_module.TimeFrequencyProcessing(params).process_group(
        BIDSFileGroup(primary=bids_file)
    )

    np.testing.assert_allclose(result.baseline_db, 11.0)
    np.testing.assert_allclose(result.power_db[0, 0, 0], [10, 12, 14, 16])


def test_process_group_sets_pyfftw_threads(monkeypatch, tmp_path) -> None:
    raw = _raw_with_annotation()
    bids_file = BIDSFile.from_path(tmp_path / "sub-01_task-test_ieeg.vhdr").attach_data(raw)
    fake_pyfftw = type("FakePyFFTW", (), {"config": type("Config", (), {"NUM_THREADS": 0})})()

    def fake_compute_power_db(*_args, **_kwargs):
        return (
            np.ones((1, 2, 1, 4), dtype=np.float32),
            TimeFrequencyGrid(
                time_s=np.array([-1.0, -0.5, 0.0, 0.5]),
                frequency_hz=np.array([10.0]),
                time_window_s=np.array([0.6]),
                smoothing_hz=np.array([10 / 3]),
                n_tapers=np.array([3]),
            ),
        )

    monkeypatch.setattr(processor_module, "_PYFFTW_AVAILABLE", True)
    monkeypatch.setattr(processor_module, "pyfftw", fake_pyfftw)
    monkeypatch.setattr(processor_module, "get_threads_for_worker", lambda: 7)
    monkeypatch.setattr(processor_module, "compute_power_db", fake_compute_power_db)

    params = TimeFrequencyParams(
        anchor_event_codes=["1"],
        tmin_s=-1.0,
        tmax_s=1.0,
        baseline_window_s=(-1.0, -0.5),
    )
    processor_module.TimeFrequencyProcessing(params).process_group(BIDSFileGroup(primary=bids_file))

    assert fake_pyfftw.config.NUM_THREADS == 7
