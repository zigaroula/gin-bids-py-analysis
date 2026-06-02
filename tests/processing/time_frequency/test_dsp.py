from __future__ import annotations

import numpy as np
from scipy.signal.windows import dpss

from bidsforge.processing.time_frequency.dsp import (
    apply_baseline_correction,
    build_frequency_grid,
    build_time_frequency_grid,
    compute_baseline_db,
    compute_power_db,
)
from bidsforge.processing.time_frequency.params import (
    TimeFrequencyMethod,
    TimeFrequencyParams,
)


def test_frequency_grid_matches_matlab_formula_count() -> None:
    params = TimeFrequencyParams(anchor_event_codes=["1"])
    freqs = build_frequency_grid(epoch_duration_s=4.0, params=params)

    assert freqs.shape == (58,)
    np.testing.assert_allclose(freqs[:3], np.round((4 * 2 ** np.array([0, 0.1, 0.2])) * 4) / 4)


def test_time_frequency_grid_parameters_match_matlab_rules() -> None:
    time = np.arange(-1.0, 1.0, 0.01)
    params = TimeFrequencyParams(
        anchor_event_codes=["1"],
        frequency_start_hz=16,
        frequency_exponent_max=2,
        frequency_exponent_step=1,
        time_decimation=5,
    )

    grid = build_time_frequency_grid(epoch_time_s=time, sampling_frequency_hz=100.0, params=params)

    np.testing.assert_allclose(grid.frequency_hz, [16.0, 32.0, 64.0])
    np.testing.assert_allclose(grid.time_window_s[:2], [6 / 16, 6 / 32])
    assert grid.time_window_s[2] == 0.1875
    np.testing.assert_allclose(grid.smoothing_hz[:2], [16 / 3, 32 / 3])
    assert grid.n_tapers[0] == 3
    assert grid.n_tapers[1] == 3
    assert len(grid.time_s) == 40
    assert params.method == TimeFrequencyMethod.FIELDTRIP


def test_fieldtrip_grid_uses_rounded_dpss_taper_count() -> None:
    fs = 512.0
    time = np.arange(3585) / fs - 1.0
    params = TimeFrequencyParams(anchor_event_codes=["1"])

    grid = build_time_frequency_grid(
        epoch_time_s=time,
        sampling_frequency_hz=fs,
        params=params,
    )

    assert np.all(grid.n_tapers[:31] == 3)


def test_fieldtrip_polyorder_zero_removes_dc_offset() -> None:
    fs = 100.0
    time = np.arange(-1.0, 1.0, 1 / fs)
    base_signal = np.sin(2 * np.pi * 10.0 * time).astype(np.float32)
    offset_signal = (base_signal + 1000.0).astype(np.float32)
    params = TimeFrequencyParams(
        anchor_event_codes=["1"],
        frequency_start_hz=10.0,
        frequency_exponent_max=0.01,
        frequency_exponent_step=1.0,
        time_decimation=10,
    )

    base_power, _ = compute_power_db(base_signal.reshape(1, 1, -1), time, fs, params)
    offset_power, _ = compute_power_db(offset_signal.reshape(1, 1, -1), time, fs, params)

    np.testing.assert_allclose(offset_power, base_power, rtol=1e-5, atol=1e-4)


def test_compute_power_db_shape_and_peak_frequency() -> None:
    fs = 100.0
    time = np.arange(-1.0, 1.0, 1 / fs)
    signal = np.sin(2 * np.pi * 10.0 * time).astype(np.float32)
    epochs = signal.reshape(1, 1, -1)
    params = TimeFrequencyParams(
        anchor_event_codes=["1"],
        frequency_start_hz=10.0,
        frequency_exponent_max=1.0,
        frequency_exponent_step=1.0,
        time_decimation=10,
        max_tapers=3,
    )

    power, grid = compute_power_db(epochs, time, fs, params)

    assert power.shape == (1, 1, 2, len(grid.time_s))
    assert power.dtype == np.float32
    assert np.any(np.isfinite(power))
    mean_by_freq = np.nanmean(power[0, 0], axis=1)
    assert mean_by_freq[0] > mean_by_freq[1]


def test_compute_power_db_masks_windows_outside_epoch_like_fieldtrip() -> None:
    fs = 100.0
    time = np.arange(-1.0, 1.0, 1 / fs)
    signal = np.sin(2 * np.pi * 10.0 * time).astype(np.float32)
    params = TimeFrequencyParams(
        anchor_event_codes=["1"],
        frequency_start_hz=10.0,
        frequency_exponent_max=0.01,
        frequency_exponent_step=1.0,
        time_decimation=10,
    )

    power, grid = compute_power_db(signal.reshape(1, 1, -1), time, fs, params)

    window_samples = int(round(grid.time_window_s[0] * fs))
    nsamplefreqoi = grid.time_window_s[0] * fs
    offset = round(time[0] * fs)
    timeboi = np.round(grid.time_s * fs - offset).astype(np.int64) + 1
    valid = (timeboi >= nsamplefreqoi / 2) & (timeboi < time.size - nsamplefreqoi / 2)

    assert window_samples % 2 == 0
    assert np.any(~valid)
    assert np.all(np.isnan(power[0, 0, 0, ~valid]))
    assert np.all(np.isfinite(power[0, 0, 0, valid]))


def test_compute_power_db_matches_direct_window_reference() -> None:
    fs = 80.0
    time = np.arange(-0.5, 0.5, 1 / fs)
    signal = (
        np.sin(2 * np.pi * 10.0 * time)
        + 0.2 * np.sin(2 * np.pi * 18.0 * time)
    ).astype(np.float32)
    epochs = signal.reshape(1, 1, -1)
    params = TimeFrequencyParams(
        anchor_event_codes=["1"],
        frequency_start_hz=10.0,
        frequency_exponent_max=0.01,
        frequency_exponent_step=1.0,
        time_decimation=8,
    )

    power, grid = compute_power_db(epochs, time, fs, params)

    window_samples = int(round(grid.time_window_s[0] * fs))
    half = window_samples // 2
    tapers = dpss(
        window_samples,
        NW=float(window_samples * grid.smoothing_hz[0] / fs),
        Kmax=int(grid.n_tapers[0]) + 1,
        sym=True,
        norm=2,
    ).astype(np.float32)[:-1]
    demeaned = signal - np.mean(signal)
    offsets = (np.arange(window_samples) - ((window_samples - 1) / 2)) / fs
    oscillation = np.exp(-2j * np.pi * float(grid.frequency_hz[0]) * offsets)
    offset = round(time[0] * fs)
    expected = []
    for center_t in grid.time_s:
        timeboi = int(round(center_t * fs - offset) + 1)
        if not (
            timeboi >= grid.time_window_s[0] * fs / 2
            and timeboi < signal.size - grid.time_window_s[0] * fs / 2
        ):
            expected.append(np.nan)
            continue
        center_idx = timeboi - 1
        start = center_idx - half
        segment = np.zeros(window_samples, dtype=np.float32)
        src_start = max(0, start)
        src_stop = min(signal.size, start + window_samples)
        dst_start = src_start - start
        segment[dst_start : dst_start + (src_stop - src_start)] = demeaned[src_start:src_stop]
        taper_power = []
        for taper in tapers:
            coeff = np.sum(segment * taper * oscillation)
            taper_power.append(np.abs(coeff * np.sqrt(2.0 / window_samples)) ** 2)
        expected.append(10 * np.log10(np.mean(taper_power)))

    np.testing.assert_allclose(power[0, 0, 0], expected, rtol=2e-3, atol=1e-2)


def test_baseline_is_time_axis_mean_without_singleton_dimension() -> None:
    power = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    time = np.array([-1.0, -0.8, -0.6, 0.0, 0.2])

    baseline = compute_baseline_db(power, time, (-1.0, -0.6))
    corrected = apply_baseline_correction(power, baseline)

    assert baseline.shape == (2, 3, 4)
    np.testing.assert_allclose(baseline, power[..., :3].mean(axis=-1))
    np.testing.assert_allclose(corrected, power - baseline[..., None])
