from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal.windows import dpss
from tqdm import tqdm

try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as _fftmod

    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(60)
    _PYFFTW_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when pyfftw is absent
    import numpy.fft as _fftmod

    _PYFFTW_AVAILABLE = False

from .params import TimeFrequencyMethod, TimeFrequencyParams

_CONVOLUTION_SERIES_CHUNK_SIZE = 512


@dataclass(frozen=True)
class TimeFrequencyGrid:
    time_s: np.ndarray
    frequency_hz: np.ndarray
    time_window_s: np.ndarray
    smoothing_hz: np.ndarray
    n_tapers: np.ndarray


def build_frequency_grid(
    *,
    epoch_duration_s: float,
    params: TimeFrequencyParams,
) -> np.ndarray:
    """Build the MATLAB-like logarithmic frequency grid."""
    exponents = np.arange(
        0.0,
        params.frequency_exponent_max + (params.frequency_exponent_step / 2.0),
        params.frequency_exponent_step,
        dtype=np.float64,
    )
    freqs = params.frequency_start_hz * np.power(2.0, exponents)
    return np.round(freqs * epoch_duration_s) / epoch_duration_s


def build_time_frequency_grid(
    *,
    epoch_time_s: np.ndarray,
    sampling_frequency_hz: float,
    params: TimeFrequencyParams,
) -> TimeFrequencyGrid:
    """Return time/frequency axes plus per-frequency multitaper settings."""
    epoch_time_s = np.asarray(epoch_time_s, dtype=np.float64)
    time_s = epoch_time_s[:: params.time_decimation].copy()
    epoch_duration_s = float(epoch_time_s.size) / float(sampling_frequency_hz)
    freqs = build_frequency_grid(epoch_duration_s=epoch_duration_s, params=params)
    if params.method == TimeFrequencyMethod.FIELDTRIP:
        endtime = _fieldtrip_endtime_s(
            n_samples=epoch_time_s.size,
            sampling_frequency_hz=sampling_frequency_hz,
            params=params,
        )
        freqs = _fieldtrip_output_frequencies(
            freqs,
            sampling_frequency_hz=sampling_frequency_hz,
            endtime_s=endtime,
        )

    time_window_s = np.full(freqs.shape, np.nan, dtype=np.float64)
    smoothing_hz = np.full(freqs.shape, np.nan, dtype=np.float64)

    low = freqs <= params.low_frequency_cutoff_hz
    time_window_s[low] = params.low_frequency_n_cycles / freqs[low]
    smoothing_hz[low] = freqs[low] * params.low_frequency_smoothing_fraction

    high = ~low
    time_window_s[high] = params.high_frequency_window_s
    if np.any(high):
        if params.high_frequency_smoothing_mode == "fixed":
            smoothing_hz[high] = params.high_frequency_fixed_smoothing_hz
        else:
            smoothing_hz[high] = freqs[high] / 3.0
            k = np.floor(2.0 * time_window_s[high] * smoothing_hz[high] - 1.0)
            smoothing_hz[high] = (k + 1.0) / (2.0 * time_window_s[high])

    if params.method == TimeFrequencyMethod.FIELDTRIP:
        n_tapers = np.asarray(
            [
                _fieldtrip_n_tapers(
                    int(round(float(tw) * float(sampling_frequency_hz))),
                    float(sm),
                    float(sampling_frequency_hz),
                    params,
                )
                for tw, sm in zip(time_window_s, smoothing_hz)
            ],
            dtype=np.int64,
        )
    else:
        n_tapers = np.asarray(
            [
                _n_tapers(float(tw), float(sm), params)
                for tw, sm in zip(time_window_s, smoothing_hz)
            ],
            dtype=np.int64,
        )
    return TimeFrequencyGrid(
        time_s=time_s,
        frequency_hz=freqs.astype(np.float64),
        time_window_s=time_window_s,
        smoothing_hz=smoothing_hz,
        n_tapers=n_tapers,
    )


def compute_power_db(
    epochs: np.ndarray,
    epoch_time_s: np.ndarray,
    sampling_frequency_hz: float,
    params: TimeFrequencyParams,
    *,
    verbose: bool = False,
    desc: str | None = None,
    progress_tracking_position: int = 0,
) -> tuple[np.ndarray, TimeFrequencyGrid]:
    """Compute TFR power in dB for epochs shaped [trial, channel, time]."""
    if params.method == TimeFrequencyMethod.FIELDTRIP:
        return _compute_power_db_fieldtrip(
            epochs,
            epoch_time_s,
            sampling_frequency_hz,
            params,
            verbose=verbose,
            desc=desc,
            progress_tracking_position=progress_tracking_position,
        )
    raise ValueError(f"Unsupported time-frequency method: {params.method!r}")


def _compute_power_db_fieldtrip(
    epochs: np.ndarray,
    epoch_time_s: np.ndarray,
    sampling_frequency_hz: float,
    params: TimeFrequencyParams,
    *,
    verbose: bool = False,
    desc: str | None = None,
    progress_tracking_position: int = 0,
) -> tuple[np.ndarray, TimeFrequencyGrid]:
    """Compute TFR power using FieldTrip ft_specest_mtmconvol conventions."""
    arr = np.asarray(epochs, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("epochs must be shaped [trial, channel, time].")

    grid = build_time_frequency_grid(
        epoch_time_s=epoch_time_s,
        sampling_frequency_hz=sampling_frequency_hz,
        params=params,
    )
    n_trials, n_channels, _ = arr.shape
    n_freqs = int(grid.frequency_hz.size)
    n_times = int(grid.time_s.size)
    out = np.empty((n_trials, n_channels, n_freqs, n_times), dtype=np.float32)

    sample_times = np.asarray(epoch_time_s, dtype=np.float64)
    fs = float(sampling_frequency_hz)
    ndatsample = int(arr.shape[-1])
    dattime = ndatsample / fs
    pad = dattime if params.fieldtrip_pad_s is None else float(params.fieldtrip_pad_s)
    if round(pad * fs) < ndatsample:
        raise ValueError("fieldtrip_pad_s cannot be shorter than the epoch duration.")
    postpad = int(round((pad - dattime) * fs))
    endnsample = int(round(pad * fs))

    timeoi, timeboi = _fieldtrip_time_bins(sample_times, grid.time_s, fs)
    if not np.array_equal(timeoi, grid.time_s):
        grid = TimeFrequencyGrid(
            time_s=timeoi,
            frequency_hz=grid.frequency_hz,
            time_window_s=grid.time_window_s,
            smoothing_hz=grid.smoothing_hz,
            n_tapers=grid.n_tapers,
        )
        n_times = int(grid.time_s.size)
        out = np.empty((n_trials, n_channels, n_freqs, n_times), dtype=np.float32)

    series = arr.reshape(n_trials * n_channels, ndatsample).astype(np.float32, copy=False)
    series = _fieldtrip_polyremoval(series, params.fieldtrip_polyorder)
    if postpad > 0:
        padded = np.pad(series, ((0, 0), (0, postpad)), mode="constant")
    else:
        padded = series
    datspectrum = _fftmod.fft(padded, n=endnsample, axis=-1)

    for fi, freq in enumerate(
        tqdm(
            grid.frequency_hz,
            desc=desc or "Time-frequency",
            unit="freq",
            disable=not verbose,
            leave=True,
            position=progress_tracking_position,
        )
    ):
        timwinsample = max(1, int(round(grid.time_window_s[fi] * fs)))
        nsamplefreqoi = float(grid.time_window_s[fi] * fs)
        valid_time = (timeboi >= (nsamplefreqoi / 2.0)) & (
            timeboi < (ndatsample - (nsamplefreqoi / 2.0))
        )
        valid_output_indices = np.flatnonzero(valid_time)
        valid_data_indices = timeboi[valid_time] - 1

        if valid_output_indices.size == 0:
            out[:, :, fi, :] = np.nan
            continue

        tapers = _fieldtrip_dpss_tapers(timwinsample, float(grid.smoothing_hz[fi]), fs, params)
        if tapers.size == 0:
            out[:, :, fi, :] = np.nan
            continue

        power_sum = np.zeros((series.shape[0], n_times), dtype=np.float64)
        count = np.zeros((n_times,), dtype=np.int32)
        wavelet_ffts = _fieldtrip_wavelet_ffts(
            tapers,
            freq=float(freq),
            timwinsample=timwinsample,
            endnsample=endnsample,
            fs=fs,
        )
        scale = math.sqrt(2.0 / float(timwinsample))
        for wavelet_fft in wavelet_ffts:
            dum = np.fft.fftshift(
                _fftmod.ifft(datspectrum * wavelet_fft[np.newaxis, :], axis=-1),
                axes=-1,
            )
            coeff = dum[:, valid_data_indices] * scale
            power_sum[:, valid_output_indices] += np.abs(coeff) ** 2
            count[valid_output_indices] += 1

        power = np.full((series.shape[0], n_times), np.nan, dtype=np.float64)
        valid_count = count > 0
        power[:, valid_count] = power_sum[:, valid_count] / count[valid_count]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, :, fi, :] = (
                10.0 * np.log10(np.maximum(power, 1e-20))
            ).astype(np.float32).reshape(n_trials, n_channels, n_times)

    return out, grid


def _compute_power_db_convolution_reference(
    epochs: np.ndarray,
    epoch_time_s: np.ndarray,
    sampling_frequency_hz: float,
    params: TimeFrequencyParams,
    *,
    verbose: bool = False,
    desc: str | None = None,
    progress_tracking_position: int = 0,
) -> tuple[np.ndarray, TimeFrequencyGrid]:
    """Previous local convolution implementation kept as a private reference."""
    arr = np.asarray(epochs, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("epochs must be shaped [trial, channel, time].")

    grid = build_time_frequency_grid(
        epoch_time_s=epoch_time_s,
        sampling_frequency_hz=sampling_frequency_hz,
        params=params,
    )
    n_trials, n_channels, _ = arr.shape
    n_freqs = int(grid.frequency_hz.size)
    n_times = int(grid.time_s.size)
    out = np.empty((n_trials, n_channels, n_freqs, n_times), dtype=np.float32)

    sample_times = np.asarray(epoch_time_s, dtype=np.float64)
    center_indices = np.asarray(
        [int(np.argmin(np.abs(sample_times - float(center_t)))) for center_t in grid.time_s],
        dtype=np.int64,
    )
    series = arr.reshape(n_trials * n_channels, arr.shape[-1])
    for fi, freq in enumerate(
        tqdm(
            grid.frequency_hz,
            desc=desc or "Time-frequency",
            unit="freq",
            disable=not verbose,
            leave=True,
            position=progress_tracking_position,
        )
    ):
        window_samples = max(1, int(round(grid.time_window_s[fi] * sampling_frequency_hz)))
        if window_samples % 2 == 0:
            window_samples += 1
        half_window = window_samples // 2
        valid_time_mask = (center_indices - half_window >= 0) & (
            center_indices + half_window < arr.shape[-1]
        )
        nw = max(float(grid.time_window_s[fi] * grid.smoothing_hz[fi]), 0.5)
        tapers = dpss(
            window_samples,
            NW=nw,
            Kmax=int(grid.n_tapers[fi]),
            sym=False,
            norm=2,
        ).astype(np.float32)
        if tapers.ndim == 1:
            tapers = tapers.reshape(1, -1)

        power_sum = np.zeros((series.shape[0], n_times), dtype=np.float32)
        offsets = (np.arange(window_samples, dtype=np.float64) - (window_samples // 2)) / float(
            sampling_frequency_hz
        )
        oscillation = np.exp(-2j * np.pi * float(freq) * offsets).astype(np.complex64)
        scale = np.sum(np.square(tapers), axis=1, dtype=np.float64)
        for taper, taper_scale in zip(tapers, scale):
            kernel = (taper.astype(np.complex64) * oscillation)[::-1]
            coeff = _convolve_and_sample(
                series,
                kernel,
                center_indices,
                chunk_size=_CONVOLUTION_SERIES_CHUNK_SIZE,
            )
            power_sum += (np.abs(coeff) ** 2 / float(taper_scale)).astype(np.float32)
        power = power_sum / float(tapers.shape[0])
        power[:, ~valid_time_mask] = np.nan
        out[:, :, fi, :] = (
            10.0 * np.log10(np.maximum(power, 1e-20))
        ).astype(np.float32).reshape(n_trials, n_channels, n_times)

    return out, grid


def compute_baseline_db(
    power_db_raw: np.ndarray,
    time_s: np.ndarray,
    baseline_window_s: tuple[float, float],
) -> np.ndarray:
    """Average raw dB power across the baseline time axis."""
    b0, b1 = baseline_window_s
    time_s = np.asarray(time_s, dtype=np.float64)
    mask = (time_s >= float(b0)) & (time_s <= float(b1))
    if not np.any(mask):
        raise ValueError(
            f"baseline_window_s={baseline_window_s!r} does not overlap the TFR time axis."
        )
    return np.nanmean(power_db_raw[..., mask], axis=-1, dtype=np.float64).astype(np.float32)


def apply_baseline_correction(power_db_raw: np.ndarray, baseline_db: np.ndarray) -> np.ndarray:
    return (np.asarray(power_db_raw, dtype=np.float32) - baseline_db[..., np.newaxis]).astype(np.float32)


def _n_tapers(time_window_s: float, smoothing_hz: float, params: TimeFrequencyParams) -> int:
    value = int(math.floor(2.0 * time_window_s * smoothing_hz - 1.0))
    value = max(int(params.min_tapers), value)
    if params.max_tapers is not None:
        value = min(int(params.max_tapers), value)
    return value


def _convolve_and_sample(
    series: np.ndarray,
    kernel_reversed: np.ndarray,
    sample_indices: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    n_series, n_samples = series.shape
    n_kernel = int(kernel_reversed.size)
    half = n_kernel // 2
    n_fft = next_fast_len(n_samples + n_kernel - 1)
    kernel_fft = _fftmod.fft(kernel_reversed, n=n_fft).astype(np.complex64, copy=False)
    out = np.empty((n_series, sample_indices.size), dtype=np.complex64)

    for start in range(0, n_series, chunk_size):
        stop = min(start + chunk_size, n_series)
        chunk_fft = _fftmod.fft(series[start:stop], n=n_fft, axis=-1)
        full = _fftmod.ifft(chunk_fft * kernel_fft[np.newaxis, :], axis=-1)
        out[start:stop] = full[:, half + sample_indices].astype(np.complex64, copy=False)

    return out


def _fieldtrip_endtime_s(
    *,
    n_samples: int,
    sampling_frequency_hz: float,
    params: TimeFrequencyParams,
) -> float:
    if params.fieldtrip_pad_s is not None:
        return float(params.fieldtrip_pad_s)
    return float(n_samples) / float(sampling_frequency_hz)


def _fieldtrip_output_frequencies(
    freqs: np.ndarray,
    *,
    sampling_frequency_hz: float,
    endtime_s: float,
) -> np.ndarray:
    endnsample = int(round(endtime_s * sampling_frequency_hz))
    freqboi = np.round(freqs / (sampling_frequency_hz / endnsample)).astype(np.int64) + 1
    freqboi = np.unique(freqboi)
    if freqboi.size and freqboi[0] == 1:
        freqboi = freqboi[1:]
    return (freqboi - 1).astype(np.float64) / float(endtime_s)


def _fieldtrip_time_bins(
    sample_times: np.ndarray,
    requested_time_s: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    timeoi = np.unique(np.round(np.asarray(requested_time_s, dtype=np.float64) * fs) / fs)
    offset = int(round(float(sample_times[0]) * fs))
    timeboi = np.round(timeoi * fs - offset).astype(np.int64) + 1
    return timeoi, timeboi


def _fieldtrip_polyremoval(series: np.ndarray, polyorder: int) -> np.ndarray:
    if polyorder < 0:
        return series.astype(np.float32, copy=True)
    x = np.arange(series.shape[-1], dtype=np.float64)
    x = x - np.mean(x)
    if np.max(np.abs(x)) > 0:
        x = x / np.max(np.abs(x))
    design = np.vander(x, N=polyorder + 1, increasing=True)
    coeff, *_ = np.linalg.lstsq(design, series.T.astype(np.float64), rcond=None)
    trend = (design @ coeff).T
    return (series.astype(np.float64) - trend).astype(np.float32)


def _fieldtrip_dpss_tapers(
    timwinsample: int,
    smoothing_hz: float,
    fs: float,
    params: TimeFrequencyParams,
) -> np.ndarray:
    nw = float(timwinsample) * float(smoothing_hz) / float(fs)
    final_k = _fieldtrip_n_tapers(timwinsample, smoothing_hz, fs, params)
    tapers = dpss(
        timwinsample,
        NW=max(nw, 0.5),
        Kmax=final_k + 1,
        sym=True,
        norm=2,
    ).astype(np.float32)
    if tapers.ndim == 1:
        tapers = tapers.reshape(1, -1)
    return tapers[:-1]


def _fieldtrip_wavelet_ffts(
    tapers: np.ndarray,
    *,
    freq: float,
    timwinsample: int,
    endnsample: int,
    fs: float,
) -> np.ndarray:
    tappad = int(math.ceil(endnsample / 2.0) - math.floor(timwinsample / 2.0))
    postzero = int(endnsample - tappad - timwinsample)
    if tappad < 0 or postzero < 0:
        return np.zeros((tapers.shape[0], endnsample), dtype=np.complex64)

    anglein = (
        (np.arange(timwinsample, dtype=np.float64) - ((timwinsample - 1.0) / 2.0))
        * ((2.0 * np.pi / fs) * float(freq))
    )
    wavelet_ffts = np.empty((tapers.shape[0], endnsample), dtype=np.complex64)
    for idx, taper in enumerate(tapers):
        wavelet = np.zeros((endnsample,), dtype=np.complex64)
        wavelet[tappad : tappad + timwinsample] = (
            taper.astype(np.float32)
            * (np.cos(anglein).astype(np.float32) + 1j * np.sin(anglein).astype(np.float32))
        )
        wavelet_ffts[idx] = _fftmod.fft(wavelet, n=endnsample).astype(np.complex64)
    return wavelet_ffts


def _fieldtrip_n_tapers(
    timwinsample: int,
    smoothing_hz: float,
    fs: float,
    params: TimeFrequencyParams,
) -> int:
    nw = float(timwinsample) * float(smoothing_hz) / float(fs)
    value = int(round(2.0 * nw)) - 1
    value = max(int(params.min_tapers), value)
    if params.max_tapers is not None:
        value = min(int(params.max_tapers), value)
    return value
