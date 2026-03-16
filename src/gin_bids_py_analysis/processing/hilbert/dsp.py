"""
Hilbert-band envelope DSP — pure-computation module.

All functions here are free of file I/O and side-effects so that they can
be called directly in unit tests and in parallel jobs without any shared
state.

Pipeline overview
-----------------
1. Build a uniform frequency-bin grid and clamp it to the Nyquist limit
   (Shannon clamp).
2. Optionally re-reference channels to a bipolar montage.
3. For each channel:
   a. For each adjacent pair of bins (subband):
      - Apply a FIR band-pass filter in the frequency domain.
      - Compute the analytic signal via Hilbert coefficients.
      - Take the magnitude → amplitude envelope (float32).
   b. Optionally resample each envelope to a target frequency.
   c. Optionally normalise to a percentage of the mid-recording baseline.
   d. Average the resulting envelopes across all subbands.
   e. Optionally apply a fixed-divisor sliding-average smoother for each requested
      window length.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from scipy.signal import resample_poly

try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as _fftmod
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(60)
except ImportError:
    import numpy.fft as _fftmod  # fall back to numpy.fft

if TYPE_CHECKING:
    from .params import HilbertParams

from gin_bids_py_analysis.processing.hilbert.fir import FirBandPass
from gin_bids_py_analysis.processing.utils.channels import build_montage
from gin_bids_py_analysis.processing.hilbert.params import NormalizationMode


# ---------------------------------------------------------------------------
# Frequency-bin utilities
# ---------------------------------------------------------------------------


def build_frequency_bins(f_min: float, f_max: float, f_step: float) -> list[float]:
    """Generate a uniform list of frequency bin edges.

    The grid starts at *f_min* and advances by *f_step* until *f_max* is
    reached (inclusive, within floating-point tolerance).

    Example::

        build_frequency_bins(50, 150, 10)
        # → [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0]

    Args:
        f_min:  Lowest bin edge in Hz.
        f_max:  Highest bin edge in Hz.
        f_step: Spacing between adjacent edges in Hz.

    Returns:
        List of bin edge values as floats.
    """
    bins: list[float] = []
    f = f_min
    # Use a small epsilon to handle floating-point rounding at the upper end
    while f <= f_max + 1e-9:
        bins.append(float(f))
        f = f_min + len(bins) * f_step  # avoid cumulative float drift
    return bins


def apply_shannon_clamp(
    bins: list[float],
    fs: float,
    f_min: float,
    f_step: float,
) -> list[float]:
    """Silently clamp the frequency bin grid to the Nyquist limit.

    If the highest bin exceeds ``fs / 2``, the maximum bin is rounded
    *down* to the nearest multiple of *f_step* that lies within Nyquist,
    and the grid is rebuilt from *f_min*.

    If no clamping is needed the original list is returned unchanged.

    Args:
        bins:   Current list of bin edges (as produced by
                :func:`build_frequency_bins`).
        fs:     Sampling frequency in Hz.
        f_min:  Lowest bin edge (used when rebuilding the clamped grid).
        f_step: Bin spacing (used when rebuilding the clamped grid).

    Returns:
        (Possibly clamped) list of bin edges.
    """
    f_max_current = bins[-1]
    if fs > 2.0 * f_max_current:
        # Nyquist is above f_max — no clamping needed
        return bins

    # Clamp: largest bin that fits inside Nyquist, aligned to f_step
    f_max_clamped = (int(fs // 2) // int(f_step)) * f_step
    return build_frequency_bins(f_min, f_max_clamped, f_step)


# ---------------------------------------------------------------------------
# Per-sample signal processing
# ---------------------------------------------------------------------------


def downsample(
    signal: np.ndarray,
    fs: float,
    fs_down: float,
) -> np.ndarray:
    """Decimate *signal* by an integer factor via index picking.

    Decimation logic:

    * ``factor = int(fs) // int(fs_down)``
    * ``n_down = len(signal) // factor``
    * output: ``signal[0 : n_down * factor : factor]``

    Note: this is *not* a proper anti-aliased downsampler

    Args:
        signal: 1-D array.
        fs:     Original sampling frequency in Hz.
        fs_down: Target sampling frequency in Hz.

    Returns:
        Decimated float32 array.
    """
    factor = int(fs) // int(fs_down)
    n_down = len(signal) // factor
    return signal[: n_down * factor : factor].astype(np.float32)


def normalize_percent(signal: np.ndarray) -> np.ndarray:
    """Normalise *signal* to a percentage of its mid-recording baseline.

    The baseline is computed as the mean of the middle half of the last axis:

    * ``value = round(signal.shape[-1] / 4)``
    * ``baseline = mean(signal[..., value : 3 * value], axis=-1)``

    A zero baseline is replaced with 1 to avoid division by zero.
    The result is scaled so that the baseline region has a mean of 100.

    Works on both 1-D ``[n_samples]`` and 2-D ``[n_subbands, n_samples]``
    arrays; in the 2-D case each row is normalised independently.

    Args:
        signal: Float array of shape ``[n_samples]`` or ``[n_subbands, n_samples]``.

    Returns:
        Normalised float32 array of the same shape (baseline region ≈ 100).
    """
    n_len = signal.shape[-1]
    value = round(n_len / 4)
    mean_mid = np.mean(signal[..., value : 3 * value], axis=-1, keepdims=True)
    fmtab = np.where(mean_mid != 0.0, mean_mid, 1.0)
    return (100.0 * signal / fmtab).astype(np.float32)


def normalize_db(signal: np.ndarray) -> np.ndarray:
    """Normalise *signal* to decibels relative to its mid-recording baseline.

    The baseline is computed as the mean of the middle half of the last axis
    (same window as :func:`normalize_percent`):

    * ``value = round(signal.shape[-1] / 4)``
    * ``baseline = mean(signal[..., value : 3 * value], axis=-1)``

    Formula: ``20 * log10(max(signal, 1e-10) / max(baseline, 1e-10))``

    Zero-valued samples or a zero baseline are guarded by the 1e-10 floor so
    that the function never raises or produces ``-inf``.

    Works on both 1-D ``[n_samples]`` and 2-D ``[n_subbands, n_samples]``
    arrays; in the 2-D case each row is normalised independently.

    Args:
        signal: Float array of shape ``[n_samples]`` or ``[n_subbands, n_samples]``.

    Returns:
        Normalised float32 array in dB of the same shape (baseline ≈ 0 dB;
        values above baseline are positive; values below are negative).
    """
    n_len = signal.shape[-1]
    value = round(n_len / 4)
    mean_mid = np.mean(signal[..., value : 3 * value], axis=-1, keepdims=True)
    baseline = np.maximum(mean_mid, 1e-10)
    return (20.0 * np.log10(np.maximum(signal, 1e-10) / baseline)).astype(np.float32)


def moving_average(signal: np.ndarray, coefficient: int) -> np.ndarray:
    """Fixed-divisor sliding average centred around each sample.

    The implementation uses samples on both sides of the current index, so it
    is not causal. Unlike a standard moving average, the divisor is always
    ``coefficient`` regardless of how many samples are actually summed near
    the edges.

    Edge logic (index = ``coefficient // 2``, weight = ``1 / coefficient``):

    * If ``i - index ≤ 0``:   ``begin = 0``,          ``end = i + index``
    * If ``i ≥ len - index``: ``begin = i - index + 1``, ``end = len - 1``
    * Otherwise:              ``begin = i - (index - 1)``, ``end = i + index``

    The output at sample *i* is ``sum(signal[begin : end + 1]) / coefficient``.

    A O(n) cumulative-sum implementation is used for efficiency while
    preserving the exact values.

    Args:
        signal:      1-D float array.
        coefficient: Window length; must be ≥ 1. If 1, the signal is returned
                     unchanged.

    Returns:
        Smoothed float32 array of the same length.
    """
    n = len(signal)
    if coefficient <= 1:
        return signal.astype(np.float32)

    index = coefficient // 2
    weight = 1.0 / coefficient

    # Build cumulative sum (prefix sums) for O(1) range queries
    # cs[i] = sum(signal[0:i])
    cs = np.zeros(n + 1, dtype=np.float64)
    cs[1:] = np.cumsum(signal.astype(np.float64))

    # Vectorised index computation
    i = np.arange(n)
    in_head = i - index <= 0
    in_tail = i >= n - index
    begin = np.where(in_head, 0,
             np.where(in_tail, i - index + 1,
                      i - (index - 1)))
    end = np.where(in_head, i + index,
           np.where(in_tail, n - 1,
                    i + index))
    end = np.minimum(end, n - 1)

    totals = cs[end + 1] - cs[begin]
    return (totals * weight).astype(np.float32)


# ---------------------------------------------------------------------------
# Channel-level pipeline
# ---------------------------------------------------------------------------


def process_channel(
    signal_1d: np.ndarray,
    fs: float,
    bins: list[float],
    params: "HilbertParams",
    filters: list["FirBandPass"] | None = None,
    combined_matrix: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Run the full Hilbert-band envelope pipeline on a single channel.

    Steps:

    1. For each adjacent bin pair (subband):
       a. Build (or retrieve from cache) a :class:`FirBandPass` filter.
       b. Apply the filter → amplitude envelope (float32).
    2. Optionally resample each envelope with
       :func:`scipy.signal.resample_poly`.
    3. Optionally normalise to percentage of baseline.
    4. Average the resulting envelopes across subbands.
    5. For each smoothing window: apply :func:`moving_average`
       (or identity for ``window_ms = 0``), then optionally subtract 100.

    Args:
        signal_1d: 1-D array ``[n_samples]``.
        fs:        Sampling frequency of *signal_1d*.
        bins:      Frequency bin edges (already Shannon-clamped).
        params:    :class:`~.params.HilbertParams` instance controlling all
                   pipeline toggles and settings.
        filters:   Optional list of pre-built :class:`FirBandPass` objects
                   (one per subband).  Used to derive *combined_matrix* when
                   that is not supplied directly.  Pass ``None`` to build
                   filters on the fly from *bins*.
        combined_matrix: Optional ``[n_subbands, n_samples]`` real float32
                   array of pre-stacked filter x Hilbert coefficients.
                   When supplied (e.g. by :func:`process_all_channels`) the
                   matrix is not rebuilt per channel.  Pass ``None`` to
                   derive it from *filters*.

    Returns:
        ``{window_ms: envelope_array}`` where each array has length
        ``n_downsampled`` (or ``n_samples`` if downsampling is disabled).
    """
    n_samples = len(signal_1d)

    # Resolve combined_matrix: accept pre-built, derive from filters,
    # or build everything from bins.  The [n_sub, n_samples] float32 matrix
    # is the only thing needed for the hot path below.
    if combined_matrix is None:
        if filters is None:
            filters = [FirBandPass(low, high, fs, n_samples)
                       for low, high in zip(bins[:-1], bins[1:])]
        combined_matrix = np.stack([fir.combined for fir in filters], axis=0)

    n_subbands = combined_matrix.shape[0]
    if n_subbands == 0:
        raise ValueError(
            "No subbands produced — bins list must have at least 2 elements."
        )

    # ------------------------------------------------------------------
    # Step 1a: FFT the signal once.
    # float32 input → complex64 (2x less data than float64/complex128).
    # The same spectrum is broadcast to all subbands.
    # ------------------------------------------------------------------
    spectrum = _fftmod.fft(signal_1d.astype(np.float32))  # complex64 [n_samples]

    # ------------------------------------------------------------------
    # Step 1b+c: batched IFFT → amplitude envelope.
    # ------------------------------------------------------------------
    analytics = _fftmod.ifft(spectrum * combined_matrix, axis=1)  # complex64 [n_sub, n_samples]
    envelopes_2d = np.abs(analytics).astype(np.float32)           # [n_sub, n_samples] float32

    # ------------------------------------------------------------------
    # Step 2: polyphase resample to exact target frequency.
    # scipy.signal.resample_poly applies an anti-aliasing FIR filter
    # and produces exactly ceil(n_samples * up / down) output samples.
    # ------------------------------------------------------------------
    if params.downsampled_frequency_hz is not None:
        _g = math.gcd(int(params.downsampled_frequency_hz), int(fs))
        _up = int(params.downsampled_frequency_hz) // _g
        _down = int(fs) // _g
        envelopes_2d = resample_poly(envelopes_2d, _up, _down, axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 3: normalisation — delegates to standalone functions which
    # handle both 1-D and 2-D arrays via axis=-1 / keepdims.
    # ------------------------------------------------------------------
    nm = params.normalization_mode
    if nm.is_percent:
        envelopes_2d = normalize_percent(envelopes_2d)
    elif nm == NormalizationMode.DB:
        envelopes_2d = normalize_db(envelopes_2d)

    # ------------------------------------------------------------------
    # Step 4: average across subbands → 1-D [n_samples_eff] float32
    # ------------------------------------------------------------------
    mean_data = np.sum(envelopes_2d, axis=0, dtype=np.float32) / n_subbands

    # ------------------------------------------------------------------
    # Step 5: smoothing windows
    # ------------------------------------------------------------------
    fs_eff = params.downsampled_frequency_hz if params.downsampled_frequency_hz is not None else fs
    result: dict[int, np.ndarray] = {}

    for window_ms in params.smoothing_windows_ms:
        if window_ms == 0:
            smoothed = mean_data.copy()
        else:
            coefficient = int((fs_eff * window_ms) / 1000)
            smoothed = moving_average(mean_data, coefficient)

        if params.normalization_mode.is_centered:
            smoothed = (smoothed - 100.0).astype(np.float32)

        result[window_ms] = smoothed

    return result


# ---------------------------------------------------------------------------
# Dataset-level pipeline
# ---------------------------------------------------------------------------


def process_all_channels(
    data_2d: np.ndarray,
    channel_names: list[str],
    fs: float,
    params: "HilbertParams",
    verbose: bool = False,
    desc: str | None = None,
    progress_tracking_position: int = 0,
) -> tuple[dict[int, np.ndarray], list[str], list[float]]:
    """Run the Hilbert-band envelope pipeline on all channels.

    Applies the montage, Shannon clamp, and then :func:`process_channel`
    to every row of *data_2d*.

    Args:
        data_2d:       2-D array of shape ``[n_channels, n_samples]``
                       (float32 recommended).
        channel_names: List of channel name strings corresponding to the rows
                       of *data_2d*.
        fs:            Sampling frequency in Hz.
        params:        :class:`~.params.HilbertParams` instance.

    Returns:
        A tuple ``(result_dict, montaged_names, bins)`` where:

        * ``result_dict`` maps each ``window_ms`` to a 2-D float32 array of
          shape ``[n_montaged_channels, n_downsampled_samples]``.
        * ``montaged_names`` is the list of channel names after montaging.
        * ``bins`` is the list of frequency bin edges actually used (after
          Shannon clamping).
    """
    # ------------------------------------------------------------------
    # Apply montage
    # ------------------------------------------------------------------
    montaged_data, montaged_names = build_montage(
        data=data_2d,
        channel_names=channel_names,
        mode=params.montage_mode,
        direction=params.bipolar_direction,
        storage=params.bipolar_storage,
    )

    # ------------------------------------------------------------------
    # Build and clamp frequency bins
    # ------------------------------------------------------------------
    bins = build_frequency_bins(params.f_min, params.f_max, params.f_step)
    bins = apply_shannon_clamp(bins, fs, params.f_min, params.f_step)

    if len(bins) < 2:
        raise ValueError(
            f"After Shannon clamping for fs={fs} Hz, fewer than 2 bins remain "
            f"(bins={bins}).  Lower f_max or increase fs."
        )

    # ------------------------------------------------------------------
    # Pre-compute filter coefficients once for all channels.
    # combined_matrix [n_sub, n_samples] float32 is built here once so
    # process_channel never rebuilds it per channel.
    # ------------------------------------------------------------------
    n_samples = montaged_data.shape[1]
    filters = [FirBandPass(low, high, fs, n_samples)
               for low, high in zip(bins[:-1], bins[1:])]
    combined_matrix = np.stack([fir.combined for fir in filters], axis=0)
    n_channels = montaged_data.shape[0]

    # ------------------------------------------------------------------
    # Process each channel and collect results per smoothing window
    # ------------------------------------------------------------------
    # Initialise the output dict with empty lists
    per_window: dict[int, list[np.ndarray]] = {
        w: [] for w in params.smoothing_windows_ms
    }

    for ch_idx in tqdm(
        range(n_channels),
        desc=desc or "Processing",
        unit="ch",
        disable=not verbose,
        leave=True,
        position=progress_tracking_position,
    ):
        ch_result = process_channel(
            montaged_data[ch_idx], fs, bins, params,
            combined_matrix=combined_matrix,
        )
        for window_ms, arr in ch_result.items():
            per_window[window_ms].append(arr)

    # Stack into 2-D arrays: [n_channels, n_down]
    result_dict: dict[int, np.ndarray] = {
        w: np.stack(arrs, axis=0) for w, arrs in per_window.items()
    }

    return result_dict, montaged_names, bins
