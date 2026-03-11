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
      - Compute the analytic signal via the Hilbert trick.
      - Take the magnitude → amplitude envelope (float32).
   b. Optionally decimate the envelope by an integer factor.
   c. Optionally normalise to a percentage of the mid-recording baseline.
   d. Average normalised envelopes across all subbands.
   e. Optionally apply a causal moving-average smoother for each requested
      window length.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .params import HilbertParams
    
from gin_bids_py_analysis.processing.utils.channels import build_montage


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _number_of_points(signal_length: int) -> int:
    """Return the FFT size used for a signal of *signal_length* samples.

    * If ``signal_length < 1024`` → return 512.
    * Otherwise → return the next power of two (``2^ceil(log2(signal_length))``).

    Args:
        signal_length: Number of samples in the input signal.

    Returns:
        FFT size (always a power of two).
    """
    if signal_length < 1024:
        return 512
    return 1 << math.ceil(math.log2(signal_length))


def _hamming_window(n: int) -> np.ndarray:
    """Compute the Hamming window of length *n*.

    The denominator is ``n`` (not ``n - 1`` as in the standard definition),
    which means the window is **not** exactly periodic

    Formula: ``w[i] = 0.54 - 0.46 · cos(2π · i / n)``

    Args:
        n: Window length.

    Returns:
        float64 array of length *n*.
    """
    i = np.arange(n, dtype=np.float64)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * i / n)


def _hilbert_coeff(n: int) -> np.ndarray:
    """Build the one-sided Hilbert frequency-domain coefficients for an
    *n*-point FFT.

    These weights turn a real spectrum into the spectrum of the analytic
    signal (positive frequencies doubled, DC and Nyquist preserved,
    negative frequencies zeroed):

    * Even *n*:  ``h[0] = 1,  h[1 : n//2] = 2,  h[n//2] = 1,  rest = 0``
    * Odd *n*:   ``h[0] = 1,  h[1 : n//2] = 2,  rest = 0``
      (upper bound is **exclusive**)

    Args:
        n: FFT size.

    Returns:
        float64 array of length *n*.
    """
    h = np.zeros(n, dtype=np.float64)
    h[0] = 1.0
    if n % 2 == 0:
        # Even: DC=1, positive freqs x 2, Nyquist=1
        h[1 : n // 2] = 2.0
        h[n // 2] = 1.0
    else:
        # Odd: DC=1, positive freqs x 2 (upper bound exclusive)
        h[1 : n // 2] = 2.0
    return h


# ---------------------------------------------------------------------------
# Band-pass + Hilbert filter
# ---------------------------------------------------------------------------


class FirBandPass:
    """Precomputed frequency-domain FIR band-pass filter for one subband.

    The filter is computed once at construction and reused for every signal
    of the same length (via :meth:`apply`).  Internally it stores the
    product of the FIR frequency response and the Hilbert coefficients so
    that a single complex multiply is needed per signal.

    Construction steps
    ------------------
    1. Define piecewise-linear *breakpoints* around the pass-band::

           [0,
            (f_low - 0.5) / Nyquist,
            f_low          / Nyquist,
            f_high         / Nyquist,
            (f_high + 0.5) / Nyquist,
            1.0]

       with corresponding *gains* ``[0, 0, 1, 1, 0, 0]``.

    2. Interpolate to a one-sided N-point response, mirror to two-sided.
    3. IFFT → time-domain taps → apply Hamming window.
    4. FFT → ``fir_coeff`` (complex, length N).
    5. ``combined = fir_coeff x hilbert_coeff``.

    Applying the filter
    -------------------
    For a signal *x* of up to *N* samples::

        Y = IFFT(FFT(zero_pad(x, N)) x combined)
        envelope = |Y|[:len(x)]

    Args:
        f_low:    Lower edge of the pass-band in Hz.
        f_high:   Upper edge of the pass-band in Hz.
        fs:       Sampling frequency in Hz.
        n_points: FFT size (must equal ``_number_of_points(len(signal))``
                  for the signals that will be filtered).
    """

    def __init__(
        self,
        f_low: float,
        f_high: float,
        fs: float,
        n_points: int,
    ) -> None:
        self.f_low = f_low
        self.f_high = f_high
        self.fs = fs
        self.n_points = n_points

        nyquist = fs / 2.0

        # ------------------------------------------------------------------
        # Step 1: piecewise-linear frequency response (one-sided, N//2+1 pts)
        # ------------------------------------------------------------------
        breakpoints = np.array([
            0.0,
            (f_low - 0.5) / nyquist,
            f_low          / nyquist,
            f_high         / nyquist,
            (f_high + 0.5) / nyquist,
            1.0,
        ])
        gains = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])

        # One-sided grid: 0 … 1 (inclusive) with N//2 + 1 points
        n_onesided = n_points // 2 + 1
        grid = np.linspace(0.0, 1.0, n_onesided)
        h_onesided = np.interp(grid, breakpoints, gains)

        # ------------------------------------------------------------------
        # Step 2: two-sided (full) frequency response
        # ------------------------------------------------------------------
        # Mirror: slot 0 = DC, slots 1..N//2-1 and their conjugate mirrors,
        # slot N//2 = Nyquist (for even N).
        h_full = np.empty(n_points, dtype=np.float64)
        h_full[:n_onesided] = h_onesided
        # Fill the second half by mirroring (negative frequencies)
        h_full[n_onesided:] = h_onesided[1 : n_points - n_onesided + 1][::-1]

        # ------------------------------------------------------------------
        # Step 3: IFFT → time-domain taps → apply Hamming window
        # ------------------------------------------------------------------
        h_time = np.real(np.fft.ifft(h_full))
        h_windowed = h_time * _hamming_window(n_points)

        # ------------------------------------------------------------------
        # Step 4: FFT → FIR frequency-domain coefficients
        # ------------------------------------------------------------------
        fir_coeff = np.fft.fft(h_windowed)

        # ------------------------------------------------------------------
        # Step 5: combine with Hilbert coefficients for one-shot multiply
        # ------------------------------------------------------------------
        hilbert = _hilbert_coeff(n_points)
        self._combined: np.ndarray = fir_coeff * hilbert  # complex128

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Apply the band-pass + Hilbert transform to *signal*.

        The signal is zero-padded to ``self.n_points`` before the FFT.
        Only the first ``len(signal)`` samples of the output are returned.

        Args:
            signal: 1-D real array (any float dtype).

        Returns:
            Amplitude envelope as a **float32** array with the same length
            as *signal*.
        """
        n = len(signal)
        # Zero-pad to FFT size
        padded = np.zeros(self.n_points, dtype=np.float64)
        padded[:n] = signal

        spectrum = np.fft.fft(padded)
        analytic = np.fft.ifft(spectrum * self._combined)
        envelope = np.abs(analytic[:n])
        return envelope.astype(np.float32)


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

    The baseline is computed as the mean of the middle half of the signal:

    * ``value = round(len(signal) / 4)``
    * ``baseline = mean(signal[value : 3 * value])``

    If the baseline is zero it is replaced with 1 to avoid division by zero.
    The result is scaled so that the baseline region has a mean of 100.

    Args:
        signal: 1-D float array (typically a downsampled envelope).

    Returns:
        Normalised float32 array (values in roughly the range [0, 200] for a
        stationary signal, centred on 100).
    """
    value = round(len(signal) / 4)
    mean_mid = np.mean(signal[value : 3 * value])
    fmtab = float(mean_mid) if float(mean_mid) != 0.0 else 1.0
    return (100.0 * signal / fmtab).astype(np.float32)


def moving_average(signal: np.ndarray, coefficient: int) -> np.ndarray:
    """Causal moving average.

    Unlike a standard moving average, the weight is always ``1 / coefficient``
    regardless of how many samples are actually summed near the edges.

    Edge logic (index = ``coefficient // 2``, weight = ``1 / coefficient``):

    * If ``i - index ≤ 0``:   ``begin = 0``,          ``end = i + index``
    * If ``i ≥ len - index``: ``begin = i - index + 1``, ``end = len - 1``
    * Otherwise:              ``begin = i - (index - 1)``, ``end = i + index``

    The output at sample *i* is ``sum(signal[begin : end + 1]) / coefficient``.

    A O(n) cumulative-sum implementation is used for efficiency while
    preserving the exact values.

    Args:
        signal:      1-D float array.
        coefficient: Window length; must be ≥ 1.  If 1, the signal is
                     returned unchanged (multiplied by 1.0/1 = identity).

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

    output = np.empty(n, dtype=np.float32)
    for i in range(n):
        if i - index <= 0:
            begin = 0
            end = i + index
        elif i >= n - index:
            begin = i - index + 1
            end = n - 1
        else:
            begin = i - (index - 1)
            end = i + index

        # Clamp end to valid range
        end = min(end, n - 1)
        total = cs[end + 1] - cs[begin]
        output[i] = float(total) * weight

    return output


# ---------------------------------------------------------------------------
# Channel-level pipeline
# ---------------------------------------------------------------------------


def process_channel(
    signal_1d: np.ndarray,
    fs: float,
    bins: list[float],
    params: "HilbertParams",
) -> dict[int, np.ndarray]:
    """Run the full Hilbert-band envelope pipeline on a single channel.

    Steps:

    1. For each adjacent bin pair (subband):
       a. Build (or retrieve from cache) a :class:`FirBandPass` filter.
       b. Apply the filter → amplitude envelope (float32).
    2. Optionally decimate each envelope.
    3. Optionally normalise to percentage of baseline.
    4. Average the normalised envelopes across subbands.
    5. For each smoothing window: apply :func:`moving_average`
       (or identity for window = 0), then optionally centre.

    Args:
        signal_1d: 1-D array ``[n_samples]``.
        fs:        Sampling frequency of *signal_1d*.
        bins:      Frequency bin edges (already Shannon-clamped).
        params:    :class:`~.params.HilbertParams` instance controlling all
                   pipeline toggles and settings.

    Returns:
        ``{window_ms: envelope_array}`` where each array has length
        ``n_downsampled`` (or ``n_samples`` if downsampling is disabled).
    """
    n_samples = len(signal_1d)
    n_points = _number_of_points(n_samples)

    # ------------------------------------------------------------------
    # Step 1 & 2: compute + (optionally) downsample envelope per subband
    # ------------------------------------------------------------------
    sub_envelopes: list[np.ndarray] = []
    for low, high in zip(bins[:-1], bins[1:]):
        fir = FirBandPass(low, high, fs, n_points)
        envelope = fir.apply(signal_1d)

        if params.do_downsample:
            envelope = downsample(envelope, fs, params.downsampled_frequency_hz)

        sub_envelopes.append(envelope)

    if not sub_envelopes:
        raise ValueError(
            "No subbands produced — bins list must have at least 2 elements."
        )

    # ------------------------------------------------------------------
    # Step 3: normalise each subband envelope
    # ------------------------------------------------------------------
    if params.do_normalize_percent:
        sub_envelopes = [normalize_percent(e) for e in sub_envelopes]

    # ------------------------------------------------------------------
    # Step 4: average across subbands
    # ------------------------------------------------------------------
    mean_data = np.sum(sub_envelopes, axis=0, dtype=np.float32) / len(sub_envelopes)

    # ------------------------------------------------------------------
    # Step 5: smoothing windows
    # ------------------------------------------------------------------
    fs_eff = params.downsampled_frequency_hz if params.do_downsample else fs
    result: dict[int, np.ndarray] = {}

    for window_ms in params.smoothing_windows_ms:
        if not params.do_smoothing or window_ms == 0:
            smoothed = mean_data.copy()
        else:
            coefficient = int((fs_eff * window_ms) / 1000)
            smoothed = moving_average(mean_data, coefficient)

        if params.centered:
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
    # Process each channel and collect results per smoothing window
    # ------------------------------------------------------------------
    # Initialise the output dict with empty lists
    per_window: dict[int, list[np.ndarray]] = {
        w: [] for w in params.smoothing_windows_ms
    }

    for ch_idx in range(montaged_data.shape[0]):
        ch_result = process_channel(montaged_data[ch_idx], fs, bins, params)
        print(f"Processed channel {ch_idx+1}/{montaged_data.shape[0]}")
        for window_ms, arr in ch_result.items():
            per_window[window_ms].append(arr)

    # Stack into 2-D arrays: [n_channels, n_down]
    result_dict: dict[int, np.ndarray] = {
        w: np.stack(arrs, axis=0) for w, arrs in per_window.items()
    }

    return result_dict, montaged_names, bins
