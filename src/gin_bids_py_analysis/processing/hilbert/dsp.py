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
import numpy.fft as _fftmod

try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as _fftmod  # type: ignore[assignment]
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(60)
except ImportError:
    pass  # fall back to numpy.fft

# Number of subbands processed per batched IFFT call.
# Smaller values reduce peak memory at the cost of more kernel launches.
_CHUNK_SIZE = 4

if TYPE_CHECKING:
    from .params import HilbertParams

from gin_bids_py_analysis.processing.utils.channels import build_montage


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

    2. Apply a linear phase delay to form a Hermitian-symmetric spectrum of
       size ``2*(npt-1)`` (where ``npt = _number_of_points(signal_length)``).
    3. IFFT → take first ``signal_length`` real taps → apply Hamming window.
    4. FFT → ``fir_coeff`` (magnitude, float32, length ``signal_length``).
    5. ``combined = fir_coeff x hilbert_coeff``.

    Applying the filter
    -------------------
    For a signal *x* of exactly *signal_length* samples::

        Y = IFFT(FFT(x) x combined)
        envelope = |Y|

    Args:
        f_low:         Lower edge of the pass-band in Hz.
        f_high:        Upper edge of the pass-band in Hz.
        fs:            Sampling frequency in Hz.
        signal_length: Number of samples in the signals that will be
                       filtered.  The coefficient array has this length.
    """

    def __init__(
        self,
        f_low: float,
        f_high: float,
        fs: float,
        signal_length: int,
    ) -> None:
        self.f_low = f_low
        self.f_high = f_high
        self.fs = fs
        self.signal_length = signal_length

        nyquist = fs / 2.0

        # Piecewise-linear breakpoints and gains for the band-pass shape.
        # Clamp to [0, 1] so that edge subbands near Nyquist (where
        # f_high + 0.5 can marginally exceed Nyquist) remain within the
        # valid normalised frequency range.
        breakpoints = np.clip(np.array([
            0.0,
            (f_low - 0.5) / nyquist,
            f_low          / nyquist,
            f_high         / nyquist,
            (f_high + 0.5) / nyquist,
            1.0,
        ], dtype=np.float32), 0.0, 1.0)
        gains = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)

        # FIR magnitude response (float32, length signal_length)
        fir_coeff = self._build_fir_coefficients(signal_length, breakpoints, gains)

        # Combine with Hilbert coefficients (real float32 x float32).
        # Values are in [0, 2] for pass-band frequencies, 0 elsewhere.
        hilbert = self._hilbert_coeff(signal_length).astype(np.float32)
        self._combined: np.ndarray = fir_coeff * hilbert  # float32

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Apply the band-pass + Hilbert transform to *signal*.

        *signal* must have exactly ``self.signal_length`` samples.

        Args:
            signal: 1-D real array of length ``signal_length`` (any float dtype).

        Returns:
            Amplitude envelope as a **float32** array of the same length.
        """
        spectrum = _fftmod.fft(signal.astype(np.float32))           # complex64
        analytic = _fftmod.ifft(spectrum * self._combined)          # complex64
        return np.abs(analytic).astype(np.float32)

    def _number_of_points(self, signal_length: int) -> int:
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

    def _hamming_window(self, n: int) -> np.ndarray:
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

    def _hilbert_coeff(self, n: int) -> np.ndarray:
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

    def _interpolate_breakpoints_grid(
        self,
        npt: int,
        frequency: np.ndarray,
        magnitude: np.ndarray,
    ) -> np.ndarray:
        """Build a piecewise-linearly interpolated one-sided magnitude grid.

        The function fills a float32 array of length ``npt + 1`` by linearly
        interpolating between consecutive ``(frequency, magnitude)`` breakpoints.
        The running start index ``nb`` is 1-based.

        Args:
            npt:       Number of one-sided frequency points (from
                    :func:`_number_of_points`).
            frequency: Six normalised frequency breakpoints in [0, 1].
            magnitude: Six corresponding gain values.

        Returns:
            float32 array of length ``npt + 1``.
        """
        hh = np.zeros(npt + 1, dtype=np.float32)
        lap = int(npt / 25)     # trunc(npt / 25)
        n_pts = float(npt + 1)  # numberOfPoints after the +1 inside the reference loop

        hh[0] = np.float32(magnitude[0])
        nb = 1.0  # 1-based running start index

        for i in range(len(frequency) - 1):
            df_i = float(frequency[i + 1]) - float(frequency[i])
            if df_i < 0.0:
                raise ValueError("Frequencies must be non-decreasing")

            if df_i == 0.0:
                nb = nb - lap // 2
                ne: float = nb + lap
            else:
                ne = float(math.trunc(float(frequency[i + 1]) * n_pts))

            # Clamp ne to the valid grid boundary.  Breakpoints that collapse to
            # identical values at the boundary (e.g. both high-side points at 1.0
            # after Nyquist clamping) can otherwise push ne past the array end.
            ne = min(ne, n_pts)

            begin = int(nb) - 1
            count = int(ne) - begin  # = ne - nb + 1 elements

            if count > 0:
                if nb == ne:
                    hh[begin : begin + count] = np.float32(magnitude[i])
                else:
                    j_arr = np.arange(count, dtype=np.float32)
                    inc = j_arr / np.float32(ne - nb)
                    hh[begin : begin + count] = (
                        np.float32(magnitude[i + 1]) * inc
                        + np.float32(magnitude[i]) * (np.float32(1.0) - inc)
                    )

            nb = ne + 1.0

        return hh

    def _build_fir_coefficients(
        self,
        n_samples: int,
        frequency: np.ndarray,
        magnitude: np.ndarray,
    ) -> np.ndarray:
        """Compute FIR frequency-domain magnitude coefficients.

        Steps:

        1. Interpolate breakpoints → one-sided magnitude grid (float32).
        2. Apply a linear phase delay to form a phase-delayed one-sided spectrum.
        3. Mirror to a Hermitian-symmetric full spectrum of size ``2*(npt-1)``.
        *Note:* indices 0 and 1 of the conjugate-mirror half are intentionally zero.
        4. IFFT → take real part of first ``n_samples`` taps → apply Hamming window.
        5. FFT → magnitude (zero-phase FIR response, float32).

        Args:
            n_samples: Length of the signal that will be filtered.  The returned
                    coefficient array has this many elements.
            frequency: Six normalised frequency breakpoints in [0, 1].
            magnitude: Six corresponding gain values.

        Returns:
            float32 array of length ``n_samples``.
        """
        npt = self._number_of_points(n_samples)
        dt = np.float32(0.5 * (n_samples - 1))

        # One-sided magnitude grid HH[0..npt] (npt+1 elements)
        hh = self._interpolate_breakpoints_grid(npt, frequency, magnitude)

        # Phase-delayed one-sided spectrum H_Complex[0..npt-1] (complex64)
        i_arr = np.arange(npt, dtype=np.float32)
        phase = -dt * np.float32(math.pi) * i_arr / np.float32(npt - 1)
        h_complex = (hh[:npt] * np.exp(1j * phase)).astype(np.complex64)

        # Conjugate-mirror array H_Complex_Conjugate[0..npt-3] (complex64).
        # Reference loop: for (i = npt-3; i > 1; i--) → indices 2..npt-3 only.
        # Indices 0 and 1 remain zero (never assigned by the reference loop).
        h_conj = np.zeros(npt - 2, dtype=np.complex64)
        if npt > 4:
            # h_conj[2..npt-3] = h_complex[npt-4..1] (reversed)
            h_conj[2 : npt - 2] = h_complex[1 : npt - 3][::-1]

        # Hermitian-symmetric full spectrum H_Complex_Final[0..2*(npt-1)-1]
        h_final = np.empty(2 * (npt - 1), dtype=np.complex64)
        h_final[:npt] = h_complex
        h_final[npt:] = np.conj(h_conj)  # indices npt..2*(npt-1)-1

        # IFFT (size 2*(npt-1)), normalised by 1/(2*(npt-1)) — matches FFTW_BACKWARD
        h_time = _fftmod.ifft(h_final)

        # Take real part of first n_samples taps, apply Hamming window (float32)
        window = self._hamming_window(n_samples).astype(np.float32)
        h_windowed = h_time[:n_samples].real.astype(np.float32) * window

        # Forward FFT → magnitude (float32, length n_samples)
        return np.abs(_fftmod.fft(h_windowed)).astype(np.float32)

    @property
    def combined(self) -> np.ndarray:
        """Pre-computed frequency-domain coefficients (FIR x Hilbert, float32)."""
        return self._combined


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
    # Step 1b+c: chunked batched IFFT → amplitude envelope.
    # _CHUNK_SIZE subbands per call caps transient memory to
    # ~(_CHUNK_SIZE x n_samples x 8 B) for the complex64 temporaries.
    # ------------------------------------------------------------------
    row_chunks: list[np.ndarray] = []
    for start in range(0, n_subbands, _CHUNK_SIZE):
        chunk = combined_matrix[start : start + _CHUNK_SIZE]     # [c, n_samples] float32
        analytics = _fftmod.ifft(spectrum * chunk, axis=1)        # complex64 [c, n_samples]
        row_chunks.append(np.abs(analytics).astype(np.float32))

    envelopes_2d = np.concatenate(row_chunks, axis=0)  # [n_sub, n_samples] float32

    # ------------------------------------------------------------------
    # Step 2: vectorised downsample — one 2-D strided slice, no Python loop.
    # ------------------------------------------------------------------
    if params.do_downsample:
        factor = int(fs) // int(params.downsampled_frequency_hz)
        n_down = n_samples // factor
        envelopes_2d = np.ascontiguousarray(
            envelopes_2d[:, :n_down * factor:factor]
        )  # [n_sub, n_down] float32

    # ------------------------------------------------------------------
    # Step 3: vectorised normalisation — axis-wise mean, no Python loop.
    # ------------------------------------------------------------------
    if params.do_normalize_percent:
        n_len = envelopes_2d.shape[1]
        value = round(n_len / 4)
        mean_mid = np.mean(
            envelopes_2d[:, value : 3 * value], axis=1, keepdims=True
        )  # [n_sub, 1]
        fmtab = np.where(mean_mid != 0.0, mean_mid, 1.0).astype(np.float32)
        envelopes_2d = (100.0 * envelopes_2d / fmtab).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 4: average across subbands → 1-D [n_samples_eff] float32
    # ------------------------------------------------------------------
    mean_data = np.sum(envelopes_2d, axis=0, dtype=np.float32) / n_subbands

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

    for ch_idx in range(n_channels):
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
