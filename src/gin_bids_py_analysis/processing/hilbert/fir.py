"""
Frequency-domain FIR band-pass filter and supporting DSP primitives.

This module contains the low-level building blocks used by
:mod:`gin_bids_py_analysis.processing.hilbert.dsp`:

- :func:`_number_of_points` — FFT size selection.
- :func:`_hamming_window`   — asymmetric Hamming window.
- :func:`_hilbert_coeff`    — one-sided analytic-signal coefficients.
- :class:`FirBandPass`      — precomputed frequency-domain FIR + Hilbert filter.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as _fftmod
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(60)
except ImportError:
    import numpy.fft as _fftmod  # fall back to numpy.fft


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
    which means the window is **not** exactly periodic.

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


def _interpolate_breakpoints_grid(
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
        hilbert = _hilbert_coeff(signal_length).astype(np.float32)
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
        npt = _number_of_points(n_samples)
        dt = np.float32(0.5 * (n_samples - 1))

        # One-sided magnitude grid HH[0..npt] (npt+1 elements)
        hh = _interpolate_breakpoints_grid(npt, frequency, magnitude)

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
        window = _hamming_window(n_samples).astype(np.float32)
        h_windowed = h_time[:n_samples].real.astype(np.float32) * window

        # Forward FFT → magnitude (float32, length n_samples)
        return np.abs(_fftmod.fft(h_windowed)).astype(np.float32)

    @property
    def combined(self) -> np.ndarray:
        """Pre-computed frequency-domain coefficients (FIR x Hilbert, float32)."""
        return self._combined
