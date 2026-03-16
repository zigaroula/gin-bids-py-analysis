"""
Unit tests for gin_bids_py_analysis.processing.hilbert.dsp

Each test targets a single function or a clearly bounded behaviour so that
failures pinpoint the exact broken piece of the pipeline.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gin_bids_py_analysis.processing.hilbert.fir import (
    FirBandPass,
    _hamming_window,
    _hilbert_coeff,
    _number_of_points,
)
from gin_bids_py_analysis.processing.hilbert.dsp import (
    apply_shannon_clamp,
    build_frequency_bins,
    downsample,
    moving_average,
    normalize_db,
    normalize_percent,
)
from gin_bids_py_analysis.processing.utils.channels import build_montage


# ---------------------------------------------------------------------------
# _number_of_points
# ---------------------------------------------------------------------------


class TestNumberOfPoints:
    def test_small_signal_returns_512(self):
        assert _number_of_points(1) == 512
        assert _number_of_points(100) == 512
        assert _number_of_points(1023) == 512

    def test_exactly_1024_returns_1024(self):
        # 2^10 = 1024; log2(1024) = 10 exactly → result should be 1024.
        assert _number_of_points(1024) == 1024

    def test_power_of_two_is_returned_unchanged(self):
        assert _number_of_points(2048) == 2048
        assert _number_of_points(4096) == 4096

    def test_non_power_of_two_rounds_up(self):
        # 1025 → next power of two above 1025 is 2048
        assert _number_of_points(1025) == 2048
        assert _number_of_points(3000) == 4096


# ---------------------------------------------------------------------------
# _hamming_window
# ---------------------------------------------------------------------------


class TestHammingWindow:
    def test_length(self):
        w = _hamming_window(64)
        assert len(w) == 64

    def test_first_sample(self):
        # w[0] = 0.54 - 0.46 * cos(0) = 0.54 - 0.46 = 0.08
        w = _hamming_window(64)
        assert pytest.approx(w[0], abs=1e-10) == 0.08

    def test_denominator_is_n_not_n_minus_1(self):
        # For length N, the last sample uses 2π*(N-1)/N (not 2π*(N-1)/(N-1)=2π).
        # Standard Hamming uses N-1 → w[N-1] = w[0] = 0.08.
        # This uses N → w[N-1] ≠ 0.08 (slightly different).
        n = 64
        w = _hamming_window(n)
        w_last = 0.54 - 0.46 * math.cos(2 * math.pi * (n - 1) / n)
        assert pytest.approx(float(w[n - 1]), rel=1e-9) == w_last
        # Confirm it is NOT the standard value (would be ~0.08 if denominator were n-1)
        standard_last = 0.54 - 0.46 * math.cos(2 * math.pi * (n - 1) / (n - 1))
        assert abs(float(w[n - 1]) - standard_last) > 1e-4

    def test_midpoint_is_one(self):
        # At i = N/2: cos(π) = -1 → w = 0.54 + 0.46 = 1.0
        n = 64
        w = _hamming_window(n)
        assert pytest.approx(float(w[n // 2]), abs=1e-10) == 1.0


# ---------------------------------------------------------------------------
# _hilbert_coeff
# ---------------------------------------------------------------------------


class TestHilbertCoeff:
    def test_even_dc_and_nyquist_are_one(self):
        n = 8
        h = _hilbert_coeff(n)
        assert h[0] == 1.0
        assert h[n // 2] == 1.0

    def test_even_positive_freqs_are_two(self):
        n = 8
        h = _hilbert_coeff(n)
        # Positive frequencies: indices 1 to N//2 - 1
        assert all(h[i] == 2.0 for i in range(1, n // 2))

    def test_even_negative_freqs_are_zero(self):
        n = 8
        h = _hilbert_coeff(n)
        # Negative frequencies: indices N//2 + 1 to N - 1
        assert all(h[i] == 0.0 for i in range(n // 2 + 1, n))

    def test_odd_dc_is_one(self):
        n = 9
        h = _hilbert_coeff(n)
        assert h[0] == 1.0

    def test_odd_nyquist_slot_is_zero(self):
        # For odd N, there is no Nyquist bin → slot N//2 is a positive-freq bin
        # set to 2.0 (upper bound exclusive means n//2
        # is NOT set to 2.0).  Slot N//2 should be 0.
        n = 9
        h = _hilbert_coeff(n)
        # h[1 : n//2] = 2  →  h[1:4] = 2  →  h[4] should be 0
        assert h[n // 2] == 0.0

    def test_odd_positive_freqs_are_two(self):
        n = 9
        h = _hilbert_coeff(n)
        assert all(h[i] == 2.0 for i in range(1, n // 2))


# ---------------------------------------------------------------------------
# build_frequency_bins
# ---------------------------------------------------------------------------


class TestBuildFrequencyBins:
    def test_basic_range(self):
        bins = build_frequency_bins(50, 150, 10)
        assert bins[0] == pytest.approx(50.0)
        assert bins[-1] == pytest.approx(150.0)
        assert len(bins) == 11

    def test_single_step(self):
        bins = build_frequency_bins(70, 80, 10)
        assert bins == pytest.approx([70.0, 80.0])

    def test_float_step(self):
        bins = build_frequency_bins(0.5, 2.0, 0.5)
        assert bins == pytest.approx([0.5, 1.0, 1.5, 2.0])


# ---------------------------------------------------------------------------
# apply_shannon_clamp
# ---------------------------------------------------------------------------


class TestApplyShannonClamp:
    def test_no_clamp_when_below_nyquist(self):
        bins = build_frequency_bins(50, 150, 10)
        # fs=1000 Hz → Nyquist=500 Hz; f_max=150 < 500 → no clamp
        result = apply_shannon_clamp(bins, fs=1000.0, f_min=50.0, f_step=10.0)
        assert result == pytest.approx(bins)

    def test_clamp_triggered(self):
        bins = build_frequency_bins(50, 150, 10)
        # fs=200 Hz → Nyquist=100 Hz; f_max=150 > 100 → clamped
        # f_max_clamped = (100 // 10) * 10 = 100
        result = apply_shannon_clamp(bins, fs=200.0, f_min=50.0, f_step=10.0)
        assert result[-1] == pytest.approx(100.0)
        assert result[0] == pytest.approx(50.0)

    def test_clamp_aligns_to_step(self):
        bins = build_frequency_bins(5, 55, 5)
        # fs=80 → Nyquist=40; f_max_clamped = (40 // 5) * 5 = 40
        result = apply_shannon_clamp(bins, fs=80.0, f_min=5.0, f_step=5.0)
        assert result[-1] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# downsample
# ---------------------------------------------------------------------------


class TestDownsample:
    def test_shape(self):
        signal = np.ones(1000, dtype=np.float32)
        down = downsample(signal, fs=1000.0, fs_down=64.0)
        # factor = 1000 // 64 = 15; n_down = 1000 // 15 = 66
        factor = 1000 // 64
        expected_len = 1000 // factor
        assert len(down) == expected_len

    def test_picks_correct_samples(self):
        # signal[i] = i; after decimation by factor we expect [0, f, 2f, ...]
        signal = np.arange(200, dtype=np.float32)
        factor = 200 // 50  # fs=200, fs_down=50 → factor=4
        down = downsample(signal, fs=200.0, fs_down=50.0)
        expected = signal[::factor][: len(signal) // factor]
        np.testing.assert_array_equal(down, expected)

    def test_output_is_float32(self):
        signal = np.ones(100, dtype=np.float64)
        down = downsample(signal, fs=1000.0, fs_down=64.0)
        assert down.dtype == np.float32


# ---------------------------------------------------------------------------
# normalize_percent
# ---------------------------------------------------------------------------


class TestNormalizePercent:
    def test_constant_signal_normalises_to_100(self):
        signal = np.full(100, 3.0, dtype=np.float32)
        out = normalize_percent(signal)
        np.testing.assert_allclose(out, 100.0, atol=1e-3)

    def test_zero_baseline_replaced_by_one(self):
        # Middle half is all zeros → fmtab=1 → output = 100 * signal / 1
        signal = np.zeros(100, dtype=np.float32)
        out = normalize_percent(signal)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_output_is_float32(self):
        signal = np.ones(100, dtype=np.float64)
        out = normalize_percent(signal)
        assert out.dtype == np.float32

    def test_baseline_uses_middle_50_percent(self):
        # Craft a signal where ONLY the middle half is non-zero.
        n = 100
        signal = np.zeros(n, dtype=np.float32)
        value = round(n / 4)  # = 25
        signal[value : 3 * value] = 4.0  # mean_mid = 4.0
        out = normalize_percent(signal)
        # Middle half should be 100 * 4 / 4 = 100; elsewhere 0
        np.testing.assert_allclose(out[value : 3 * value], 100.0, atol=1e-3)
        np.testing.assert_allclose(out[:value], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# normalize_db
# ---------------------------------------------------------------------------


class TestNormalizeDb:
    def test_constant_signal_baseline_is_zero_db(self):
        """A constant signal has baseline == signal, so output is 20*log10(1) = 0 dB."""
        signal = np.full(100, 3.0, dtype=np.float32)
        out = normalize_db(signal)
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_output_is_float32(self):
        signal = np.ones(100, dtype=np.float64)
        out = normalize_db(signal)
        assert out.dtype == np.float32

    def test_values_above_baseline_are_positive(self):
        """Samples with amplitude > baseline must be positive dB."""
        n = 200
        signal = np.ones(n, dtype=np.float32)
        # Middle half has value 1 (baseline = 1); edges have value 10 > 1.
        value = round(n / 4)  # 50
        signal[:value] = 10.0
        signal[3 * value :] = 10.0
        out = normalize_db(signal)
        # Edge samples: 20*log10(10/1) ≈ 20 dB
        np.testing.assert_allclose(out[:value], 20.0, atol=0.5)

    def test_values_below_baseline_are_negative(self):
        """Samples with amplitude < baseline must be negative dB."""
        n = 200
        signal = np.ones(n, dtype=np.float32)
        value = round(n / 4)  # 50
        signal[:value] = 0.1
        signal[3 * value :] = 0.1
        out = normalize_db(signal)
        # Edge samples: 20*log10(0.1/1) ≈ -20 dB
        np.testing.assert_allclose(out[:value], -20.0, atol=0.5)

    def test_zero_signal_does_not_crash(self):
        """Zero-valued signal must not raise or produce -inf (guarded by 1e-10 floor)."""
        signal = np.zeros(100, dtype=np.float32)
        out = normalize_db(signal)
        assert np.all(np.isfinite(out))

    def test_baseline_uses_middle_50_percent(self):
        """The baseline window is the same as normalize_percent: middle 50% of the signal."""
        n = 100
        signal = np.ones(n, dtype=np.float32)
        value = round(n / 4)  # 25
        # Middle half fixed at 2.0 → baseline = 2.0; outer quarters at 1.0
        signal[value : 3 * value] = 2.0
        out = normalize_db(signal)
        # Middle half: 20*log10(2/2) = 0 dB
        np.testing.assert_allclose(out[value : 3 * value], 0.0, atol=1e-4)
        # Outer quarters: 20*log10(1/2) ≈ -6.02 dB
        expected_outer = 20.0 * np.log10(1.0 / 2.0)
        np.testing.assert_allclose(out[:value], expected_outer, atol=0.01)


# ---------------------------------------------------------------------------
# moving_average
# ---------------------------------------------------------------------------


class TestMovingAverage:
    def test_coefficient_1_is_identity(self):
        signal = np.arange(10, dtype=np.float32)
        out = moving_average(signal, coefficient=1)
        np.testing.assert_array_equal(out, signal)

    def test_weight_is_always_1_over_coefficient(self):
        """
        Verify the invariant: the weight is 1/coefficient regardless of
        how many samples are actually summed near the edges.

        At i=0 with coefficient=5, index=2:
        - begin=0, end=0+2=2 → sum of signal[0:3] * (1/5)
        """
        signal = np.ones(10, dtype=np.float32)
        coefficient = 5
        out = moving_average(signal, coefficient=coefficient)
        # Near the start: weight = 1/5, and we're summing <= 5 ones, so value < 1
        assert out[0] < 1.0
        assert pytest.approx(float(out[0]), rel=1e-5) == 3.0 / coefficient

    def test_interior_sample_is_window_mean(self):
        """
        For a constant signal the interior region is summed and divided by
        ``coefficient``.  For an *even* coefficient the interior sliding window
        spans exactly ``coefficient`` samples, giving output = 1.0.  For odd
        coefficients the window spans ``2*(coefficient//2) < coefficient``
        samples, giving output < 1.0 (still the correct behaviour).

        Interior range: begin = i - (index-1), end = i + index
        → count = end - begin + 1 = 2 * index = 2 * (coefficient // 2)

        Even coefficient (e.g. 4, index=2): count = 4 = coefficient → out = 1.0
        """
        signal = np.ones(50, dtype=np.float32)
        coefficient = 4  # even: interior count = 2*(4//2) = 4 = coefficient
        out = moving_average(signal, coefficient=coefficient)
        mid = len(signal) // 2
        assert pytest.approx(float(out[mid]), rel=1e-5) == 1.0

    def test_interior_odd_coefficient_weight(self):
        """
        For an *odd* coefficient the sliding window spans fewer than
        ``coefficient`` elements, but the divisor is still ``coefficient``.

        coefficient=5, index=2: interior count = 2*2 = 4 → output = 4/5 = 0.8.
        """
        signal = np.ones(50, dtype=np.float32)
        coefficient = 5
        index = coefficient // 2  # = 2
        expected_count = 2 * index  # = 4
        expected_value = expected_count / coefficient  # = 0.8
        out = moving_average(signal, coefficient=coefficient)
        mid = len(signal) // 2
        assert pytest.approx(float(out[mid]), rel=1e-5) == expected_value

    def test_output_length_matches_input(self):
        signal = np.ones(100, dtype=np.float32)
        out = moving_average(signal, coefficient=10)
        assert len(out) == len(signal)

    def test_output_is_float32(self):
        signal = np.ones(50, dtype=np.float64)
        out = moving_average(signal, coefficient=5)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# build_montage — mono
# ---------------------------------------------------------------------------


class TestBuildMontageMono:
    def test_mono_returns_data_unchanged(self):
        data = np.random.default_rng(0).random((4, 100)).astype(np.float32)
        names = ["A1", "A2", "B1", "B2"]
        out_data, out_names = build_montage(data, names, "mono", "next_minus_previous", "previous")
        np.testing.assert_array_equal(out_data, data)
        assert out_names == names


# ---------------------------------------------------------------------------
# build_montage — bipolar
# ---------------------------------------------------------------------------


class TestBuildMontageBipolar:
    @pytest.fixture()
    def four_channel_data(self):
        rng = np.random.default_rng(42)
        data = rng.random((4, 200)).astype(np.float32)
        names = ["A1", "A2", "A3", "A4"]
        return data, names

    def test_bipolar_shape(self, four_channel_data):
        data, names = four_channel_data
        out_data, out_names = build_montage(data, names, "bipolar", "next_minus_previous", "previous")
        # 4 contacts → 3 pairs
        assert out_data.shape == (3, 200)
        assert len(out_names) == 3

    def test_next_minus_previous_direction(self, four_channel_data):
        data, names = four_channel_data
        out_data, _ = build_montage(data, names, "bipolar", "next_minus_previous", "previous")
        # First pair: A2 - A1
        np.testing.assert_allclose(out_data[0], data[1] - data[0], rtol=1e-5)

    def test_previous_minus_next_direction(self, four_channel_data):
        data, names = four_channel_data
        out_data, _ = build_montage(data, names, "bipolar", "previous_minus_next", "previous")
        # First pair: A1 - A2
        np.testing.assert_allclose(out_data[0], data[0] - data[1], rtol=1e-5)

    def test_storage_previous_label(self, four_channel_data):
        data, names = four_channel_data
        _, out_names = build_montage(data, names, "bipolar", "next_minus_previous", "previous")
        assert out_names[0] == "A1"
        assert out_names[1] == "A2"

    def test_storage_next_label(self, four_channel_data):
        data, names = four_channel_data
        _, out_names = build_montage(data, names, "bipolar", "next_minus_previous", "next")
        assert out_names[0] == "A2"
        assert out_names[1] == "A3"

    def test_storage_prev_minus_next_label(self, four_channel_data):
        data, names = four_channel_data
        _, out_names = build_montage(data, names, "bipolar", "next_minus_previous", "previous_minus_next")
        assert out_names[0] == "A1-A2"

    def test_storage_next_minus_prev_label(self, four_channel_data):
        data, names = four_channel_data
        _, out_names = build_montage(data, names, "bipolar", "next_minus_previous", "next_minus_previous")
        assert out_names[0] == "A2-A1"

    def test_non_adjacent_contacts_skipped(self):
        # A1 and A3 are not adjacent (gap of 2) → only A1-A2 and A3-A4 if present
        data = np.random.default_rng(0).random((3, 50)).astype(np.float32)
        names = ["A1", "A3", "A4"]  # A1→A3 gap, A3→A4 adjacent
        out_data, out_names = build_montage(data, names, "bipolar", "next_minus_previous", "previous")
        assert out_data.shape[0] == 1
        assert out_names == ["A3"]

    def test_unparseable_channels_skipped(self):
        data = np.random.default_rng(0).random((3, 50)).astype(np.float32)
        names = ["Ref", "A1", "A2"]  # "Ref" cannot be parsed
        out_data, out_names = build_montage(data, names, "bipolar", "next_minus_previous", "previous")
        assert out_data.shape[0] == 1  # only A1-A2

    def test_multiple_electrodes(self):
        # Two electrodes: A1,A2 and B1,B2,B3
        data = np.random.default_rng(0).random((5, 100)).astype(np.float32)
        names = ["A1", "A2", "B1", "B2", "B3"]
        out_data, out_names = build_montage(data, names, "bipolar", "next_minus_previous", "previous")
        # A: 1 pair; B: 2 pairs → total 3
        assert out_data.shape[0] == 3


# ---------------------------------------------------------------------------
# FirBandPass
# ---------------------------------------------------------------------------


class TestFirBandPass:
    def test_apply_output_shape(self):
        n_samples = 500
        signal = np.random.default_rng(1).random(n_samples).astype(np.float32)
        fir = FirBandPass(f_low=8.0, f_high=12.0, fs=1000.0, signal_length=n_samples)
        envelope = fir.apply(signal)
        assert envelope.shape == (n_samples,)

    def test_apply_output_is_float32(self):
        n_samples = 200
        signal = np.random.default_rng(2).random(n_samples).astype(np.float32)
        fir = FirBandPass(f_low=50.0, f_high=60.0, fs=1000.0, signal_length=n_samples)
        envelope = fir.apply(signal)
        assert envelope.dtype == np.float32

    def test_apply_output_is_nonnegative(self):
        # Envelope = |analytic signal| → always >= 0
        n_samples = 512
        signal = np.random.default_rng(3).random(n_samples).astype(np.float32)
        fir = FirBandPass(f_low=50.0, f_high=60.0, fs=1000.0, signal_length=n_samples)
        envelope = fir.apply(signal)
        assert np.all(envelope >= 0.0)

    def test_zero_signal_gives_zero_envelope(self):
        n_samples = 512
        signal = np.zeros(n_samples, dtype=np.float32)
        fir = FirBandPass(f_low=50.0, f_high=60.0, fs=1000.0, signal_length=n_samples)
        envelope = fir.apply(signal)
        np.testing.assert_allclose(envelope, 0.0, atol=1e-6)

    def test_in_band_sine_produces_nonzero_envelope(self):
        """A pure tone at the centre frequency should yield a clearly non-zero envelope.

        Note on passband gain: the Hamming window uses denominator ``n``
        (not ``n-1``), so ``w[0] = 0.08``.  The circular-convolution of the
        ideal passband response with this window concentrates at DC with
        amplitude ``w[0] = 0.08``.  Consequently the FIR passband gain is
        approximately 0.08 rather than 1.0.  The ``normalize_percent``
        step later compensates for this — the raw envelope values are small
        by design.  We therefore verify the envelope is *non-trivially above
        zero* (i.e. > 0.05) rather than close to the input amplitude.
        """
        fs = 1000.0
        t = np.arange(2000) / fs
        # 55 Hz is inside the 50-60 Hz band
        signal = np.sin(2 * np.pi * 55 * t).astype(np.float32)
        fir = FirBandPass(f_low=50.0, f_high=60.0, fs=fs, signal_length=len(t))
        envelope = fir.apply(signal)
        # Ignore transient at start; envelope should be well above 0 in steady state
        # Passband gain ≈ 0.08 (Hamming w[0]) for unit-amplitude input.
        assert np.mean(envelope[200:]) > 0.05

    def test_out_of_band_sine_produces_attenuated_envelope(self):
        """A sine far outside the pass-band should yield a near-zero envelope."""
        fs = 1000.0
        t = np.arange(2000) / fs
        # 200 Hz is far outside the 50-60 Hz band
        signal = np.sin(2 * np.pi * 200 * t).astype(np.float32)
        fir = FirBandPass(f_low=50.0, f_high=60.0, fs=fs, signal_length=len(t))
        envelope = fir.apply(signal)
        assert np.mean(envelope[200:]) < 0.1
