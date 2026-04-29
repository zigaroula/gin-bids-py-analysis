"""
Integration test for the Hilbert-band envelope pipeline.

Uses synthetic numpy data only — no file I/O, no BIDS indexing, no MNE.
Verifies that :func:`process_all_channels` produces outputs of the correct
shape, dtype, and approximate range for a well-controlled input signal.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.hilbert.dsp import process_all_channels
from gin_bids_py_analysis.processing.hilbert.params import HilbertParams, MontageMode, NormalizationMode
import gin_bids_py_analysis.processing.hilbert.processor as processor_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_params() -> HilbertParams:
    """HilbertParams covering a small frequency range for fast tests."""
    return HilbertParams(
        f_min=50,
        f_max=100,
        f_step=10,
        downsampled_frequency_hz=64.0,
        montage_mode=MontageMode.MONO,
        smoothing_windows_ms=[0, 250, 1000],
        normalization_mode=NormalizationMode.PERCENT,
    )


def test_hilbert_params_normalize_notch_filter_freqs() -> None:
    assert HilbertParams(
        f_min=50,
        f_max=100,
        f_step=10,
        notch_filter_freqs=50,
    ).notch_filter_freqs == [50.0]
    assert HilbertParams(
        f_min=50,
        f_max=100,
        f_step=10,
        notch_filter_freqs="50, 150",
    ).notch_filter_freqs == [50.0, 150.0]


@pytest.mark.parametrize("freqs", [0, -50, float("nan"), float("inf"), "bad"])
def test_hilbert_params_reject_invalid_notch_filter_freqs(freqs: object) -> None:
    with pytest.raises(ValueError, match="notch_filter_freqs"):
        HilbertParams(
            f_min=50,
            f_max=100,
            f_step=10,
            notch_filter_freqs=freqs,
        )


@pytest.fixture()
def synthetic_data():
    """
    4-channel synthetic float32 iEEG signal at 1000 Hz.

    Each channel is a mixture of band-limited noise designed to produce a
    stable envelope so that normalization converges to values near 100.
    """
    rng = np.random.default_rng(seed=0)
    fs = 1000.0
    n_samples = 2000  # 2 seconds
    n_channels = 4

    # Broadband white noise — contains energy across all bands including 50-100 Hz
    data = rng.standard_normal((n_channels, n_samples)).astype(np.float32)
    return data, fs


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestProcessAllChannels:
    def test_output_keys_match_smoothing_windows(self, synthetic_data, default_params):
        data, fs = synthetic_data
        ch_names = [f"A{i+1}" for i in range(data.shape[0])]
        result, _, _ = process_all_channels(data, ch_names, fs, default_params)

        assert set(result.keys()) == set(default_params.smoothing_windows_ms)

    def test_output_shape(self, synthetic_data, default_params):
        data, fs = synthetic_data
        n_channels = data.shape[0]
        ch_names = [f"A{i+1}" for i in range(n_channels)]
        result, montaged_names, _ = process_all_channels(data, ch_names, fs, default_params)

        # Expected n_down: resample_poly uses polyphase rational resampling.
        # Output length = ceil(n_samples * up / down) where up/down = target_fs/fs.
        g = math.gcd(int(default_params.downsampled_frequency_hz), int(fs))
        up = int(default_params.downsampled_frequency_hz) // g
        down = int(fs) // g
        expected_n_down = math.ceil(data.shape[1] * up / down)

        for window_ms, array in result.items():
            assert array.shape == (n_channels, expected_n_down), (
                f"Wrong shape for sm{window_ms}: {array.shape}"
            )

    def test_output_dtype_is_float32(self, synthetic_data, default_params):
        data, fs = synthetic_data
        ch_names = [f"A{i+1}" for i in range(data.shape[0])]
        result, _, _ = process_all_channels(data, ch_names, fs, default_params)

        for window_ms, array in result.items():
            assert array.dtype == np.float32, f"sm{window_ms} dtype: {array.dtype}"

    def test_output_is_finite(self, synthetic_data, default_params):
        data, fs = synthetic_data
        ch_names = [f"A{i+1}" for i in range(data.shape[0])]
        result, _, _ = process_all_channels(data, ch_names, fs, default_params)

        for window_ms, array in result.items():
            assert np.all(np.isfinite(array)), (
                f"sm{window_ms} contains non-finite values"
            )

    def test_normalized_values_near_100(self, synthetic_data, default_params):
        """
        After percent-normalization the mean of the middle portion of the
        unsmoothed output (sm0) should be close to 100 for all channels.
        The normalization is based on the middle half of the signal, so the
        grand mean of the entire trace should also be reasonably close.
        """
        data, fs = synthetic_data
        ch_names = [f"A{i+1}" for i in range(data.shape[0])]
        result, _, _ = process_all_channels(data, ch_names, fs, default_params)

        sm0 = result[0]  # unsmoothed, float32 [n_channels, n_down]
        for ch_idx in range(sm0.shape[0]):
            n = sm0.shape[1]
            mid_mean = float(np.mean(sm0[ch_idx, n // 4 : 3 * n // 4]))
            # The middle half average should be within 50% of 100
            assert 50 < mid_mean < 200, (
                f"Channel {ch_idx}: middle-half mean = {mid_mean:.1f}, expected ~100"
            )

    def test_montaged_names_returned(self, synthetic_data, default_params):
        data, fs = synthetic_data
        ch_names = [f"A{i+1}" for i in range(data.shape[0])]
        _, names, _ = process_all_channels(data, ch_names, fs, default_params)
        # Mono mode → names unchanged
        assert names == ch_names

    def test_centered_option_shifts_baseline(self, synthetic_data):
        """PERCENT_CENTERED should shift the mean of sm0 from ~100 to ~0."""
        params_centered = HilbertParams(
            f_min=50, f_max=100, f_step=10,
            montage_mode=MontageMode.MONO,
            smoothing_windows_ms=[0],
            normalization_mode=NormalizationMode.PERCENT_CENTERED,
        )
        data, fs = synthetic_data
        ch_names = [f"A{i+1}" for i in range(data.shape[0])]
        result, _, _ = process_all_channels(data, ch_names, fs, params_centered)

        sm0 = result[0]
        for ch_idx in range(sm0.shape[0]):
            n = sm0.shape[1]
            mid_mean = float(np.mean(sm0[ch_idx, n // 4 : 3 * n // 4]))
            # Centred: baseline should be near 0 (was ~100, minus 100)
            assert -50 < mid_mean < 50, (
                f"Channel {ch_idx}: centered mid-mean = {mid_mean:.1f}, expected ~0"
            )

    def test_no_downsample_returns_original_length(self, synthetic_data):
        params_no_down = HilbertParams(
            f_min=50, f_max=100, f_step=10,
            montage_mode=MontageMode.MONO,
            smoothing_windows_ms=[0],
            downsampled_frequency_hz=None,
            normalization_mode=NormalizationMode.PERCENT,
        )
        data, fs = synthetic_data
        ch_names = [f"A{i+1}" for i in range(data.shape[0])]
        result, _, _ = process_all_channels(data, ch_names, fs, params_no_down)

        sm0 = result[0]
        assert sm0.shape[1] == data.shape[1]

    def test_shannon_clamp_applied_automatically(self):
        """
        If f_max exceeds Nyquist the pipeline should silently clamp and still
        produce valid output (no exception).
        """
        fs = 200.0  # Nyquist = 100 Hz
        # f_max=150 exceeds Nyquist → should be clamped to 100
        params = HilbertParams(
            f_min=50, f_max=150, f_step=10,
            montage_mode=MontageMode.MONO,
            smoothing_windows_ms=[0],
            normalization_mode=NormalizationMode.PERCENT,
            downsampled_frequency_hz=16.0,
        )
        n_samples = 500
        data = np.random.default_rng(7).standard_normal((2, n_samples)).astype(np.float32)
        ch_names = ["A1", "A2"]
        result, _, _ = process_all_channels(data, ch_names, fs, params)
        assert 0 in result
        assert np.all(np.isfinite(result[0]))


class _FakeRaw:
    def __init__(self, data: np.ndarray, ch_names: list[str], fs: float) -> None:
        self._data = data
        self.ch_names = ch_names
        self.info = {"sfreq": fs}
        self.annotations = []

    def get_data(self) -> np.ndarray:
        return self._data


class TestHilbertProcessingChannelSelection:
    def test_process_group_can_use_events_tsv_secondary(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
        mock_bids_file,
    ) -> None:
        fs = 512.0
        data = np.zeros((1, 32), dtype=np.float32)
        raw = _FakeRaw(data, ["A1"], fs)
        events_file = BIDSFile.from_path(
            tmp_path / "sub-01_ses-01_task-rest_run-1_events.tsv"
        ).attach_data(
            [
                {
                    "onset": "152.13746632495895",
                    "duration": "0",
                    "trial_type": "Trigger",
                    "code": "11",
                }
            ]
        )

        def fake_process_all_channels(
            data_2d: np.ndarray,
            channel_names: list[str],
            fs: float,
            params: HilbertParams,
            **_: object,
        ):
            del data_2d, fs, params
            return (
                {0: np.zeros((len(channel_names), 10), dtype=np.float32)},
                list(channel_names),
                [50.0, 60.0],
            )

        mock_bids_file.attach_data(raw)
        monkeypatch.setattr(processor_module, "process_all_channels", fake_process_all_channels)

        params = HilbertParams(
            f_min=50,
            f_max=60,
            f_step=10,
            montage_mode=MontageMode.MONO,
            smoothing_windows_ms=[0],
            downsampled_frequency_hz=100.0,
            normalization_mode=NormalizationMode.NONE,
            events_source="events_tsv",
        )

        result = processor_module.HilbertProcessing(params).process_group(
            BIDSFileGroup(primary=mock_bids_file, secondaries=[events_file])
        )

        assert result.original_events[0]["onset"] == pytest.approx(152.13746632495895)
        assert result.metadata["events_source_resolved"] == "events_tsv"
        assert result.metadata["events_onset_precision"] == "exact_time"

    def test_process_group_subsets_data_and_names_together(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_bids_file,
    ) -> None:
        fs = 1000.0
        n_samples = 32
        data = np.vstack([
            np.full(n_samples, fill_value=row_idx, dtype=np.float32)
            for row_idx in range(5)
        ])
        ch_names = ["A1", "A2", "A3", "B1", "B2"]
        raw = _FakeRaw(data, ch_names, fs)
        captured: dict[str, np.ndarray | list[str] | float] = {}

        def fake_process_all_channels(
            data_2d: np.ndarray,
            channel_names: list[str],
            fs: float,
            params: HilbertParams,
            **_: object,
        ):
            captured["data"] = data_2d.copy()
            captured["channel_names"] = list(channel_names)
            captured["fs"] = fs
            return (
                {0: np.zeros((len(channel_names), data_2d.shape[1]), dtype=np.float32)},
                list(channel_names),
                [50.0, 60.0],
            )

        mock_bids_file.attach_data(raw)
        monkeypatch.setattr(processor_module, "process_all_channels", fake_process_all_channels)

        params = HilbertParams(
            f_min=50,
            f_max=60,
            f_step=10,
            montage_mode=MontageMode.MONO,
            smoothing_windows_ms=[0],
            downsampled_frequency_hz=None,
            normalization_mode=NormalizationMode.NONE,
            channels_for_montage=["B1", "B2"],
        )

        result = processor_module.HilbertProcessing(params).process_group(
            BIDSFileGroup(primary=mock_bids_file)
        )

        np.testing.assert_array_equal(captured["data"], data[[3, 4], :])
        assert captured["channel_names"] == ["B1", "B2"]
        assert captured["fs"] == fs
        assert result.channel_names == ["B1", "B2"]

    def test_process_group_applies_notch_filter_before_processing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_bids_file,
    ) -> None:
        fs = 1000.0
        n_samples = 32
        raw = _FakeRaw(
            np.zeros((1, n_samples), dtype=np.float32),
            ["A1"],
            fs,
        )
        filtered_raw = _FakeRaw(
            np.ones((1, n_samples), dtype=np.float32),
            ["A1"],
            fs,
        )
        captured: dict[str, object] = {}

        def fake_apply_notch_filter(raw_arg, freqs):
            captured["raw_arg"] = raw_arg
            captured["freqs"] = list(freqs)
            return filtered_raw

        def fake_process_all_channels(
            data_2d: np.ndarray,
            channel_names: list[str],
            fs: float,
            params: HilbertParams,
            **_: object,
        ):
            del channel_names, fs, params
            captured["data"] = data_2d.copy()
            return (
                {0: np.zeros((1, data_2d.shape[1]), dtype=np.float32)},
                ["A1"],
                [50.0, 60.0],
            )

        mock_bids_file.attach_data(raw)
        monkeypatch.setattr(processor_module, "apply_notch_filter", fake_apply_notch_filter)
        monkeypatch.setattr(processor_module, "process_all_channels", fake_process_all_channels)

        params = HilbertParams(
            f_min=50,
            f_max=60,
            f_step=10,
            montage_mode=MontageMode.MONO,
            smoothing_windows_ms=[0],
            downsampled_frequency_hz=None,
            normalization_mode=NormalizationMode.NONE,
            notch_filter_freqs=[50.0],
        )

        result = processor_module.HilbertProcessing(params).process_group(
            BIDSFileGroup(primary=mock_bids_file)
        )

        assert captured["raw_arg"] is raw
        assert captured["freqs"] == [50.0]
        np.testing.assert_array_equal(captured["data"], np.ones((1, n_samples), dtype=np.float32))
        assert result.metadata["notch_filter_freqs"] == [50.0]
        assert result.metadata["notch_filter_applied"] is True

    def test_process_group_applies_exclusion_after_inclusion(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_bids_file,
    ) -> None:
        fs = 1000.0
        n_samples = 32
        data = np.vstack([
            np.full(n_samples, fill_value=row_idx, dtype=np.float32)
            for row_idx in range(5)
        ])
        ch_names = ["A1", "A2", "A3", "B1", "B2"]
        raw = _FakeRaw(data, ch_names, fs)
        captured: dict[str, np.ndarray | list[str] | float] = {}

        def fake_process_all_channels(
            data_2d: np.ndarray,
            channel_names: list[str],
            fs: float,
            params: HilbertParams,
            **_: object,
        ):
            captured["data"] = data_2d.copy()
            captured["channel_names"] = list(channel_names)
            captured["fs"] = fs
            return (
                {0: np.zeros((len(channel_names), data_2d.shape[1]), dtype=np.float32)},
                list(channel_names),
                [50.0, 60.0],
            )

        mock_bids_file.attach_data(raw)
        monkeypatch.setattr(processor_module, "process_all_channels", fake_process_all_channels)

        params = HilbertParams(
            f_min=50,
            f_max=60,
            f_step=10,
            montage_mode=MontageMode.MONO,
            smoothing_windows_ms=[0],
            downsampled_frequency_hz=None,
            normalization_mode=NormalizationMode.NONE,
            channels_for_montage=["A1", "A2", "A3", "B1"],
            channels_to_exclude_for_montage=["A2", "B1"],
        )

        result = processor_module.HilbertProcessing(params).process_group(
            BIDSFileGroup(primary=mock_bids_file)
        )

        np.testing.assert_array_equal(captured["data"], data[[0, 2], :])
        assert captured["channel_names"] == ["A1", "A3"]
        assert captured["fs"] == fs
        assert result.channel_names == ["A1", "A3"]

    def test_process_group_raises_when_no_requested_channel_matches(
        self,
        mock_bids_file,
    ) -> None:
        raw = _FakeRaw(
            np.zeros((2, 16), dtype=np.float32),
            ["A1", "A2"],
            1000.0,
        )

        mock_bids_file.attach_data(raw)

        params = HilbertParams(
            f_min=50,
            f_max=60,
            f_step=10,
            montage_mode=MontageMode.MONO,
            smoothing_windows_ms=[0],
            downsampled_frequency_hz=None,
            normalization_mode=NormalizationMode.NONE,
            channels_for_montage=["B1", "B2"],
        )

        with pytest.raises(ValueError, match="channels_for_montage did not match any input channels"):
            processor_module.HilbertProcessing(params).process_group(
                BIDSFileGroup(primary=mock_bids_file)
            )


# ---------------------------------------------------------------------------
# NormalizationMode tests
# ---------------------------------------------------------------------------


class TestNormalizationModes:
    def test_db_mode_output_near_zero_at_baseline(self, synthetic_data):
        """With DB normalization the middle portion of the unsmoothed output should be near 0 dB."""
        data, fs = synthetic_data
        ch_names = [f"A{i+1}" for i in range(data.shape[0])]
        params = HilbertParams(
            f_min=50, f_max=100, f_step=10,
            montage_mode=MontageMode.MONO,
            smoothing_windows_ms=[0],
            normalization_mode=NormalizationMode.DB,
        )
        result, _, _ = process_all_channels(data, ch_names, fs, params)
        sm0 = result[0]
        for ch_idx in range(sm0.shape[0]):
            n = sm0.shape[1]
            mid_mean = float(np.mean(sm0[ch_idx, n // 4 : 3 * n // 4]))
            # Middle-half mean should be near 0 dB
            assert -10 < mid_mean < 10, (
                f"Channel {ch_idx}: DB baseline mean = {mid_mean:.2f} dB, expected ~0"
            )

    def test_db_mode_output_is_finite(self, synthetic_data):
        """DB normalization must never produce NaN or inf."""
        data, fs = synthetic_data
        ch_names = [f"A{i+1}" for i in range(data.shape[0])]
        params = HilbertParams(
            f_min=50, f_max=100, f_step=10,
            montage_mode=MontageMode.MONO,
            smoothing_windows_ms=[0],
            normalization_mode=NormalizationMode.DB,
        )
        result, _, _ = process_all_channels(data, ch_names, fs, params)
        for arr in result.values():
            assert np.all(np.isfinite(arr))

    def test_none_mode_output_is_amplitude(self, synthetic_data):
        """NONE normalization should leave the envelope in raw amplitude units (>0, not ~100)."""
        data, fs = synthetic_data
        ch_names = [f"A{i+1}" for i in range(data.shape[0])]
        params = HilbertParams(
            f_min=50, f_max=100, f_step=10,
            montage_mode=MontageMode.MONO,
            smoothing_windows_ms=[0],
            normalization_mode=NormalizationMode.NONE,
        )
        result, _, _ = process_all_channels(data, ch_names, fs, params)
        sm0 = result[0]
        # Raw amplitude from broadband noise at unit variance is small, not ~100
        assert float(np.mean(sm0)) < 10.0
