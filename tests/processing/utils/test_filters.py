from __future__ import annotations

import numpy as np
import pytest
from mne import create_info
from mne.io import BaseRaw, RawArray

from bidsforge.processing.utils.filters import apply_notch_filter


def _make_raw() -> RawArray:
    info = create_info(ch_names=["A1"], sfreq=100.0, ch_types=["seeg"])
    return RawArray(np.zeros((1, 100), dtype=np.float64), info, verbose="ERROR")


def test_apply_notch_filter_disabled_returns_original_raw() -> None:
    raw = _make_raw()

    assert apply_notch_filter(raw, []) is raw
    assert apply_notch_filter(raw, None) is raw


def test_apply_notch_filter_enabled_copies_and_preloads(monkeypatch) -> None:
    raw = _make_raw()
    calls: list[list[float]] = []

    def fake_notch_filter(self, freqs, **kwargs):
        del kwargs
        calls.append(list(freqs))
        return self

    monkeypatch.setattr(BaseRaw, "notch_filter", fake_notch_filter)

    filtered = apply_notch_filter(raw, [40.0])

    assert filtered is not raw
    assert filtered.preload is True
    assert calls == [[40.0]]


def test_apply_notch_filter_rejects_freq_at_or_above_nyquist() -> None:
    raw = _make_raw()

    with pytest.raises(ValueError, match="Nyquist"):
        apply_notch_filter(raw, [50.0])



