"""Signal filtering helpers shared by processing pipelines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from mne.io import BaseRaw


def apply_notch_filter(raw: BaseRaw, freqs: Sequence[float] | None) -> BaseRaw:
    """Return a Raw object with optional notch filtering applied.

    Disabled filtering returns *raw* unchanged. Enabled filtering operates on a
    preloaded copy so callers do not mutate cached or attached ``Raw`` objects.
    """
    normalized_freqs = _normalize_notch_freqs(freqs)
    if not normalized_freqs:
        return raw

    sfreq = float(raw.info["sfreq"])
    nyquist = sfreq / 2.0
    invalid = [freq for freq in normalized_freqs if freq >= nyquist]
    if invalid:
        raise ValueError(
            "notch_filter_freqs must be strictly below the Nyquist frequency "
            f"({nyquist:g} Hz for sfreq={sfreq:g} Hz). Got {invalid!r}."
        )

    filtered = raw.copy().load_data()
    filtered.notch_filter(freqs=normalized_freqs, verbose=False)
    return filtered


def _normalize_notch_freqs(freqs: Sequence[float] | None) -> list[float]:
    if freqs is None:
        return []
    normalized: list[float] = []
    for freq in freqs:
        value = float(freq)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("notch_filter_freqs values must be finite and strictly positive.")
        normalized.append(value)
    return normalized
