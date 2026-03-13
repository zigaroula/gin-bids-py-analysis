from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mne import Annotations
import numpy as np

from gin_bids_py_analysis.processing.base import BaseProcessingResult


@dataclass
class HilbertProcessingResult(BaseProcessingResult):
    """Result of the Hilbert-band envelope pipeline for one iEEG file.

    Attributes:
        smoothed:        ``{window_ms: array}`` mapping each smoothing window
                         (in milliseconds) to a float32 array of shape
                         ``[n_channels, n_downsampled_samples]``.  Key ``0``
                         is the unsmoothed result.
        channel_names:   Ordered list of channel labels after montaging.
        bins:            Frequency bin edges actually used (after Shannon
                         clamping).  Adjacent pairs define the subbands, e.g.
                         ``[50., 60., 70.]`` means subbands 50-60 Hz and
                         60-70 Hz were processed.
        downsampled_fs:  Effective sampling rate of the envelope output in Hz.
        original_fs:     Sampling rate of the raw input signal in Hz.
    """

    smoothed: dict[int, np.ndarray] = field(default_factory=dict)
    channel_names: list[str] = field(default_factory=list)
    bins: list[float] = field(default_factory=list)
    downsampled_fs: float = 0.0
    original_fs: float = 0.0
    original_events: Annotations | Any = field(default=None)
