"""
Result container for Delphos HFO/spike detection processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gin_bids_py_analysis.processing.base import BaseProcessingResult


@dataclass
class DelphosProcessingResult(BaseProcessingResult):
    """
    Result of the Delphos detection pipeline for one processed file group.

    Attributes:
        markers:
            List of detected event markers. Each marker is a dict containing:
            - 'onset': Event onset time in seconds
            - 'duration': Event duration in seconds
            - 'channel': Channel name/label
            - 'event_type': Type of event (e.g., 'Ripple', 'Fast Ripple', 'Spike')
            - 'peak_frequency': Peak frequency of the event in Hz
            - 'amplitude': Amplitude characteristics
            - Additional event-specific features
        
        freq_band:
            Frequency bands used for oscillation detection. Shape: (n_bands, 2)
        
        n_spk:
            Number of spikes detected per channel. Shape: (n_channels,)
        
        n_osc:
            Number of oscillations detected per channel per band.
            Shape: (n_channels, n_bands)
        
        channel_names:
            Ordered list of channel labels after montaging.
        
        detection_charac:
            Array of detection characteristics for all events.
            Shape and content depend on the detection algorithm.
        
        event_rates:
            Dictionary containing event rate statistics:
            - Event counts per type
            - Event rates per second per type
            - Total event statistics
            - Signal duration
        
        algorithm_config:
            Dictionary containing the algorithm parameters used for detection.
        
        original_fs:
            Sampling rate of the input signal in Hz.
    """

    markers: list[dict[str, Any]] = field(default_factory=list)
    freq_band: np.ndarray = field(default_factory=lambda: np.array([]))
    n_spk: np.ndarray = field(default_factory=lambda: np.array([]))
    n_osc: np.ndarray = field(default_factory=lambda: np.array([]))
    channel_names: list[str] = field(default_factory=list)
    detection_charac: np.ndarray = field(default_factory=lambda: np.array([]))
    event_rates: dict[str, Any] = field(default_factory=dict)
    algorithm_config: dict[str, Any] = field(default_factory=dict)
    original_fs: float = 0.0
