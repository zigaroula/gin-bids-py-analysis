"""
Result container for HFO/spike detection processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gin_bids_py_analysis.processing.base import BaseProcessingResult
from gin_bids_py_analysis.processing.utils.serialization import OutputTree, compressed


@dataclass
class HfoSpikeDetectorProcessingResult(BaseProcessingResult):
    """
    Result of the HFO/spike detection pipeline for one processed file group.

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

    def to_output_tree(self, *, pipeline_version: str = "unknown") -> OutputTree:
        """Return the canonical serialisable output tree for this result.

        Args:
            pipeline_version: Package version string written into provenance.

        Returns:
            A nested mapping suitable for :func:`write_hdf5_tree` or
            :func:`write_matlab_tree`.
        """
        n_events = len(self.markers)
        n_samples = self.metadata.get("n_samples", 0)
        duration = n_samples / self.original_fs if self.original_fs > 0 else 0.0

        onsets = np.array(
            [e.get("onset_time_seconds", 0) for e in self.markers], dtype=np.float64
        )
        durations = np.array(
            [e.get("duration", 0) for e in self.markers], dtype=np.float64
        )
        channels = np.array(
            [e.get("channel_label", "") for e in self.markers], dtype=object
        )
        event_types = np.array(
            [e.get("event_type", "") for e in self.markers], dtype=object
        )
        peak_freqs = np.array(
            [e.get("peak_frequency_hz", 0) for e in self.markers], dtype=np.float64
        )
        sample_indices = np.array(
            [e.get("sample_index", 0) for e in self.markers], dtype=np.int64
        )
        freq_indices = np.array(
            [e.get("frequency_index", 0) for e in self.markers], dtype=np.int64
        )
        strengths = np.array(
            [e.get("detection_strength", 0) for e in self.markers], dtype=np.float64
        )
        colors = np.array(
            [e.get("visualization_color", "#808080") for e in self.markers], dtype=object
        )

        return {
            "events": {
                "onset": onsets,
                "duration": durations,
                "channel": channels,
                "event_type": event_types,
                "peak_frequency": peak_freqs,
                "sample_index": sample_indices,
                "frequency_index": freq_indices,
                "detection_strength": strengths,
                "color": colors,
            },
            "counts": {
                "n_spk": self.n_spk.astype(np.int64) if self.n_spk.size > 0 else None,
                "n_osc": self.n_osc.astype(np.int64) if self.n_osc.size > 0 else None,
                "detection_charac": (
                    compressed(self.detection_charac.astype(np.float64))
                    if self.detection_charac.size > 0
                    else None
                ),
            },
            "axes": {
                "channel_names": np.array(self.channel_names, dtype=object),
                "freq_band": (
                    self.freq_band.astype(np.float32) if self.freq_band.size > 0 else None
                ),
            },
            "meta": {
                "schema_name": "hfo_spike_detection",
                "schema_version": "2.0",
                "original_fs": float(self.original_fs),
                "montage_mode": str(self.metadata.get("montage_mode", "")),
                "duration_seconds": float(duration),
                "n_channels": np.int64(len(self.channel_names)),
                "n_events": np.int64(n_events),
            },
            "provenance": {
                "raw_bids_path": str(self.source_group.primary.path),
                "pipeline_name": "hfo_spike_detection",
                "pipeline_version": pipeline_version,
            },
        }
