"""
Result container for HFO/spike detection processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bidsforge.processing.base import BaseProcessingResult
from bidsforge.processing.utils.serialization import OutputTree, compressed


DETECTION_CHARAC_COLUMNS = (
    "sample_index",
    "frequency_index",
    "z_value",
    "duration_half_height_samples",
    "frequency_extent_bins",
    "area_half_height",
    "right_duration_samples",
    "left_duration_samples",
    "upper_frequency_extent_bins",
    "lower_frequency_extent_bins",
    "fwhm_time_samples",
)

DETECTION_CHARAC_DESCRIPTIONS = (
    "Sample index of the detected time-frequency maximum.",
    "Frequency-bin index of the detected time-frequency maximum.",
    "Z-scored detection strength at the maximum.",
    "Full duration at half height, in samples.",
    "Full frequency extent at half height, in frequency bins.",
    "Half-height region area in sample-bin units.",
    "Right-side duration from the maximum at half height, in samples.",
    "Left-side duration from the maximum at half height, in samples.",
    "Upper frequency extent from the maximum at half height, in bins.",
    "Lower frequency extent from the maximum at half height, in bins.",
    "Expected time full-width at half maximum, in samples.",
)


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

    def __post_init__(self) -> None:
        self.freq_band = _coerce_freq_band(self.freq_band)
        self.n_spk = np.asarray(self.n_spk)
        self.n_osc = _coerce_n_osc(self.n_osc, self.freq_band)
        self.detection_charac = _coerce_detection_charac(self.detection_charac)

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

        n_spk_export = self.n_spk.astype(np.int64) if self.n_spk.size > 0 else None
        n_osc_export = self.n_osc.astype(np.int64) if self.n_osc.size > 0 else None
        freq_band_export = (
            self.freq_band.astype(np.float32)
            if self.freq_band.size > 0
            else None
        )
        detection_charac_export = (
            self.detection_charac.astype(np.float64)
            if self.detection_charac.size > 0
            else np.empty((0, len(DETECTION_CHARAC_COLUMNS)), dtype=np.float64)
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
                "n_spk": n_spk_export,
                "n_osc": n_osc_export,
                "freq_band": freq_band_export,
            },
            "features": {
                "detection_charac": compressed(detection_charac_export),
                "detection_charac_columns": np.array(
                    DETECTION_CHARAC_COLUMNS, dtype=object
                ),
                "detection_charac_descriptions": np.array(
                    DETECTION_CHARAC_DESCRIPTIONS, dtype=object
                ),
            },
            "axes": {
                "channel_names": np.array(self.channel_names, dtype=object),
            },
            "meta": {
                "schema_name": "hfo_spike_detection",
                "schema_version": "2.1",
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


def _coerce_detection_charac(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0, 11), dtype=np.float64)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return np.empty((0, 11), dtype=np.float64)
        if value.ndim == 1:
            return value.reshape(1, -1)
        return value
    if isinstance(value, (list, tuple)):
        rows = [
            np.asarray(item, dtype=np.float64)
            for item in value
            if np.asarray(item).size > 0
        ]
        if not rows:
            return np.empty((0, 11), dtype=np.float64)
        return np.vstack(rows)
    arr = np.asarray(value)
    if arr.ndim == 1 and arr.size > 0:
        return arr.reshape(1, -1)
    return arr


def _coerce_freq_band(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.size == 0:
        return np.array([])
    if arr.ndim == 1 and arr.size == 2:
        return arr.reshape(1, 2)
    return arr


def _coerce_n_osc(value: Any, freq_band: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.size == 0:
        return np.array([])
    if arr.ndim == 1 and freq_band.shape == (1, 2):
        return arr.reshape(-1, 1)
    return arr
