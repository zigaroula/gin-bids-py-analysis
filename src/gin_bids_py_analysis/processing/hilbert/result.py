from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mne import Annotations
import numpy as np

from gin_bids_py_analysis.processing.base import BaseProcessingResult
from gin_bids_py_analysis.processing.utils.serialization import OutputTree, compressed


@dataclass
class HilbertProcessingResult(BaseProcessingResult):
    """Result of the Hilbert-band envelope pipeline for one processed file group.

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
        original_events: Source annotations carried forward for optional
                         BrainVision marker export.
    """

    smoothed: dict[int, np.ndarray] = field(default_factory=dict)
    channel_names: list[str] = field(default_factory=list)
    bins: list[float] = field(default_factory=list)
    downsampled_fs: float = 0.0
    original_fs: float = 0.0
    original_events: Annotations | Any = field(default=None)

    def to_output_tree(self, *, pipeline_version: str = "unknown") -> OutputTree:
        """Return the canonical serialisable output tree for this result.

        All smoothing windows are stacked into a single 3-D ``data/envelope``
        array with shape ``[n_smoothing_windows, n_channels, n_down]``, sorted
        by window size ascending.

        Args:
            pipeline_version: Package version string written into provenance.

        Returns:
            A nested mapping suitable for :func:`write_hdf5_tree` or
            :func:`write_matlab_tree`.
        """
        sorted_windows = sorted(self.smoothed.keys())
        envelope_3d = np.stack(
            [self.smoothed[w] for w in sorted_windows], axis=0
        ).astype(np.float32)
        n_down = envelope_3d.shape[2]

        meta: dict[str, object] = {
            "schema_name": "hilbert",
            "schema_version": "2.0",
            "sampling_frequency_hz": float(self.original_fs),
            "downsampled_frequency_hz": float(self.downsampled_fs),
            "montage_mode": str(self.metadata.get("montage_mode", "")),
            "unit": str(self.metadata.get("unit", "amplitude")),
            "dimension_order": "smoothing_window x channel x time",
            "centered": bool(self.metadata.get("centered", False)),
        }
        for key in (
            "events_source_requested",
            "events_source_resolved",
            "events_onset_precision",
            "events_file",
            "event_sample_shift_samples",
        ):
            value = self.metadata.get(key)
            if value is not None:
                meta[key] = str(value)

        return {
            "data": {"envelope": compressed(envelope_3d)},
            "axes": {
                "channel": np.array(self.channel_names, dtype=object),
                "smoothing_window_ms": np.array(sorted_windows, dtype=np.int32),
                "band_limits_hz": np.array(self.bins, dtype=np.float32),
                "time_s": np.arange(n_down, dtype=np.float64) / self.downsampled_fs,
            },
            "meta": meta,
            "provenance": {
                "raw_bids_path": str(self.source_group.primary.path),
                "pipeline_name": "hilbert",
                "pipeline_version": pipeline_version,
            },
        }
