"""
Writer for HFO/spike detection results.

Supports two output formats:

- **TSV** (``"tsv"``) — two BIDS-compatible TSV files written side by side:
    ``*_events.tsv`` (one row per detected event) and ``*_rates.tsv``
    (per-channel event rates in events / second).
- **HDF5** (``"hdf5"``, default) — a single structured ``.h5`` file
    containing all detection data, counts, axes, metadata, and provenance.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from gin_bids_py_analysis.processing.base import (
    BaseProcessingResult,
    BaseProcessingWriter,
)
from gin_bids_py_analysis.processing.utils.serialization import write_hdf5_tree, write_matlab_tree

from .result import HfoSpikeDetectorProcessingResult


def _package_version() -> str:
    """Return the installed package version, or ``'unknown'`` if not found."""
    try:
        return version("gin-bids-py-analysis")
    except PackageNotFoundError:
        return "unknown"


class HfoSpikeDetectorProcessingWriter(BaseProcessingWriter):
    """Writes a :class:`HfoSpikeDetectorProcessingResult` to BIDS derivatives.

    The output format is chosen via
    :attr:`~gin_bids_py_analysis.processing.hfo_spike_detection.HfoSpikeDetectorWriterParams.output_format`:

    * **TSV** (``"tsv"``) — two files written to the same directory:

        - ``*_events.tsv`` — one row per detected event with columns:
          ``onset``, ``channel``, ``event_type``, ``peak_frequency``,
          ``sample_index``, ``frequency_index``, ``detection_strength``.
        - ``*_rates.tsv`` — per-channel detection rates in events / second;
          one column per unique event type (sorted), one row per channel.

    * **HDF5** (``"hdf5"``, default) — a single structured ``.h5`` file.

    HDF5 schema
    -----------
    ::

        /markers
            /onset                float64  [n_events]   — event onset in seconds
            /channel              str      [n_events]   — channel label
            /event_type           str      [n_events]   — type of event
            /peak_frequency       float64  [n_events]   — peak frequency in Hz
            /sample_index         int64    [n_events]   — onset sample index
            /frequency_index      int64    [n_events]   — frequency bin index
            /detection_strength   float64  [n_events]   — detection strength score
        /detection_counts
            /n_spk                int64    [n_channels]          — spikes per channel
            /n_osc                int64    [n_channels, n_bands] — oscillations per channel/band
            /detection_charac     float64  [...]                 — raw characteristics (omitted if empty)
        /axes
            /channel_names        str      [n_channels] — ordered channel labels
            /freq_band            float32  [n_bands, 2] — frequency band edges in Hz
        /meta
            /original_fs          scalar float64  — source recording sampling rate in Hz
            /montage_mode         str             — mono / bipolar
            /duration_seconds     scalar float64  — recording duration in seconds
            /n_channels           scalar int64    — number of channels
            /n_events             scalar int64    — total number of detected events
        /provenance
            /raw_bids_path        str  — path to the source iEEG file
            /pipeline_name        str  = "hfo_spike_detection"
            /pipeline_version     str  — package version

    Example::

        from pathlib import Path
        from gin_bids_py_analysis.processing.hfo_spike_detection import (
            HfoSpikeDetectorProcessingWriter,
            HfoSpikeDetectorWriterParams,
        )

        writer = HfoSpikeDetectorProcessingWriter(
            HfoSpikeDetectorWriterParams(bids_root=Path("/data"))
        )
        output_path = writer.write(result)   # → …_events.h5

        writer_tsv = HfoSpikeDetectorProcessingWriter(
            HfoSpikeDetectorWriterParams(bids_root=Path("/data"), output_format="tsv")
        )
        writer_tsv.write(result)   # writes …_events.tsv and …_rates.tsv
    """

    # ------------------------------------------------------------------
    # Abstract method implementation (required by BaseProcessingWriter)
    # ------------------------------------------------------------------

    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        """Dispatch serialisation to the format selected by ``output_format``."""
        if not isinstance(result, HfoSpikeDetectorProcessingResult):
            raise TypeError(
                f"Expected HfoSpikeDetectorProcessingResult, got {type(result).__name__}"
            )

        if self.params.output_format == "tsv":
            self._write_events_tsv(result, output_path)
            self._write_rates_tsv(result, output_path)
        else:
            tree = result.to_output_tree(pipeline_version=_package_version())
            if self.params.output_format == "matlab":
                write_matlab_tree(output_path, tree)
            else:
                write_hdf5_tree(output_path, tree)

    # ------------------------------------------------------------------
    # TSV writers
    # ------------------------------------------------------------------

    def _write_events_tsv(self, result: HfoSpikeDetectorProcessingResult, output_path: Path) -> None:
        """Write one TSV row per detected event to *output_path*.

        Columns (in order): ``onset``, ``channel``, ``event_type``,
        ``peak_frequency``, ``sample_index``, ``frequency_index``,
        ``detection_strength``.

        Args:
            result: Detection result containing :attr:`~HfoSpikeDetectorProcessingResult.markers`.
            output_path: Destination ``.tsv`` path constructed by the base writer.
        """
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow([
                "onset", "duration", "channel", "event_type", "peak_frequency",
                "sample_index", "frequency_index", "detection_strength", "color"
            ])
            for event in result.markers:
                writer.writerow([
                    event.get("onset_time_seconds", 0),
                    event.get("duration", 0),
                    event.get("channel_label", ""),
                    event.get("event_type", ""),
                    event.get("peak_frequency_hz", 0),
                    event.get("sample_index", 0),
                    event.get("frequency_index", 0),
                    event.get("detection_strength", 0),
                    event.get("visualization_color", "#808080"),
                ])

    def _write_rates_tsv(self, result: HfoSpikeDetectorProcessingResult, events_path: Path) -> None:
        """Write per-channel event rates (events / second) to a companion TSV.

        The output file is placed in the same directory as *events_path*, with
        ``_events`` in the filename replaced by ``_rates``.  Columns are
        ``channel`` followed by one column per unique event type (sorted
        ascending).  Channels with no events for a given type default to
        ``0.0``.

        Args:
            result: Detection result with :attr:`~HfoSpikeDetectorProcessingResult.markers`,
                    :attr:`~HfoSpikeDetectorProcessingResult.channel_names`,
                    :attr:`~HfoSpikeDetectorProcessingResult.metadata`, and
                    :attr:`~HfoSpikeDetectorProcessingResult.original_fs`.
            events_path: Path of the events TSV (used to derive the rates filename).
        """
        suffix = events_path.suffix
        rates_path = events_path.with_name(
            events_path.name.replace(f"_events{suffix}", f"_rates{suffix}")
        )

        n_samples = result.metadata.get("n_samples", 0)
        duration = n_samples / result.original_fs if result.original_fs > 0 else 0.0

        # Count events per (channel, event_type)
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        event_types: set[str] = set()
        for event in result.markers:
            ch = event.get("channel_label", "")
            et = event.get("event_type", "")
            counts[ch][et] += 1
            event_types.add(et)

        sorted_types = sorted(event_types)

        with open(rates_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(["channel"] + sorted_types)
            for channel in result.channel_names:
                row: list = [channel]
                for et in sorted_types:
                    count = counts.get(channel, {}).get(et, 0)
                    row.append(count / duration if duration > 0 else 0.0)
                writer.writerow(row)

