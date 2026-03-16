"""
Writer for Delphos detection results.

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

import numpy as np

from gin_bids_py_analysis.processing.base import (
    BaseProcessingResult,
    BaseProcessingWriter,
)

from .result import DelphosProcessingResult


def _package_version() -> str:
    """Return the installed package version, or ``'unknown'`` if not found."""
    try:
        return version("gin-bids-py-analysis")
    except PackageNotFoundError:
        return "unknown"


class DelphosProcessingWriter(BaseProcessingWriter):
    """Writes a :class:`DelphosProcessingResult` to BIDS derivatives.

    The output format is chosen via
    :attr:`~gin_bids_py_analysis.processing.delphos.DelphosWriterParams.output_format`:

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
            /pipeline_name        str  = "delphos"
            /pipeline_version     str  — package version

    Example::

        from pathlib import Path
        from gin_bids_py_analysis.processing.delphos import (
            DelphosProcessingWriter,
            DelphosWriterParams,
        )

        writer = DelphosProcessingWriter(
            DelphosWriterParams(bids_root=Path("/data"))
        )
        output_path = writer.write(result)   # → …_events.h5

        writer_tsv = DelphosProcessingWriter(
            DelphosWriterParams(bids_root=Path("/data"), output_format="tsv")
        )
        writer_tsv.write(result)   # writes …_events.tsv and …_rates.tsv
    """

    # ------------------------------------------------------------------
    # Abstract method implementation (required by BaseProcessingWriter)
    # ------------------------------------------------------------------

    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        """Dispatch serialisation to the format selected by ``output_format``."""
        if not isinstance(result, DelphosProcessingResult):
            raise TypeError(
                f"Expected DelphosProcessingResult, got {type(result).__name__}"
            )

        if self.params.output_format == "tsv":
            self._write_events_tsv(result, output_path)
            self._write_rates_tsv(result, output_path)
        else:
            self._write_hdf5(result, output_path)

    # ------------------------------------------------------------------
    # TSV writers
    # ------------------------------------------------------------------

    def _write_events_tsv(self, result: DelphosProcessingResult, output_path: Path) -> None:
        """Write one TSV row per detected event to *output_path*.

        Columns (in order): ``onset``, ``channel``, ``event_type``,
        ``peak_frequency``, ``sample_index``, ``frequency_index``,
        ``detection_strength``.

        Args:
            result: Detection result containing :attr:`~DelphosProcessingResult.markers`.
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

    def _write_rates_tsv(self, result: DelphosProcessingResult, events_path: Path) -> None:
        """Write per-channel event rates (events / second) to a companion TSV.

        The output file is placed in the same directory as *events_path*, with
        ``_events`` in the filename replaced by ``_rates``.  Columns are
        ``channel`` followed by one column per unique event type (sorted
        ascending).  Channels with no events for a given type default to
        ``0.0``.

        Args:
            result: Detection result with :attr:`~DelphosProcessingResult.markers`,
                    :attr:`~DelphosProcessingResult.channel_names`,
                    :attr:`~DelphosProcessingResult.metadata`, and
                    :attr:`~DelphosProcessingResult.original_fs`.
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

    # ------------------------------------------------------------------
    # HDF5 writer
    # ------------------------------------------------------------------

    def _write_hdf5(self, result: DelphosProcessingResult, output_path: Path) -> None:
        """Write *result* to a structured HDF5 file at *output_path*.

        See the :class:`DelphosProcessingWriter` class docstring for the full
        file schema.

        Args:
            result: Detection result to serialise.
            output_path: Destination ``.h5`` path.
        """
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required for HDF5 output. Install it with: pip install h5py"
            ) from exc

        str_dtype = h5py.string_dtype(encoding="utf-8")

        n_samples = result.metadata.get("n_samples", 0)
        duration = n_samples / result.original_fs if result.original_fs > 0 else 0.0

        # Pre-extract marker columns for efficient array creation.
        n_events = len(result.markers)
        onsets = np.array([e.get("onset_time_seconds", 0) for e in result.markers], dtype=np.float64)
        durations = np.array([e.get("duration", 0) for e in result.markers], dtype=np.float64)
        channels = np.array([e.get("channel_label", "") for e in result.markers], dtype=object)
        event_types = np.array([e.get("event_type", "") for e in result.markers], dtype=object)
        peak_freqs = np.array([e.get("peak_frequency_hz", 0) for e in result.markers], dtype=np.float64)
        sample_indices = np.array([e.get("sample_index", 0) for e in result.markers], dtype=np.int64)
        freq_indices = np.array([e.get("frequency_index", 0) for e in result.markers], dtype=np.int64)
        strengths = np.array([e.get("detection_strength", 0) for e in result.markers], dtype=np.float64)
        colors = np.array([e.get("visualization_color", "#808080") for e in result.markers], dtype=object)

        with h5py.File(output_path, "w") as fh:

            # /markers
            markers_grp = fh.create_group("markers")
            markers_grp.create_dataset("onset", data=onsets)
            markers_grp.create_dataset("duration", data=durations)
            markers_grp.create_dataset("channel", data=channels, dtype=str_dtype)
            markers_grp.create_dataset("event_type", data=event_types, dtype=str_dtype)
            markers_grp.create_dataset("peak_frequency", data=peak_freqs)
            markers_grp.create_dataset("sample_index", data=sample_indices)
            markers_grp.create_dataset("frequency_index", data=freq_indices)
            markers_grp.create_dataset("detection_strength", data=strengths)
            markers_grp.create_dataset("color", data=colors, dtype=str_dtype)

            # /detection_counts
            counts_grp = fh.create_group("detection_counts")
            if result.n_spk.size > 0:
                counts_grp.create_dataset("n_spk", data=result.n_spk.astype(np.int64))
            if result.n_osc.size > 0:
                counts_grp.create_dataset("n_osc", data=result.n_osc.astype(np.int64))
            if result.detection_charac.size > 0:
                counts_grp.create_dataset(
                    "detection_charac",
                    data=result.detection_charac.astype(np.float64),
                    compression="gzip",
                    compression_opts=4,
                )

            # /axes
            axes_grp = fh.create_group("axes")
            axes_grp.create_dataset(
                "channel_names",
                data=np.array(result.channel_names, dtype=object),
                dtype=str_dtype,
            )
            if result.freq_band.size > 0:
                axes_grp.create_dataset(
                    "freq_band", data=result.freq_band.astype(np.float32)
                )

            # /meta
            meta_grp = fh.create_group("meta")
            meta_grp.create_dataset("original_fs", data=float(result.original_fs))
            meta_grp.create_dataset(
                "montage_mode",
                data=result.metadata.get("montage_mode", ""),
                dtype=str_dtype,
            )
            meta_grp.create_dataset("duration_seconds", data=duration)
            meta_grp.create_dataset("n_channels", data=np.int64(len(result.channel_names)))
            meta_grp.create_dataset("n_events", data=np.int64(n_events))

            # /provenance
            prov_grp = fh.create_group("provenance")
            prov_grp.create_dataset(
                "raw_bids_path",
                data=str(result.source_group.primary.path),
                dtype=str_dtype,
            )
            prov_grp.create_dataset("pipeline_name", data="delphos", dtype=str_dtype)
            prov_grp.create_dataset(
                "pipeline_version", data=_package_version(), dtype=str_dtype
            )
