from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import h5py
import mne
import numpy as np
import pybv

from gin_bids_py_analysis.bids.helpers import modify_entities
from gin_bids_py_analysis.processing.base import (
    BaseProcessingResult,
    BaseProcessingWriter,
)

from .result import HilbertProcessingResult

def _package_version() -> str:
    try:
        return version("gin-bids-py-analysis")
    except PackageNotFoundError:
        return "unknown"


def _downsample_events(original_events: mne.Annotations | Any, downsampled_fs: float) -> list[dict] | None:
    """Convert source annotations to sample indices at the envelope rate.

    Args:
        original_events: MNE Annotations object or any list of dicts with keys
                         "onset" (in seconds) and "duration" (in seconds).

    Returns:
        List of dicts with keys "onset" (in samples), "duration" (in samples),
        "type", and "description", or ``None`` if no events are present.
    """
    if not original_events:
        return None

    downsampled_events = []
    for ann in original_events:
        full_desc: str = ann["description"]

        # MNE encodes BrainVision markers as "Type/Description" (e.g. "Stimulus/S  1").
        # Split on the first "/" to recover the BrainVision type and description fields.
        if "/" in full_desc:
            ann_type, ann_desc = full_desc.split("/", 1)
        else:
            ann_type, ann_desc = "Stimulus", full_desc

        # pybv requires description to be an int for Stimulus/Response markers.
        # MNE formats them as "S  1" or "R  2" — strip the leading letter and whitespace.
        description: int | str
        if ann_type in ("Stimulus", "Response"):
            numeric_part = ann_desc.lstrip("SRsr").strip()
            try:
                description = int(numeric_part)
            except ValueError:
                # Non-numeric: fall back to Comment type so pybv accepts a string.
                ann_type = "Comment"
                description = ann_desc
        else:
            description = ann_desc  # Comment / unknown — string is fine

        onset_samples = round(ann["onset"] * downsampled_fs)
        duration_samples = round(ann["duration"] * downsampled_fs)

        downsampled_events.append(
            {
                "onset": onset_samples,
                "duration": duration_samples,
                "type": ann_type,
                "description": description,
            }
        )

    return downsampled_events if downsampled_events else None


class HilbertProcessingWriter(BaseProcessingWriter):
    """Writes a :class:`HilbertProcessingResult` to BIDS derivatives.

    The output format is selected automatically from
    :attr:`~gin_bids_py_analysis.processing.hilbert.HilbertWriterParams.output_format`:

    * **HDF5** (``"hdf5"``, default) — a single file containing all smoothing
      windows stacked along the first axis.  See :meth:`_write_hdf5` for the
      full schema.
    * **BrainVision** (``"brainvision"``) — one file triplet
      (``.vhdr`` / ``.vmrk`` / ``.eeg``) is written per smoothing window using
      a per-window ``desc-<output_description>sm{N}`` entity in the filename.
      Annotations are taken from ``result.original_events`` and remapped to the
      envelope sampling rate.

    Pass a :class:`~gin_bids_py_analysis.processing.hilbert.HilbertWriterParams`
    instance to the constructor — only ``bids_root`` is required.

    Example — HDF5 (default)::

        writer = HilbertProcessingWriter(
            HilbertWriterParams(bids_root=Path("/data/my_study"))
        )
        out_path = writer.write(result)   # → …_hilbert.h5

    Example — BrainVision::

        writer = HilbertProcessingWriter(
            HilbertWriterParams(bids_root=Path("/data/my_study"), output_format="brainvision")
        )
        writer.write(result)  # writes per-window files such as …desc-hilbertsm0_….vhdr

    HDF5 schema
    -----------
    ::

        /data
            /envelope             float32  [n_smoothing_windows, n_channels, n_down]
        /axes
            /channel              str      [n_channels]          — labels after montaging
            /smoothing_window_ms  int32    [n_smoothing_windows] — sorted ascending
            /band_limits_hz       float32  [n_bins]              — bin edges (clamped)
            /time_s               float64  [n_down]              — envelope time axis
        /meta
            /sampling_frequency_hz     scalar float64  — original recording fs
            /downsampled_frequency_hz  scalar float64  — envelope fs
            /montage_mode              str
            /unit                      str  ("percent" or "amplitude")
            /dimension_order           str  "smoothing_window × channel × time"
            /centered                  scalar bool
        /provenance
            /raw_bids_path      str  — path to the source iEEG file
            /pipeline_name      str  = "hilbert"
            /pipeline_version   str  — package version
    """

    # ------------------------------------------------------------------
    # Abstract method implementation (required by BaseProcessingWriter)
    # ------------------------------------------------------------------

    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        """Serialize *result* using the format selected by ``output_format``."""
        if not isinstance(result, HilbertProcessingResult):
            raise TypeError(
                f"Expected HilbertProcessingResult, got {type(result).__name__!r}"
            )
        
        if self.params.output_format == "brainvision":
            self._write_brainvision_all_windows(result, output_path)
        else:
            self._write_hdf5(result, output_path)

    # ------------------------------------------------------------------
    # HDF5 writer
    # ------------------------------------------------------------------

    def _write_hdf5(self, result: HilbertProcessingResult, output_path: Path) -> None:
        """Write *result* to an HDF5 file at *output_path*.

        All smoothing windows are stacked into a single 3-D dataset
        ``/data/envelope`` with shape ``[n_smoothing_windows, n_channels, n_down]``.
        """
        # Sort smoothing windows so axis order in the file is deterministic.
        sorted_windows = sorted(result.smoothed.keys())

        # Stack into a single 3-D array: [n_smoothing_windows, n_channels, n_down]
        envelope_3d = np.stack(
            [result.smoothed[w] for w in sorted_windows], axis=0
        ).astype(np.float32)

        n_down = envelope_3d.shape[2]
        str_dtype = h5py.string_dtype(encoding="utf-8")

        with h5py.File(output_path, "w") as fh:

            # ------------------------------------------------------------------
            # /data
            # ------------------------------------------------------------------
            data_grp = fh.create_group("data")
            data_grp.create_dataset(
                "envelope",
                data=envelope_3d,
                compression="gzip",
                compression_opts=4,
            )

            # ------------------------------------------------------------------
            # /axes
            # ------------------------------------------------------------------
            axes_grp = fh.create_group("axes")

            axes_grp.create_dataset(
                "channel",
                data=np.array(result.channel_names, dtype=object),
                dtype=str_dtype,
            )
            axes_grp.create_dataset(
                "smoothing_window_ms",
                data=np.array(sorted_windows, dtype=np.int32),
            )
            axes_grp.create_dataset(
                "band_limits_hz",
                data=np.array(result.bins, dtype=np.float32),
            )
            axes_grp.create_dataset(
                "time_s",
                data=np.arange(n_down, dtype=np.float64) / result.downsampled_fs,
            )

            # ------------------------------------------------------------------
            # /meta
            # ------------------------------------------------------------------
            meta_grp = fh.create_group("meta")
            meta_grp.create_dataset("sampling_frequency_hz", data=result.original_fs)
            meta_grp.create_dataset(
                "downsampled_frequency_hz", data=result.downsampled_fs
            )
            meta_grp.create_dataset(
                "montage_mode",
                data=result.metadata.get("montage_mode", ""),
                dtype=str_dtype,
            )
            meta_grp.create_dataset(
                "unit",
                data=result.metadata.get("unit", "amplitude"),
                dtype=str_dtype,
            )
            meta_grp.create_dataset(
                "dimension_order",
                data="smoothing_window \u00d7 channel \u00d7 time",
                dtype=str_dtype,
            )
            meta_grp.create_dataset(
                "centered",
                data=bool(result.metadata.get("centered", False)),
            )

            # ------------------------------------------------------------------
            # /provenance
            # ------------------------------------------------------------------
            prov_grp = fh.create_group("provenance")
            prov_grp.create_dataset(
                "raw_bids_path",
                data=str(result.source_group.primary.path),
                dtype=str_dtype,
            )
            prov_grp.create_dataset("pipeline_name", data="hilbert", dtype=str_dtype)
            prov_grp.create_dataset(
                "pipeline_version", data=_package_version(), dtype=str_dtype
            )

    # ------------------------------------------------------------------
    # BrainVision writer
    # ------------------------------------------------------------------

    def _write_brainvision_all_windows(
        self, result: BaseProcessingResult, output_path: Path
    ) -> list[Path]:
        """Write one BrainVision file triplet per smoothing window.

        For each smoothing window *N* (sorted ascending) pybv writes three files:

        * ``…desc-<output_description>sm{N}_….vhdr`` — text header
        * ``…desc-<output_description>sm{N}_….vmrk`` — marker file
        * ``…desc-<output_description>sm{N}_….eeg``  — binary float32 data

        Annotations are taken from ``result.original_events`` and remapped to
        the envelope sampling rate via :func:`_downsample_events`.

        Args:
            result: A :class:`HilbertProcessingResult` to serialise.
            output_path: Base BIDS-derived path used to construct the per-window
                filenames.

        Returns:
            List of ``.vhdr`` paths, one per smoothing window, sorted by
            window size ascending (smallest window first).
        """
        if not isinstance(result, HilbertProcessingResult):
            raise TypeError(
                f"Expected HilbertProcessingResult, got {type(result).__name__!r}"
            )

        events = _downsample_events(result.original_events, result.downsampled_fs)

        unit = result.metadata.get("unit", "µV")
        scale_factor = result.metadata.get("scale_factor", 1e-6)

        sorted_windows = sorted(result.smoothed.keys())
        out_paths: list[Path] = []

        for w in sorted_windows:
            fname_base = modify_entities(output_path.stem, desc=f"{self.params.output_description}sm{w}")
            pybv.write_brainvision(
                data=result.smoothed[w] * scale_factor,
                sfreq=result.downsampled_fs,
                ch_names=result.channel_names,
                fname_base=fname_base,
                folder_out=str(output_path.parent),
                events=events,
                unit=unit,
                overwrite=True,
            )
            out_paths.append(output_path.parent / f"{fname_base}.vhdr")

        return out_paths

