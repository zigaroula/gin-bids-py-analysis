from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.bids.helpers import modify_entities
from gin_bids_py_analysis.processing.base import (
    BaseProcessingResult,
    BaseProcessingWriter,
)
from gin_bids_py_analysis.processing.utils.events import downsample_events
from gin_bids_py_analysis.processing.utils.serialization import write_hdf5_tree, write_matlab_tree

from .result import HilbertProcessingResult

_downsample_events = downsample_events


def _package_version() -> str:
    try:
        return version("gin-bids-py-analysis")
    except PackageNotFoundError:
        return "unknown"


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
      Events are taken from ``result.events``, already projected to the envelope
      sampling rate by the processor.

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
            tree = result.to_output_tree(pipeline_version=_package_version())
            if self.params.output_format == "matlab":
                write_matlab_tree(output_path, tree)
            else:
                write_hdf5_tree(output_path, tree)

    def get_output_path(self, group: BIDSFileGroup) -> Path:
        """Override to check per-window BrainVision files when skipping existing output.

        For BrainVision format, the actual output is a set of per-smoothing-window
        ``.vhdr`` files whose ``desc`` entity is suffixed with ``sm{N}``
        (e.g. ``desc-bgasm0``).  The base-class path (e.g. ``desc-bga_….h5``)
        never exists on disk, so ``skip_existing`` would never fire.

        This override globs for any matching per-window ``.vhdr`` file.  If at
        least one is found the first match is returned (it exists → skip).  If
        none are found a predictable ``sm0`` canary path is returned so that
        ``skip_existing`` correctly triggers processing.

        For HDF5 format the default base-class behaviour is preserved.
        """
        base_path = self._build_output_path(group.primary.entities)
        if self.params.output_format == "brainvision":
            glob_stem = modify_entities(
                base_path.stem, desc=f"{self.params.output_description}sm*"
            )
            matches = sorted(base_path.parent.glob(f"{glob_stem}.vhdr"))
            if matches:
                return matches[0]
            # No files on disk yet — return a non-existent canary path.
            fname_base = modify_entities(
                base_path.stem, desc=f"{self.params.output_description}sm0"
            )
            return base_path.parent / f"{fname_base}.vhdr"
        return base_path

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

        Events are taken from ``result.events``, already projected to the
        envelope sampling rate by the processor.

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

        try:
            import pybv
        except ImportError as exc:
            raise ImportError(
                "pybv is required for BrainVision output. Install it with: pip install pybv"
            ) from exc

        events = result.events
        if events is None and result.original_events is not None:
            events = downsample_events(
                result.original_events,
                result.downsampled_fs,
                original_fs=result.original_fs,
                event_sample_shift_samples=int(
                    result.metadata.get("event_sample_shift_samples", 0)
                ),
                event_onset_precision=str(
                    result.metadata.get("events_onset_precision", "sample_quantized")
                ),
            )

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

