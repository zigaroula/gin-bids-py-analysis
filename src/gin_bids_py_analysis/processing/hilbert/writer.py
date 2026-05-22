from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import mne

from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.bids.helpers import modify_entities
from gin_bids_py_analysis.processing.base import (
    BaseProcessingResult,
    BaseProcessingWriter,
)
from gin_bids_py_analysis.processing.utils.events import coerce_annotation_events
from gin_bids_py_analysis.processing.utils.matlab import matlab_round
from gin_bids_py_analysis.processing.utils.serialization import write_hdf5_tree, write_matlab_tree

from .result import HilbertProcessingResult

def _package_version() -> str:
    try:
        return version("gin-bids-py-analysis")
    except PackageNotFoundError:
        return "unknown"


def _downsample_events(
    original_events: mne.Annotations | Any,
    downsampled_fs: float,
    *,
    original_fs: float | None = None,
    event_sample_shift_samples: int = 0,
    event_onset_precision: str = "sample_quantized",
) -> list[dict] | None:
    """Convert source annotations to sample indices at the envelope rate.

    Args:
        original_events: MNE Annotations object or any list of dicts with keys
                         "onset" (in seconds) and "duration" (in seconds).
        downsampled_fs: Sampling frequency of the exported envelope signal.
        original_fs: Source sampling frequency, required when applying a source
                     sample offset before projection to ``downsampled_fs``.
        event_sample_shift_samples: Offset to apply in source samples before
                                    converting events to the envelope rate.
        event_onset_precision: ``"sample_quantized"`` preserves the historical
                               annotation path by snapping to a source sample
                               before downsampling. ``"exact_time"`` keeps
                               exact TSV onsets in seconds and applies the
                               sample shift as a time offset.
                               ``"spm_continuous_sample"`` reproduces SPM's
                               continuous-file event-to-sample conversion after
                               downsampling: source samples are treated as
                               MATLAB/SPM 1-based samples, then converted to
                               BrainVision zero-based marker onsets.

    Returns:
        List of dicts with keys "onset" (in samples), "duration" (in samples),
        "type", and "description", or ``None`` if no events are present.
    """
    if not original_events:
        return None

    if event_sample_shift_samples and original_fs is None:
        raise ValueError(
            "original_fs is required when event_sample_shift_samples is non-zero."
        )
    if event_onset_precision not in {
        "sample_quantized",
        "exact_time",
        "spm_continuous_sample",
    }:
        raise ValueError(
            "event_onset_precision must be 'sample_quantized', 'exact_time', "
            "or 'spm_continuous_sample'."
        )

    downsampled_events = []
    for ann in coerce_annotation_events(original_events):
        description: int | str
        ann_type = ann.event_type
        if ann_type in ("Stimulus", "Response") and ann.code is not None:
            description = int(ann.code)
        else:
            if ann_type in ("Stimulus", "Response"):
                ann_type = "Comment"
            description = ann.description

        if event_onset_precision == "spm_continuous_sample":
            if original_fs is None:
                raise ValueError(
                    "original_fs is required for event_onset_precision="
                    "'spm_continuous_sample'."
                )
            shifted_onset_s = ann.onset_s + (
                (event_sample_shift_samples + 1) / float(original_fs)
            )
            onset_samples = matlab_round(shifted_onset_s * downsampled_fs + 1) - 1
        elif event_onset_precision == "exact_time":
            shifted_onset_s = ann.onset_s
            if event_sample_shift_samples:
                shifted_onset_s += event_sample_shift_samples / float(original_fs)
            onset_samples = matlab_round(shifted_onset_s * downsampled_fs)
        elif event_sample_shift_samples:
            source_onset_sample = (
                matlab_round(ann.onset_s * float(original_fs))
                + event_sample_shift_samples
            )
            onset_samples = matlab_round(
                source_onset_sample * downsampled_fs / float(original_fs)
            )
        else:
            onset_samples = matlab_round(ann.onset_s * downsampled_fs)
        duration_samples = matlab_round(ann.duration_s * downsampled_fs)

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

        try:
            import pybv
        except ImportError as exc:
            raise ImportError(
                "pybv is required for BrainVision output. Install it with: pip install pybv"
            ) from exc

        event_sample_shift_samples = int(
            result.metadata.get("event_sample_shift_samples", 0)
        )
        event_onset_precision = str(
            result.metadata.get("events_onset_precision", "sample_quantized")
        )
        events = _downsample_events(
            result.original_events,
            result.downsampled_fs,
            original_fs=result.original_fs,
            event_sample_shift_samples=event_sample_shift_samples,
            event_onset_precision=event_onset_precision,
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

