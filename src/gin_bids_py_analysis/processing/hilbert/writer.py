from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py
import mne
import numpy as np
import pybv

from gin_bids_py_analysis.bids.helpers import build_bids_path
from gin_bids_py_analysis.processing.base import (
    BaseProcessingResult,
    BaseProcessingWriter,
    _PROVENANCE_ENTITIES,
)

from .result import HilbertProcessingResult

# Extensions that trigger BrainVision output instead of HDF5.
_BV_EXTENSIONS = frozenset({".vhdr", ".eeg"})


def _package_version() -> str:
    try:
        return version("gin-bids-py-analysis")
    except PackageNotFoundError:
        return "unknown"


def _read_bv_events(
    source_path: Path,
    original_fs: float,
    downsampled_fs: float,
) -> list[dict] | None:
    """Read annotations from a BrainVision source file and remap to the downsampled rate.

    Markers are imported via MNE from the ``.vmrk`` file paired with the
    source ``.vhdr`` header.  Sample indices are remapped from *original_fs*
    to *downsampled_fs* so they match the envelope time axis.

    MNE encodes BrainVision marker type and description as ``"type/description"``
    (e.g. ``"Stimulus/S  1"``).  This function splits them back and converts the
    numeric part to an ``int`` (required by pybv for ``Stimulus`` / ``Response``
    markers).  Non-numeric descriptions are written as ``"Comment"`` type.
    Structural ``New Segment`` annotations added by MNE are silently dropped.

    Args:
        source_path:    Path to the source BIDS iEEG file.  Only ``.vhdr``
                        files are processed; any other extension returns ``None``.
        original_fs:    Sampling frequency of the raw recording in Hz.
        downsampled_fs: Sampling frequency of the envelope output in Hz.

    Returns:
        ``None`` if the source is not a BrainVision file or has no annotations.
        Otherwise a **list of dicts** ready for :func:`pybv.write_brainvision`,
        each containing keys ``onset``, ``duration``, ``type``, and
        ``description``.
    """
    if source_path.suffix.lower() != ".vhdr":
        return None

    raw = mne.io.read_raw_brainvision(str(source_path), preload=False, verbose=False)

    if len(raw.annotations) == 0:
        return None

    events_out: list[dict] = []
    for ann in raw.annotations:
        full_desc: str = ann["description"]

        # MNE adds a "New Segment" annotation at t=0; skip it.
        if full_desc.startswith("New Segment"):
            continue

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
        events_out.append(
            {
                "onset": onset_samples,
                "duration": duration_samples,
                "type": ann_type,
                "description": description,
            }
        )

    return events_out if events_out else None


class HilbertProcessingWriter(BaseProcessingWriter):
    """Writes a :class:`HilbertProcessingResult` to BIDS derivatives.

    The output format is selected automatically from
    :attr:`~gin_bids_py_analysis.processing.hilbert.HilbertWriterParams.output_extension`:

    * **HDF5** (``.h5``, default) — a single file containing all smoothing
      windows stacked along the first axis.  See :meth:`_write_hdf5` for the
      full schema.
    * **BrainVision** (``.vhdr`` or ``.eeg``) — one file triplet
      (``.vhdr`` / ``.vmrk`` / ``.eeg``) is written per smoothing window.
      Files are named ``…_smwin-{N}ms_<suffix>.vhdr``.  If the source file is
      itself a BrainVision file, its annotations are imported via MNE and
      remapped to the envelope sampling rate.

    Pass a :class:`~gin_bids_py_analysis.processing.hilbert.HilbertWriterParams`
    instance to the constructor — only ``bids_root`` is required.

    Example — HDF5 (default)::

        writer = HilbertProcessingWriter(
            HilbertWriterParams(bids_root=Path("/data/my_study"))
        )
        out_path = writer.write(result)   # → …_hilbert.h5

    Example — BrainVision::

        writer = HilbertProcessingWriter(
            HilbertWriterParams(bids_root=Path("/data/my_study"), output_extension=".vhdr")
        )
        out_path = writer.write(result)   # → …_smwin-0ms_hilbert.vhdr (smallest window)

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

    def write(self, result: BaseProcessingResult) -> Path:
        """Write *result* to disk, dispatching on ``output_extension``.

        BrainVision output is triggered when ``output_extension`` is
        ``.vhdr`` or ``.eeg``; all other extensions fall through to the
        base-class HDF5 path.

        Returns:
            For HDF5: path to the single ``.h5`` file.
            For BrainVision: path to the ``.vhdr`` header of the
            **smallest** smoothing window.
        """
        if self.params.output_extension.lower() in _BV_EXTENSIONS:
            paths = self._write_brainvision_all_windows(result)
            return paths[0]
        return super().write(result)

    # ------------------------------------------------------------------
    # Abstract method implementation (required by BaseProcessingWriter)
    # ------------------------------------------------------------------

    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        """Serialize *result* as HDF5 (called by the base-class :meth:`write`)."""
        if not isinstance(result, HilbertProcessingResult):
            raise TypeError(
                f"Expected HilbertProcessingResult, got {type(result).__name__!r}"
            )
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
        self, result: BaseProcessingResult
    ) -> list[Path]:
        """Write one BrainVision file triplet per smoothing window.

        For each smoothing window *N* (sorted ascending) pybv writes three files:

        * ``…_smwin-{N}ms_<suffix>.vhdr`` — text header
        * ``…_smwin-{N}ms_<suffix>.vmrk`` — marker file
        * ``…_smwin-{N}ms_<suffix>.eeg``  — binary float32 data

        Annotations are read from the source file (if it is a BrainVision
        ``.vhdr``) via MNE and remapped to the envelope sampling rate using
        :func:`_read_bv_events`.

        Args:
            result: A :class:`HilbertProcessingResult` to serialise.

        Returns:
            List of ``.vhdr`` paths, one per smoothing window, sorted by
            window size ascending (smallest window first).
        """
        if not isinstance(result, HilbertProcessingResult):
            raise TypeError(
                f"Expected HilbertProcessingResult, got {type(result).__name__!r}"
            )

        # Reconstruct BIDS path using the same logic as the base-class write().
        output_root = (
            self.params.bids_root / "derivatives" / self.params.pipeline_label
        )
        primary = result.source_group.primary
        entities = {
            k: v
            for k, v in primary.entities.items()
            if k not in _PROVENANCE_ENTITIES
        }
        entities["desc"] = self.params.pipeline_label

        # Build canonical base path with .vhdr extension to obtain folder + stem.
        base_path = build_bids_path(
            entities=entities,
            root=output_root,
            suffix=self.params.output_suffix,
            extension=".vhdr",
        )
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Strip the trailing _{suffix} from the stem so we can inject _smwin-{N}ms
        # before it.  rfind ensures we strip only the last (suffix) occurrence even
        # if the suffix string appears elsewhere in the entity block.
        suffix = self.params.output_suffix
        entity_stem = base_path.stem[: base_path.stem.rfind(f"_{suffix}")]

        # Import markers from the source file once; reused for every window.
        events = _read_bv_events(
            source_path=primary.path,
            original_fs=result.original_fs,
            downsampled_fs=result.downsampled_fs,
        )

        sorted_windows = sorted(result.smoothed.keys())
        out_paths: list[Path] = []

        for w in sorted_windows:
            fname_base = f"{entity_stem}_smwin-{w}ms_{suffix}"
            pybv.write_brainvision(
                data=result.smoothed[w] * 1e-6,  # convert from µV to V for pybv
                sfreq=result.downsampled_fs,
                ch_names=result.channel_names,
                fname_base=fname_base,
                folder_out=str(base_path.parent),
                events=events,
                unit="µV",
                overwrite=True,
            )
            out_paths.append(base_path.parent / f"{fname_base}.vhdr")

        return out_paths

