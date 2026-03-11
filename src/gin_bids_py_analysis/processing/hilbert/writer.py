from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py
import numpy as np

from gin_bids_py_analysis.processing.base import BaseProcessingResult, BaseProcessingWriter

from .result import HilbertProcessingResult


def _package_version() -> str:
    try:
        return version("gin-bids-py-analysis")
    except PackageNotFoundError:
        return "unknown"


class HilbertProcessingWriter(BaseProcessingWriter):
    """Writes a :class:`HilbertProcessingResult` to a BIDS derivatives HDF5 file.

    Path construction, directory creation, and BIDS naming are all handled
    by :meth:`BaseProcessingWriter.write`.  This class only implements
    :meth:`_write_data`.

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

    Pass a :class:`~gin_bids_py_analysis.processing.hilbert.HilbertWriterParams`
    instance to the constructor — only ``bids_root`` is required.

    Example::

        from pathlib import Path
        from gin_bids_py_analysis.processing.hilbert import (
            HilbertWriterParams, HilbertProcessingWriter,
        )

        writer = HilbertProcessingWriter(
            HilbertWriterParams(bids_root=Path("/data/my_study"))
        )
        out_path = writer.write(result)
    """

    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        if not isinstance(result, HilbertProcessingResult):
            raise TypeError(
                f"Expected HilbertProcessingResult, got {type(result).__name__!r}"
            )

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

