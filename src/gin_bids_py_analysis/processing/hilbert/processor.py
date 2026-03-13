from __future__ import annotations

import os

import numpy as np

try:
    import pyfftw
    _PYFFTW_AVAILABLE = True
except ImportError:
    _PYFFTW_AVAILABLE = False
    
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.base import BaseProcessing
from gin_bids_py_analysis.data.loader import load_ieeg

from .dsp import process_all_channels
from .params import HilbertParams
from .result import HilbertProcessingResult


def _fftw_threads_for_worker() -> int:
    """Return how many pyfftw threads this worker should use.

    When joblib spawns N worker processes each should use cpu_count/N threads
    so the total thread count stays close to the number of physical cores.
    joblib exposes the worker count via the LOKY_MAX_CPU_COUNT / joblib env
    variables; if we can't determine it we default to all cores.
    """
    cpu = os.cpu_count() or 1
    # joblib sets this env var in each worker process
    n_workers_str = os.environ.get("LOKY_MAX_CPU_COUNT") or os.environ.get("JOBLIB_NPROCS")
    try:
        n_workers = int(n_workers_str) if n_workers_str else 1
    except ValueError:
        n_workers = 1
    return max(1, cpu // n_workers)


def _select_channels_for_montage(
    data: np.ndarray,
    channel_names: list[str],
    selected_names: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    """Subset *data* and *channel_names* using the same row indices.

    The selection preserves the original file order rather than the order of
    *selected_names*.  This keeps channel adjacency intact for bipolar montage
    construction and prevents name/data mismatches when only a subset of
    channels should be processed.
    """
    if not selected_names:
        return data, list(channel_names)

    selected_lookup = set(selected_names)
    keep_indices = [
        idx for idx, name in enumerate(channel_names)
        if name in selected_lookup
    ]

    if not keep_indices:
        raise ValueError(
            "channels_for_montage did not match any input channels: "
            f"{selected_names!r}"
        )

    return data[keep_indices, :], [channel_names[idx] for idx in keep_indices]


class HilbertProcessing(BaseProcessing):
    """Hilbert-band envelope processor.

    Instantiate with a :class:`HilbertParams` object; then call
    :meth:`run` (process + write) or :meth:`execute` (process only) with a
    list of BIDS files or file groups.

    Example::

        from pathlib import Path
        from gin_bids_py_analysis.processing.hilbert import (
            HilbertParams, HilbertProcessing,
            HilbertProcessingWriter, HilbertWriterParams,
        )

        params = HilbertParams(f_min=50, f_max=150, f_step=10)
        processor = HilbertProcessing(params)
        writer = HilbertProcessingWriter(HilbertWriterParams(bids_root=Path("/data")))
        out_paths = processor.run(ieeg_files, writer)
    """

    def __init__(self, params: HilbertParams) -> None:
        self.params = params

    def process_group(self, group: BIDSFileGroup) -> HilbertProcessingResult:
        """Run the Hilbert-band envelope pipeline on one file group.

        Reads the iEEG file via MNE (format auto-detected from the extension),
        extracts the sampling frequency and channel data, applies the full
        pipeline defined in :mod:`.dsp`, and returns a
        :class:`HilbertProcessingResult`.

        For single-file analyses ``group.primary`` is the iEEG file.
        ``group.secondaries`` is not used by this processor.

        Args:
            group: The file group to process.

        Returns:
            :class:`HilbertProcessingResult` with ``smoothed`` arrays,
            channel names, and frequency metadata.
        """
        raw = load_ieeg(group.primary)
        fs: float = raw.info["sfreq"]

        if _PYFFTW_AVAILABLE:
            pyfftw.config.NUM_THREADS = _fftw_threads_for_worker()

        # get_data() returns shape [n_channels, n_times] as float64
        data: np.ndarray = raw.get_data().astype(np.float32)
        ch_names: list[str] = list(raw.ch_names)
        data, ch_names = _select_channels_for_montage(
            data,
            ch_names,
            self.params.channels_for_montage,
        )

        smoothed, montaged_names, bins = process_all_channels(
            data_2d=data,
            channel_names=ch_names,
            fs=fs,
            params=self.params,
        )

        if self.params.do_downsample:
            factor = int(fs) // int(self.params.downsampled_frequency_hz)
            downsampled_fs = fs / factor
        else:
            downsampled_fs = fs

        return HilbertProcessingResult(
            source_group=group,
            smoothed=smoothed,
            channel_names=montaged_names,
            bins=bins,
            downsampled_fs=downsampled_fs,
            original_fs=fs,
            metadata={
                "montage_mode": self.params.montage_mode.value,
                "centered": self.params.centered,
                "unit": "percent" if self.params.do_normalize_percent else "amplitude",
            },
            original_events=raw.annotations,
        )
