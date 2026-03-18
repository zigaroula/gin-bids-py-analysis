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
from ..utils.channels import select_channels_for_montage
from ..utils.multithreading import get_threads_for_worker


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

    def __init__(self, params: HilbertParams, verbose: bool = True) -> None:
        self.params = params
        self.verbose = verbose

    def process_group(self, group: BIDSFileGroup, progress_tracking_position: int = 0) -> HilbertProcessingResult:
        """Run the Hilbert-band envelope pipeline on one file group.

        Reads the iEEG file via MNE (format auto-detected from the extension),
        extracts the sampling frequency and channel data, applies the full
        pipeline defined in :mod:`.dsp`, and returns a
        :class:`HilbertProcessingResult`.

        For single-file analyses ``group.primary`` is the iEEG file.
        ``group.secondaries`` is not used by this processor.

        Args:
            group: The file group to process.
            progress_tracking_position: Optional position index for progress tracking (e.g. with tqdm).

        Returns:
            :class:`HilbertProcessingResult` with ``smoothed`` arrays,
            channel names, and frequency metadata.
        """
        raw = load_ieeg(group.primary)
        fs: float = raw.info["sfreq"]

        if _PYFFTW_AVAILABLE:
            pyfftw.config.NUM_THREADS = get_threads_for_worker()

        # get_data() returns shape [n_channels, n_times] as float64
        data: np.ndarray = raw.get_data().astype(np.float32)
        ch_names: list[str] = list(raw.ch_names)
        data, ch_names = select_channels_for_montage(
            data,
            ch_names,
            self.params.channels_for_montage,
            self.params.channels_to_exclude_for_montage,
        )

        smoothed, montaged_names, bins = process_all_channels(
            data_2d=data,
            channel_names=ch_names,
            fs=fs,
            params=self.params,
            verbose=self.verbose,
            desc=group.primary.path.name,
            progress_tracking_position=progress_tracking_position,
        )

        downsampled_fs = self.params.downsampled_frequency_hz if self.params.downsampled_frequency_hz is not None else fs

        return HilbertProcessingResult(
            source_group=group,
            smoothed=smoothed,
            channel_names=montaged_names,
            bins=bins,
            downsampled_fs=downsampled_fs,
            original_fs=fs,
            metadata={
                "montage_mode": self.params.montage_mode.value,
                "centered": self.params.normalization_mode.is_centered,
                "unit_label": self.params.normalization_mode.unit_label,
                "unit": self.params.normalization_mode.unit,
                "scale_factor": self.params.normalization_mode.scale_factor,
            },
            original_events=raw.annotations,
        )
